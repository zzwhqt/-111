from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .core import ClipEncoder, DATA_DIR, IndexStore, encode_video, validate_remote_url
from .label_study import POOL_DIR as LABEL_POOL_DIR
from .label_study import router as label_router
from .multimodal_study import router as multimodal_study_router
from .scene_errors import router as scene_errors_router


app = FastAPI(title="Task Scene Duplicate Search MVP", version="0.2.0")
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAIL_DIR)), name="thumbnails")
LABEL_THUMBNAIL_DIR = LABEL_POOL_DIR / "thumbnails"
LABEL_THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/label-thumbnails",
    StaticFiles(directory=str(LABEL_THUMBNAIL_DIR)),
    name="label-thumbnails",
)
app.include_router(label_router)
app.include_router(multimodal_study_router)
app.include_router(scene_errors_router)


@lru_cache(maxsize=1)
def encoder() -> ClipEncoder:
    return ClipEncoder()


@lru_cache(maxsize=1)
def store() -> IndexStore:
    return IndexStore(DATA_DIR)


class UrlSearchRequest(BaseModel):
    url: str = Field(min_length=1, max_length=8192)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, gt=0)
    fps: float = Field(default=30.0, gt=0, le=240)
    # Kept for API compatibility. Frame bounds take priority when supplied.
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    description: str | None = Field(default=None, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)


def _allowed_oss_buckets() -> set[str]:
    return {
        item.strip()
        for item in os.getenv("DEDUP_ALLOWED_OSS_BUCKETS", "egoscale-v3").split(",")
        if item.strip()
    }


def _oss_signing_ready() -> bool:
    return bool(os.getenv("OSS_ACCESS_KEY_ID") and os.getenv("OSS_ACCESS_KEY_SECRET") and os.getenv("OSS_ENDPOINT"))


def _sign_oss_url(uri: str) -> str:
    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    object_key = parsed.path.lstrip("/")
    if parsed.scheme != "oss" or not bucket_name or not object_key:
        raise ValueError("OSS address must use oss://bucket/object-key")
    if bucket_name not in _allowed_oss_buckets():
        raise ValueError(f"OSS bucket is not allowed: {bucket_name}")
    if not _oss_signing_ready():
        raise ValueError("OSS signing is not configured on this server")
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError("oss2 is not installed") from exc
    auth = oss2.Auth(os.environ["OSS_ACCESS_KEY_ID"], os.environ["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, os.environ["OSS_ENDPOINT"], bucket_name)
    ttl = min(max(int(os.getenv("DEDUP_SIGNED_URL_TTL", "900")), 60), 3600)
    return bucket.sign_url("GET", object_key, ttl, slash_safe=True)


def _resolve_video_source(source: str) -> str:
    source = source.strip()
    if source.startswith("oss://"):
        return _sign_oss_url(source)
    return validate_remote_url(source)


def _query_bounds(request: UrlSearchRequest) -> tuple[float | None, float | None, dict[str, Any]]:
    has_frames = request.start_frame is not None or request.end_frame is not None
    if has_frames:
        if request.start_frame is None or request.end_frame is None:
            raise HTTPException(400, "start_frame and end_frame must be supplied together")
        if request.end_frame <= request.start_frame:
            raise HTTPException(400, "end_frame must be greater than start_frame")
        start = request.start_frame / request.fps
        end = request.end_frame / request.fps
        return start, end, {
            "start_frame": request.start_frame,
            "end_frame": request.end_frame,
            "fps": request.fps,
            "start_seconds": round(start, 6),
            "end_seconds": round(end, 6),
            "frame_interval": "[start_frame, end_frame)",
        }
    if (request.start_seconds is None) != (request.end_seconds is None):
        raise HTTPException(400, "start_seconds and end_seconds must be supplied together")
    if request.start_seconds is not None and request.end_seconds is not None:
        if request.end_seconds <= request.start_seconds:
            raise HTTPException(400, "end_seconds must be greater than start_seconds")
        return request.start_seconds, request.end_seconds, {
            "start_frame": round(request.start_seconds * request.fps),
            "end_frame": round(request.end_seconds * request.fps),
            "fps": request.fps,
            "start_seconds": request.start_seconds,
            "end_seconds": request.end_seconds,
            "frame_interval": "derived from seconds",
        }
    return None, None, {"fps": request.fps, "frame_interval": "whole video"}


def _candidate_playback(hit: dict[str, Any], fps: float) -> dict[str, Any]:
    metadata = hit.get("metadata") or {}
    source_video = str(metadata.get("source_video") or "")
    start = float(metadata.get("start_seconds") or 0.0)
    end_raw = metadata.get("end_seconds")
    end = float(end_raw) if end_raw is not None else None
    hit["candidate_range"] = {
        "start_frame": round(start * fps),
        "end_frame": round(end * fps) if end is not None else None,
        "fps": fps,
        "start_seconds": start,
        "end_seconds": end,
        "frame_interval": "[start_frame, end_frame)",
    }
    if not source_video:
        return hit
    try:
        playable_url = _resolve_video_source(source_video)
    except (ValueError, RuntimeError):
        return hit
    fragment = f"#t={start:.3f}" if end is None else f"#t={start:.3f},{end:.3f}"
    hit["video_url"] = playable_url
    hit["video_fragment_url"] = playable_url + fragment
    thumbnail = str(metadata.get("thumbnail") or "")
    if thumbnail:
        hit["thumbnail_url"] = "/" + quote(thumbnail.lstrip("/"), safe="/")
    return hit


def _run_search(
    source: str,
    top_k: int,
    start: float | None = None,
    end: float | None = None,
    description: str | None = None,
    query_range: dict[str, Any] | None = None,
    display_fps: float = 30.0,
) -> dict:
    if (start is None) != (end is None):
        raise HTTPException(400, "start_seconds and end_seconds must be supplied together")
    if start is not None and end is not None and end <= start:
        raise HTTPException(400, "end_seconds must be greater than start_seconds")
    temp_root: Path | None = None
    try:
        task_vector, frame_vectors, thumbnail = encode_video(encoder(), source, start, end)
        temp_root = thumbnail.parent
        query_text = encoder().encode_text(description.strip()) if description and description.strip() else None
        hits = [
            _candidate_playback(item.as_dict(), display_fps)
            for item in store().search(task_vector, frame_vectors, top_k, query_text=query_text)
        ]
        return {
            "index_size": len(store().metadata),
            "score_semantics": "0-100 corpus-normalized retrieval score; it is not a calibrated duplicate probability",
            "query_mode": "video_plus_task_text" if query_text is not None else "video_only_corpus_normalized",
            "query_range": query_range,
            "results": hits,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"video search failed: {exc}") from exc
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_size": len(store().metadata),
        "model_loaded": encoder.cache_info().currsize > 0,
        "oss_signing_ready": _oss_signing_ready(),
    }


@app.get("/stats")
def stats() -> dict:
    current = store()
    return {
        "index_size": len(current.metadata),
        "vector_dimension": int(current.task_vectors.shape[1]),
        "frames_per_task": int(current.frame_vectors.shape[1]),
        "text_vector_dimension": int(current.text_vectors.shape[1]),
        "index_type": "Faiss IndexFlatIP (exact cosine search)",
        "visual_random_negative_p95": round(current.visual_p95, 6),
        "text_random_negative_p95": round(current.text_p95, 6),
    }


@app.post("/search/upload")
def search_upload(
    video: UploadFile = File(...),
    top_k: int = Form(default=10),
    description: str | None = Form(default=None),
) -> dict:
    suffix = Path(video.filename or "query.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(400, "unsupported video extension")
    fd, name = tempfile.mkstemp(prefix="task-query-", suffix=suffix)
    os.close(fd)
    path = Path(name)
    try:
        with path.open("wb") as output:
            shutil.copyfileobj(video.file, output, length=1024 * 1024)
        if path.stat().st_size > 500 * 1024 * 1024:
            raise HTTPException(413, "video exceeds 500MB")
        return _run_search(str(path), min(max(top_k, 1), 50), description=description)
    finally:
        path.unlink(missing_ok=True)


@app.post("/search/url")
def search_url(request: UrlSearchRequest) -> dict:
    try:
        url = _resolve_video_source(request.url)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    start, end, query_range = _query_bounds(request)
    return _run_search(
        url,
        request.top_k,
        start,
        end,
        request.description,
        query_range=query_range,
        display_fps=request.fps,
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _HOME_HTML


_HOME_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task 场景重复检索</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#111a2b;--line:#26334a;--muted:#93a4bd;--text:#edf3ff;--blue:#5994ff;--cyan:#5eead4}
*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,"PingFang SC",sans-serif;background:radial-gradient(circle at 15% -5%,#19315a 0,transparent 32%),var(--bg);color:var(--text)}
main{max-width:1180px;margin:0 auto;padding:44px 22px 80px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.16em;text-transform:uppercase;font-weight:700}h1{font-size:clamp(30px,5vw,52px);margin:10px 0;line-height:1.08}p{color:var(--muted);line-height:1.65}.panel{background:#111a2be8;border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 18px 60px #0006;backdrop-filter:blur(12px)}
.grid{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:14px}.wide{grid-column:1/-1}label{display:block;color:#becbe0;font-size:13px;margin-bottom:7px}input,textarea,select{width:100%;border:1px solid #34435f;border-radius:10px;padding:12px 13px;background:#0a1120;color:var(--text);font:inherit;outline:none}input:focus,textarea:focus,select:focus{border-color:var(--blue);box-shadow:0 0 0 3px #5994ff22}textarea{resize:vertical;min-height:78px}button{border:0;border-radius:11px;padding:13px 20px;background:linear-gradient(135deg,#3977ff,#6d5dfc);color:white;font-weight:700;font-size:15px;cursor:pointer;min-width:150px}button:disabled{opacity:.55;cursor:wait}.actions{display:flex;align-items:center;gap:16px}.status{font-size:13px;color:var(--muted)}
.summary{margin:24px 0 14px;display:flex;align-items:center;justify-content:space-between;gap:16px}.summary h2{margin:0;font-size:21px}.legend{font-size:12px;color:var(--muted)}.results{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.result{overflow:hidden;padding:0}.media{aspect-ratio:16/9;background:#050912;position:relative}.media video{width:100%;height:100%;object-fit:contain;background:#050912}.rank{position:absolute;left:12px;top:12px;background:#050912d9;border:1px solid #ffffff2e;border-radius:999px;padding:5px 9px;font-size:12px;z-index:2}.content{padding:19px}.topline{display:flex;align-items:flex-start;justify-content:space-between;gap:14px}.title{font-weight:750;font-size:17px;line-height:1.4}.score{white-space:nowrap;font-size:28px;font-weight:800;color:var(--cyan)}.score small{font-size:12px;color:var(--muted);font-weight:500}.badge{display:inline-flex;margin-top:8px;border:1px solid;border-radius:999px;padding:3px 8px;font-size:11px;text-transform:uppercase}.high{color:#ff858e;background:#ff4d5e14}.review{color:#ffd166;background:#ffd16612}.low{color:#74ddb8;background:#52d39a12}.desc{font-size:13px;color:#adbbcf;line-height:1.55;margin:12px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.metric{background:#0a1120;border:1px solid #202d44;border-radius:9px;padding:9px}.metric b{display:block;font-size:13px}.metric span{font-size:10px;color:var(--muted)}.range{font-size:11px;color:var(--muted);margin-top:12px;word-break:break-all}.empty{grid-column:1/-1;padding:42px;text-align:center;color:var(--muted)}.error{color:#ff8d96}
@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:1/-1}.results{grid-template-columns:1fr}}@media(max-width:520px){main{padding:28px 14px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.panel{padding:17px}.metrics{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<div class="eyebrow">CPU-only · CLIP + Faiss</div><h1>Task 场景重复检索</h1><p>输入 OSS 视频地址与 Task 帧范围，系统抽取该片段的 8 帧并在 1000 条资产中返回 Top-K 相似场景。帧区间采用 <code>[start_frame, end_frame)</code>，相似度是库内归一化检索分，不等同于重复概率。 <a href="/label" style="color:#5eead4">进入2000条Task人工盲标工作台 →</a></p>
<form id="searchForm" class="panel"><div class="grid">
<div class="wide"><label for="videoUrl">视频地址（oss:// 或 OSS HTTPS）</label><input id="videoUrl" required placeholder="oss://egoscale-v3/path/to/video.mp4"></div>
<div><label for="startFrame">开始帧（包含）</label><input id="startFrame" type="number" min="0" step="1" value="0" required></div>
<div><label for="endFrame">结束帧（不包含）</label><input id="endFrame" type="number" min="1" step="1" value="300" required></div>
<div><label for="fps">FPS</label><input id="fps" type="number" min="0.01" max="240" step="0.01" value="30" required></div>
<div><label for="topK">返回数量</label><select id="topK"><option>5</option><option selected>10</option><option>20</option></select></div>
<div class="wide"><label for="description">Task 描述（可选，填写后启用文本语义精排）</label><textarea id="description" placeholder="例如：工人在装配台拿取电动螺丝刀并拧紧零件"></textarea></div>
<div class="wide actions"><button id="submitButton" type="submit">开始匹配</button><span id="status" class="status">等待输入</span></div>
</div></form>
<div class="summary"><h2 id="resultTitle">匹配结果</h2><div class="legend">HIGH：强重复预警 · REVIEW：建议人工复核 · LOW：低风险</div></div><section id="results" class="results"><div class="panel empty">提交一个 Task 后在这里查看相似视频</div></section>
</main><script>
const form=document.getElementById('searchForm'),results=document.getElementById('results'),statusEl=document.getElementById('status'),button=document.getElementById('submitButton'),title=document.getElementById('resultTitle');
const node=(tag,className,text)=>{const n=document.createElement(tag);if(className)n.className=className;if(text!==undefined)n.textContent=text;return n};
const metric=(label,value)=>{const d=node('div','metric'),b=node('b','',value),s=node('span','',label);d.append(b,s);return d};
function render(items,indexSize){results.replaceChildren();title.textContent=`匹配结果 · ${items.length} / ${indexSize}`;if(!items.length){results.append(node('div','panel empty','没有找到候选'));return}items.forEach((item,i)=>{const meta=item.metadata||{},card=node('article','panel result'),media=node('div','media'),rank=node('span','rank',`TOP ${i+1}`);media.append(rank);if(item.video_fragment_url){const video=node('video');video.controls=true;video.preload='metadata';video.src=item.video_fragment_url;if(item.thumbnail_url)video.poster=item.thumbnail_url;media.append(video)}else if(item.thumbnail_url){const img=node('img');img.src=item.thumbnail_url;img.alt='候选缩略图';img.style='width:100%;height:100%;object-fit:cover';media.append(img)}else{media.append(node('div','empty','候选视频暂不可播放'))}const content=node('div','content'),top=node('div','topline'),left=node('div'),name=node('div','title',meta.task_name||meta.task_id||item.task_id),badge=node('span',`badge ${item.warning_level}`,item.warning_level);left.append(name,badge);const score=node('div','score',String(item.similarity_percent));score.append(node('small','', ' / 100'));top.append(left,score);const desc=node('div','desc',[meta.scene,meta.details].filter(Boolean).join(' · ')||'暂无场景描述'),metrics=node('div','metrics');metrics.append(metric('整体余弦',Number(item.raw_cosine).toFixed(4)),metric('帧覆盖度',Number(item.frame_coverage).toFixed(4)),metric('库内百分位',Number(item.corpus_percentile).toFixed(2)+'%'));const r=item.candidate_range||{},range=node('div','range',`候选范围：[${r.start_frame ?? '-'}, ${r.end_frame ?? '-'}) 帧 · ${r.start_seconds ?? '-'}s → ${r.end_seconds ?? '-'}s · ${item.score_mode}`);content.append(top,desc,metrics,range);card.append(media,content);results.append(card)})}
form.addEventListener('submit',async e=>{e.preventDefault();const payload={url:document.getElementById('videoUrl').value.trim(),start_frame:Number(document.getElementById('startFrame').value),end_frame:Number(document.getElementById('endFrame').value),fps:Number(document.getElementById('fps').value),top_k:Number(document.getElementById('topK').value),description:document.getElementById('description').value.trim()||null};button.disabled=true;statusEl.className='status';statusEl.textContent='正在读取片段、抽帧、编码并匹配…';results.replaceChildren(node('div','panel empty','检索处理中，CPU 首次加载模型可能需要几十秒'));try{const response=await fetch('/search/url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(!response.ok)throw new Error(data.detail||`HTTP ${response.status}`);statusEl.textContent=`完成 · ${data.query_mode}`;render(data.results||[],data.index_size)}catch(err){statusEl.className='status error';statusEl.textContent='检索失败';results.replaceChildren(node('div','panel empty error',String(err.message||err)))}finally{button.disabled=false}});
</script></body></html>"""
