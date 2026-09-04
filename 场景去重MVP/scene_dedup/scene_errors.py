from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .label_study import _signed_video
from .multimodal_study import _scene_asset


DB_PATH = Path(
    os.getenv(
        "DEDUP_MULTIMODAL_LABEL_DB",
        "/opt/task-dedup/labeling/multimodal_study.sqlite3",
    )
)
DEVELOPMENT_QUERY_COUNT = 80
REASON_CODES = {
    "same_type_different_place",
    "action_or_tool_bias",
    "person_or_occlusion",
    "viewpoint_or_lighting",
    "missed_similar_candidate",
    "label_ambiguous",
    "not_model_error",
    "other",
}
router = APIRouter(prefix="/study/errors", tags=["scene-error-review"])


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_error_reviews (
          query_task_id TEXT PRIMARY KEY REFERENCES query_tasks(task_id),
          reason_code TEXT NOT NULL,
          note TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def _asset(task_id: str, metadata_json: str) -> dict[str, Any]:
    metadata = json.loads(metadata_json)
    asset = _scene_asset(task_id, metadata)
    asset["video_url"] = _signed_video(metadata)
    return asset


def _error_cases(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    queries = connection.execute(
        """
        SELECT task_id,display_order,metadata_json
        FROM query_tasks
        WHERE display_order<=?
        ORDER BY display_order
        """,
        (DEVELOPMENT_QUERY_COUNT,),
    ).fetchall()
    cases: list[dict[str, Any]] = []
    for query in queries:
        candidates = connection.execute(
            """
            SELECT r.task_id,r.metadata_json,p.scene_raw_cosine,a.relation_code
            FROM candidate_pairs p
            JOIN reference_assets r ON r.task_id=p.reference_task_id
            JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name='scene'
            WHERE p.query_task_id=?
            ORDER BY p.scene_raw_cosine DESC,r.task_id
            """,
            (query["task_id"],),
        ).fetchall()
        if len(candidates) != 5:
            continue
        model_top = candidates[0]
        human_best = max(
            candidates,
            key=lambda row: (int(row["relation_code"]), float(row["scene_raw_cosine"])),
        )
        model_grade = int(model_top["relation_code"])
        best_grade = int(human_best["relation_code"])
        if model_grade >= best_grade:
            continue
        cases.append(
            {
                "query_task_id": str(query["task_id"]),
                "display_order": int(query["display_order"]),
                "model_grade": model_grade,
                "best_grade": best_grade,
                "model_score": round(float(model_top["scene_raw_cosine"]), 6),
                "better_score": round(float(human_best["scene_raw_cosine"]), 6),
                "query": _asset(str(query["task_id"]), str(query["metadata_json"])),
                "model_top": _asset(str(model_top["task_id"]), str(model_top["metadata_json"])),
                "human_best": _asset(
                    str(human_best["task_id"]), str(human_best["metadata_json"])
                ),
            }
        )
    return cases


def _progress(connection: sqlite3.Connection, cases: list[dict[str, Any]]) -> dict[str, int]:
    case_ids = {case["query_task_id"] for case in cases}
    reviewed = {
        str(row[0])
        for row in connection.execute("SELECT query_task_id FROM scene_error_reviews").fetchall()
    }
    completed = len(case_ids & reviewed)
    return {
        "development_queries": DEVELOPMENT_QUERY_COUNT,
        "locked_queries": 40,
        "total_errors": len(cases),
        "reviewed_errors": completed,
        "remaining_errors": max(0, len(cases) - completed),
    }


@router.get("", response_class=HTMLResponse)
def error_home() -> str:
    return _ERROR_HTML


@router.get("/safari", response_class=HTMLResponse)
def safari_translation_home() -> str:
    """English document variant so Safari offers its built-in page translation."""
    return _ERROR_HTML_SAFARI


@router.get("/api/next")
def next_error() -> dict[str, Any]:
    with _connect() as connection:
        cases = _error_cases(connection)
        progress = _progress(connection, cases)
        reviewed = {
            str(row[0])
            for row in connection.execute("SELECT query_task_id FROM scene_error_reviews").fetchall()
        }
        case = next((item for item in cases if item["query_task_id"] not in reviewed), None)
        if case is None:
            return {"complete": True, "progress": progress}
        return {"complete": False, "case": case, "progress": progress}


class ErrorReview(BaseModel):
    query_task_id: str = Field(min_length=1, max_length=512)
    reason_code: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=1000)


@router.post("/api/review")
def review_error(review: ErrorReview) -> dict[str, Any]:
    if review.reason_code not in REASON_CODES:
        raise HTTPException(400, "invalid reason code")
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cases = _error_cases(connection)
        if review.query_task_id not in {item["query_task_id"] for item in cases}:
            raise HTTPException(404, "development error case not found")
        connection.execute(
            """
            INSERT INTO scene_error_reviews(query_task_id,reason_code,note,created_at,updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(query_task_id) DO UPDATE SET
              reason_code=excluded.reason_code,
              note=excluded.note,
              updated_at=excluded.updated_at
            """,
            (review.query_task_id, review.reason_code, review.note.strip(), now, now),
        )
        connection.commit()
    return next_error()


@router.get("/api/summary")
def error_summary() -> dict[str, Any]:
    with _connect() as connection:
        cases = _error_cases(connection)
        progress = _progress(connection, cases)
        rows = connection.execute(
            """
            SELECT reason_code,COUNT(*) AS count
            FROM scene_error_reviews
            GROUP BY reason_code ORDER BY count DESC,reason_code
            """
        ).fetchall()
        return {
            "progress": progress,
            "reason_counts": {str(row["reason_code"]): int(row["count"]) for row in rows},
        }


_ERROR_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>场景模型错题分析</title><style>
:root{color-scheme:dark;--bg:#07101d;--panel:#101c2e;--line:#2a3b55;--text:#edf5ff;--muted:#91a6c0;--cyan:#58e1ca;--blue:#5d8dff;--red:#ef6b78}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#17375e 0,transparent 31%),var(--bg);color:var(--text);font-family:Inter,system-ui,"PingFang SC",sans-serif}header{position:sticky;top:0;z-index:10;background:#07101dee;border-bottom:1px solid var(--line)}.bar,main{max-width:1500px;margin:auto;padding:15px 22px}.bar{display:flex;align-items:center;gap:16px}.bar b{white-space:nowrap}.progress{height:8px;flex:1;background:#1a2940;border-radius:99px;overflow:hidden}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--blue))}.muted{color:var(--muted);font-size:13px}main{padding-top:25px}h1{margin:0 0 8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:15px;overflow:hidden}.card h2{font-size:15px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#020610}.body{padding:11px}.score{font-size:12px;color:var(--muted);margin-bottom:8px}.desc{font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;background:#0a1424;border-radius:9px;padding:9px}.replay{margin-top:8px;width:100%;border:1px solid #5d8dff66;border-radius:8px;padding:8px;background:#172b4c;color:#cbd9ff;cursor:pointer}.review{margin-top:16px;padding:16px;background:var(--panel);border:1px solid var(--line);border-radius:15px}.reasons{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.reasons button{border:1px solid var(--line);border-radius:9px;padding:10px 6px;background:#152239;color:var(--text);cursor:pointer}.reasons button.selected{border-color:var(--cyan);background:#174b53}textarea{width:100%;margin-top:10px;min-height:70px;border:1px solid var(--line);border-radius:9px;background:#091321;color:var(--text);padding:10px}.submit{margin-top:10px;border:0;border-radius:10px;padding:12px 20px;background:linear-gradient(135deg,#187b71,#416fdf);color:white;font-weight:800;cursor:pointer}.submit:disabled{opacity:.4}.empty{text-align:center;padding:100px 20px}.summary{white-space:pre-wrap;margin-top:12px;color:var(--muted)}@media(max-width:950px){.grid{grid-template-columns:1fr}.reasons{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><div class="bar"><b>场景模型错题分析</b><div class="progress"><i id="bar"></i></div><span class="muted" id="count">读取中</span></div></header><main><h1>模型第一名为什么不是人工最优？</h1><div class="muted">这里只展示前80条开发数据中的排序错题；后40条测试数据保持锁定。</div><div id="workspace"></div></main><script>
const labels={same_type_different_place:'同类场景，但不是同一地点',action_or_tool_bias:'被动作或工具干扰',person_or_occlusion:'人物/遮挡影响',viewpoint_or_lighting:'角度或光线变化',missed_similar_candidate:'漏掉更相似的候选',label_ambiguous:'人工标签有歧义',not_model_error:'这条不算模型错误',other:'其他原因'};let current=null,reason='';const workspace=document.getElementById('workspace'),count=document.getElementById('count'),bar=document.getElementById('bar');
function video(asset){const v=document.createElement('video');v.controls=true;v.playsInline=true;v.preload='metadata';v.src=asset.video_url;v.dataset.start=String(asset.start_seconds||0);v.defaultPlaybackRate=2;v.playbackRate=2;v.addEventListener('loadedmetadata',()=>{v.currentTime=Number(v.dataset.start||0);v.playbackRate=2});return v}
function desc(asset){return ['场景：'+(asset.scene||'暂无'),'任务：'+(asset.task_name||'暂无'),'详情：'+(asset.details||'暂无')].join('\n')}
function card(title,asset,score){const c=document.createElement('section');c.className='card';const h=document.createElement('h2');h.textContent=title;const v=video(asset),body=document.createElement('div');body.className='body';const s=document.createElement('div');s.className='score';s.textContent=score||'';const d=document.createElement('div');d.className='desc';d.textContent=desc(asset);const b=document.createElement('button');b.className='replay';b.textContent='↺ 从Task开头播放';b.onclick=()=>{v.currentTime=Number(v.dataset.start||0);v.play()};body.append(s,d,b);c.append(h,v,body);return c}
function update(p){count.textContent=`已分析 ${p.reviewed_errors}/${p.total_errors} · 锁定测试40条`;bar.style.width=(p.total_errors?100*p.reviewed_errors/p.total_errors:0)+'%'}
function render(data){update(data.progress);if(data.complete){fetch('/study/errors/api/summary').then(r=>r.json()).then(s=>{workspace.innerHTML='<div class="card empty"><h2>错题分析已完成</h2><div class="summary">'+JSON.stringify(s.reason_counts,null,2)+'</div></div>'});return}current=data.case;reason='';const grid=document.createElement('div');grid.className='grid';grid.append(card('查询视频',current.query,''),card('模型第一名',current.model_top,`模型余弦 ${current.model_score} · 人工等级 ${current.model_grade}`),card('人工认为更好的候选',current.human_best,`模型余弦 ${current.better_score} · 人工等级 ${current.best_grade}`));const review=document.createElement('section');review.className='review';const reasons=document.createElement('div');reasons.className='reasons';const buttons=[];for(const [code,text] of Object.entries(labels)){const b=document.createElement('button');b.textContent=text;b.onclick=()=>{reason=code;buttons.forEach(x=>x.classList.toggle('selected',x===b));submit.disabled=false};buttons.push(b);reasons.append(b)}const note=document.createElement('textarea');note.placeholder='可选：补充具体原因';const submit=document.createElement('button');submit.className='submit';submit.disabled=true;submit.textContent='保存原因并查看下一条';submit.onclick=async()=>{submit.disabled=true;const r=await fetch('/study/errors/api/review',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({query_task_id:current.query_task_id,reason_code:reason,note:note.value})});render(await r.json())};review.append(reasons,note,submit);workspace.replaceChildren(grid,review)}
fetch('/study/errors/api/next').then(r=>r.json()).then(render).catch(e=>{workspace.innerHTML='<div class="card empty">加载失败：'+e.message+'</div>'});
</script></body></html>"""


_ERROR_HTML_SAFARI = (
    _ERROR_HTML.replace('<html lang="zh-CN">', '<html lang="en">')
    .replace('<body>', '<body translate="no">')
    .replace(
        "d.className='desc';d.textContent=desc(asset)",
        "d.className='desc';d.lang='en';d.setAttribute('translate','yes');d.textContent=desc(asset)",
    )
)
