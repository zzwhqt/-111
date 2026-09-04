from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from .core import l2_normalize
from .label_study import POOL_DIR, _signed_video
from .multimodal_study import _scene_asset


DB_PATH = Path(
    os.getenv(
        "DEDUP_MULTIMODAL_LABEL_DB",
        "/opt/task-dedup/labeling/multimodal_study.sqlite3",
    )
)
TEXT_WEIGHT = float(os.getenv("DEDUP_RERANK_TEXT_WEIGHT", "0.70"))
VISUAL_WEIGHT = float(os.getenv("DEDUP_RERANK_VISUAL_WEIGHT", "0.30"))
DEVELOPMENT_QUERY_COUNT = 80
router = APIRouter(prefix="/study/rerank", tags=["text-heavy-rerank"])


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


@lru_cache(maxsize=1)
def _vectors() -> tuple[np.ndarray, np.ndarray]:
    task_vectors = l2_normalize(np.load(POOL_DIR / "task_vectors.npy").astype(np.float32))
    text_vectors = l2_normalize(np.load(POOL_DIR / "text_vectors.npy").astype(np.float32))
    return task_vectors, text_vectors


def _weights() -> tuple[float, float]:
    total = max(1e-12, TEXT_WEIGHT + VISUAL_WEIGHT)
    return TEXT_WEIGHT / total, VISUAL_WEIGHT / total


def _asset(task_id: str, metadata_json: str) -> dict[str, Any]:
    metadata = json.loads(metadata_json)
    asset = _scene_asset(task_id, metadata)
    asset["video_url"] = _signed_video(metadata)
    return asset


@router.get("", response_class=HTMLResponse)
def rerank_home() -> str:
    return _RERANK_HTML


@router.get("/api/query")
def rerank_query(
    position: int = Query(default=1, ge=1, le=DEVELOPMENT_QUERY_COUNT),
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    with _connect() as connection:
        query = connection.execute(
            """
            SELECT task_id,vector_index,display_order,metadata_json
            FROM query_tasks WHERE display_order=?
            """,
            (position,),
        ).fetchone()
        if query is None:
            raise HTTPException(404, "development query not found")
        references = connection.execute(
            """
            SELECT task_id,vector_index,display_index,metadata_json
            FROM reference_assets ORDER BY display_index
            """
        ).fetchall()
    task_vectors, text_vectors = _vectors()
    query_index = int(query["vector_index"])
    reference_indices = np.asarray([int(row["vector_index"]) for row in references], dtype=np.int64)
    visual_scores = task_vectors[reference_indices] @ task_vectors[query_index]
    text_scores = text_vectors[reference_indices] @ text_vectors[query_index]
    text_weight, visual_weight = _weights()
    combined_scores = text_weight * text_scores + visual_weight * visual_scores
    order = np.argsort(-combined_scores, kind="stable")[:top_k]
    candidates = []
    for rank, reference_position in enumerate(order, start=1):
        idx = int(reference_position)
        row = references[idx]
        candidate = _asset(str(row["task_id"]), str(row["metadata_json"]))
        candidate.update(
            {
                "rank": rank,
                "combined_score": round(float(combined_scores[idx]), 6),
                "text_score": round(float(text_scores[idx]), 6),
                "visual_score": round(float(visual_scores[idx]), 6),
            }
        )
        candidates.append(candidate)
    old_top_position = int(np.argmax(visual_scores))
    old_top_row = references[old_top_position]
    old_top = _asset(str(old_top_row["task_id"]), str(old_top_row["metadata_json"]))
    old_top.update(
        {
            "visual_score": round(float(visual_scores[old_top_position]), 6),
            "text_score": round(float(text_scores[old_top_position]), 6),
        }
    )
    return {
        "position": position,
        "development_queries": DEVELOPMENT_QUERY_COUNT,
        "locked_queries": 40,
        "weights": {"text": text_weight, "visual": visual_weight},
        "query": _asset(str(query["task_id"]), str(query["metadata_json"])),
        "old_visual_top1": old_top,
        "candidates": candidates,
    }


def _ndcg(labels: list[int]) -> float:
    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))

    ideal = dcg(sorted(labels, reverse=True))
    return dcg(labels) / ideal if ideal > 0 else 1.0


def _evaluate(rows: list[sqlite3.Row], score_name: str) -> dict[str, Any]:
    by_query: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_task_id"]), []).append(row)
    ndcgs: list[float] = []
    exact_best = 0
    top1_relevant = 0
    evaluated = 0
    for query_rows in by_query.values():
        if len(query_rows) != 5:
            continue
        ranked = sorted(query_rows, key=lambda row: float(row[score_name]), reverse=True)
        labels = [int(row["relation_code"]) for row in ranked]
        ndcgs.append(_ndcg(labels))
        exact_best += int(labels[0] == max(labels))
        top1_relevant += int(labels[0] >= 2)
        evaluated += 1
    return {
        "queries": evaluated,
        "ndcg_at_5": round(statistics.mean(ndcgs), 4) if ndcgs else None,
        "top1_exact_best_rate": round(exact_best / evaluated, 4) if evaluated else None,
        "top1_relation_ge_2_rate": round(top1_relevant / evaluated, 4) if evaluated else None,
    }


@router.get("/api/summary")
def rerank_summary() -> dict[str, Any]:
    text_weight, visual_weight = _weights()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT q.task_id AS query_task_id,p.scene_raw_cosine,p.text_raw_cosine,
                   a.relation_code
            FROM query_tasks q
            JOIN candidate_pairs p ON p.query_task_id=q.task_id
            JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name='scene'
            WHERE q.display_order<=?
            """,
            (DEVELOPMENT_QUERY_COUNT,),
        ).fetchall()
    materialized = []
    for row in rows:
        item = dict(row)
        item["text_heavy_score"] = (
            text_weight * float(row["text_raw_cosine"])
            + visual_weight * float(row["scene_raw_cosine"])
        )
        materialized.append(item)
    weight_sweep = {}
    for step in range(11):
        candidate_text_weight = step / 10.0
        for item in materialized:
            item["sweep_score"] = (
                candidate_text_weight * float(item["text_raw_cosine"])
                + (1.0 - candidate_text_weight) * float(item["scene_raw_cosine"])
            )
        weight_sweep[f"text_{candidate_text_weight:.1f}"] = _evaluate(
            materialized, "sweep_score"
        )
    return {
        "scope": "original five candidates of the first 80 development queries",
        "weights": {"text": text_weight, "visual": visual_weight},
        "visual_only": _evaluate(materialized, "scene_raw_cosine"),
        "text_only": _evaluate(materialized, "text_raw_cosine"),
        "text_heavy": _evaluate(materialized, "text_heavy_score"),
        "weight_sweep": weight_sweep,
        "warning": "The all-100-reference rerank contains unlabeled candidates and requires manual inspection.",
    }


_RERANK_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>文本优先召回 V2</title><style>
:root{color-scheme:dark;--bg:#07101d;--panel:#101c2e;--line:#2a3b55;--text:#edf5ff;--muted:#91a6c0;--cyan:#58e1ca;--blue:#5d8dff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#17375e 0,transparent 31%),var(--bg);color:var(--text);font-family:Inter,system-ui,"PingFang SC",sans-serif}header{position:sticky;top:0;z-index:10;background:#07101dee;border-bottom:1px solid var(--line)}.bar,main{max-width:1500px;margin:auto;padding:14px 22px}.bar{display:flex;align-items:center;gap:10px}.bar b{margin-right:auto}.bar button{border:1px solid var(--line);border-radius:8px;padding:8px 13px;background:#172b4c;color:white;cursor:pointer}.bar button:disabled{opacity:.35}main{padding-top:22px}.intro{color:var(--muted);font-size:13px;line-height:1.6}.query{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}.results{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}.card h2{margin:0;padding:11px 13px;border-bottom:1px solid var(--line);font-size:14px}.new-top{border-color:#58e1ca99}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#020610}.body{padding:11px}.scores{color:var(--cyan);font-size:12px;margin-bottom:8px}.desc{background:#091422;border-radius:8px;padding:9px;font-size:12px;line-height:1.55;white-space:pre-wrap}.replay{width:100%;margin-top:8px;border:1px solid #5d8dff66;border-radius:8px;padding:8px;background:#172b4c;color:white;cursor:pointer}.summary{margin:13px 0;padding:12px;background:#0a1424;border:1px solid var(--line);border-radius:10px;white-space:pre-wrap;color:#cad8eb;font-size:12px}@media(max-width:900px){.query,.results{grid-template-columns:1fr}}
</style></head><body><header><div class="bar"><b>文本优先召回 V2 · 文字70% + 视觉30%</b><button id="prev">上一条</button><span id="count">1/80</span><button id="next">下一条</button></div></header><main><div class="intro">每条查询都重新与全部100个参考资产比较。下方先显示旧视觉Top1，再展示文本优先的新Top5；后40条测试数据不展示。</div><div id="summary" class="summary">正在读取原5候选离线对比…</div><div id="workspace"></div></main><script>
let position=1;const workspace=document.getElementById('workspace'),count=document.getElementById('count'),prev=document.getElementById('prev'),next=document.getElementById('next'),summary=document.getElementById('summary');
function video(asset){const v=document.createElement('video');v.controls=true;v.playsInline=true;v.preload='metadata';v.src=asset.video_url;v.dataset.start=String(asset.start_seconds||0);v.defaultPlaybackRate=2;v.playbackRate=2;v.addEventListener('loadedmetadata',()=>{v.currentTime=Number(v.dataset.start||0);v.playbackRate=2});return v}
function description(asset){return ['场景：'+(asset.scene||'暂无'),'任务：'+(asset.task_name||'暂无'),'详情：'+(asset.details||'暂无')].join('\n')}
function card(title,asset,scores,highlight=false){const c=document.createElement('article');c.className='card'+(highlight?' new-top':'');const h=document.createElement('h2');h.textContent=title;const v=video(asset),body=document.createElement('div');body.className='body';const s=document.createElement('div');s.className='scores';s.textContent=scores;const d=document.createElement('div');d.className='desc';d.textContent=description(asset);const b=document.createElement('button');b.className='replay';b.textContent='↺ 从Task开头播放';b.onclick=()=>{v.currentTime=Number(v.dataset.start||0);v.play()};body.append(s,d,b);c.append(h,v,body);return c}
async function load(){prev.disabled=next.disabled=true;const r=await fetch('/study/rerank/api/query?position='+position+'&top_k=5'),d=await r.json();count.textContent=`${position}/${d.development_queries}`;const q=card('查询视频',d.query,'');const old=card('旧版：纯视觉Top1',d.old_visual_top1,`视觉 ${d.old_visual_top1.visual_score} · 文本 ${d.old_visual_top1.text_score}`);const top=document.createElement('div');top.className='query';top.append(q,old);const results=document.createElement('div');results.className='results';d.candidates.forEach((c,i)=>results.append(card(`新版 TOP ${i+1}`,c,`综合 ${c.combined_score} · 文本 ${c.text_score} · 视觉 ${c.visual_score}`,i===0)));workspace.replaceChildren(top,results);prev.disabled=position<=1;next.disabled=position>=d.development_queries}
prev.onclick=()=>{if(position>1){position--;load()}};next.onclick=()=>{if(position<80){position++;load()}};
fetch('/study/rerank/api/summary').then(r=>r.json()).then(d=>{const best=Object.entries(d.weight_sweep).sort((a,b)=>b[1].ndcg_at_5-a[1].ndcg_at_5)[0];summary.textContent=`原5候选离线对比：视觉 NDCG@5=${d.visual_only.ndcg_at_5}，文字70%版本=${d.text_heavy.ndcg_at_5}；权重扫描最佳为 ${best[0].replace('_',' ')}，NDCG@5=${best[1].ndcg_at_5}。当前页面仍按你的要求展示文字70%+视觉30%的全库Top5。`});load();
</script></body></html>"""
