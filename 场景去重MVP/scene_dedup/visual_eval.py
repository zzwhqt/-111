from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .core import l2_normalize
from .label_study import POOL_DIR


DB_PATH = Path(
    os.getenv(
        "DEDUP_MULTIMODAL_LABEL_DB",
        "/opt/task-dedup/labeling/multimodal_study.sqlite3",
    )
)
DEVELOPMENT_QUERY_COUNT = 80
router = APIRouter(prefix="/study/visual-eval", tags=["pure-visual-evaluation"])


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


@lru_cache(maxsize=1)
def _frame_vectors() -> np.ndarray:
    return l2_normalize(np.load(POOL_DIR / "frame_vectors.npy").astype(np.float32))


@lru_cache(maxsize=1)
def _visual_p95() -> float:
    task_vectors = l2_normalize(np.load(POOL_DIR / "task_vectors.npy").astype(np.float32))
    pair_scores = task_vectors @ task_vectors.T
    upper = np.triu_indices(len(task_vectors), k=1)
    return float(np.quantile(pair_scores[upper], 0.95))


def _ndcg(labels: list[int]) -> float:
    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))

    ideal = dcg(sorted(labels, reverse=True))
    return dcg(labels) / ideal if ideal > 0 else 1.0


@lru_cache(maxsize=1)
def _records() -> tuple[dict[str, Any], ...]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT q.task_id AS query_task_id,q.display_order,
                   q.vector_index AS query_vector_index,
                   r.vector_index AS reference_vector_index,
                   p.scene_raw_cosine,a.relation_code
            FROM query_tasks q
            JOIN candidate_pairs p ON p.query_task_id=q.task_id
            JOIN reference_assets r ON r.task_id=p.reference_task_id
            JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name='scene'
            ORDER BY q.display_order,p.candidate_order
            """
        ).fetchall()
    frames = _frame_vectors()
    visual_p95 = _visual_p95()
    records = []
    for row in rows:
        query_frames = frames[int(row["query_vector_index"])]
        reference_frames = frames[int(row["reference_vector_index"])]
        matrix = query_frames @ reference_frames.T
        frame_coverage = float(
            (matrix.max(axis=1).mean() + matrix.max(axis=0).mean()) / 2.0
        )
        records.append(
            {
                "query_task_id": str(row["query_task_id"]),
                "display_order": int(row["display_order"]),
                "task_cosine": float(row["scene_raw_cosine"]),
                "task_distinctiveness": float(
                    np.clip(
                        (float(row["scene_raw_cosine"]) - visual_p95)
                        / max(1e-6, 1.0 - visual_p95),
                        0.0,
                        1.0,
                    )
                ),
                "frame_coverage": frame_coverage,
                "relation_code": int(row["relation_code"]),
            }
        )
    return tuple(records)


def _evaluate(
    records: list[dict[str, Any]],
    task_weight: float,
    task_field: str = "task_cosine",
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["query_task_id"]].append(record)
    ndcgs = []
    exact_best = 0
    top1_relevant = 0
    hit3_relevant = 0
    relevant_queries = 0
    for query_records in grouped.values():
        if len(query_records) != 5:
            continue
        ranked = sorted(
            query_records,
            key=lambda item: (
                task_weight * item[task_field]
                + (1.0 - task_weight) * item["frame_coverage"]
            ),
            reverse=True,
        )
        labels = [int(item["relation_code"]) for item in ranked]
        ndcgs.append(_ndcg(labels))
        exact_best += int(labels[0] == max(labels))
        top1_relevant += int(labels[0] >= 2)
        if any(label >= 2 for label in labels):
            relevant_queries += 1
            hit3_relevant += int(any(label >= 2 for label in labels[:3]))
    count = len(ndcgs)
    return {
        "queries": count,
        "ndcg_at_5": round(statistics.mean(ndcgs), 4) if ndcgs else None,
        "top1_exact_best_rate": round(exact_best / count, 4) if count else None,
        "top1_relation_ge_2_rate": round(top1_relevant / count, 4) if count else None,
        "hit_at_3_given_relevant": (
            round(hit3_relevant / relevant_queries, 4) if relevant_queries else None
        ),
        "queries_with_relation_ge_2": relevant_queries,
    }


@router.get("", response_class=HTMLResponse)
def visual_eval_home() -> str:
    return _VISUAL_EVAL_HTML


@router.get("/api/summary")
def visual_eval_summary() -> dict[str, Any]:
    records = list(_records())
    development = [r for r in records if r["display_order"] <= DEVELOPMENT_QUERY_COUNT]
    locked = [r for r in records if r["display_order"] > DEVELOPMENT_QUERY_COUNT]
    sweep = {}
    for step in range(11):
        task_weight = step / 10.0
        sweep[f"task_{task_weight:.1f}_frame_{1.0-task_weight:.1f}"] = _evaluate(
            development, task_weight
        )
    best_key, best_metrics = max(
        sweep.items(), key=lambda item: (item[1]["ndcg_at_5"], item[1]["top1_exact_best_rate"])
    )
    best_task_weight = float(best_key.split("_")[1])
    normalized_sweep = {}
    for step in range(11):
        task_weight = step / 10.0
        normalized_sweep[f"task_{task_weight:.1f}_frame_{1.0-task_weight:.1f}"] = _evaluate(
            development, task_weight, "task_distinctiveness"
        )
    normalized_best_key, normalized_best_metrics = max(
        normalized_sweep.items(),
        key=lambda item: (item[1]["ndcg_at_5"], item[1]["top1_exact_best_rate"]),
    )
    normalized_best_task_weight = float(normalized_best_key.split("_")[1])
    return {
        "protocol": {
            "input": "video frames only",
            "text_fields_used": False,
            "development_queries": 80,
            "locked_queries": 40,
            "candidate_pairs_per_query": 5,
        },
        "label_counts": dict(sorted(Counter(r["relation_code"] for r in records).items())),
        "development": {
            "task_cosine_only": _evaluate(development, 1.0),
            "frame_coverage_only": _evaluate(development, 0.0),
            "current_65_task_35_frame": _evaluate(development, 0.65),
            "deployed_normalized_70_task_30_frame": _evaluate(
                development, 0.70, "task_distinctiveness"
            ),
            "weight_sweep": sweep,
            "best": {
                "task_weight": best_task_weight,
                "frame_weight": 1.0 - best_task_weight,
                "metrics": best_metrics,
            },
            "normalized_weight_sweep": normalized_sweep,
            "normalized_best": {
                "task_weight": normalized_best_task_weight,
                "frame_weight": 1.0 - normalized_best_task_weight,
                "metrics": normalized_best_metrics,
            },
        },
        "locked_test_baseline": {
            "task_cosine_only": _evaluate(locked, 1.0),
            "current_65_task_35_frame": _evaluate(locked, 0.65),
            "deployed_normalized_70_task_30_frame": _evaluate(
                locked, 0.70, "task_distinctiveness"
            ),
            "development_best_applied_once": _evaluate(locked, best_task_weight),
            "normalized_development_best_applied_once": _evaluate(
                locked, normalized_best_task_weight, "task_distinctiveness"
            ),
        },
        "all_120_baseline": _evaluate(records, 1.0),
        "all_120_tuned_60_task_40_frame": _evaluate(records, 0.60),
    }


_VISUAL_EVAL_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>纯视觉场景评测</title><style>
:root{color-scheme:dark;--bg:#07101d;--panel:#101c2e;--line:#2a3b55;--text:#edf5ff;--muted:#91a6c0;--cyan:#58e1ca;--blue:#5d8dff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#17375e 0,transparent 31%),var(--bg);color:var(--text);font-family:Inter,system-ui,"PingFang SC",sans-serif}main{max-width:1180px;margin:auto;padding:42px 22px}h1{font-size:38px;margin:0 0 8px}.intro{color:var(--muted);line-height:1.6}.notice{margin:18px 0;padding:14px;border:1px solid #58e1ca55;border-radius:12px;background:#0b2130;color:#c8fff4}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}.card b{display:block;color:var(--cyan);font-size:24px;margin:6px 0}.card span{color:var(--muted);font-size:12px}.section{margin-top:24px}.section h2{font-size:20px}.table{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{padding:11px 13px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}th{color:var(--muted)}tr.best{background:#174b5344}.links{margin-top:22px;display:flex;gap:10px;flex-wrap:wrap}.links a{padding:10px 14px;border-radius:9px;background:#172b4c;color:#dce8ff;text-decoration:none}@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1>纯视觉场景评测</h1><div class="intro">固定使用120个查询、600个纯视觉人工标签。模型只读取8帧视觉向量，完全不读取scene、task_name或details。</div><div class="notice">前80条用于选择算法和权重；后40条仅用于阶段验收。当前页面评估的是5候选内部排序，不等同于全部100个资产的完整召回率。</div><div id="cards" class="grid"></div><section class="section"><h2>80条开发集：整体向量与帧覆盖权重扫描</h2><div class="table"><table><thead><tr><th>整体向量</th><th>帧覆盖</th><th>NDCG@5</th><th>Top1人工最优</th><th>Top1等级≥2</th></tr></thead><tbody id="rows"></tbody></table></div></section><div class="links"><a href="/study/errors">查看开发集错题</a><a href="/">纯视觉视频检索</a></div></main><script>
const pct=v=>v==null?'-':(v*100).toFixed(2)+'%';fetch('/study/visual-eval/api/summary').then(r=>r.json()).then(d=>{const base=d.all_120_baseline,best=d.development.best,test=d.locked_test_baseline.development_best_applied_once,cards=document.getElementById('cards');for(const [title,value,sub] of [['120条基线 NDCG',base.ndcg_at_5,'纯整体视觉余弦'],['开发集最佳 NDCG',best.metrics.ndcg_at_5,`整体${Math.round(best.task_weight*100)}% + 帧${Math.round(best.frame_weight*100)}%`],['40条保留集 NDCG',test.ndcg_at_5,'仅应用开发集选出的权重']]){const c=document.createElement('div');c.className='card';c.innerHTML=`<span>${title}</span><b>${value}</b><span>${sub}</span>`;cards.append(c)}const rows=document.getElementById('rows');for(const [key,m] of Object.entries(d.development.weight_sweep)){const parts=key.match(/task_(\d\.\d)_frame_(\d\.\d)/),tw=Number(parts[1]),tr=document.createElement('tr');if(Math.abs(tw-best.task_weight)<.001)tr.className='best';tr.innerHTML=`<td>${Math.round(tw*100)}%</td><td>${Math.round((1-tw)*100)}%</td><td>${m.ndcg_at_5}</td><td>${pct(m.top1_exact_best_rate)}</td><td>${pct(m.top1_relation_ge_2_rate)}</td>`;rows.append(tr)}}).catch(e=>document.body.textContent='加载失败：'+e.message);
</script></body></html>"""
