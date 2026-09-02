from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .core import l2_normalize
from .label_study import POOL_DIR, _signed_video, _thumbnail_url


DB_PATH = Path(
    os.getenv(
        "DEDUP_MULTIMODAL_LABEL_DB",
        "/opt/task-dedup/labeling/multimodal_study.sqlite3",
    )
)
ROUND_SCENE = "scene"
SCENE_CODES = {0, 1, 2, 3, 9}
router = APIRouter(prefix="/study", tags=["multimodal-blind-study"])


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS study_settings (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reference_assets (
          task_id TEXT PRIMARY KEY,
          vector_index INTEGER NOT NULL,
          display_index INTEGER NOT NULL UNIQUE,
          metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_tasks (
          task_id TEXT PRIMARY KEY,
          vector_index INTEGER NOT NULL,
          display_order INTEGER NOT NULL UNIQUE,
          sample_quadrant TEXT NOT NULL,
          metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidate_pairs (
          pair_id TEXT PRIMARY KEY,
          query_task_id TEXT NOT NULL REFERENCES query_tasks(task_id),
          reference_task_id TEXT NOT NULL REFERENCES reference_assets(task_id),
          candidate_order INTEGER NOT NULL,
          candidate_sources_json TEXT NOT NULL,
          scene_raw_cosine REAL NOT NULL,
          text_raw_cosine REAL NOT NULL,
          UNIQUE(query_task_id, reference_task_id),
          UNIQUE(query_task_id, candidate_order)
        );
        CREATE TABLE IF NOT EXISTS annotations (
          pair_id TEXT NOT NULL REFERENCES candidate_pairs(pair_id),
          round_name TEXT NOT NULL CHECK(round_name IN ('scene','motion','text','final')),
          relation_code INTEGER NOT NULL,
          query_coverage INTEGER,
          reference_coverage INTEGER,
          quality INTEGER,
          confidence INTEGER,
          comment TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(pair_id, round_name)
        );
        CREATE INDEX IF NOT EXISTS idx_query_display ON query_tasks(display_order);
        CREATE INDEX IF NOT EXISTS idx_pair_query ON candidate_pairs(query_task_id,candidate_order);
        CREATE INDEX IF NOT EXISTS idx_annotation_round ON annotations(round_name,pair_id);
        """
    )


def _metadata(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(row["metadata_json"])


def _scene_asset(task_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    # Descriptions are existing asset metadata. They are shown for inspection,
    # while candidate sources and model scores remain hidden during labeling.
    return {
        "task_id": task_id,
        "scene": metadata.get("scene") or "",
        "task_name": metadata.get("task_name") or "",
        "details": metadata.get("details") or metadata.get("description") or "",
        "start_seconds": metadata.get("start_seconds"),
        "end_seconds": metadata.get("end_seconds"),
        "duration_seconds": metadata.get("duration_seconds"),
        "thumbnail_url": _thumbnail_url(metadata),
    }


def _progress(connection: sqlite3.Connection) -> dict[str, int]:
    total_pairs = int(connection.execute("SELECT COUNT(*) FROM candidate_pairs").fetchone()[0])
    labeled_pairs = int(
        connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE round_name=?", (ROUND_SCENE,)
        ).fetchone()[0]
    )
    total_queries = int(connection.execute("SELECT COUNT(*) FROM query_tasks").fetchone()[0])
    completed_queries = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT p.query_task_id
              FROM candidate_pairs p
              LEFT JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name=?
              GROUP BY p.query_task_id
              HAVING COUNT(*)=COUNT(a.pair_id)
            )
            """,
            (ROUND_SCENE,),
        ).fetchone()[0]
    )
    return {
        "total_pairs": total_pairs,
        "labeled_pairs": labeled_pairs,
        "remaining_pairs": max(0, total_pairs - labeled_pairs),
        "total_queries": total_queries,
        "completed_queries": completed_queries,
        "remaining_queries": max(0, total_queries - completed_queries),
    }


def _next_payload(connection: sqlite3.Connection) -> dict[str, Any]:
    query = connection.execute(
        """
        SELECT q.* FROM query_tasks q
        WHERE EXISTS (
          SELECT 1 FROM candidate_pairs p
          LEFT JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name=?
          WHERE p.query_task_id=q.task_id AND a.pair_id IS NULL
        )
        ORDER BY q.display_order
        LIMIT 1
        """,
        (ROUND_SCENE,),
    ).fetchone()
    progress = _progress(connection)
    if query is None:
        return {"complete": True, "progress": progress}
    query_metadata = _metadata(query)
    query_asset = _scene_asset(query["task_id"], query_metadata)
    query_asset["video_url"] = _signed_video(query_metadata)
    rows = connection.execute(
        """
        SELECT p.pair_id,p.candidate_order,r.task_id,r.metadata_json
        FROM candidate_pairs p
        JOIN reference_assets r ON r.task_id=p.reference_task_id
        WHERE p.query_task_id=?
        ORDER BY p.candidate_order
        """,
        (query["task_id"],),
    ).fetchall()
    candidates = []
    for row in rows:
        item = _scene_asset(row["task_id"], json.loads(row["metadata_json"]))
        item["pair_id"] = row["pair_id"]
        item["candidate_order"] = int(row["candidate_order"])
        candidates.append(item)
    return {
        "complete": False,
        "round": ROUND_SCENE,
        "query": query_asset,
        "candidates": candidates,
        "progress": progress,
    }


class ScenePairLabel(BaseModel):
    pair_id: str = Field(min_length=8, max_length=128)
    relation_code: int


class SceneBatchSubmission(BaseModel):
    query_task_id: str = Field(min_length=1, max_length=512)
    labels: list[ScenePairLabel] = Field(min_length=1, max_length=20)
    latency_ms: int | None = Field(default=None, ge=0, le=86_400_000)


@router.get("", response_class=HTMLResponse)
def study_home() -> str:
    return _SCENE_HTML


@router.get("/api/progress")
def progress() -> dict[str, int]:
    with _connect() as connection:
        _init_schema(connection)
        return _progress(connection)


@router.get("/api/next")
def next_scene_query() -> dict[str, Any]:
    with _connect() as connection:
        _init_schema(connection)
        return _next_payload(connection)


@router.get("/api/playback/{task_id}")
def playback(task_id: str) -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT task_id,metadata_json FROM reference_assets WHERE task_id=?
            UNION ALL
            SELECT task_id,metadata_json FROM query_tasks WHERE task_id=?
            LIMIT 1
            """,
            (task_id, task_id),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "study asset not found")
        metadata = json.loads(row["metadata_json"])
        return {
            "video_url": _signed_video(metadata),
            "start_seconds": float(metadata.get("start_seconds") or 0.0),
            "end_seconds": float(metadata.get("end_seconds") or 0.0),
        }


@router.post("/api/submit-scene")
def submit_scene(submission: SceneBatchSubmission) -> dict[str, Any]:
    labels = {item.pair_id: item.relation_code for item in submission.labels}
    if len(labels) != len(submission.labels):
        raise HTTPException(400, "duplicate pair_id in submission")
    if any(code not in SCENE_CODES for code in labels.values()):
        raise HTTPException(400, "invalid scene relation code")
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT pair_id FROM candidate_pairs WHERE query_task_id=? ORDER BY candidate_order",
            (submission.query_task_id,),
        ).fetchall()
        expected = {row["pair_id"] for row in rows}
        if not expected:
            raise HTTPException(404, "query task not found")
        if set(labels) != expected:
            raise HTTPException(400, "all and only the query's candidate pairs must be labeled")
        for pair_id, relation_code in labels.items():
            quality = 0 if relation_code == 9 else 2
            connection.execute(
                """
                INSERT INTO annotations(
                  pair_id,round_name,relation_code,quality,comment,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(pair_id,round_name) DO UPDATE SET
                  relation_code=excluded.relation_code,
                  quality=excluded.quality,
                  updated_at=excluded.updated_at
                """,
                (pair_id, ROUND_SCENE, relation_code, quality, "", now, now),
            )
        connection.commit()
        return _next_payload(connection)


@router.post("/api/undo-last")
def undo_last() -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT p.query_task_id,MAX(a.updated_at) AS last_updated
            FROM annotations a JOIN candidate_pairs p USING(pair_id)
            WHERE a.round_name=?
            GROUP BY p.query_task_id
            ORDER BY last_updated DESC LIMIT 1
            """,
            (ROUND_SCENE,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "no scene batch to undo")
        connection.execute(
            """
            DELETE FROM annotations
            WHERE round_name=? AND pair_id IN (
              SELECT pair_id FROM candidate_pairs WHERE query_task_id=?
            )
            """,
            (ROUND_SCENE, row["query_task_id"]),
        )
        connection.commit()
        return {"undone_query_task_id": row["query_task_id"], **_progress(connection)}


def _ndcg(relevances: list[int]) -> float | None:
    valid = [value for value in relevances if value != 9]
    if not valid:
        return None

    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))

    ideal = dcg(sorted(valid, reverse=True))
    return dcg(valid) / ideal if ideal > 0 else 1.0


@router.get("/api/scene-report")
def scene_report() -> dict[str, Any]:
    with _connect() as connection:
        current = _progress(connection)
        if current["total_pairs"] == 0 or current["remaining_pairs"] > 0:
            return {
                "complete": False,
                "progress": current,
                "message": "场景盲标未完成，模型分数继续隐藏。",
            }
        rows = connection.execute(
            """
            SELECT q.task_id,q.sample_quadrant,p.pair_id,p.scene_raw_cosine,
                   a.relation_code
            FROM query_tasks q
            JOIN candidate_pairs p ON p.query_task_id=q.task_id
            JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name=?
            ORDER BY q.display_order,p.scene_raw_cosine DESC
            """,
            (ROUND_SCENE,),
        ).fetchall()
    by_query: dict[str, list[sqlite3.Row]] = defaultdict(list)
    scores_by_label: dict[int, list[float]] = defaultdict(list)
    counts = Counter()
    for row in rows:
        by_query[row["task_id"]].append(row)
        counts[int(row["relation_code"])] += 1
        if int(row["relation_code"]) != 9:
            scores_by_label[int(row["relation_code"])].append(float(row["scene_raw_cosine"]))
    ndcgs = []
    top1_relevances = []
    quadrant_ndcg: dict[str, list[float]] = defaultdict(list)
    for query_rows in by_query.values():
        relevance = [int(row["relation_code"]) for row in query_rows]
        value = _ndcg(relevance)
        if value is not None:
            ndcgs.append(value)
            quadrant_ndcg[str(query_rows[0]["sample_quadrant"])].append(value)
        if relevance and relevance[0] != 9:
            top1_relevances.append(relevance[0])
    score_summary = {
        str(label): {
            "count": len(values),
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
        }
        for label, values in sorted(scores_by_label.items())
    }
    return {
        "complete": True,
        "progress": current,
        "scene_label_counts": dict(sorted(counts.items())),
        "ndcg_at_5": round(statistics.mean(ndcgs), 4) if ndcgs else None,
        "top1_mean_human_relevance": (
            round(statistics.mean(top1_relevances), 4) if top1_relevances else None
        ),
        "top1_relevant_rate_relation_ge_2": (
            round(sum(value >= 2 for value in top1_relevances) / len(top1_relevances), 4)
            if top1_relevances
            else None
        ),
        "scene_cosine_by_human_label": score_summary,
        "ndcg_by_query_quadrant": {
            key: round(statistics.mean(values), 4)
            for key, values in sorted(quadrant_ndcg.items())
            if values
        },
    }


def _select_references(
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[int]:
    kmeans = faiss.Kmeans(
        vectors.shape[1], count, niter=30, nredo=3, seed=seed, verbose=False, spherical=True
    )
    kmeans.train(vectors)
    scores = kmeans.centroids @ vectors.T
    chosen: list[int] = []
    chosen_set: set[int] = set()
    used_sources: set[str] = set()
    for centroid_index in range(count):
        for index in np.argsort(-scores[centroid_index]):
            idx = int(index)
            source = str(metadata[idx].get("source_video") or "")
            if idx not in chosen_set and source not in used_sources:
                chosen.append(idx)
                chosen_set.add(idx)
                used_sources.add(source)
                break
    if len(chosen) != count:
        raise RuntimeError(f"could only select {len(chosen)} unique-source references")
    return chosen


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    if len(values) > 1:
        ranks /= float(len(values) - 1)
    return ranks


def _pick_queries(
    remaining: list[int],
    visual_max: np.ndarray,
    text_max: np.ndarray,
    metadata: list[dict[str, Any]],
    per_quadrant: int,
) -> list[tuple[int, str]]:
    visual_rank = _percentile_ranks(visual_max)
    text_rank = _percentile_ranks(text_max)
    strategies = [
        ("visual_high_text_high", np.minimum(visual_rank, text_rank)),
        ("visual_high_text_low", visual_rank - text_rank),
        ("visual_low_text_high", text_rank - visual_rank),
        ("visual_low_text_low", -np.maximum(visual_rank, text_rank)),
    ]
    selected_positions: set[int] = set()
    source_counts: Counter[str] = Counter()
    result: list[tuple[int, str]] = []
    for name, score in strategies:
        picked = 0
        ranked_positions = [int(value) for value in np.argsort(-score, kind="stable")]
        for max_per_source in (2, 4, 10_000):
            for position in ranked_positions:
                if position in selected_positions:
                    continue
                vector_index = remaining[position]
                source = str(metadata[vector_index].get("source_video") or "")
                if source_counts[source] >= max_per_source:
                    continue
                selected_positions.add(position)
                source_counts[source] += 1
                result.append((vector_index, name))
                picked += 1
                if picked >= per_quadrant:
                    break
            if picked >= per_quadrant:
                break
        if picked != per_quadrant:
            raise RuntimeError(f"could only select {picked} queries for {name}")
    return result


def _candidate_positions(
    query_index: int,
    reference_indices: list[int],
    task_vectors: np.ndarray,
    text_vectors: np.ndarray,
    seed: int,
) -> list[tuple[int, list[str]]]:
    visual = task_vectors[reference_indices] @ task_vectors[query_index]
    text = text_vectors[reference_indices] @ text_vectors[query_index]
    visual_order = [int(value) for value in np.argsort(-visual, kind="stable")]
    text_order = [int(value) for value in np.argsort(-text, kind="stable")]
    sources: dict[int, list[str]] = {}

    def add(position: int, source: str) -> None:
        sources.setdefault(position, [])
        if source not in sources[position]:
            sources[position].append(source)

    for position in visual_order[:2]:
        add(position, "visual_top2")
    for position in text_order[:2]:
        add(position, "text_top2")

    rng = random.Random(seed ^ query_index)
    difficult_negative_pool = [
        position
        for position in range(len(reference_indices))
        if position not in sources
        and visual[position] <= float(np.median(visual))
        and text[position] <= float(np.median(text))
    ]
    if not difficult_negative_pool:
        difficult_negative_pool = [
            position for position in range(len(reference_indices)) if position not in sources
        ]
    add(rng.choice(difficult_negative_pool), "random_negative")

    fill_order = []
    for position in visual_order + text_order:
        if position not in fill_order:
            fill_order.append(position)
    for position in fill_order:
        if len(sources) >= 5:
            break
        if position not in sources:
            add(position, "rank_fill")
    if len(sources) != 5:
        raise RuntimeError("failed to build five unique candidates")
    selected = list(sources.items())
    rng.shuffle(selected)
    return selected


def initialize_study(
    pool_dir: Path,
    db_path: Path,
    seed: int = 20260829,
    reference_count: int = 100,
    query_count: int = 120,
) -> dict[str, Any]:
    global DB_PATH
    if query_count % 4:
        raise ValueError("query_count must be divisible by four")
    DB_PATH = db_path
    metadata = json.loads((pool_dir / "metadata.json").read_text("utf-8"))
    task_vectors = l2_normalize(np.load(pool_dir / "task_vectors.npy").astype(np.float32))
    text_vectors = l2_normalize(np.load(pool_dir / "text_vectors.npy").astype(np.float32))
    if len(metadata) != len(task_vectors) or len(metadata) != len(text_vectors):
        raise RuntimeError("metadata/task/text vector counts do not match")
    if len(metadata) < reference_count + query_count:
        raise RuntimeError("pool is too small for requested study")

    reference_indices = _select_references(task_vectors, metadata, reference_count, seed)
    reference_set = set(reference_indices)
    remaining = [index for index in range(len(metadata)) if index not in reference_set]
    visual_matrix = task_vectors[remaining] @ task_vectors[reference_indices].T
    text_matrix = text_vectors[remaining] @ text_vectors[reference_indices].T
    targets = _pick_queries(
        remaining,
        visual_matrix.max(axis=1),
        text_matrix.max(axis=1),
        metadata,
        query_count // 4,
    )
    rng = random.Random(seed)
    rng.shuffle(targets)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        _init_schema(connection)
        existing = sum(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("study_settings", "reference_assets", "query_tasks", "candidate_pairs", "annotations")
        )
        if existing:
            raise RuntimeError(f"multimodal study database is already initialized: {db_path}")
        settings = {
            "schema_version": "multimodal_pair_v1",
            "seed": str(seed),
            "pool_size": str(len(metadata)),
            "reference_count": str(reference_count),
            "query_count": str(query_count),
            "candidate_count_per_query": "5",
            "pair_count": str(query_count * 5),
            "active_round": ROUND_SCENE,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.executemany("INSERT INTO study_settings(key,value) VALUES(?,?)", settings.items())
        for display_index, vector_index in enumerate(reference_indices, start=1):
            connection.execute(
                "INSERT INTO reference_assets VALUES(?,?,?,?)",
                (
                    str(metadata[vector_index]["task_id"]),
                    vector_index,
                    display_index,
                    json.dumps(metadata[vector_index], ensure_ascii=False),
                ),
            )
        for display_order, (query_index, quadrant) in enumerate(targets, start=1):
            query_task_id = str(metadata[query_index]["task_id"])
            connection.execute(
                "INSERT INTO query_tasks VALUES(?,?,?,?,?)",
                (
                    query_task_id,
                    query_index,
                    display_order,
                    quadrant,
                    json.dumps(metadata[query_index], ensure_ascii=False),
                ),
            )
            candidates = _candidate_positions(
                query_index, reference_indices, task_vectors, text_vectors, seed
            )
            for candidate_order, (reference_position, sources) in enumerate(candidates, start=1):
                reference_index = reference_indices[reference_position]
                reference_task_id = str(metadata[reference_index]["task_id"])
                pair_id = hashlib.sha256(
                    f"multimodal_pair_v1\0{query_task_id}\0{reference_task_id}".encode("utf-8")
                ).hexdigest()[:24]
                connection.execute(
                    "INSERT INTO candidate_pairs VALUES(?,?,?,?,?,?,?)",
                    (
                        pair_id,
                        query_task_id,
                        reference_task_id,
                        candidate_order,
                        json.dumps(sources, ensure_ascii=False),
                        float(task_vectors[query_index] @ task_vectors[reference_index]),
                        float(text_vectors[query_index] @ text_vectors[reference_index]),
                    ),
                )
        connection.commit()
    return {
        "pool": len(metadata),
        "references": reference_count,
        "queries": query_count,
        "pairs": query_count * 5,
        "quadrants": dict(Counter(quadrant for _, quadrant in targets)),
        "database": str(db_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    init.add_argument("--db", type=Path, default=DB_PATH)
    init.add_argument("--seed", type=int, default=20260829)
    init.add_argument("--references", type=int, default=100)
    init.add_argument("--queries", type=int, default=120)
    args = parser.parse_args()
    if args.command == "init":
        print(
            json.dumps(
                initialize_study(
                    args.pool_dir,
                    args.db,
                    seed=args.seed,
                    reference_count=args.references,
                    query_count=args.queries,
                ),
                ensure_ascii=False,
            )
        )
    return 0


_SCENE_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多模态 Task 去重 · 场景轮</title><style>
:root{color-scheme:dark;--bg:#07101d;--panel:#101c2e;--line:#2a3b55;--text:#edf5ff;--muted:#91a6c0;--cyan:#58e1ca;--blue:#5d8dff;--orange:#f2ad4b;--red:#ef6b78}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#17375e 0,transparent 31%),var(--bg);color:var(--text);font-family:Inter,system-ui,"PingFang SC",sans-serif}header{position:sticky;top:0;z-index:20;background:#07101dec;backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}.bar,main{max-width:1480px;margin:auto;padding:15px 22px}.bar{display:flex;gap:16px;align-items:center}.brand{font-weight:850;white-space:nowrap}.progress{height:8px;background:#1a2940;border-radius:99px;overflow:hidden;flex:1}.progress i{display:block;width:0;height:100%;background:linear-gradient(90deg,var(--cyan),var(--blue))}.count{color:var(--muted);font-size:13px;white-space:nowrap}main{padding-top:24px}.intro{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:16px}h1{font-size:27px;margin:0 0 8px}.hint{color:var(--muted);font-size:13px;line-height:1.55}.legend{display:flex;gap:8px;flex-wrap:wrap}.legend span{border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:var(--muted);font-size:12px}.query{display:grid;grid-template-columns:minmax(440px,.85fr) minmax(0,1.15fr);gap:18px}.panel{background:#101c2ee8;border:1px solid var(--line);border-radius:17px;overflow:hidden}.panel-title{padding:13px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.query-video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#020610}.query-foot{padding:10px 16px;color:var(--muted);font-size:12px;display:flex;align-items:center;justify-content:space-between;gap:10px}.asset-description{margin:0 0 8px;padding:8px 10px;border:1px solid #2a3b5599;border-radius:9px;background:#101c2e;color:#dce8f7;font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word}.asset-description b{color:var(--cyan);font-weight:750}.candidate-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:13px}.candidate{background:#0a1424;border:1px solid var(--line);border-radius:13px;overflow:hidden}.candidate.done{border-color:#58e1ca99}.media{position:relative;cursor:pointer}.media img,.media video{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#020610}.play-tip{position:absolute;inset:auto 8px 8px auto;background:#020610cc;border-radius:7px;padding:5px 8px;font-size:11px}.candidate-body{padding:9px}.candidate-head{font-size:12px;color:var(--muted);margin-bottom:7px}.replay-button{border:1px solid #5d8dff66;border-radius:8px;padding:7px 10px;background:#172b4c;color:#bcd1ff;font-size:11px;font-weight:750;cursor:pointer;white-space:nowrap}.replay-button:hover{border-color:var(--blue);color:white}.candidate-replay{width:100%;margin-bottom:7px}.choices{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.choices button{border:1px solid var(--line);border-radius:8px;padding:8px 3px;background:#152239;color:var(--text);font-size:11px;cursor:pointer}.choices button:hover{border-color:var(--blue)}.choices button.selected{background:#174b53;border-color:var(--cyan);color:#d9fff8}.choices button[data-code="9"].selected{background:#57303a;border-color:var(--red)}.toolbar{margin-top:15px;display:flex;align-items:center;gap:10px}.primary,.secondary{border:0;border-radius:10px;padding:12px 18px;color:white;font-weight:800;cursor:pointer}.primary{background:linear-gradient(135deg,#187b71,#416fdf)}.primary:disabled{opacity:.4;cursor:not-allowed}.secondary{background:#202e45}.status{color:var(--muted);font-size:12px;margin-left:auto}.empty{padding:90px 20px;text-align:center;color:var(--muted)}@media(max-width:1050px){.query{grid-template-columns:1fr}.candidate-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.bar,main{padding-left:11px;padding-right:11px}.candidate-grid{grid-template-columns:1fr}.choices button{font-size:10px}.intro{display:block}.legend{margin-top:10px}.query-foot{align-items:flex-start;flex-direction:column}}
</style></head><body><header><div class="bar"><div class="brand">Task 多模态盲标 · 场景轮</div><div class="progress"><i id="progressBar"></i></div><div class="count" id="count">读取中…</div></div></header>
<main><div class="intro"><div><h1>只判断工作场景，不判断人在做什么</h1><div class="hint">本轮展示库内已有的视频描述，但隐藏模型分数和候选来源。请结合视频与描述判断场景，再给每一对选择一个场景等级。点击画面只负责播放，不会选择标签。</div></div><div class="legend"><span>3 几乎同一工位</span><span>2 工位布局明显相似</span><span>1 同类环境</span><span>0 环境不同</span><span>9 无法判断</span></div></div><div id="workspace"></div><div class="toolbar"><button class="primary" id="submit" disabled>提交这5对并进入下一条</button><button class="secondary" id="undo">撤销上一组</button><span class="status" id="status"></span></div></main>
<script>
const workspace=document.getElementById('workspace'),submit=document.getElementById('submit'),undo=document.getElementById('undo'),statusEl=document.getElementById('status'),countEl=document.getElementById('count'),progressBar=document.getElementById('progressBar');const playbackRateKey='taskDedupPlaybackRate';let current=null,labels={},startedAt=0,preferredPlaybackRate=readPlaybackRate();
function readPlaybackRate(){try{const value=Number(localStorage.getItem(playbackRateKey));return Number.isFinite(value)&&value>=.25&&value<=4?value:2}catch(_){return 2}}
function rememberPlaybackRate(rate,source){if(!Number.isFinite(rate)||rate<.25||rate>4)return;preferredPlaybackRate=rate;try{localStorage.setItem(playbackRateKey,String(rate))}catch(_){}document.querySelectorAll('video').forEach(video=>{if(video!==source&&Math.abs(video.playbackRate-rate)>.001){video.defaultPlaybackRate=rate;video.playbackRate=rate}});statusEl.textContent=`已记住 ${rate}× 倍速，后续视频自动沿用`}
function updateProgress(p){const pct=p.total_pairs?100*p.labeled_pairs/p.total_pairs:0;progressBar.style.width=pct+'%';countEl.textContent=`查询 ${p.completed_queries}/${p.total_queries} · Pair ${p.labeled_pairs}/${p.total_pairs}`}
function pauseResetAll(){document.querySelectorAll('video').forEach(v=>{try{v.pause();const s=Number(v.dataset.start||0);if(Number.isFinite(s))v.currentTime=s}catch(_){}})}
function pauseResetCandidates(except=null){document.querySelectorAll('.candidate video').forEach(v=>{if(v===except)return;try{v.pause();const s=Number(v.dataset.start||0);if(Number.isFinite(s))v.currentTime=s}catch(_){}})}
function restartTaskVideo(video){if(!video)return;const restart=()=>{try{video.currentTime=Number(video.dataset.start||0);video.play().catch(()=>{})}catch(_){}};if(video.readyState>=1)restart();else video.addEventListener('loadedmetadata',restart,{once:true})}
function assetVideo(url,start,cls=''){const v=document.createElement('video');v.controls=true;v.playsInline=true;v.preload='metadata';v.className=cls;v.dataset.start=String(start||0);v.defaultPlaybackRate=preferredPlaybackRate;v.playbackRate=preferredPlaybackRate;v.src=url;v.addEventListener('loadedmetadata',()=>{try{v.currentTime=Number(v.dataset.start||0);v.defaultPlaybackRate=preferredPlaybackRate;v.playbackRate=preferredPlaybackRate}catch(_){}});v.addEventListener('ratechange',()=>rememberPlaybackRate(v.playbackRate,v));return v}
function descriptionBlock(asset){const box=document.createElement('div');box.className='asset-description';const title=document.createElement('b');title.textContent='库内视频描述';box.append(title);const lines=[['场景',asset.scene],['任务',asset.task_name],['详情',asset.details]].filter(([,value])=>String(value||'').trim());if(!lines.length){const empty=document.createElement('span');empty.textContent='暂无描述';box.append(document.createElement('br'),empty);return box}for(const [label,value] of lines){const row=document.createElement('div');row.textContent=`${label}：${value}`;box.append(document.createElement('br'),row)}return box}
async function playCandidate(media,candidate){const loaded=media.querySelector('video');pauseResetCandidates(loaded);if(loaded){loaded.play().catch(()=>{});return}media.classList.add('loading');try{const r=await fetch('/study/api/playback/'+encodeURIComponent(candidate.task_id));if(!r.ok)throw new Error(await r.text());const d=await r.json(),v=assetVideo(d.video_url,d.start_seconds);media.replaceChildren(v);v.play().catch(()=>{})}catch(e){statusEl.textContent='视频加载失败：'+e.message}finally{media.classList.remove('loading')}}
function choose(pairId,code,card){labels[pairId]=code;card.classList.add('done');card.querySelectorAll('.choices button').forEach(b=>b.classList.toggle('selected',Number(b.dataset.code)===code));submit.disabled=Object.keys(labels).length!==(current?.candidates?.length||0)}
function render(d){pauseResetAll();current=d;labels={};startedAt=Date.now();updateProgress(d.progress);statusEl.textContent='';if(d.complete){workspace.innerHTML='<div class="panel empty"><h2>场景轮已完成</h2><p>600个Pair已经全部标注，场景模型指标现已可以解锁。</p></div>';submit.disabled=true;return}const wrap=document.createElement('div');wrap.className='query';const left=document.createElement('section');left.className='panel';const lt=document.createElement('div');lt.className='panel-title';lt.innerHTML='<b>查询 Task</b><span>请循环查看完整片段</span>';const qv=assetVideo(d.query.video_url,d.query.start_seconds,'query-video');const qf=document.createElement('div');qf.className='query-foot';const qinfo=document.createElement('span');qinfo.textContent=`片段时长约 ${Number(d.query.duration_seconds||0).toFixed(1)} 秒 · 当前默认 ${preferredPlaybackRate}× · 修改任意视频倍速后会自动记忆`;const qReplay=document.createElement('button');qReplay.type='button';qReplay.className='replay-button';qReplay.textContent='↺ 从 Task 开头播放';qReplay.addEventListener('click',()=>restartTaskVideo(qv));qf.append(qinfo,qReplay);left.append(lt,qv,descriptionBlock(d.query),qf);const right=document.createElement('section');right.className='panel';const rt=document.createElement('div');rt.className='panel-title';rt.innerHTML='<b>固定候选 Pair（5个都要标）</b><span>只比较环境、设备、工位和布局</span>';const grid=document.createElement('div');grid.className='candidate-grid';d.candidates.forEach((c,i)=>{const card=document.createElement('article');card.className='candidate';const media=document.createElement('div');media.className='media';const img=document.createElement('img');img.src=c.thumbnail_url||'';img.alt='候选场景封面';const tip=document.createElement('span');tip.className='play-tip';tip.textContent='点击播放';media.append(img,tip);media.addEventListener('click',event=>{if(event.target.closest('video'))return;playCandidate(media,c)});const body=document.createElement('div');body.className='candidate-body';const head=document.createElement('div');head.className='candidate-head';head.textContent=`候选 ${i+1} · 片段约 ${Number(c.duration_seconds||0).toFixed(1)} 秒`;const replay=document.createElement('button');replay.type='button';replay.className='replay-button candidate-replay';replay.textContent='↺ 从 Task 开头播放';replay.addEventListener('click',async()=>{await playCandidate(media,c);restartTaskVideo(media.querySelector('video'))});const choices=document.createElement('div');choices.className='choices';[[3,'3 同一'],[2,'2 相似'],[1,'1 同类'],[0,'0 不同'],[9,'9 不清']].forEach(([code,text])=>{const b=document.createElement('button');b.type='button';b.dataset.code=String(code);b.textContent=text;b.addEventListener('click',()=>choose(c.pair_id,code,card));choices.append(b)});body.append(head,descriptionBlock(c),replay,choices);card.append(media,body);grid.append(card)});right.append(rt,grid);wrap.append(left,right);workspace.replaceChildren(wrap);submit.disabled=true}
async function loadNext(){statusEl.textContent='正在加载…';const r=await fetch('/study/api/next');if(!r.ok)throw new Error(await r.text());render(await r.json())}
submit.addEventListener('click',async()=>{if(!current||submit.disabled)return;submit.disabled=true;statusEl.textContent='正在保存…';try{const body={query_task_id:current.query.task_id,labels:current.candidates.map(c=>({pair_id:c.pair_id,relation_code:labels[c.pair_id]})),latency_ms:Date.now()-startedAt};const r=await fetch('/study/api/submit-scene',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());render(await r.json())}catch(e){statusEl.textContent='保存失败：'+e.message;submit.disabled=false}});
undo.addEventListener('click',async()=>{if(!confirm('撤销上一组5个场景标签？'))return;try{const r=await fetch('/study/api/undo-last',{method:'POST'});if(!r.ok)throw new Error(await r.text());await loadNext();statusEl.textContent='已撤销上一组'}catch(e){statusEl.textContent='撤销失败：'+e.message}});
loadNext().catch(e=>{workspace.innerHTML='<div class="panel empty">加载失败：'+e.message+'</div>';statusEl.textContent=e.message});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
