from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import faiss
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .core import l2_normalize


POOL_DIR = Path(os.getenv("DEDUP_LABEL_POOL_DIR", "/opt/task-dedup/labeling/pool"))
DB_PATH = Path(os.getenv("DEDUP_LABEL_DB", "/opt/task-dedup/labeling/study.sqlite3"))
router = APIRouter(prefix="/label", tags=["human-label-study"])


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
          display_order INTEGER,
          sample_band TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          model_top5_json TEXT NOT NULL,
          model_top1_warning TEXT,
          model_top1_score REAL,
          is_label_target INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS human_labels (
          task_id TEXT PRIMARY KEY REFERENCES query_tasks(task_id),
          human_label TEXT NOT NULL CHECK(human_label IN ('duplicate','review','novel','bad')),
          selected_reference_id TEXT REFERENCES reference_assets(task_id),
          comment TEXT NOT NULL DEFAULT '',
          latency_ms INTEGER,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_query_order ON query_tasks(is_label_target, display_order);
        """
    )


def _metadata(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(row["metadata_json"])


def _thumbnail_url(metadata: dict[str, Any]) -> str | None:
    thumbnail = str(metadata.get("thumbnail") or "")
    if not thumbnail:
        return None
    name = Path(thumbnail).name
    path = POOL_DIR / "thumbnails" / name
    version = path.stat().st_mtime_ns if path.is_file() else 0
    return "/label-thumbnails/" + quote(name) + f"?v={version}"


def _signed_video(metadata: dict[str, Any]) -> str:
    uri = str(metadata.get("source_video") or "")
    parsed = urlparse(uri)
    allowed = {
        item.strip()
        for item in os.getenv("DEDUP_ALLOWED_OSS_BUCKETS", "egoscale-v3").split(",")
        if item.strip()
    }
    if parsed.scheme != "oss" or not parsed.netloc or not parsed.path:
        raise HTTPException(422, "label asset has no valid OSS source")
    if parsed.netloc not in allowed:
        raise HTTPException(403, "label asset bucket is not allowed")
    try:
        import oss2
    except ImportError as exc:
        raise HTTPException(500, "oss2 is unavailable") from exc
    access_key = os.getenv("OSS_ACCESS_KEY_ID")
    secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    endpoint = os.getenv("OSS_ENDPOINT")
    if not access_key or not secret or not endpoint:
        raise HTTPException(503, "OSS signing is not configured")
    auth = oss2.Auth(access_key, secret)
    bucket = oss2.Bucket(auth, endpoint, parsed.netloc)
    ttl = min(max(int(os.getenv("DEDUP_SIGNED_URL_TTL", "900")), 60), 3600)
    url = bucket.sign_url("GET", parsed.path.lstrip("/"), ttl, slash_safe=True)
    start = float(metadata.get("start_seconds") or 0.0)
    end = float(metadata.get("end_seconds") or start + 1.0)
    return f"{url}#t={start:.3f},{end:.3f}"


def _public_asset(task_id: str, metadata: dict[str, Any], display_index: int | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "display_index": display_index,
        "scene": metadata.get("scene") or "",
        "task_name": metadata.get("task_name") or "",
        "details": metadata.get("details") or "",
        "start_seconds": metadata.get("start_seconds"),
        "end_seconds": metadata.get("end_seconds"),
        "duration_seconds": metadata.get("duration_seconds"),
        "thumbnail_url": _thumbnail_url(metadata),
    }


def _progress(connection: sqlite3.Connection) -> dict[str, int]:
    target = int(connection.execute("SELECT COUNT(*) FROM query_tasks WHERE is_label_target=1").fetchone()[0])
    labeled = int(
        connection.execute(
            "SELECT COUNT(*) FROM human_labels h JOIN query_tasks q USING(task_id) WHERE q.is_label_target=1"
        ).fetchone()[0]
    )
    return {"target": target, "labeled": labeled, "remaining": max(0, target - labeled)}


def _next_payload(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT q.* FROM query_tasks q
        LEFT JOIN human_labels h USING(task_id)
        WHERE q.is_label_target=1 AND h.task_id IS NULL
        ORDER BY q.display_order LIMIT 1
        """
    ).fetchone()
    progress = _progress(connection)
    if row is None:
        return {"complete": True, "progress": progress}
    metadata = _metadata(row)
    candidates = json.loads(row["model_top5_json"])
    # Do not expose rank or score before labeling. A stable shuffle prevents
    # the candidate card order from revealing the model's top-1 prediction.
    seed = int(hashlib.sha256(row["task_id"].encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(candidates)
    public_candidates = []
    for candidate in candidates:
        ref = connection.execute(
            "SELECT * FROM reference_assets WHERE task_id=?", (candidate["task_id"],)
        ).fetchone()
        if ref is None:
            continue
        public_candidates.append(_public_asset(ref["task_id"], _metadata(ref), ref["display_index"]))
    query = _public_asset(row["task_id"], metadata)
    query["video_url"] = _signed_video(metadata)
    return {
        "complete": False,
        "query": query,
        "quick_candidates": public_candidates,
        "progress": progress,
    }


class LabelSubmission(BaseModel):
    task_id: str
    human_label: str
    selected_reference_id: str | None = None
    comment: str = Field(default="", max_length=1000)
    latency_ms: int | None = Field(default=None, ge=0, le=86_400_000)


@router.get("", response_class=HTMLResponse)
def label_home() -> str:
    return _LABEL_HTML


@router.get("/report", response_class=HTMLResponse)
def label_report_home() -> str:
    return _REPORT_HTML


@router.get("/api/progress")
def progress() -> dict[str, int]:
    with _connect() as connection:
        _init_schema(connection)
        return _progress(connection)


@router.get("/api/next")
def next_query() -> dict[str, Any]:
    with _connect() as connection:
        _init_schema(connection)
        return _next_payload(connection)


@router.get("/api/references")
def references() -> dict[str, Any]:
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM reference_assets ORDER BY display_index").fetchall()
        return {
            "references": [
                _public_asset(row["task_id"], _metadata(row), row["display_index"]) for row in rows
            ]
        }


@router.get("/api/reference/{task_id}/playback")
def reference_playback(task_id: str) -> dict[str, str]:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM reference_assets WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "reference asset not found")
        return {"video_url": _signed_video(_metadata(row))}


@router.post("/api/submit")
def submit_label(submission: LabelSubmission) -> dict[str, Any]:
    if submission.human_label not in {"duplicate", "review", "novel", "bad"}:
        raise HTTPException(400, "invalid human_label")
    if submission.human_label in {"duplicate", "review"} and not submission.selected_reference_id:
        raise HTTPException(400, "duplicate/review requires a selected reference")
    if submission.human_label in {"novel", "bad"}:
        submission.selected_reference_id = None
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        query = connection.execute(
            "SELECT task_id FROM query_tasks WHERE task_id=? AND is_label_target=1", (submission.task_id,)
        ).fetchone()
        if query is None:
            raise HTTPException(404, "query task not found")
        if submission.selected_reference_id:
            reference = connection.execute(
                "SELECT task_id FROM reference_assets WHERE task_id=?", (submission.selected_reference_id,)
            ).fetchone()
            if reference is None:
                raise HTTPException(400, "selected reference not found")
        connection.execute(
            """
            INSERT INTO human_labels(task_id,human_label,selected_reference_id,comment,latency_ms,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(task_id) DO UPDATE SET
              human_label=excluded.human_label,
              selected_reference_id=excluded.selected_reference_id,
              comment=excluded.comment,
              latency_ms=excluded.latency_ms,
              updated_at=excluded.updated_at
            """,
            (
                submission.task_id,
                submission.human_label,
                submission.selected_reference_id,
                submission.comment.strip(),
                submission.latency_ms,
                now,
                now,
            ),
        )
        if submission.human_label == "bad":
            current_order = connection.execute(
                "SELECT display_order FROM query_tasks WHERE task_id=?", (submission.task_id,)
            ).fetchone()[0]
            reserve = connection.execute(
                """
                SELECT q.task_id FROM query_tasks q LEFT JOIN human_labels h USING(task_id)
                WHERE q.is_label_target=0 AND q.sample_band LIKE 'reserve_%' AND h.task_id IS NULL
                ORDER BY q.sample_band,q.task_id LIMIT 1
                """
            ).fetchone()
            if reserve is not None:
                connection.execute(
                    "UPDATE query_tasks SET is_label_target=0 WHERE task_id=?", (submission.task_id,)
                )
                connection.execute(
                    "UPDATE query_tasks SET is_label_target=1,display_order=? WHERE task_id=?",
                    (current_order, reserve["task_id"]),
                )
        connection.commit()
        return _next_payload(connection)


@router.post("/api/undo-last")
def undo_last() -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT h.task_id,h.human_label,q.is_label_target,q.display_order
            FROM human_labels h JOIN query_tasks q USING(task_id)
            ORDER BY h.updated_at DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise HTTPException(404, "no label to undo")
        if row["human_label"] == "bad" and not row["is_label_target"]:
            promoted = connection.execute(
                """
                SELECT q.task_id FROM query_tasks q LEFT JOIN human_labels h USING(task_id)
                WHERE q.is_label_target=1 AND q.display_order=?
                  AND q.sample_band LIKE 'reserve_%' AND h.task_id IS NULL
                LIMIT 1
                """,
                (row["display_order"],),
            ).fetchone()
            if promoted is not None:
                connection.execute(
                    "UPDATE query_tasks SET is_label_target=0,display_order=NULL WHERE task_id=?",
                    (promoted["task_id"],),
                )
                connection.execute(
                    "UPDATE query_tasks SET is_label_target=1 WHERE task_id=?", (row["task_id"],)
                )
        connection.execute("DELETE FROM human_labels WHERE task_id=?", (row["task_id"],))
        connection.commit()
        return {"undone_task_id": row["task_id"], **_progress(connection)}


def _report_payload(connection: sqlite3.Connection) -> dict[str, Any]:
    progress_value = _progress(connection)
    counts = {
        row["human_label"]: int(row["count"])
        for row in connection.execute(
            "SELECT human_label,COUNT(*) AS count FROM human_labels GROUP BY human_label"
        ).fetchall()
    }
    if progress_value["labeled"] < progress_value["target"]:
        return {
            "complete": False,
            "progress": progress_value,
            "human_label_counts": counts,
            "message": "为避免人工判断被模型影响，完整模型指标会在1800条盲标完成后解锁。",
        }
    rows = connection.execute(
        """
        SELECT q.model_top5_json,q.model_top1_warning,q.model_top1_score,
               h.human_label,h.selected_reference_id
        FROM query_tasks q JOIN human_labels h USING(task_id)
        WHERE q.is_label_target=1 AND h.human_label!='bad'
        """
    ).fetchall()
    evaluable = len(rows)
    positive = 0
    top1_hits = 0
    top5_hits = 0
    reciprocal_rank_sum = 0.0
    high_total = 0
    high_true = 0
    novel_total = 0
    novel_high = 0
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        top5 = json.loads(row["model_top5_json"])
        ids = [item["task_id"] for item in top5]
        human = row["human_label"]
        predicted = row["model_top1_warning"] or "low"
        confusion.setdefault(human, {}).setdefault(predicted, 0)
        confusion[human][predicted] += 1
        if human == "duplicate" and row["selected_reference_id"]:
            positive += 1
            top1_hits += int(bool(ids) and ids[0] == row["selected_reference_id"])
            top5_hits += int(row["selected_reference_id"] in ids)
            if row["selected_reference_id"] in ids:
                reciprocal_rank_sum += 1.0 / (ids.index(row["selected_reference_id"]) + 1)
        if predicted == "high":
            high_total += 1
            high_true += int(
                human == "duplicate"
                and bool(ids)
                and row["selected_reference_id"] == ids[0]
            )
        if human == "novel":
            novel_total += 1
            novel_high += int(predicted == "high")
    return {
        "complete": True,
        "progress": progress_value,
        "human_label_counts": counts,
        "evaluable": evaluable,
        "duplicate_queries": positive,
        "recall_at_1": round(top1_hits / positive, 4) if positive else None,
        "recall_at_5": round(top5_hits / positive, 4) if positive else None,
        "mrr_at_5": round(reciprocal_rank_sum / positive, 4) if positive else None,
        "high_precision": round(high_true / high_total, 4) if high_total else None,
        "novel_high_false_positive_rate": round(novel_high / novel_total, 4) if novel_total else None,
        "confusion": confusion,
    }


@router.get("/api/report")
def report() -> dict[str, Any]:
    with _connect() as connection:
        return _report_payload(connection)


@router.get("/export.csv")
def export_csv() -> StreamingResponse:
    with _connect() as connection:
        current_progress = _progress(connection)
        complete = current_progress["target"] > 0 and current_progress["labeled"] >= current_progress["target"]
        if complete:
            rows = connection.execute(
                """
                SELECT q.task_id,q.is_label_target,q.sample_band,q.model_top1_warning,q.model_top1_score,
                       h.human_label,h.selected_reference_id,h.comment,h.latency_ms,h.updated_at
                FROM query_tasks q LEFT JOIN human_labels h USING(task_id)
                WHERE q.is_label_target=1 OR h.task_id IS NOT NULL
                ORDER BY COALESCE(q.display_order,999999),h.updated_at
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT q.task_id,h.human_label,h.selected_reference_id,h.comment,h.latency_ms,h.updated_at
                FROM query_tasks q LEFT JOIN human_labels h USING(task_id)
                WHERE q.is_label_target=1 OR h.task_id IS NOT NULL
                ORDER BY COALESCE(q.display_order,999999),h.updated_at
                """
            ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(rows[0].keys() if rows else [])
    writer.writerows([tuple(row) for row in rows])
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=task_scene_labels.csv"},
    )


def _select_references(vectors: np.ndarray, metadata: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    dimension = vectors.shape[1]
    kmeans = faiss.Kmeans(dimension, count, niter=30, nredo=3, seed=seed, verbose=False, spherical=True)
    kmeans.train(vectors)
    centroid_scores = kmeans.centroids @ vectors.T
    chosen: list[int] = []
    used_sources: set[str] = set()
    for centroid_index in range(count):
        for index in np.argsort(-centroid_scores[centroid_index]):
            idx = int(index)
            source = str(metadata[idx].get("source_video") or "")
            if idx not in chosen and source not in used_sources:
                chosen.append(idx)
                used_sources.add(source)
                break
    if len(chosen) != count:
        raise RuntimeError(f"could only select {len(chosen)} references")
    return chosen


def _score_query(
    query_index: int,
    reference_indices: list[int],
    task_vectors: np.ndarray,
    frame_vectors: np.ndarray,
    negative_scores: np.ndarray,
    visual_p95: float,
) -> list[dict[str, Any]]:
    raw_all = task_vectors[reference_indices] @ task_vectors[query_index]
    recall_positions = np.argsort(-raw_all)[: min(30, len(reference_indices))]
    hits = []
    query_frames = frame_vectors[query_index]
    for position in recall_positions:
        reference_index = reference_indices[int(position)]
        raw = float(raw_all[int(position)])
        pairwise = query_frames @ frame_vectors[reference_index].T
        coverage = float((pairwise.max(axis=1).mean() + pairwise.max(axis=0).mean()) / 2.0)
        distinctiveness = float(np.clip((raw - visual_p95) / max(1e-6, 1.0 - visual_p95), 0.0, 1.0))
        score = float(np.clip(0.70 * distinctiveness + 0.30 * coverage, 0.0, 1.0) * 100.0)
        percentile = float(100.0 * np.searchsorted(negative_scores, raw, side="right") / len(negative_scores))
        if raw >= 0.995 and coverage >= 0.98:
            warning = "high"
        elif score >= 55.0 or percentile >= 99.0:
            warning = "review"
        else:
            warning = "low"
        hits.append(
            {
                "reference_index": reference_index,
                "raw_cosine": round(raw, 6),
                "frame_coverage": round(coverage, 6),
                "similarity_percent": round(score, 2),
                "corpus_percentile": round(percentile, 3),
                "warning_level": warning,
            }
        )
    hits.sort(key=lambda item: item["similarity_percent"], reverse=True)
    return hits[:5]


def initialize_study(pool_dir: Path, db_path: Path, seed: int = 20260828) -> dict[str, Any]:
    global DB_PATH
    DB_PATH = db_path
    metadata = json.loads((pool_dir / "metadata.json").read_text("utf-8"))
    task_vectors = l2_normalize(np.load(pool_dir / "task_vectors.npy").astype(np.float32))
    frame_vectors = l2_normalize(np.load(pool_dir / "frame_vectors.npy").astype(np.float32))
    if len(metadata) != 2000 or len(task_vectors) != 2000 or len(frame_vectors) != 2000:
        raise RuntimeError("label study requires exactly 2000 consolidated tasks")
    reference_indices = _select_references(task_vectors, metadata, 50, seed)
    reference_set = set(reference_indices)
    remaining = [idx for idx in range(len(metadata)) if idx not in reference_set]
    reference_matrix = task_vectors[reference_indices]
    max_raw = task_vectors[remaining] @ reference_matrix.T
    ordered = [remaining[int(pos)] for pos in np.argsort(max_raw.max(axis=1))]
    bands = [ordered[:650], ordered[650:1300], ordered[1300:]]
    rng = random.Random(seed)
    targets: list[tuple[int, str]] = []
    reserves: list[tuple[int, str]] = []
    for name, band in zip(("low", "medium", "high"), bands):
        rng.shuffle(band)
        targets.extend((idx, name) for idx in band[:600])
        reserves.extend((idx, f"reserve_{name}") for idx in band[600:])
    rng.shuffle(targets)
    reference_sources = {str(metadata[idx].get("source_video") or "") for idx in reference_indices}

    reference_pair_scores = reference_matrix @ reference_matrix.T
    upper = np.triu_indices(len(reference_indices), k=1)
    negatives = np.sort(reference_pair_scores[upper].astype(np.float32))
    visual_p95 = float(np.quantile(negatives, 0.95))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        _init_schema(connection)
        existing = sum(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("study_settings", "reference_assets", "query_tasks", "human_labels")
        )
        if existing:
            raise RuntimeError(f"study database is already initialized: {db_path}")
        settings = {
            "seed": str(seed),
            "pool_size": "2000",
            "reference_count": "50",
            "label_target_count": "1800",
            "reserve_count": "150",
            "visual_p95": repr(visual_p95),
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
        for display_order, (vector_index, band) in enumerate(targets, start=1):
            effective_band = (
                f"{band}_same_source"
                if str(metadata[vector_index].get("source_video") or "") in reference_sources
                else band
            )
            hits = _score_query(vector_index, reference_indices, task_vectors, frame_vectors, negatives, visual_p95)
            for hit in hits:
                hit["task_id"] = str(metadata[hit.pop("reference_index")]["task_id"])
            connection.execute(
                "INSERT INTO query_tasks VALUES(?,?,?,?,?,?,?,?,1)",
                (
                    str(metadata[vector_index]["task_id"]),
                    vector_index,
                    display_order,
                    effective_band,
                    json.dumps(metadata[vector_index], ensure_ascii=False),
                    json.dumps(hits, ensure_ascii=False),
                    hits[0]["warning_level"],
                    hits[0]["similarity_percent"],
                ),
            )
        for vector_index, band in reserves:
            effective_band = (
                f"{band}_same_source"
                if str(metadata[vector_index].get("source_video") or "") in reference_sources
                else band
            )
            hits = _score_query(vector_index, reference_indices, task_vectors, frame_vectors, negatives, visual_p95)
            for hit in hits:
                hit["task_id"] = str(metadata[hit.pop("reference_index")]["task_id"])
            connection.execute(
                "INSERT INTO query_tasks VALUES(?,?,?,?,?,?,?,?,0)",
                (
                    str(metadata[vector_index]["task_id"]),
                    vector_index,
                    None,
                    effective_band,
                    json.dumps(metadata[vector_index], ensure_ascii=False),
                    json.dumps(hits, ensure_ascii=False),
                    hits[0]["warning_level"],
                    hits[0]["similarity_percent"],
                ),
            )
        connection.commit()
    return {
        "pool": len(metadata),
        "references": len(reference_indices),
        "targets": len(targets),
        "reserves": len(reserves),
        "visual_p95": round(visual_p95, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    init.add_argument("--db", type=Path, default=DB_PATH)
    init.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize_study(args.pool_dir, args.db, args.seed), ensure_ascii=False))
    return 0


_LABEL_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task 场景盲标工作台</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#111a2b;--line:#283650;--text:#edf3ff;--muted:#96a8c1;--cyan:#5eead4;--blue:#5b8cff;--yellow:#ffd166;--red:#ff7d88}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#17325b 0,transparent 30%),var(--bg);color:var(--text);font-family:Inter,system-ui,"PingFang SC",sans-serif}header{position:sticky;top:0;z-index:10;background:#080d18e8;backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}.bar,main{max-width:1320px;margin:auto;padding:16px 22px}.bar{display:flex;align-items:center;gap:18px}.brand{font-weight:800}.progress{height:8px;background:#1b2840;border-radius:99px;overflow:hidden;flex:1}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--blue));width:0}.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}main{padding-top:28px}.intro{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:18px}h1{margin:0;font-size:28px}.hint{color:var(--muted);font-size:13px;line-height:1.55;max-width:740px}.layout{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(420px,.85fr);gap:20px}.panel{background:#111a2be8;border:1px solid var(--line);border-radius:18px;overflow:hidden}.panel-title{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px}.query-video{width:100%;aspect-ratio:16/9;background:#030711;object-fit:contain}.query-meta{padding:15px 18px;color:var(--muted);font-size:12px}.candidate-list{padding:14px;display:grid;grid-template-columns:1fr 1fr;gap:12px;max-height:610px;overflow:auto}.candidate{border:1px solid var(--line);border-radius:13px;overflow:hidden;background:#0b1322;cursor:pointer;transition:.15s}.candidate:hover{border-color:#5b8cff99}.candidate.selected{border-color:var(--cyan);box-shadow:0 0 0 2px #5eead433}.candidate img,.candidate video{width:100%;aspect-ratio:16/9;object-fit:cover;background:#030711;display:block}.candidate .body{padding:10px}.candidate b{font-size:13px}.candidate p{font-size:11px;color:var(--muted);margin:5px 0 0;line-height:1.35}.candidate .pick{width:100%;margin-top:9px;border:1px solid #5eead466;border-radius:8px;padding:8px;background:#17364a;color:var(--cyan);font-weight:750;cursor:pointer}.actions{margin-top:18px;padding:18px;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px}.actions button,.secondary{border:0;border-radius:11px;padding:13px 10px;color:white;font-weight:750;cursor:pointer}.duplicate{background:#1ea97c}.review{background:#b98718}.novel{background:#386ee8}.bad{background:#ad4250}.secondary{background:#202d44}.footer-tools{display:flex;gap:10px;margin-top:12px;align-items:center}.footer-tools input{flex:1;background:#0b1322;border:1px solid var(--line);border-radius:10px;color:var(--text);padding:11px}.footer-tools button{min-width:100px}.status{color:var(--muted);font-size:12px;margin-left:auto}.modal{position:fixed;inset:0;background:#030711e8;z-index:30;display:none;padding:24px;overflow:auto}.modal.open{display:block}.modal-head{max-width:1200px;margin:auto;display:flex;justify-content:space-between;align-items:center}.all-grid{max-width:1200px;margin:18px auto;display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.empty{padding:80px;text-align:center;color:var(--muted)}@media(max-width:900px){.layout{grid-template-columns:1fr}.candidate-list{max-height:none}.all-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.bar,main{padding-left:12px;padding-right:12px}.actions{grid-template-columns:1fr 1fr}.candidate-list{grid-template-columns:1fr}.all-grid{grid-template-columns:1fr}}
</style></head><body><header><div class="bar"><div class="brand">场景盲标</div><div class="progress"><i id="progressBar"></i></div><div id="progressCount" class="count">0 / 1800</div></div></header><main><div class="intro"><div><h1>人工判断 Task 是否重复</h1><div class="hint">模型分数和真实排名在1800条完成前隐藏。先看左侧查询，再从右侧选择最接近的参考资产；候选没有命中时打开全部50个参考。重复/相似必须先选参考资产。</div></div><div><a href="/label/report" style="color:var(--cyan);margin-right:16px">评测仪表盘</a><a href="/label/export.csv" style="color:var(--cyan)">导出标注CSV</a></div></div><div id="workspace"><div class="empty panel">正在加载评测集…</div></div></main><div id="allModal" class="modal"><div class="modal-head"><h2>全部50个参考资产</h2><button id="closeModal" class="secondary">关闭</button></div><div id="allGrid" class="all-grid"></div></div><script>
let current=null,selected=null,started=0,referencesCache=null;const workspace=document.getElementById('workspace'),bar=document.getElementById('progressBar'),count=document.getElementById('progressCount'),modal=document.getElementById('allModal'),allGrid=document.getElementById('allGrid');
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n};const escText=v=>String(v||'');
function progress(p){count.textContent=`${p.labeled} / ${p.target}`;bar.style.width=`${p.target?100*p.labeled/p.target:0}%`}
function resetReferenceVideos(){modal.classList.remove('open');document.querySelectorAll('.candidate video').forEach(video=>{video.pause();const start=Number(video.dataset.startSeconds||0);const reset=()=>{video.pause();try{video.currentTime=start}catch(_){}};if(video.readyState>=1)reset();else video.addEventListener('loadedmetadata',reset,{once:true})})}
async function loadVideo(card,asset){let video=card.querySelector('video');if(video)return video;const r=await fetch(`/label/api/reference/${encodeURIComponent(asset.task_id)}/playback`),d=await r.json();video=el('video');video.controls=true;video.preload='metadata';video.dataset.startSeconds=String(Number(asset.start_seconds||0));video.src=d.video_url;video.onclick=e=>e.stopPropagation();const old=card.querySelector('img');if(old)old.replaceWith(video);else card.prepend(video);return video}
function chooseAsset(card,asset,closeModal){document.querySelectorAll('.candidate').forEach(x=>x.classList.toggle('selected',x.dataset.taskId===asset.task_id));selected=asset;const status=document.getElementById('status');if(status)status.textContent=`已选择参考 ${asset.display_index}`;if(closeModal)modal.classList.remove('open')}
function card(asset,quick=false){const c=el('div','candidate'),img=el('img');c.dataset.taskId=asset.task_id;img.src=asset.thumbnail_url||'';img.alt=`参考 ${asset.display_index}`;const body=el('div','body'),title=el('b','',`参考 ${asset.display_index} · ${escText(asset.task_name)||'未命名任务'}`),desc=el('p','',[asset.scene,asset.details].filter(Boolean).join(' · ')||'点击画面播放参考片段');body.append(title,desc);c.append(img,body);img.onclick=async e=>{e.stopPropagation();const video=await loadVideo(c,asset);try{await video.play()}catch(_){}};if(quick){c.onclick=async()=>{chooseAsset(c,asset,false);await loadVideo(c,asset)}}else{c.style.cursor='default';const pick=el('button','pick','选择该资产');pick.type='button';pick.onclick=e=>{e.stopPropagation();chooseAsset(c,asset,true)};body.append(pick)}return c}
async function showAll(){modal.classList.add('open');if(referencesCache)return;const r=await fetch('/label/api/references'),d=await r.json();referencesCache=d.references;allGrid.replaceChildren(...referencesCache.map(x=>card(x,false)))}
function render(data){resetReferenceVideos();progress(data.progress);selected=null;started=Date.now();if(data.complete){current=null;workspace.innerHTML='<div class="empty panel"><h2>1800条盲标已完成</h2><p><a href="/label/report" style="color:var(--cyan)">打开评测仪表盘</a>查看完整指标。</p></div>';return}current=data.query;const layout=el('div','layout'),left=el('section','panel'),right=el('section','panel'),lh=el('div','panel-title','查询Task'),rh=el('div','panel-title');rh.append(el('span','','可能相关的参考资产（顺序已打乱）'));const all=el('button','secondary','查看全部50个');all.onclick=showAll;rh.append(all);const video=el('video','query-video');video.controls=true;video.autoplay=false;video.preload='metadata';video.src=current.video_url;if(current.thumbnail_url)video.poster=current.thumbnail_url;left.append(lh,video,el('div','query-meta',`${current.task_id} · ${current.start_seconds}s → ${current.end_seconds}s`));const list=el('div','candidate-list');data.quick_candidates.forEach(x=>list.append(card(x,true)));right.append(rh,list);layout.append(left,right);const actions=el('div','panel actions');[['duplicate','重复 D'],['review','相似/复核 R'],['novel','全新 N'],['bad','坏片 B']].forEach(([value,text])=>{const b=el('button',value,text);b.onclick=()=>submit(value);actions.append(b)});const tools=el('div','footer-tools'),comment=el('input');comment.id='comment';comment.placeholder='可选备注：为什么重复/不重复';const undo=el('button','secondary','撤销上一条');undo.onclick=undoLast;const status=el('span','status','请选择标签');status.id='status';tools.append(comment,undo,status);workspace.replaceChildren(layout,actions,tools)}
async function submit(label){if(!current||modal.classList.contains('open'))return;const status=document.getElementById('status');if((label==='duplicate'||label==='review')&&!selected){status.textContent='请先选择一个参考资产';status.style.color='var(--red)';return}document.querySelectorAll('.actions button').forEach(b=>b.disabled=true);const payload={task_id:current.task_id,human_label:label,selected_reference_id:selected?.task_id||null,comment:document.getElementById('comment').value,latency_ms:Date.now()-started};try{const r=await fetch('/label/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),d=await r.json();if(!r.ok)throw new Error(d.detail||'提交失败');render(d)}catch(e){status.textContent=e.message;document.querySelectorAll('.actions button').forEach(b=>b.disabled=false)}}
async function undoLast(){const r=await fetch('/label/api/undo-last',{method:'POST'}),d=await r.json();if(!r.ok){alert(d.detail||'无法撤销');return}await loadNext()}
async function loadNext(){const r=await fetch('/label/api/next'),d=await r.json();if(!r.ok){workspace.textContent=d.detail||'加载失败';return}render(d)}
document.getElementById('closeModal').onclick=()=>modal.classList.remove('open');document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key.toLowerCase()==='d')submit('duplicate');if(e.key.toLowerCase()==='r')submit('review');if(e.key.toLowerCase()==='n')submit('novel');if(e.key.toLowerCase()==='b')submit('bad')});loadNext();
</script></body></html>"""


_REPORT_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>场景检索评测仪表盘</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#111a2b;--line:#293851;--text:#edf3ff;--muted:#96a8c1;--cyan:#5eead4}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#18335c,transparent 30%),var(--bg);color:var(--text);font-family:Inter,system-ui,"PingFang SC",sans-serif}main{max-width:1100px;margin:auto;padding:46px 22px}a{color:var(--cyan)}h1{font-size:36px;margin:12px 0}.muted{color:var(--muted);line-height:1.6}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px}.card b{font-size:30px;display:block;margin-bottom:6px}.card span{color:var(--muted);font-size:12px}pre{white-space:pre-wrap;word-break:break-word;background:#080e1a;border:1px solid var(--line);border-radius:14px;padding:18px;color:#c7d5eb}@media(max-width:760px){.grid{grid-template-columns:1fr 1fr}}</style></head><body><main><a href="/label">← 返回盲标工作台</a><h1>场景检索评测仪表盘</h1><p id="message" class="muted">正在读取评测进度…</p><section id="cards" class="grid"></section><pre id="details">完整混淆矩阵将在1800条盲标完成后显示。</pre></main><script>
const cards=document.getElementById('cards'),msg=document.getElementById('message'),details=document.getElementById('details');const card=(v,n)=>{const d=document.createElement('div');d.className='card';const b=document.createElement('b');b.textContent=v??'-';const s=document.createElement('span');s.textContent=n;d.append(b,s);return d};fetch('/label/api/report').then(r=>r.json()).then(d=>{msg.textContent=d.complete?'盲标已完成，以下是冻结模型预测与人工标签的对比。':d.message;cards.replaceChildren(card(`${d.progress.labeled}/${d.progress.target}`,'标注进度'),card(d.recall_at_1==null?'-':(100*d.recall_at_1).toFixed(1)+'%','Recall@1'),card(d.recall_at_5==null?'-':(100*d.recall_at_5).toFixed(1)+'%','Recall@5'),card(d.high_precision==null?'-':(100*d.high_precision).toFixed(1)+'%','High 预警精确率'));details.textContent=JSON.stringify({mrr_at_5:d.mrr_at_5,human_label_counts:d.human_label_counts,novel_high_false_positive_rate:d.novel_high_false_positive_rate,confusion:d.confusion},null,2)}).catch(e=>msg.textContent=e.message);
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
