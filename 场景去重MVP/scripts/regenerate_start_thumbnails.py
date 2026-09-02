#!/usr/bin/env python3
"""Atomically rebuild Task posters from each Task's start timestamp."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import oss2


def ordered_records(pool_dir: Path, study_db: Path | None) -> list[Path]:
    records = sorted((pool_dir / "records").glob("*.json"))
    if study_db is None or not study_db.is_file():
        return records
    with sqlite3.connect(study_db) as connection:
        priority = [
            row[0]
            for row in connection.execute(
                "SELECT task_id FROM reference_assets ORDER BY display_index"
            )
        ]
        current = connection.execute(
            """
            SELECT q.task_id FROM query_tasks q
            LEFT JOIN human_labels h USING(task_id)
            WHERE q.is_label_target=1 AND h.task_id IS NULL
            ORDER BY q.display_order LIMIT 1
            """
        ).fetchone()
    if current:
        priority.append(current[0])
    rank = {task_id: index for index, task_id in enumerate(priority)}
    return sorted(
        records,
        key=lambda path: (
            rank.get(json.loads(path.read_text("utf-8"))["task_id"], len(rank)),
            path.name,
        ),
    )


def rebuild_one(
    record_path: Path,
    pool_dir: Path,
    endpoint: str,
    auth: oss2.Auth,
    ffmpeg: str,
    buckets: dict[str, oss2.Bucket],
) -> str:
    metadata = json.loads(record_path.read_text("utf-8"))
    task_id = str(metadata["task_id"])
    parsed = urlparse(str(metadata["source_video"]))
    if parsed.scheme != "oss" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid OSS source for {task_id}")
    bucket = buckets.setdefault(parsed.netloc, oss2.Bucket(auth, endpoint, parsed.netloc))
    signed_url = bucket.sign_url("GET", parsed.path.lstrip("/"), 1800, slash_safe=True)
    start = max(0.0, float(metadata.get("start_seconds") or 0.0))
    thumbnail = pool_dir / str(metadata["thumbnail"])
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{thumbnail.stem}.", suffix=".jpg", dir=thumbnail.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                signed_url,
                "-frames:v",
                "1",
                "-vf",
                "scale=336:336:force_original_aspect_ratio=decrease,"
                "pad=336:336:(ow-iw)/2:(oh-ih)/2",
                "-q:v",
                "3",
                "-y",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        if temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty thumbnail")
        os.replace(temporary, thumbnail)
        return task_id
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--study-db", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default=os.getenv("DEDUP_FFMPEG", "ffmpeg"))
    parser.add_argument("--endpoint", default=os.getenv("OSS_ENDPOINT"))
    args = parser.parse_args()
    access_key = os.getenv("OSS_ACCESS_KEY_ID")
    secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    if not access_key or not secret or not args.endpoint:
        raise SystemExit("OSS credentials and endpoint are required")

    records = ordered_records(args.pool_dir, args.study_db)
    auth = oss2.Auth(access_key, secret)
    buckets: dict[str, oss2.Bucket] = {}
    completed = 0
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                rebuild_one,
                record,
                args.pool_dir,
                args.endpoint,
                auth,
                args.ffmpeg,
                buckets,
            ): record
            for record in records
        }
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(records):
                    print(
                        json.dumps(
                            {"completed": completed, "total": len(records)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            except Exception as exc:
                failures.append({"record": futures[future].name, "error": str(exc)})

    print(
        json.dumps(
            {"completed": completed, "total": len(records), "failures": failures},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
