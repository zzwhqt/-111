from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import oss2

from .core import ClipEncoder, extract_frames, pick_task, safe_name, task_bounds


def oss_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "oss" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid OSS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def task_index(sample_id: str) -> int:
    match = re.search(r"_task_(\d+)$", sample_id)
    if not match:
        raise ValueError(f"sample_id has no task suffix: {sample_id}")
    return int(match.group(1))


def tasks_key(video_key: str) -> str:
    directory, filename = video_key.rsplit("/", 1)
    base = re.sub(r"_(left|right)(?:_undistorted)?(?:_proxy)?\.mp4$", "", filename)
    return f"{directory}/{base}_left_tasks.json"


def load_manifest(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("task_name") or "")
            if not sample_id or sample_id in seen or not str(row.get("video_url", "")).startswith("oss://"):
                continue
            seen.add(sample_id)
            rows.append(row)
    return rows


def consolidate(records_dir: Path, data_dir: Path) -> int:
    records = []
    task_vectors = []
    frame_vectors = []
    text_vectors = []
    for meta_path in sorted(records_dir.glob("*.json")):
        vector_path = meta_path.with_suffix(".npz")
        if not vector_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text("utf-8"))
        arrays = np.load(vector_path)
        records.append(metadata)
        task_vectors.append(arrays["task_vector"])
        frame_vectors.append(arrays["frame_vectors"])
        text_vectors.append(arrays["text_vector"])
    if not records:
        raise RuntimeError("no successful records to consolidate")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "task_vectors.npy", np.stack(task_vectors).astype(np.float32))
    np.save(data_dir / "frame_vectors.npy", np.stack(frame_vectors).astype(np.float32))
    np.save(data_dir / "text_vectors.npy", np.stack(text_vectors).astype(np.float32))
    (data_dir / "metadata.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.getenv("OSS_ENDPOINT", "https://oss-cn-shenzhen.aliyuncs.com"))
    parser.add_argument("--data-dir", type=Path, default=Path("/opt/task-dedup/data"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sign-ttl", type=int, default=3600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="write per-task checkpoints only; consolidate once after all parallel shards finish",
    )
    args = parser.parse_args()

    access_key = os.getenv("OSS_ACCESS_KEY_ID")
    secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    if not access_key or not secret:
        raise SystemExit("OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET are required")

    rows = load_manifest(args.manifest)
    records_dir = args.data_dir / "records"
    thumbs_dir = args.data_dir / "thumbnails"
    records_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    encoder = ClipEncoder()
    auth = oss2.Auth(access_key, secret)
    buckets: dict[str, oss2.Bucket] = {}
    task_cache: dict[tuple[str, str], object] = {}
    failures = args.data_dir / "failures.jsonl"
    success = len(list(records_dir.glob("*.npz")))

    for row in rows:
        if success >= args.limit:
            break
        sample_id = str(row.get("sample_id") or row.get("task_name"))
        stem = safe_name(sample_id)
        if (records_dir / f"{stem}.npz").is_file() and (records_dir / f"{stem}.json").is_file():
            continue
        started = time.time()
        work_dir: Path | None = None
        try:
            idx = task_index(sample_id)
            bucket_name, video_key = oss_parts(str(row["video_url"]))
            bucket = buckets.setdefault(bucket_name, oss2.Bucket(auth, args.endpoint, bucket_name))
            metadata_key = tasks_key(video_key)
            cache_key = (bucket_name, metadata_key)
            if cache_key not in task_cache:
                task_cache[cache_key] = json.loads(bucket.get_object(metadata_key).read().decode("utf-8"))
            task = pick_task(task_cache[cache_key], idx)
            if task is None:
                raise ValueError(f"task {idx} not found in {metadata_key}")
            bounds = task_bounds(task, fps=args.fps)
            if bounds is None:
                raise ValueError(f"task {idx} has no valid time/frame range")
            signed_url = bucket.sign_url("GET", video_key, args.sign_ttl, slash_safe=True)
            work_dir = Path(tempfile.mkdtemp(prefix="ingest-task-"))
            frames = extract_frames(signed_url, work_dir, bounds[0], bounds[1])
            task_vector, frame_vectors = encoder.encode(frames)
            scene = str(task.get("scene") or "").strip()
            task_name = str(task.get("task_name") or task.get("task_label") or "").strip()
            details = str(task.get("details") or task.get("description") or "").strip()
            canonical_text = ". ".join(item for item in (scene, task_name, details) if item)
            text_vector = encoder.encode_text(canonical_text or task_name or sample_id)
            thumb_name = f"{stem}.jpg"
            # The labeling/search poster represents the Task boundary, so keep
            # the first sampled frame instead of the visually easier midpoint.
            shutil.copyfile(frames[0], thumbs_dir / thumb_name)
            metadata = {
                "task_id": sample_id,
                "source_video": row["video_url"],
                "task_json": f"oss://{bucket_name}/{metadata_key}",
                "task_index": idx,
                "start_seconds": round(bounds[0], 3),
                "end_seconds": round(bounds[1], 3),
                "duration_seconds": round(bounds[1] - bounds[0], 3),
                "scene": scene,
                "task_name": task_name,
                "details": details,
                "thumbnail": f"thumbnails/{thumb_name}",
            }
            (records_dir / f"{stem}.json").write_text(json.dumps(metadata, ensure_ascii=False), "utf-8")
            # The vector file is the completion marker.  Writing it last makes
            # interrupted ingestion safe to resume without a half-record.
            np.savez_compressed(
                records_dir / f"{stem}.npz",
                task_vector=task_vector,
                frame_vectors=frame_vectors,
                text_vector=text_vector,
            )
            success += 1
            print(json.dumps({"status": "indexed", "count": success, "task_id": sample_id, "seconds": round(time.time() - started, 2)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            with failures.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"task_id": sample_id, "error": str(exc)}, ensure_ascii=False) + "\n")
            print(json.dumps({"status": "failed", "task_id": sample_id, "error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        finally:
            if work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    if args.skip_consolidate:
        print(json.dumps({"status": "shard_complete", "local_success_counter": success}, ensure_ascii=False))
        return 0
    count = consolidate(records_dir, args.data_dir)
    print(json.dumps({"status": "complete", "index_size": count}, ensure_ascii=False))
    return 0 if count >= args.limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
