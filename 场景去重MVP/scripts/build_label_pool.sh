#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/task-dedup/app
SOURCE_DIR=/opt/task-dedup/data
POOL_DIR=/opt/task-dedup/labeling/pool
MANIFEST_DIR=/opt/task-dedup/manifests
PYTHON=/opt/micromamba/root/envs/task-dedup/bin/python
TARGET_TOTAL=2000
export DEDUP_FFMPEG=/opt/micromamba/root/envs/task-dedup/bin/ffmpeg
export DEDUP_FFPROBE=/opt/micromamba/root/envs/task-dedup/bin/ffprobe

if [[ -f /etc/task-scene-dedup.env ]]; then
  # shellcheck disable=SC1091
  set -a
  source /etc/task-scene-dedup.env
  set +a
fi
if [[ -z "${OSS_ACCESS_KEY_ID:-}" || -z "${OSS_ACCESS_KEY_SECRET:-}" ]]; then
  echo "OSS credentials are required" >&2
  exit 2
fi

mkdir -p "$POOL_DIR/records" "$POOL_DIR/thumbnails"
if [[ ! -e "$POOL_DIR/.seeded_from_main" ]]; then
  cp -al "$SOURCE_DIR/records/." "$POOL_DIR/records/"
  cp -al "$SOURCE_DIR/thumbnails/." "$POOL_DIR/thumbnails/"
  touch "$POOL_DIR/.seeded_from_main"
fi

current=$(find "$POOL_DIR/records" -name '*.npz' | wc -l)
remaining=$((TARGET_TOTAL - current))
if (( remaining > 0 )); then
  add_a=$(((remaining + 1) / 2))
  add_b=$((remaining / 2))
  target_a=$((current + add_a))
  target_b=$((current + add_b))

  cd "$APP_DIR"
  DEDUP_TORCH_THREADS=2 "$PYTHON" -m scene_dedup.ingest \
    --manifest "$MANIFEST_DIR/shard_a.jsonl" --data-dir "$POOL_DIR" \
    --limit "$target_a" --skip-consolidate \
    > /opt/task-dedup/labeling/ingest_a.log 2>&1 &
  pid_a=$!
  DEDUP_TORCH_THREADS=2 "$PYTHON" -m scene_dedup.ingest \
    --manifest "$MANIFEST_DIR/shard_b.jsonl" --data-dir "$POOL_DIR" \
    --limit "$target_b" --skip-consolidate \
    > /opt/task-dedup/labeling/ingest_b.log 2>&1 &
  pid_b=$!
  wait "$pid_a"
  wait "$pid_b"
fi

cd "$APP_DIR"
"$PYTHON" -c \
  'from pathlib import Path; from scene_dedup.ingest import consolidate; print("consolidated", consolidate(Path("/opt/task-dedup/labeling/pool/records"), Path("/opt/task-dedup/labeling/pool")))'

final_count=$(find "$POOL_DIR/records" -name '*.npz' | wc -l)
if (( final_count != TARGET_TOTAL )); then
  echo "unexpected final checkpoint count: $final_count" >&2
  exit 4
fi
echo "label_pool_complete final_count=$final_count"
