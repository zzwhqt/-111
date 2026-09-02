#!/usr/bin/env bash
set -uo pipefail

APP_DIR=/opt/task-dedup/app
DATA_DIR=/opt/task-dedup/data
PYTHON=/opt/micromamba/root/envs/task-dedup/bin/python
TARGET_TOTAL=1000

if [[ -z "${OSS_ACCESS_KEY_ID:-}" || -z "${OSS_ACCESS_KEY_SECRET:-}" ]]; then
  echo "OSS credentials are not present in the tmux session" >&2
  exit 2
fi

current=$(find "$DATA_DIR/records" -name '*.npz' | wc -l)
remaining=$((TARGET_TOTAL - current))
if (( remaining <= 0 )); then
  echo "checkpoint target already reached: $current"
else
  add_a=$(((remaining + 1) / 2))
  add_b=$((remaining / 2))
  target_a=$((current + add_a))
  target_b=$((current + add_b))
  echo "parallel start current=$current add_a=$add_a add_b=$add_b"

  DEDUP_TORCH_THREADS=2 "$PYTHON" -m scene_dedup.ingest \
    --manifest /opt/task-dedup/manifests/shard_a.jsonl \
    --limit "$target_a" --skip-consolidate \
    > /opt/task-dedup/ingest_a.log 2>&1 &
  pid_a=$!

  DEDUP_TORCH_THREADS=2 "$PYTHON" -m scene_dedup.ingest \
    --manifest /opt/task-dedup/manifests/shard_b.jsonl \
    --limit "$target_b" --skip-consolidate \
    > /opt/task-dedup/ingest_b.log 2>&1 &
  pid_b=$!

  wait "$pid_a"
  rc_a=$?
  wait "$pid_b"
  rc_b=$?
  if (( rc_a != 0 || rc_b != 0 )); then
    echo "parallel shard failed rc_a=$rc_a rc_b=$rc_b" >&2
    exit 3
  fi
fi

cd "$APP_DIR"
"$PYTHON" -c \
  'from pathlib import Path; from scene_dedup.ingest import consolidate; print("consolidated", consolidate(Path("/opt/task-dedup/data/records"), Path("/opt/task-dedup/data")))'

final_count=$(find "$DATA_DIR/records" -name '*.npz' | wc -l)
if (( final_count != TARGET_TOTAL )); then
  echo "unexpected final checkpoint count: $final_count" >&2
  exit 4
fi
echo "parallel_complete final_count=$final_count"
