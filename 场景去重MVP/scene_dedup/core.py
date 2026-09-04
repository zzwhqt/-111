from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import faiss
import numpy as np
from PIL import Image


FRAME_COUNT = int(os.getenv("DEDUP_FRAME_COUNT", "8"))
MODEL_NAME = os.getenv("DEDUP_MODEL_NAME", "ViT-B-32")
MODEL_PRETRAINED = os.getenv("DEDUP_MODEL_PRETRAINED", "openai")
DATA_DIR = Path(os.getenv("DEDUP_DATA_DIR", "/opt/task-dedup/data"))
FFMPEG = os.getenv("DEDUP_FFMPEG", "ffmpeg")
FFPROBE = os.getenv("DEDUP_FFPROBE", "ffprobe")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:180] or "task"


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def parse_time(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        nums = [float(item) for item in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]


def task_bounds(task: dict[str, Any], fps: float = 30.0) -> tuple[float, float] | None:
    start_frame = task.get("start_frame", task.get("frame_start"))
    end_frame = task.get("end_frame", task.get("frame_end"))
    if start_frame is not None and end_frame is not None:
        start = float(start_frame) / fps
        end = float(end_frame) / fps
        return (start, end) if end > start else None
    start = parse_time(task.get("start", task.get("start_time")))
    end = parse_time(task.get("end", task.get("end_time")))
    if start is None or end is None or end <= start:
        return None
    return start, end


def task_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("tasks", "task_list", "actions", "data", "segments"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for nested in ("tasks", "items", "data"):
                rows = value.get(nested)
                if isinstance(rows, list):
                    return [item for item in rows if isinstance(item, dict)]
    return []


def pick_task(payload: Any, task_index: int) -> dict[str, Any] | None:
    rows = task_rows(payload)
    for pos, row in enumerate(rows):
        raw_id = row.get("task_id", row.get("id", row.get("index", pos)))
        match = re.search(r"(\d+)$", str(raw_id))
        normalized = int(match.group(1)) if match else pos
        if normalized == task_index:
            return row
    return rows[task_index] if 0 <= task_index < len(rows) else None


def validate_remote_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only http(s) video URLs are accepted")
    allowed = tuple(
        item.strip().lower()
        for item in os.getenv("DEDUP_ALLOWED_URL_SUFFIXES", ".aliyuncs.com").split(",")
        if item.strip()
    )
    host = parsed.hostname.lower()
    if allowed and not any(host == suffix.lstrip(".") or host.endswith(suffix) for suffix in allowed):
        raise ValueError(f"URL host is not allowed: {host}")
    return url


def _video_duration(source: str) -> float:
    command = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video has no valid duration")
    return duration


def extract_frames(
    source: str,
    output_dir: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    frame_count: int = FRAME_COUNT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        validate_remote_url(source)
    elif not Path(source).is_file():
        raise FileNotFoundError(source)

    if start_seconds is None or end_seconds is None:
        duration = _video_duration(source)
        start_seconds = 0.0
        end_seconds = duration
    duration = max(0.15, float(end_seconds) - float(start_seconds))
    fps = max(0.01, frame_count / duration)
    pattern = str(output_dir / "frame_%03d.jpg")
    command = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_seconds > 0:
        command += ["-ss", f"{start_seconds:.3f}"]
    command += [
        "-i",
        source,
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"fps={fps:.8f},scale=336:336:force_original_aspect_ratio=decrease,pad=336:336:(ow-iw)/2:(oh-ih)/2",
        "-frames:v",
        str(frame_count),
        "-q:v",
        "3",
        "-y",
        pattern,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("FFmpeg did not extract any frame")
    while len(frames) < frame_count:
        duplicate = output_dir / f"frame_{len(frames) + 1:03d}.jpg"
        shutil.copyfile(frames[-1], duplicate)
        frames.append(duplicate)
    return frames[:frame_count]


class ClipEncoder:
    def __init__(self) -> None:
        import open_clip
        import torch

        threads = int(os.getenv("DEDUP_TORCH_THREADS", "4"))
        torch.set_num_threads(max(1, threads))
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=MODEL_PRETRAINED, device="cpu"
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model.eval()
        self._lock = threading.Lock()

    def encode(self, frame_paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
        images = [self.preprocess(Image.open(path).convert("RGB")) for path in frame_paths]
        batch = self.torch.stack(images)
        with self._lock, self.torch.inference_mode():
            vectors = self.model.encode_image(batch).float().cpu().numpy()
        frame_vectors = l2_normalize(vectors)
        task_vector = l2_normalize(frame_vectors.mean(axis=0))
        return task_vector.astype(np.float32), frame_vectors.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text])
        with self._lock, self.torch.inference_mode():
            vector = self.model.encode_text(tokens).float().cpu().numpy()[0]
        return l2_normalize(vector).astype(np.float32)


@dataclass(frozen=True)
class SearchHit:
    task_id: str
    raw_cosine: float
    frame_coverage: float
    text_alignment: float
    visual_similarity: float
    corpus_percentile: float
    text_distinctiveness: float | None
    score_mode: str
    similarity_percent: float
    warning_level: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "raw_cosine": round(self.raw_cosine, 6),
            "frame_coverage": round(self.frame_coverage, 6),
            "text_alignment": round(self.text_alignment, 6),
            "visual_similarity": round(self.visual_similarity, 6),
            "corpus_percentile": round(self.corpus_percentile, 3),
            "text_distinctiveness": (
                round(self.text_distinctiveness, 6) if self.text_distinctiveness is not None else None
            ),
            "score_mode": self.score_mode,
            "similarity_percent": round(self.similarity_percent, 2),
            "warning_level": self.warning_level,
            "metadata": self.metadata,
        }


class IndexStore:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        metadata_path = data_dir / "metadata.json"
        task_path = data_dir / "task_vectors.npy"
        frame_path = data_dir / "frame_vectors.npy"
        text_path = data_dir / "text_vectors.npy"
        if not (metadata_path.is_file() and task_path.is_file() and frame_path.is_file() and text_path.is_file()):
            raise RuntimeError(f"index is incomplete under {data_dir}")
        self.metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text("utf-8"))
        self.task_vectors = l2_normalize(np.load(task_path).astype(np.float32))
        self.frame_vectors = l2_normalize(np.load(frame_path).astype(np.float32))
        self.text_vectors = l2_normalize(np.load(text_path).astype(np.float32))
        if (
            len(self.metadata) != len(self.task_vectors)
            or len(self.metadata) != len(self.frame_vectors)
            or len(self.metadata) != len(self.text_vectors)
        ):
            raise RuntimeError("metadata/vector counts do not match")
        self.index = faiss.IndexFlatIP(self.task_vectors.shape[1])
        self.index.add(self.task_vectors)

        visual_pairs = self.task_vectors @ self.task_vectors.T
        text_pairs = self.text_vectors @ self.text_vectors.T
        upper = np.triu_indices(len(self.metadata), k=1)
        self.visual_negative_scores = np.sort(visual_pairs[upper].astype(np.float32))
        self.text_negative_scores = np.sort(text_pairs[upper].astype(np.float32))
        self.visual_p95 = float(np.quantile(self.visual_negative_scores, 0.95))
        self.text_p95 = float(np.quantile(self.text_negative_scores, 0.95))

    def search(
        self,
        query_task: np.ndarray,
        query_frames: np.ndarray,
        top_k: int = 10,
        query_text: np.ndarray | None = None,
    ) -> list[SearchHit]:
        if not self.metadata:
            return []
        recall_k = min(len(self.metadata), max(top_k * 5, 30))
        scores, ids = self.index.search(l2_normalize(query_task)[None, :], recall_k)
        hits: list[SearchHit] = []
        query_frames = l2_normalize(query_frames)
        for raw_score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            candidate_frames = self.frame_vectors[idx]
            pairwise = query_frames @ candidate_frames.T
            q_to_c = float(pairwise.max(axis=1).mean())
            c_to_q = float(pairwise.max(axis=0).mean())
            frame_coverage = (q_to_c + c_to_q) / 2.0
            # Pure-visual ranking weights selected on the frozen 80-query
            # development split, then checked once on the 40-query holdout.
            visual_similarity = 0.60 * float(raw_score) + 0.40 * frame_coverage
            corpus_percentile = float(
                100.0
                * np.searchsorted(self.visual_negative_scores, raw_score, side="right")
                / max(1, len(self.visual_negative_scores))
            )
            visual_distinctiveness = float(
                np.clip((float(raw_score) - self.visual_p95) / max(1e-6, 1.0 - self.visual_p95), 0.0, 1.0)
            )
            if query_text is not None:
                text_alignment = float(query_text @ self.text_vectors[idx])
                text_distinctiveness = float(
                    np.clip(
                        (text_alignment - self.text_p95) / max(1e-6, 1.0 - self.text_p95),
                        0.0,
                        1.0,
                    )
                )
                combined = 0.55 * visual_distinctiveness + 0.25 * frame_coverage + 0.20 * text_distinctiveness
                score_mode = "video_plus_task_text"
            else:
                text_alignment = float(query_task @ self.text_vectors[idx])
                text_distinctiveness = None
                combined = visual_similarity
                score_mode = "video_only_60_task_40_frame"
            combined = float(np.clip(combined, 0.0, 1.0))
            similarity_percent = float(np.clip(combined, 0.0, 1.0) * 100.0)
            if float(raw_score) >= 0.995 and frame_coverage >= 0.98:
                warning = "high"
            elif corpus_percentile >= 99.0:
                warning = "review"
            else:
                warning = "low"
            metadata = self.metadata[idx]
            hits.append(
                SearchHit(
                    task_id=str(metadata["task_id"]),
                    raw_cosine=float(raw_score),
                    frame_coverage=frame_coverage,
                    text_alignment=text_alignment,
                    visual_similarity=visual_similarity,
                    corpus_percentile=corpus_percentile,
                    text_distinctiveness=text_distinctiveness,
                    score_mode=score_mode,
                    similarity_percent=similarity_percent,
                    warning_level=warning,
                    metadata=metadata,
                )
            )
        hits.sort(key=lambda item: item.similarity_percent, reverse=True)
        return hits[:top_k]


def encode_video(
    encoder: ClipEncoder,
    source: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="task-dedup-"))
    frames = extract_frames(source, temp_dir, start_seconds, end_seconds)
    task_vector, frame_vectors = encoder.encode(frames)
    return task_vector, frame_vectors, frames[len(frames) // 2]
