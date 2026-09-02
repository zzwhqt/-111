#!/usr/bin/env python3
"""Read-only interim metrics for the still-blinded labeling study."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def proportion(successes: int, total: int) -> dict[str, Any] | None:
    if total <= 0:
        return None
    value = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (value + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(value * (1.0 - value) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "value": round(value, 4),
        "successes": successes,
        "total": total,
        "wilson_95": [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    rows = connection.execute(
        """
        SELECT q.sample_band,q.model_top5_json,q.model_top1_warning,q.model_top1_score,
               h.human_label,h.selected_reference_id
        FROM query_tasks q JOIN human_labels h USING(task_id)
        WHERE q.is_label_target=1 AND h.human_label!='bad'
        ORDER BY q.display_order
        """
    ).fetchall()
    all_label_counts = dict(
        connection.execute(
            "SELECT human_label,COUNT(*) FROM human_labels GROUP BY human_label"
        ).fetchall()
    )

    label_counts = Counter(row["human_label"] for row in rows)
    band_counts = Counter(str(row["sample_band"]).split("_same_source")[0] for row in rows)
    warning_counts = Counter(row["model_top1_warning"] or "low" for row in rows)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    scores: dict[str, list[float]] = defaultdict(list)
    duplicate = top1 = top5_hits = 0
    reciprocal_rank = 0.0
    review = review_top1 = review_top5 = 0
    high_total = high_strict_true = high_duplicate_recall = 0
    novel = novel_high = 0

    for row in rows:
        candidates = json.loads(row["model_top5_json"])
        ids = [candidate["task_id"] for candidate in candidates]
        human = row["human_label"]
        selected = row["selected_reference_id"]
        warning = row["model_top1_warning"] or "low"
        confusion[human][warning] += 1
        if row["model_top1_score"] is not None:
            scores[human].append(float(row["model_top1_score"]))
        exact_top1 = bool(ids) and selected == ids[0]
        in_top5 = bool(selected) and selected in ids
        if human == "duplicate" and selected:
            duplicate += 1
            top1 += int(exact_top1)
            top5_hits += int(in_top5)
            if in_top5:
                reciprocal_rank += 1.0 / (ids.index(selected) + 1)
            high_duplicate_recall += int(warning == "high" and exact_top1)
        if human == "review" and selected:
            review += 1
            review_top1 += int(exact_top1)
            review_top5 += int(in_top5)
        if warning == "high":
            high_total += 1
            high_strict_true += int(human == "duplicate" and exact_top1)
        if human == "novel":
            novel += 1
            novel_high += int(warning == "high")

    score_summary = {
        label: {
            "count": len(values),
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
        }
        for label, values in sorted(scores.items())
        if values
    }
    result = {
        "warning": "INTERIM_ONLY_SMALL_SAMPLE_DO_NOT_TUNE_ON_THIS_REPORT",
        "database_integrity": integrity,
        "effective_labeled": len(rows),
        "all_human_label_counts": all_label_counts,
        "effective_label_counts": dict(sorted(label_counts.items())),
        "sample_band_counts": dict(sorted(band_counts.items())),
        "model_warning_counts": dict(sorted(warning_counts.items())),
        "duplicate_queries": duplicate,
        "recall_at_1": proportion(top1, duplicate),
        "recall_at_5": proportion(top5_hits, duplicate),
        "mrr_at_5": round(reciprocal_rank / duplicate, 4) if duplicate else None,
        "duplicate_high_exact_recall": proportion(high_duplicate_recall, duplicate),
        "review_queries": review,
        "review_reference_at_1": proportion(review_top1, review),
        "review_reference_at_5": proportion(review_top5, review),
        "high_alert_precision_strict": proportion(high_strict_true, high_total),
        "novel_high_false_positive_rate": proportion(novel_high, novel),
        "top1_score_by_human_label": score_summary,
        "confusion_human_by_warning": {
            label: dict(sorted(counts.items())) for label, counts in sorted(confusion.items())
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
