#!/usr/bin/env python3
"""Compare scene rankers on already labeled pairs without mutating the study."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from interim_multimodal_scene_report import _auc, _ndcg, _pearson, _rankdata, _ratio


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _evaluate(records: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    valid = [record for record in records if record["relation"] != 9]
    scores = [float(record[score_key]) for record in valid]
    relations = [int(record["relation"]) for record in valid]
    score_by_label: dict[int, list[float]] = defaultdict(list)
    for score, relation in zip(scores, relations):
        score_by_label[relation].append(score)

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_query[str(record["query_task_id"])].append(record)

    ndcgs: list[float] = []
    exact_best = 0
    evaluated_queries = 0
    relevant_queries = 0
    hit1 = 0
    hit3 = 0
    reciprocal_rank_sum = 0.0
    quadrant_values: dict[str, list[float]] = defaultdict(list)
    quadrant_relevant: dict[str, list[int]] = defaultdict(list)
    for query_records in by_query.values():
        if len(query_records) != 5:
            continue
        ranked = sorted(query_records, key=lambda item: float(item[score_key]), reverse=True)
        relevance = [int(item["relation"]) for item in ranked]
        value = _ndcg(relevance)
        evaluated_queries += 1
        if value is not None:
            ndcgs.append(value)
            quadrant_values[str(ranked[0]["quadrant"])].append(value)
        exact_best += int(relevance[0] == max(relevance))
        first_relevant = next(
            (index + 1 for index, item in enumerate(relevance) if item >= 2), None
        )
        if first_relevant is not None:
            relevant_queries += 1
            hit1 += int(first_relevant == 1)
            hit3 += int(first_relevant <= 3)
            reciprocal_rank_sum += 1.0 / first_relevant
            quadrant_relevant[str(ranked[0]["quadrant"])].append(first_relevant)

    spearman = _pearson(_rankdata(scores), _rankdata([float(value) for value in relations]))
    auc = _auc(scores, [int(relation >= 2) for relation in relations])
    return {
        "score_by_human_label": {
            str(label): {
                "count": len(values),
                "mean": round(statistics.mean(values), 6),
                "median": round(statistics.median(values), 6),
            }
            for label, values in sorted(score_by_label.items())
        },
        "spearman_vs_graded_label": round(spearman, 4) if spearman is not None else None,
        "pair_auc_relation_ge_2": round(auc, 4) if auc is not None else None,
        "mean_ndcg_at_5": round(statistics.mean(ndcgs), 4) if ndcgs else None,
        "top1_exact_best_grade": _ratio(exact_best, evaluated_queries),
        "relevant_queries": relevant_queries,
        "hit_at_1_given_relevant": _ratio(hit1, relevant_queries),
        "hit_at_3_given_relevant": _ratio(hit3, relevant_queries),
        "mrr_given_relevant": (
            round(reciprocal_rank_sum / relevant_queries, 4) if relevant_queries else None
        ),
        "ndcg_by_quadrant": {
            key: round(statistics.mean(values), 4)
            for key, values in sorted(quadrant_values.items())
            if values
        },
        "relevant_rank_by_quadrant": {
            key: {
                "queries": len(ranks),
                "hit_at_1": _ratio(sum(rank == 1 for rank in ranks), len(ranks)),
                "hit_at_3": _ratio(sum(rank <= 3 for rank in ranks), len(ranks)),
            }
            for key, ranks in sorted(quadrant_relevant.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    rows = connection.execute(
        """
        SELECT q.task_id AS query_task_id,q.vector_index AS query_vector_index,
               q.sample_quadrant,r.vector_index AS reference_vector_index,
               p.scene_raw_cosine,a.relation_code
        FROM query_tasks q
        JOIN candidate_pairs p ON p.query_task_id=q.task_id
        JOIN reference_assets r ON r.task_id=p.reference_task_id
        JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name='scene'
        ORDER BY q.display_order,p.candidate_order
        """
    ).fetchall()
    frame_vectors = _normalize(
        np.load(args.pool_dir / "frame_vectors.npy").astype(np.float32)
    )
    task_vectors = _normalize(np.load(args.pool_dir / "task_vectors.npy").astype(np.float32))
    reference_indices = [
        int(row[0])
        for row in connection.execute(
            "SELECT vector_index FROM reference_assets ORDER BY display_index"
        ).fetchall()
    ]
    reference_matrix = task_vectors[reference_indices]
    pair_scores = reference_matrix @ reference_matrix.T
    upper = np.triu_indices(len(reference_indices), k=1)
    visual_p95 = float(np.quantile(pair_scores[upper], 0.95))

    records: list[dict[str, Any]] = []
    for row in rows:
        query_index = int(row["query_vector_index"])
        reference_index = int(row["reference_vector_index"])
        raw = float(row["scene_raw_cosine"])
        pairwise = frame_vectors[query_index] @ frame_vectors[reference_index].T
        query_to_reference = float(pairwise.max(axis=1).mean())
        reference_to_query = float(pairwise.max(axis=0).mean())
        frame_coverage = (query_to_reference + reference_to_query) / 2.0
        visual_similarity = 0.65 * raw + 0.35 * frame_coverage
        distinctiveness = float(
            np.clip((raw - visual_p95) / max(1e-6, 1.0 - visual_p95), 0.0, 1.0)
        )
        current_scene_score = 0.70 * distinctiveness + 0.30 * frame_coverage
        records.append(
            {
                "query_task_id": str(row["query_task_id"]),
                "quadrant": str(row["sample_quadrant"]),
                "relation": int(row["relation_code"]),
                "raw_cosine": raw,
                "frame_coverage": frame_coverage,
                "visual_similarity": visual_similarity,
                "current_scene_score": current_scene_score,
            }
        )

    result = {
        "warning": "INTERIM_ONLY_CURRENT_SCORE_IS_NOT_A_DUPLICATE_PROBABILITY",
        "database_integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
        "complete_queries": len(records) // 5,
        "labeled_pairs": len(records),
        "visual_reference_p95": round(visual_p95, 6),
        "rankers": {
            "raw_task_cosine": _evaluate(records, "raw_cosine"),
            "raw_plus_frame_coverage": _evaluate(records, "visual_similarity"),
            "current_corpus_normalized_scene_score": _evaluate(
                records, "current_scene_score"
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
