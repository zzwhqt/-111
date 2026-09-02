#!/usr/bin/env python3
"""Read-only aggregate report for a partially labeled multimodal scene round."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in order[cursor:end]:
            ranks[position] = average_rank
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _ndcg(relevances: list[int]) -> float | None:
    if not relevances:
        return None

    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))

    ideal = dcg(sorted(relevances, reverse=True))
    return dcg(relevances) / ideal if ideal > 0 else 1.0


def _ratio(successes: int, total: int) -> dict[str, Any] | None:
    if not total:
        return None
    return {"successes": successes, "total": total, "value": round(successes / total, 4)}


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
        SELECT q.task_id,q.sample_quadrant,p.pair_id,p.scene_raw_cosine,
               p.candidate_sources_json,a.relation_code
        FROM query_tasks q
        JOIN candidate_pairs p ON p.query_task_id=q.task_id
        JOIN annotations a ON a.pair_id=p.pair_id AND a.round_name='scene'
        ORDER BY q.display_order,p.scene_raw_cosine DESC
        """
    ).fetchall()
    total_pairs = int(connection.execute("SELECT COUNT(*) FROM candidate_pairs").fetchone()[0])
    total_queries = int(connection.execute("SELECT COUNT(*) FROM query_tasks").fetchone()[0])

    by_query: dict[str, list[sqlite3.Row]] = defaultdict(list)
    label_counts: Counter[int] = Counter()
    score_by_label: dict[int, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    all_relations: list[int] = []
    for row in rows:
        by_query[str(row["task_id"])].append(row)
        relation = int(row["relation_code"])
        label_counts[relation] += 1
        if relation != 9:
            score = float(row["scene_raw_cosine"])
            score_by_label[relation].append(score)
            all_scores.append(score)
            all_relations.append(relation)

    ndcgs: list[float] = []
    top1_relevances: list[int] = []
    exact_best = 0
    relevant_queries = 0
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_sum = 0.0
    quadrant_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"queries": 0, "ndcg": [], "relevant_queries": 0, "hit1": 0, "hit3": 0}
    )
    complete_queries = 0
    for query_rows in by_query.values():
        if len(query_rows) != 5:
            continue
        complete_queries += 1
        valid_rows = [row for row in query_rows if int(row["relation_code"]) != 9]
        if not valid_rows:
            continue
        relevances = [int(row["relation_code"]) for row in valid_rows]
        value = _ndcg(relevances)
        if value is not None:
            ndcgs.append(value)
        top1_relevance = relevances[0]
        top1_relevances.append(top1_relevance)
        exact_best += int(top1_relevance == max(relevances))
        has_relevant = any(relevance >= 2 for relevance in relevances)
        first_relevant_rank = next(
            (index + 1 for index, relevance in enumerate(relevances) if relevance >= 2), None
        )
        if has_relevant:
            relevant_queries += 1
            hit_at_1 += int(first_relevant_rank == 1)
            hit_at_3 += int(first_relevant_rank is not None and first_relevant_rank <= 3)
            reciprocal_rank_sum += 1.0 / first_relevant_rank
        quadrant = str(valid_rows[0]["sample_quadrant"])
        metrics = quadrant_metrics[quadrant]
        metrics["queries"] += 1
        if value is not None:
            metrics["ndcg"].append(value)
        if has_relevant:
            metrics["relevant_queries"] += 1
            metrics["hit1"] += int(first_relevant_rank == 1)
            metrics["hit3"] += int(first_relevant_rank is not None and first_relevant_rank <= 3)

    spearman = _pearson(_rankdata(all_scores), _rankdata([float(value) for value in all_relations]))
    binary_labels = [int(relation >= 2) for relation in all_relations]
    binary_auc = _auc(all_scores, binary_labels)
    score_summary = {
        str(label): {
            "count": len(values),
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
        }
        for label, values in sorted(score_by_label.items())
        if values
    }
    quadrant_summary = {}
    for quadrant, metrics in sorted(quadrant_metrics.items()):
        quadrant_summary[quadrant] = {
            "queries": metrics["queries"],
            "mean_ndcg_at_5": (
                round(statistics.mean(metrics["ndcg"]), 4) if metrics["ndcg"] else None
            ),
            "relevant_queries": metrics["relevant_queries"],
            "hit_at_1": _ratio(metrics["hit1"], metrics["relevant_queries"]),
            "hit_at_3": _ratio(metrics["hit3"], metrics["relevant_queries"]),
        }

    result = {
        "warning": "INTERIM_ONLY_SMALL_SAMPLE_DO_NOT_TUNE_OR_UNBLIND_REMAINING_CASES",
        "database_integrity": integrity,
        "progress": {
            "complete_queries": complete_queries,
            "total_queries": total_queries,
            "labeled_pairs": len(rows),
            "total_pairs": total_pairs,
        },
        "scene_label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "unjudgeable_pairs": label_counts.get(9, 0),
        "scene_cosine_by_human_label": score_summary,
        "spearman_score_vs_graded_label": round(spearman, 4) if spearman is not None else None,
        "pair_auc_relation_ge_2": round(binary_auc, 4) if binary_auc is not None else None,
        "mean_ndcg_at_5": round(statistics.mean(ndcgs), 4) if ndcgs else None,
        "top1_exact_best_grade": _ratio(exact_best, len(top1_relevances)),
        "top1_relation_ge_2_all_queries": _ratio(
            sum(relevance >= 2 for relevance in top1_relevances), len(top1_relevances)
        ),
        "queries_with_any_relation_ge_2": relevant_queries,
        "hit_at_1_given_relevant": _ratio(hit_at_1, relevant_queries),
        "hit_at_3_given_relevant": _ratio(hit_at_3, relevant_queries),
        "mrr_given_relevant": (
            round(reciprocal_rank_sum / relevant_queries, 4) if relevant_queries else None
        ),
        "mean_top1_human_grade": (
            round(statistics.mean(top1_relevances), 4) if top1_relevances else None
        ),
        "by_query_sampling_quadrant": quadrant_summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
