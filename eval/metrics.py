"""Standard ranking metrics for offline evaluation."""
from __future__ import annotations

import math


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(recommended[:k]):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def coverage(all_recs: list[list], catalog_size: int) -> float:
    seen = set()
    for r in all_recs:
        seen.update(r)
    return len(seen) / catalog_size if catalog_size else 0.0
