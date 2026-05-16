"""Business-rule re-ranking: no-repeat, diversity, recency."""
from __future__ import annotations

import pandas as pd


def rerank(
    candidates: list[tuple[str, float]],
    items: pd.DataFrame,
    seen_items: set[str] | None = None,
    diversity_lambda: float = 0.3,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    seen_items = seen_items or set()
    items_idx = items.set_index("item_id")

    pool = [(i, s) for i, s in candidates if i not in seen_items]
    if not pool:
        return []

    selected: list[tuple[str, float]] = []
    used_categories: dict[str, int] = {}

    while pool and len(selected) < top_k:
        best, best_score = None, -1e9
        for cand, score in pool:
            cat = items_idx.loc[cand, "category"] if cand in items_idx.index else None
            penalty = diversity_lambda * used_categories.get(cat, 0)
            adj = score - penalty
            if adj > best_score:
                best, best_score = (cand, score), adj
        if not best:
            break
        selected.append(best)
        cat = items_idx.loc[best[0], "category"] if best[0] in items_idx.index else None
        used_categories[cat] = used_categories.get(cat, 0) + 1
        pool = [(i, s) for (i, s) in pool if i != best[0]]

    return selected
