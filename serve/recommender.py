"""Top-level recommender — orchestrates cold-start, model scoring, and re-ranking."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import cold_start
from .reranker import rerank


def recommend(user_id: str, top_k: int = 10, data_dir: str = "data") -> list[tuple[str, float]]:
    base = Path(data_dir)
    interactions = pd.read_parquet(base / "interactions.parquet")
    users = pd.read_parquet(base / "users.parquet")
    items = pd.read_parquet(base / "items.parquet")
    seen = set(interactions.loc[interactions["user_id"] == user_id, "item_id"])

    if cold_start.is_cold_user(user_id, interactions):
        # Cold path: popularity within demographic segment
        cands = cold_start.popular_in_segment(user_id, users, interactions, top_k=top_k * 5)
        candidates = [(c, 1.0 / (rank + 1)) for rank, c in enumerate(cands)]
    else:
        # Warm path: in production this is Two-Tower scored against item embeddings (ANN).
        # Here we approximate by category preference + popularity.
        prefs = items.set_index("item_id").loc[list(seen), "category"].value_counts()
        if prefs.empty:
            top_cats = items["category"].value_counts().head(2).index.tolist()
        else:
            top_cats = prefs.head(2).index.tolist()
        cand_pool = items[items["category"].isin(top_cats)]
        pop = interactions["item_id"].value_counts()
        candidates = [
            (i, float(pop.get(i, 0)) + 0.1)
            for i in cand_pool["item_id"]
            if i not in seen
        ]
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[: top_k * 5]

    return rerank(candidates, items, seen_items=seen, top_k=top_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    recs = recommend(args.user_id, args.top_k, args.data_dir)
    print(f"\nTop {args.top_k} recommendations for {args.user_id}:")
    for i, (item, score) in enumerate(recs, 1):
        print(f"  {i:2d}. {item:10s}  score={score:.3f}")


if __name__ == "__main__":
    main()
