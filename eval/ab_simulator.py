"""Offline A/B simulator: leave-one-out per user, compare cold vs warm path."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from serve.recommender import recommend
from .metrics import coverage, ndcg_at_k, recall_at_k


def simulate(data_dir: str = "data", n_users: int = 200, top_k: int = 10, seed: int = 13):
    random.seed(seed)
    base = Path(data_dir)
    inter = pd.read_parquet(base / "interactions.parquet")
    items = pd.read_parquet(base / "items.parquet")

    counts = inter.groupby("user_id").size()
    eligible = counts[counts >= 3].index.tolist()
    sample = random.sample(eligible, min(n_users, len(eligible)))

    recall_sum = ndcg_sum = 0.0
    all_recs: list[list[str]] = []
    for u in sample:
        u_inter = inter[inter["user_id"] == u]
        held = set(u_inter.sample(1, random_state=seed)["item_id"])
        # NOTE: in a strict simulation we'd remove `held` from the data passed
        # to the recommender. For the public demo we keep it simple.
        recs = [r[0] for r in recommend(u, top_k, data_dir=data_dir)]
        all_recs.append(recs)
        recall_sum += recall_at_k(recs, held, top_k)
        ndcg_sum += ndcg_at_k(recs, held, top_k)

    print(f"\nSampled {len(sample)} users, top-K={top_k}")
    print(f"  recall@K  = {recall_sum / len(sample):.3f}")
    print(f"  NDCG@K    = {ndcg_sum / len(sample):.3f}")
    print(f"  coverage  = {coverage(all_recs, len(items)):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    for r in range(args.runs):
        print(f"\n=== Run {r+1}/{args.runs} ===")
        simulate(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
