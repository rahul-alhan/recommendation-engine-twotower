"""Generate synthetic implicit-feedback interactions with item attributes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["electronics", "books", "fashion", "home", "sports", "beauty"]


def generate(n_users: int, n_items: int, density: float = 0.02, seed: int = 7):
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(n_users)],
        "age_bucket": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_users),
        "gender": rng.choice(["F", "M", "X"], n_users, p=[0.48, 0.48, 0.04]),
    })
    items = pd.DataFrame({
        "item_id": [f"i_{i}" for i in range(n_items)],
        "category": rng.choice(CATEGORIES, n_items),
        "price_bucket": rng.choice(["low", "mid", "high"], n_items, p=[0.5, 0.35, 0.15]),
    })
    user_pref = {u: rng.choice(CATEGORIES, 2, replace=False) for u in users["user_id"]}

    rows = []
    n_interactions = int(n_users * n_items * density)
    for _ in range(n_interactions):
        u = rng.choice(users["user_id"])
        prefs = user_pref[u]
        cand = items[items["category"].isin(prefs)] if rng.random() < 0.7 else items
        i = rng.choice(cand["item_id"].values)
        rows.append({"user_id": u, "item_id": i, "implicit": 1})

    df = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"])
    return users, items, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--items", type=int, default=500)
    p.add_argument("--density", type=float, default=0.02)
    p.add_argument("--out", default="data/interactions.parquet")
    args = p.parse_args()

    users, items, inter = generate(args.users, args.items, args.density)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(args.out, index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    print(f"users={len(users):,}  items={len(items):,}  interactions={len(inter):,}")
    print(f"Wrote → {out_dir}")


if __name__ == "__main__":
    main()
