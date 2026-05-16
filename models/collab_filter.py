"""Implicit ALS collaborative-filtering baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class CollabFilter:
    def __init__(self, factors: int = 32, regularization: float = 0.01, iterations: int = 15):
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as e:
            raise ImportError("Install `implicit` to use the CF baseline.") from e
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        self.user_idx: dict[str, int] = {}
        self.item_idx: dict[str, int] = {}

    def fit(self, interactions: pd.DataFrame) -> "CollabFilter":
        users = interactions["user_id"].unique().tolist()
        items = interactions["item_id"].unique().tolist()
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {i: idx for idx, i in enumerate(items)}

        rows = interactions["user_id"].map(self.user_idx).to_numpy()
        cols = interactions["item_id"].map(self.item_idx).to_numpy()
        data = np.ones(len(interactions), dtype=np.float32)
        mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        self.model.fit(mat)
        self._user_item = mat
        return self

    def recommend(self, user_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        if user_id not in self.user_idx:
            return []
        u = self.user_idx[user_id]
        ids, scores = self.model.recommend(u, self._user_item[u], N=top_k)
        inv = {v: k for k, v in self.item_idx.items()}
        return [(inv[int(i)], float(s)) for i, s in zip(ids, scores)]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--interactions", required=True)
    t.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "train":
        inter = pd.read_parquet(args.interactions)
        cf = CollabFilter().fit(inter)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cf, args.out)
        print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
