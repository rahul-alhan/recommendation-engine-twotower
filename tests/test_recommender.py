"""Smoke tests for cold-start logic and re-ranker."""
from __future__ import annotations

import pandas as pd

from serve.cold_start import is_cold_user, popular_in_segment
from serve.reranker import rerank


def _toy():
    users = pd.DataFrame({"user_id": ["u1", "u2"], "age_bucket": ["25-34", "25-34"], "gender": ["F", "F"]})
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "i4"],
        "category": ["a", "a", "b", "b"],
        "price_bucket": ["mid", "mid", "low", "low"],
    })
    inter = pd.DataFrame({
        "user_id": ["u1"] * 8 + ["u2"] * 1,
        "item_id": ["i1"] * 4 + ["i2"] * 4 + ["i1"],
        "implicit": [1] * 9,
    })
    return users, items, inter


def test_cold_user_detection():
    users, items, inter = _toy()
    assert is_cold_user("u2", inter) is True
    assert is_cold_user("u1", inter) is False


def test_popular_in_segment():
    users, items, inter = _toy()
    recs = popular_in_segment("u2", users, inter, top_k=2)
    assert len(recs) <= 2
    assert all(r in {"i1", "i2", "i3", "i4"} for r in recs)


def test_reranker_diversity():
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "i4"],
        "category": ["a", "a", "b", "b"],
        "price_bucket": ["mid", "mid", "low", "low"],
    })
    cands = [("i1", 1.0), ("i2", 0.9), ("i3", 0.8), ("i4", 0.7)]
    out = rerank(cands, items, top_k=3, diversity_lambda=0.5)
    assert len(out) == 3
    assert out[0][0] == "i1"
