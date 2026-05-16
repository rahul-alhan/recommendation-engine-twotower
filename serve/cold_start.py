"""Cold-start fallback strategies."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COLD_THRESHOLD = 5


def is_cold_user(user_id: str, interactions: pd.DataFrame) -> bool:
    return (interactions["user_id"] == user_id).sum() < COLD_THRESHOLD


def is_cold_item(item_id: str, interactions: pd.DataFrame) -> bool:
    return (interactions["item_id"] == item_id).sum() < COLD_THRESHOLD


def popular_in_segment(user_id: str, users: pd.DataFrame, interactions: pd.DataFrame, top_k: int = 10) -> list[str]:
    if user_id not in users["user_id"].values:
        return interactions["item_id"].value_counts().head(top_k).index.tolist()
    seg = users.loc[users["user_id"] == user_id, ["age_bucket", "gender"]].iloc[0]
    peer_users = users[(users["age_bucket"] == seg["age_bucket"]) & (users["gender"] == seg["gender"])]
    peer_ids = set(peer_users["user_id"])
    peer_interactions = interactions[interactions["user_id"].isin(peer_ids)]
    return peer_interactions["item_id"].value_counts().head(top_k).index.tolist()


def content_neighbors(item_id: str, items: pd.DataFrame, top_k: int = 10) -> list[str]:
    if item_id not in items["item_id"].values:
        return items["item_id"].head(top_k).tolist()
    items = items.copy()
    items["text"] = items["category"].astype(str) + " " + items["price_bucket"].astype(str)
    vec = TfidfVectorizer().fit_transform(items["text"])
    idx = items.index[items["item_id"] == item_id][0]
    sim = cosine_similarity(vec[idx], vec).flatten()
    items = items.assign(sim=sim).sort_values("sim", ascending=False)
    return items[items["item_id"] != item_id]["item_id"].head(top_k).tolist()
