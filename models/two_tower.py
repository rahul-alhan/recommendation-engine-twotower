"""Two-Tower model: separate MLPs for users and items, dot-product similarity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Tower(nn.Module):
    def __init__(self, n_ids: int, n_cat_feats: list[int], emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, emb_dim)
        self.cat_embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in n_cat_feats])
        in_dim = emb_dim * (1 + len(n_cat_feats))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, ids: torch.Tensor, cat_feats: list[torch.Tensor]) -> torch.Tensor:
        parts = [self.id_emb(ids)] + [emb(c) for emb, c in zip(self.cat_embs, cat_feats)]
        x = torch.cat(parts, dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, user_cats: list[int], item_cats: list[int], out_dim: int = 64):
        super().__init__()
        self.user_tower = Tower(n_users, user_cats, out_dim=out_dim)
        self.item_tower = Tower(n_items, item_cats, out_dim=out_dim)


class InteractionsDataset(Dataset):
    def __init__(self, inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
        self.inter = inter.reset_index(drop=True)
        self.user_idx = {u: i for i, u in enumerate(users["user_id"])}
        self.item_idx = {it: i for i, it in enumerate(items["item_id"])}
        self.users = users.set_index("user_id")
        self.items = items.set_index("item_id")
        self._encode_cats()

    def _encode_cats(self):
        self.user_cats = []
        self.user_cat_maps = []
        for col in ["age_bucket", "gender"]:
            cats = sorted(self.users[col].unique().tolist())
            m = {c: i for i, c in enumerate(cats)}
            self.user_cat_maps.append((col, m))
            self.user_cats.append(len(cats))
        self.item_cats = []
        self.item_cat_maps = []
        for col in ["category", "price_bucket"]:
            cats = sorted(self.items[col].unique().tolist())
            m = {c: i for i, c in enumerate(cats)}
            self.item_cat_maps.append((col, m))
            self.item_cats.append(len(cats))

    def _user_features(self, uid: str):
        row = self.users.loc[uid]
        return [m[row[col]] for col, m in self.user_cat_maps]

    def _item_features(self, iid: str):
        row = self.items.loc[iid]
        return [m[row[col]] for col, m in self.item_cat_maps]

    def __len__(self):
        return len(self.inter)

    def __getitem__(self, idx):
        r = self.inter.iloc[idx]
        u = self.user_idx[r["user_id"]]
        i_pos = self.item_idx[r["item_id"]]
        i_neg = np.random.randint(0, len(self.item_idx))
        return {
            "user_id": u,
            "user_feats": self._user_features(r["user_id"]),
            "pos_item_id": i_pos,
            "pos_item_feats": self._item_features(r["item_id"]),
            "neg_item_id": i_neg,
            "neg_item_feats": self._item_features(self.items.index[i_neg]),
        }


def _collate(batch):
    def stack(key):
        return torch.tensor([b[key] for b in batch], dtype=torch.long)
    def stack_feats(key, n):
        feats = [b[key] for b in batch]
        return [torch.tensor([f[i] for f in feats], dtype=torch.long) for i in range(n)]
    n_uf = len(batch[0]["user_feats"])
    n_if = len(batch[0]["pos_item_feats"])
    return {
        "user_id": stack("user_id"),
        "user_feats": stack_feats("user_feats", n_uf),
        "pos_item_id": stack("pos_item_id"),
        "pos_item_feats": stack_feats("pos_item_feats", n_if),
        "neg_item_id": stack("neg_item_id"),
        "neg_item_feats": stack_feats("neg_item_feats", n_if),
    }


def train(interactions_path: str, out_path: str, epochs: int = 5, batch_size: int = 512):
    inter = pd.read_parquet(interactions_path)
    base = Path(interactions_path).parent
    users = pd.read_parquet(base / "users.parquet")
    items = pd.read_parquet(base / "items.parquet")

    ds = InteractionsDataset(inter, users, items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    model = TwoTower(len(users), len(items), ds.user_cats, ds.item_cats)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for batch in dl:
            u_emb = model.user_tower(batch["user_id"], batch["user_feats"])
            pos_emb = model.item_tower(batch["pos_item_id"], batch["pos_item_feats"])
            neg_emb = model.item_tower(batch["neg_item_id"], batch["neg_item_feats"])
            pos_score = (u_emb * pos_emb).sum(dim=-1)
            neg_score = (u_emb * neg_emb).sum(dim=-1)
            loss = F.softplus(neg_score - pos_score).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch["user_id"])
        print(f"epoch {ep+1}/{epochs}  loss={total/len(ds):.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "user_idx": ds.user_idx,
            "item_idx": ds.item_idx,
            "user_cats": ds.user_cats,
            "item_cats": ds.item_cats,
            "user_cat_maps": ds.user_cat_maps,
            "item_cat_maps": ds.item_cat_maps,
        },
        out_path,
    )
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--interactions", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    if args.cmd == "train":
        train(args.interactions, args.out, args.epochs)


if __name__ == "__main__":
    main()
