# Recommendation Engine — Two-Tower + Collaborative Filtering + Cold-Start

End-to-end recommendation system that combines a **PyTorch Two-Tower** embedding model, an **implicit-feedback collaborative-filtering** baseline, a **content-based cold-start fallback**, and a **business-rule re-ranking** layer.

> Mirrors a production engine I built in a prior role (2022–2024) — delivered ~20% engagement lift and ~12% 30-day retention improvement (A/B validated).

---

## Architecture

```
                         ┌────────────────────┐
   user_id ──┬───────▶  │  user_tower (MLP)   │ ──▶  user_embedding (d=64)
             │           └────────────────────┘
             │                                            │
             │           ┌────────────────────┐           │
   item_id ──┴───────▶  │  item_tower (MLP)   │ ──▶  item_embedding (d=64)
                         └────────────────────┘           │
                                                          ▼
                                              dot-product score
                                                          │
                                                          ▼
                              top-N candidates ───▶  re_ranker  ───▶  served list
                                                       │
                                                       ├─ business rules (no-repeat,
                                                       │   diversity, recency)
                                                       └─ cold-start fallback for
                                                          users/items < N interactions
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic interaction data
python -m scripts.generate_data --users 2000 --items 500 --out data/interactions.parquet

# 2. Train the Two-Tower model
python -m models.two_tower train \
  --interactions data/interactions.parquet \
  --out artifacts/two_tower.pt

# 3. Train the collaborative-filtering baseline
python -m models.collab_filter train \
  --interactions data/interactions.parquet \
  --out artifacts/cf.pkl

# 4. Generate recs for a user (with cold-start handling)
python -m serve.recommender \
  --user-id u_42 \
  --top-k 10

# 5. Run A/B simulation
python -m eval.ab_simulator --runs 5
```

---

## Cold-Start Strategy

The hardest part of recommendation isn't the model — it's the long tail of users and items with little or no interaction history. This system handles three cold-start regimes:

| Regime | Trigger | Fallback |
|---|---|---|
| **Cold user** | < 5 interactions | popularity within demographic segment |
| **Cold item** | < 5 interactions | content-based: TF-IDF over item attributes → nearest hot items |
| **Both cold** | new user + new item | category-level popularity ranking |

Cold-start logic lives in `serve/cold_start.py` and is invoked by the recommender *before* the Two-Tower scoring step, so the rest of the pipeline never sees an unrecoverable miss.

---

## Why Two-Tower (vs. classic matrix factorization)

| Aspect | Matrix Factorization | Two-Tower |
|---|---|---|
| New users | retrain or special-case | feed user features through tower |
| New items | retrain or special-case | feed item features through tower |
| Side features | hard | natural |
| Scaling to many items | dot-product at serve | ANN on item embeddings |
| Online retraining | weekly batch | incremental, daily |

The CF baseline still ships — it's a reliable sanity check and the cold-start regime falls back to it for anonymous users.

---

## Repository Layout

```
recommendation-engine-twotower/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   └── generate_data.py
├── models/
│   ├── __init__.py
│   ├── two_tower.py             # PyTorch model + training loop
│   └── collab_filter.py         # implicit ALS baseline
├── serve/
│   ├── __init__.py
│   ├── cold_start.py
│   ├── reranker.py              # business-rule re-ranking
│   └── recommender.py           # top-level serving entrypoint
├── eval/
│   ├── __init__.py
│   ├── metrics.py               # recall@k, ndcg@k, coverage
│   └── ab_simulator.py          # offline A/B test harness
└── tests/
    └── test_recommender.py
```

---

## Evaluation Metrics

- **Recall@K**, **NDCG@K** — standard ranking metrics
- **Coverage** — % of catalog ever recommended (anti-popularity-bias check)
- **Cold-start hit rate** — % of cold users/items getting non-empty recs
- **Diversity** (intra-list) — average pairwise cosine distance in served list

---

## Production Notes

- Trained nightly on **AWS SageMaker** (PyTorch container, single GPU)
- Item embeddings indexed in **OpenSearch kNN** for sub-50ms top-K retrieval
- User tower runs at request time on Lambda (CPU is plenty for a 64-d MLP)
- Re-ranker applies session-level diversity + recency penalties last
- A/B traffic split through **GrowthBook** with engagement / retention as primary metrics

---

## License

MIT
