# Instacart Next-Basket Reorder Predictor

**Question:** a customer is about to place their next grocery order. Which of the products they've bought before will they buy again?

Same problem quick-commerce apps (Instamart, Zepto, BigBasket) solve for "Buy again" carousels and smart-cart pre-fill.

```text
docker compose up
│
├── db         Postgres 16 (data in a named volume)
│
├── pipeline   runs once, re-runs skip finished steps
│     1. download   6 Instacart CSVs from a public Hugging Face mirror
│     2. load       COPY 37M rows into raw.*, add PK/FK/CHECK + indexes
│                   (EXPLAIN ANALYZE before vs after indexing)
│     3. features   sql/03_features.sql -> feat.* materialized views
│                   -> feat.candidates (one row per user x product)
│     4. train      export splits -> LightGBM baseline + PyTorch GRU
│     5. evaluate   held-out test users -> artifacts/ (models, metrics)
│
└── api        FastAPI on :8000/docs
                loads artifacts/, reads feat.candidates WHERE user_id = $1
```

## Results

Held-out **test users: 5,314** (never seen in training), 340,845 candidate (user, product) pairs, 9.8% positive.
Trained on a 20% user sample (1.19M rows) on a 2-CPU / 7 GB box; set `USER_SAMPLE_PCT=100` to use all 131k users.

| Model | AUC | PR-AUC | Basket F1 | Precision@10 | Recall@10 | NDCG@10 | Hit-rate@10 |
|---|---|---|---|---|---|---|---|
| Heuristic (share of past orders with the item) | 0.768 | 0.320 | 0.327 | 0.297 | 0.524 | 0.499 | 0.912 |
| LightGBM (123 trees) | 0.833 | **0.428** | 0.379 | 0.333 | 0.585 | 0.561 | 0.936 |
| PyTorch GRU + embeddings (826k params) | 0.830 | 0.418 | 0.378 | 0.331 | 0.583 | 0.558 | 0.938 |
| **Ensemble (mean of both)** — API default | **0.833** | 0.427 | **0.381** | **0.334** | **0.589** | **0.563** | **0.939** |

**Honest read:**

- Both models beat the no-ML heuristic by a wide margin (+16% relative basket F1, +13% NDCG@10).
- The neural net **does not beat** LightGBM on its own. That's the usual result on tabular data, because the SQL features already summarize most of the purchase timeline the GRU reads (LightGBM's top features by gain: `up_last5_rate` 28%, `up_last5` 19%, `up_rate_since_first` 11%).
- The two models make *different* mistakes, though: averaging them is the best on every user-level metric, so the API serves the ensemble.
- **Basket F1** uses a probability threshold tuned on validation users (0.19 for the ensemble), then scores per user the Kaggle way.

Serving: one `/predict` call = indexed fetch of the user's candidates (~9 ms in Postgres) + scoring both models → ~15–35 ms end-to-end on a warm process.

## What each part demonstrates

### SQL (Postgres) — `sql/`
- **Bulk load done right**: tables created bare, loaded with `COPY` (32M basket rows), *then* primary/foreign keys, `CHECK` constraints and indexes are added — building a B-tree once is far cheaper than maintaining it per row.
- **Measured indexing**: the API's hot queries are run with `EXPLAIN ANALYZE` before and after indexing; the plans flip from `Seq Scan` to `Index Scan` (`artifacts/sql_benchmark.json`):

| Query | Before (seq scan) | After (index) | Speedup |
|---|---|---|---|
| User's last 5 orders with product names | 3,067 ms | 0.24 ms | ~12,600× |
| One basket by `order_id` | 916 ms | 0.014 ms | ~65,000× |
| One user's reorder rate from raw tables | 2,569 ms | 0.10 ms | ~25,000× |

- **Whole feature store in SQL** (`03_features.sql`, 11 materialized views): window functions (`ROW_NUMBER`, running `SUM … OVER` as a "day clock", `LAG` for re-buy intervals, windowed shares over aggregates), `FILTER` aggregates, `ARRAY_AGG` for the GRU's gap timeline, and a **bit-packed 20-order purchase history** per (user, product) computed with `1::bigint << (recency_rank - 1)`.
- **One view for training and serving** — `feat.candidates` is read in bulk for training and `WHERE user_id = $1` by the API, so there is no train/serve skew. Leakage-safe: every feature uses *prior* orders only.
- **Analysis queries** with business answers in `artifacts/analysis.md` (e.g. dairy/eggs reorders at 67% vs. a 45.6% reorder rate when a customer returns after 30+ days).

### Python / ML — `src/reorder/`
- **Baselines first**: a no-ML heuristic ("share of past orders containing this item") and LightGBM with early stopping.
- **Neural net** (`nn.py`, PyTorch): for every (user, product) the model reads a 20-step timeline — *was the item in that order? how many days apart were the orders?* — through a **GRU**, joins it with learned **product / aisle / department embeddings** and the scaled SQL features, then an MLP outputs P(reorder). Memory-light: sequences are decoded per batch from the 64-bit mask instead of materializing an (N, 20, 3) tensor.
- **Metrics that match the product**: pair-level (AUC, PR-AUC, log-loss) and user-level (basket F1 with a threshold tuned on validation, Precision/Recall/NDCG/Hit-rate @10). Users are split 70/10/20 by hash — no user leaks across splits.

### FastAPI — `src/reorder/api/`
| Endpoint | What it does |
|---|---|
| `POST /predict` | `{user_id, k, model, [days_since_last, order_dow, order_hour]}` → top-k products with probabilities and whether each is in the predicted basket. Optional fields let you ask "what if they order on Sunday morning?" |
| `GET /users/{id}/history` | last N orders with items, live from Postgres |
| `GET /users/sample` | random held-out user ids to try |
| `GET /model-info` | model sizes + test metrics |
| `GET /health` | DB + model check (used by the Docker healthcheck) |

Models load once at startup (lifespan), DB access goes through a connection pool, blocking work runs in a threadpool, all queries are parameterized, and requests are validated by Pydantic (bad `user_id`, `k`, hour → 422).

### Docker — `Dockerfile`, `docker-compose.yml`
- **Multi-stage build**: deps installed into a venv in a builder stage; the runtime image is `python:3.12-slim` + venv, runs as a **non-root** user, with a `HEALTHCHECK`. CPU-only PyTorch wheels keep it small.
- **Compose orchestration**: `db` (healthchecked) → `pipeline` (one-shot, idempotent — re-runs skip finished steps) → `api` (starts only after `pipeline` completes successfully).
- **CI**: GitHub Actions runs ruff + pytest, then builds both image targets.

## Run it

```bash
cp .env.example .env          # optional: USER_SAMPLE_PCT=20 for laptops with <8 GB RAM
docker compose up --build     # downloads data, builds features, trains, serves
# first run: ~30-60 min depending on machine; later runs reuse everything

open http://localhost:8000/docs
make demo                     # predict for a random held-out user
```

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"user_id": 4242, "k": 5, "model": "lightgbm"}'
```

Real response (abridged), user 4242:

```json
{"user_id": 4242, "model": "ensemble", "threshold": 0.19, "candidates_scored": 29,
 "predicted_basket_size": 19, "expected_reorders": 8.57,
 "products": [
   {"rank": 1, "product_name": "Clementines",  "department": "produce", "probability": 0.82, "in_predicted_basket": true},
   {"rank": 2, "product_name": "Trail Mix",    "department": "snacks",  "probability": 0.80, "in_predicted_basket": true},
   {"rank": 3, "product_name": "Whole Grain Cheddar Baked Snack Crackers", "probability": 0.67, "in_predicted_basket": true}],
 "latency_ms": 34.9}
```

Without Docker (Postgres running locally):

```bash
pip install -r requirements-dev.txt
export PYTHONPATH=src DATABASE_URL=postgresql://instacart:instacart@localhost:5432/instacart
python -m reorder.pipeline all          # or one step: download|load|features|analysis|train-lgbm|train-nn|evaluate
uvicorn reorder.api.main:app --reload
pytest                                  # 27 tests, no DB needed (repo + scorer are faked)
```

## Repo layout

```
sql/01_schema.sql              bare tables for fast COPY
sql/02_constraints_indexes.sql PK/FK/CHECK + indexes
sql/03_features.sql            feature store (materialized views + candidates view)
sql/analysis.sql               business EDA queries -> artifacts/analysis.md
sql/benchmark_queries.sql      queries timed before/after indexing
src/reorder/pipeline.py        CLI: download -> load -> features -> analysis -> train -> evaluate
src/reorder/ingest.py          download + COPY + benchmark
src/reorder/features.py        feature lists, scaler, bitmask decoding, split export
src/reorder/lgbm.py            LightGBM baseline
src/reorder/nn.py              PyTorch GRU + embeddings model
src/reorder/metrics.py         AUC / basket F1 / P@k / R@k / NDCG@k
src/reorder/api/               FastAPI app, SQL repository, scorer, schemas
tests/                         API, metrics, features, NN, SQL-file tests
artifacts/                     metrics, benchmark, analysis, trained models
```

## Data

[Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) (3.4M orders, 206k users, 50k products, 32M basket rows), downloaded automatically from a public Hugging Face mirror (`DATA_BASE_URL` in `config.py`). For each user the Kaggle "train" order is the label; all features come from their earlier ("prior") orders. Kaggle "test" users have no labels and are served by the API only.
