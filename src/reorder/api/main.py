"""FastAPI service: "what will this user reorder next?"

Run locally:  uvicorn reorder.api.main:app --reload
Front-end:    http://localhost:8000/
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from itertools import groupby
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, config
from .repository import Repository
from .schemas import HealthResponse, HistoryResponse, PredictRequest, PredictResponse, ProductScore
from .scorer import Scorer

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("reorder.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy things once per process, not per request.
    app.state.scorer = Scorer(config.ARTIFACT_DIR)
    app.state.repo = Repository(config.DATABASE_URL)
    log.info("models loaded from %s", config.ARTIFACT_DIR)
    yield
    app.state.repo.close()


app = FastAPI(
    title="Instacart Reorder Predictor",
    version=__version__,
    description="Predicts which previously bought products a user will reorder in their next order.",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    """The front-end: a single page that calls the endpoints below."""
    return FileResponse(STATIC_DIR / "index.html")


def get_repo(request: Request) -> Repository:
    return request.app.state.repo


def get_scorer(request: Request) -> Scorer:
    return request.app.state.scorer


@app.get("/health", response_model=HealthResponse)
def health(repo: Repository = Depends(get_repo), scorer: Scorer = Depends(get_scorer)):
    try:
        db_ok = repo.ping()
    except Exception:  # noqa: BLE001 — health must never raise
        db_ok = False
    return HealthResponse(status="ok" if db_ok else "degraded", database=db_ok,
                          models_loaded=scorer is not None)


@app.get("/model-info")
def model_info(scorer: Scorer = Depends(get_scorer)):
    return {"version": __version__, **scorer.info()}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, repo: Repository = Depends(get_repo),
                  scorer: Scorer = Depends(get_scorer)):
    t0 = time.perf_counter()
    df = await run_in_threadpool(repo.candidates, req.user_id)
    if df.empty:
        raise HTTPException(404, f"user {req.user_id} not found or has no purchase history")

    # Optional "what-if" order context overrides the stored target order.
    if req.days_since_last is not None:
        df["t_days_since_last"] = req.days_since_last
    if req.order_dow is not None:
        df["t_dow"] = req.order_dow
    if req.order_hour is not None:
        df["t_hour"] = req.order_hour

    probs = await run_in_threadpool(scorer.score, df, req.model)
    thr = scorer.threshold(req.model)
    df = df.assign(prob=probs).sort_values("prob", ascending=False)
    top = df.head(req.k)

    products = [
        ProductScore(rank=i + 1, product_id=int(r.product_id), product_name=r.product_name,
                     aisle=r.aisle, department=r.department, probability=round(float(r.prob), 4),
                     in_predicted_basket=bool(r.prob >= thr))
        for i, r in enumerate(top.itertuples())
    ]
    latency = (time.perf_counter() - t0) * 1000
    log.info("predict user=%s model=%s candidates=%d %.1fms", req.user_id, req.model, len(df), latency)
    return PredictResponse(
        user_id=req.user_id, model=req.model, threshold=thr, candidates_scored=len(df),
        predicted_basket_size=int((df.prob >= thr).sum()),
        expected_reorders=round(float(df.prob.sum()), 2),
        products=products, latency_ms=round(latency, 1),
    )


@app.get("/users/{user_id}/history", response_model=HistoryResponse)
def history(user_id: int, n_orders: int = Query(5, ge=1, le=50),
            repo: Repository = Depends(get_repo)):
    rows = repo.history(user_id, n_orders)
    if not rows:
        raise HTTPException(404, f"user {user_id} not found")
    orders = []
    for _, items in groupby(rows, key=lambda r: r["order_number"]):
        items = list(items)
        head = items[0]
        orders.append({
            "order_number": head["order_number"],
            "order_dow": head["order_dow"],
            "order_hour_of_day": head["order_hour_of_day"],
            "days_since_prior_order": head["days_since_prior_order"],
            "items": [{"product_id": r["product_id"], "product_name": r["product_name"],
                       "add_to_cart_order": r["add_to_cart_order"], "reordered": bool(r["reordered"])}
                      for r in items],
        })
    return {"user_id": user_id, "orders": orders}


@app.get("/users/sample")
def sample_users(n: int = Query(5, ge=1, le=50), repo: Repository = Depends(get_repo)):
    """Random user ids to try (held-out test users and Kaggle's unlabeled users)."""
    return repo.sample_users(n)
