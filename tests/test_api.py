"""API tests with the database and models faked out (no Postgres needed)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from reorder.api.main import app, get_repo, get_scorer

from .conftest import make_candidates


class FakeRepo:
    def ping(self):
        return True

    def candidates(self, user_id):
        return make_candidates(user_id) if user_id == 7 else pd.DataFrame()

    def history(self, user_id, n):
        if user_id != 7:
            return []
        return [
            dict(order_number=3, order_dow=1, order_hour_of_day=9, days_since_prior_order=7.0,
                 add_to_cart_order=1, product_id=1, product_name="Banana", reordered=1),
            dict(order_number=3, order_dow=1, order_hour_of_day=9, days_since_prior_order=7.0,
                 add_to_cart_order=2, product_id=2, product_name="Milk", reordered=0),
            dict(order_number=2, order_dow=0, order_hour_of_day=10, days_since_prior_order=None,
                 add_to_cart_order=1, product_id=1, product_name="Banana", reordered=0),
        ]

    def sample_users(self, n):
        return [{"user_id": 7, "split": "test"}][:n]


class FakeScorer:
    def threshold(self, model):
        return 0.5

    def score(self, df, model):
        self.last_df = df
        return np.linspace(0.1, 0.9, len(df))   # last product scores highest

    def info(self):
        return {"models": ["neural_net"]}


@pytest.fixture
def client():
    scorer = FakeScorer()
    app.dependency_overrides[get_repo] = lambda: FakeRepo()
    app.dependency_overrides[get_scorer] = lambda: scorer
    c = TestClient(app)  # not used as a context manager -> real lifespan is skipped
    c.scorer = scorer
    yield c
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "database": True, "models_loaded": True}


def test_predict_ranks_by_probability(client):
    r = client.post("/predict", json={"user_id": 7, "k": 3})
    assert r.status_code == 200
    body = r.json()
    probs = [p["probability"] for p in body["products"]]
    assert probs == sorted(probs, reverse=True) and len(probs) == 3
    assert body["products"][0]["product_id"] == 6
    assert body["candidates_scored"] == 6
    assert body["predicted_basket_size"] == 3        # 0.58, 0.74, 0.9 >= 0.5


def test_predict_context_override(client):
    client.post("/predict", json={"user_id": 7, "days_since_last": 3, "order_hour": 20})
    df = client.scorer.last_df
    assert (df["t_days_since_last"] == 3).all() and (df["t_hour"] == 20).all()


def test_predict_unknown_user_404(client):
    assert client.post("/predict", json={"user_id": 999}).status_code == 404


@pytest.mark.parametrize("payload", [
    {"user_id": -1}, {"user_id": 7, "k": 0}, {"user_id": 7, "model": "xgboost"},
    {"user_id": 7, "order_hour": 25},
])
def test_predict_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_history_groups_orders(client):
    body = client.get("/users/7/history").json()
    assert [o["order_number"] for o in body["orders"]] == [3, 2]
    assert [i["product_name"] for i in body["orders"][0]["items"]] == ["Banana", "Milk"]


def test_history_unknown_user(client):
    assert client.get("/users/999/history").status_code == 404
