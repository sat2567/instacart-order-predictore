from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    user_id: int = Field(..., gt=0, examples=[4242])
    k: int = Field(10, ge=1, le=100, description="How many products to return")
    model: Literal["neural_net", "lightgbm", "ensemble"] = "ensemble"
    # Optional order context; defaults to the user's actual next order in the data.
    days_since_last: float | None = Field(None, ge=0, le=30)
    order_dow: int | None = Field(None, ge=0, le=6)
    order_hour: int | None = Field(None, ge=0, le=23)


class ProductScore(BaseModel):
    rank: int
    product_id: int
    product_name: str
    aisle: str
    department: str
    probability: float
    in_predicted_basket: bool = Field(..., description="probability >= model's tuned threshold")


class PredictResponse(BaseModel):
    user_id: int
    model: str
    threshold: float
    candidates_scored: int
    predicted_basket_size: int
    expected_reorders: float = Field(..., description="sum of probabilities over all candidates")
    products: list[ProductScore]
    latency_ms: float


class HistoryItem(BaseModel):
    product_id: int
    product_name: str
    add_to_cart_order: int
    reordered: bool


class HistoryOrder(BaseModel):
    order_number: int
    order_dow: int
    order_hour_of_day: int
    days_since_prior_order: float | None
    items: list[HistoryItem]


class HistoryResponse(BaseModel):
    user_id: int
    orders: list[HistoryOrder]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    models_loaded: bool
