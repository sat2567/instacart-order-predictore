"""Evaluation metrics.

Two views of quality:
  * pair level  — is each (user, product) probability good?  (AUC, log-loss)
  * user level  — is each user's predicted basket good?       (F1, P@k, R@k, NDCG@k)
The user-level ones are what a product manager cares about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def pair_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p)),
    }


def basket_f1(df: pd.DataFrame, threshold: float) -> float:
    """Mean per-user F1 when the predicted basket = items with p >= threshold.
    Kaggle convention: a user with no reorders and an empty predicted basket
    ('None') scores 1.0."""
    d = df.assign(pred=(df["prob"] >= threshold).astype(np.int8))
    g = d.assign(tp=d["pred"] * d["label"]).groupby("user_id").agg(
        tp=("tp", "sum"), n_pred=("pred", "sum"), n_true=("label", "sum"))
    precision = np.where(g.n_pred > 0, g.tp / g.n_pred.clip(lower=1), 0.0)
    recall = np.where(g.n_true > 0, g.tp / g.n_true.clip(lower=1), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    f1 = np.where((g.n_true == 0) & (g.n_pred == 0), 1.0, f1)
    return float(np.mean(f1))


def tune_threshold(df: pd.DataFrame, grid=None) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.51, 0.01), 2) if grid is None else grid
    scores = [(t, basket_f1(df, t)) for t in grid]
    return max(scores, key=lambda x: x[1])


def topk_metrics(df: pd.DataFrame, k: int = 10) -> dict:
    """Ranking metrics over users with at least one reorder."""
    d = df[df.groupby("user_id")["label"].transform("sum") > 0].copy()
    d["rank"] = d.groupby("user_id")["prob"].rank(method="first", ascending=False)
    d["gain"] = np.where(d["rank"] <= k, d["label"] / np.log2(d["rank"] + 1), 0.0)
    top = d[d["rank"] <= k]
    hits = top.groupby("user_id")["label"].sum()
    n_true = d.groupby("user_id")["label"].sum()
    dcg = d.groupby("user_id")["gain"].sum()
    ideal = n_true.clip(upper=k).apply(lambda n: np.sum(1.0 / np.log2(np.arange(2, n + 2))))
    n_cand = d.groupby("user_id").size()
    return {
        f"precision@{k}": float((hits / np.minimum(k, n_cand)).mean()),
        f"recall@{k}": float((hits / n_true).mean()),
        f"ndcg@{k}": float((dcg / ideal).mean()),
        f"hit_rate@{k}": float((hits > 0).mean()),
        "users_evaluated": int(len(n_true)),
    }


def full_report(df: pd.DataFrame, threshold: float, k: int = 10) -> dict:
    return {
        **pair_metrics(df["label"].to_numpy(), df["prob"].to_numpy()),
        "basket_f1": basket_f1(df, threshold),
        "threshold": threshold,
        **topk_metrics(df, k),
    }
