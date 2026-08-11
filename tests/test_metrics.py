import numpy as np
import pandas as pd
import pytest

from reorder.metrics import basket_f1, pair_metrics, topk_metrics, tune_threshold


def frame(rows):
    return pd.DataFrame(rows, columns=["user_id", "label", "prob"])


def test_basket_f1_perfect_and_none_convention():
    df = frame([
        (1, 1, 0.9), (1, 0, 0.1),        # user 1: perfect
        (2, 0, 0.1), (2, 0, 0.05),       # user 2: no reorders, predicts None -> 1.0
    ])
    assert basket_f1(df, 0.5) == pytest.approx(1.0)


def test_basket_f1_half():
    # predicts {a, b}, truth {a}: P = .5, R = 1, F1 = 2/3
    df = frame([(1, 1, 0.9), (1, 0, 0.8), (1, 0, 0.1)])
    assert basket_f1(df, 0.5) == pytest.approx(2 / 3)


def test_topk_perfect_ranking():
    df = frame([(1, 1, 0.9), (1, 1, 0.8), (1, 0, 0.1), (2, 0, 0.3), (2, 1, 0.7)])
    m = topk_metrics(df, k=2)
    assert m["ndcg@2"] == pytest.approx(1.0)
    assert m["recall@2"] == pytest.approx(1.0)


def test_topk_skips_users_without_positives():
    df = frame([(1, 1, 0.9), (2, 0, 0.5)])
    assert topk_metrics(df, k=5)["users_evaluated"] == 1


def test_pair_metrics_random_is_half_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    m = pair_metrics(y, rng.random(5000))
    assert 0.45 < m["auc"] < 0.55


def test_tune_threshold_picks_separating_value():
    df = frame([(1, 1, 0.4), (1, 0, 0.2), (2, 1, 0.45), (2, 0, 0.1)])
    thr, f1 = tune_threshold(df)
    assert 0.2 < thr <= 0.4 and f1 == pytest.approx(1.0)
