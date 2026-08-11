import numpy as np
import pandas as pd
import pytest

from reorder.features import NUMERIC_COLS


def make_candidates(user_id: int = 7, n: int = 6, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows shaped exactly like feat.candidates (+ names)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) * 5 for c in NUMERIC_COLS})
    df["user_id"] = user_id
    df["product_id"] = np.arange(1, n + 1)
    df["aisle_id"] = rng.integers(1, 10, n)
    df["department_id"] = rng.integers(1, 5, n)
    df["up_bitmask"] = rng.integers(0, 2**20, n)
    df["gaps_last20"] = [[7.0, 14.0, 3.0, 30.0]] * n
    df["product_name"] = [f"Product {i}" for i in range(1, n + 1)]
    df["aisle"] = "aisle"
    df["department"] = "dept"
    return df


@pytest.fixture
def candidates():
    return make_candidates()
