"""Feature definitions shared by training and serving.

The SQL view feat.candidates produces every column; this module only decides
which columns each model uses and how to turn them into arrays/tensors.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db

ID_COLS = ["user_id", "product_id"]
CAT_COLS = ["product_id", "aisle_id", "department_id"]

NUMERIC_COLS = [
    # user
    "u_n_orders", "u_avg_gap", "u_std_gap", "u_avg_basket", "u_n_distinct",
    "u_distinct_ratio", "u_reorder_ratio", "u_avg_hour",
    # target-order context
    "t_days_since_last", "t_dow", "t_hour",
    # product
    "p_purchases", "p_reorder_rate", "p_avg_cart_pos", "p_users",
    "p_repeat_user_share", "p_avg_buys_per_user",
    # user x product
    "up_orders", "up_order_rate", "up_rate_since_first", "up_orders_since_last",
    "up_days_since_last", "up_avg_interval", "up_avg_cart_pos", "up_last5",
    "up_last5_rate", "ua_share", "ud_share",
]

# Heavy-tailed counts get log1p before standardising (NN only; trees don't care).
LOG_COLS = {
    "u_n_orders", "u_n_distinct", "p_purchases", "p_users", "up_orders",
    "up_orders_since_last", "p_avg_buys_per_user",
}

# LightGBM sees the tabular features + categorical ids it can split on.
LGBM_COLS = NUMERIC_COLS + ["aisle_id", "department_id"]

SEQ_LEN = config.SEQ_LEN


# ---------------------------------------------------------------------
# Sequence decoding
# ---------------------------------------------------------------------
def decode_bitmask(mask: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """bit (k-1) set = bought in k-th most recent order  ->  (n, seq_len) 0/1,
    ordered oldest -> newest so the GRU's last step is the latest order."""
    mask = np.asarray(mask, dtype=np.int64)
    bits = (mask[:, None] >> np.arange(seq_len, dtype=np.int64)) & 1  # col 0 = most recent
    return bits[:, ::-1].astype(np.float32)


def gaps_matrix(gaps: pd.Series, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad each user's gap list to seq_len. Returns (gaps, exists_mask)."""
    n = len(gaps)
    out = np.zeros((n, seq_len), dtype=np.float32)
    exists = np.zeros((n, seq_len), dtype=np.float32)
    for i, g in enumerate(gaps):
        g = parse_pg_array(g)[-seq_len:]
        if len(g):
            out[i, seq_len - len(g):] = g
            exists[i, seq_len - len(g):] = 1.0
    return out, exists


def parse_pg_array(v) -> list[float]:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    s = str(v).strip("{}")
    return [float(x) for x in s.split(",")] if s else []


# ---------------------------------------------------------------------
# Numeric scaling (fit on train split, saved with the model)
# ---------------------------------------------------------------------
class NumericScaler:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    @staticmethod
    def _raw(df: pd.DataFrame) -> np.ndarray:
        x = df[NUMERIC_COLS].astype("float64").fillna(0.0).to_numpy(copy=True)
        for j, c in enumerate(NUMERIC_COLS):
            if c in LOG_COLS:
                x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
        return x

    def fit(self, df: pd.DataFrame) -> NumericScaler:
        x = self._raw(df)
        self.mean = x.mean(axis=0).tolist()
        self.std = (x.std(axis=0) + 1e-6).tolist()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = (self._raw(df) - np.array(self.mean)) / np.array(self.std)
        return np.clip(x, -8, 8).astype(np.float32)

    def to_dict(self):
        return {"cols": NUMERIC_COLS, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d):
        assert d["cols"] == NUMERIC_COLS, "feature list changed since training"
        return cls(d["mean"], d["std"])


# ---------------------------------------------------------------------
# Export training data from Postgres
# ---------------------------------------------------------------------
SPLITS = ("train", "valid", "test")

TRAINING_QUERY = """
SELECT c.*, COALESCE(l.label, 0) AS label
FROM feat.candidates c
JOIN feat.user_split s USING (user_id)
LEFT JOIN feat.labels l USING (user_id, product_id)
WHERE s.split = {split} AND c.user_id % 100 < {pct}
"""


def export_split(split: str, pct: int) -> pd.DataFrame:
    """Stream one split out of Postgres with COPY ... TO STDOUT
    (far faster and lighter than fetchall on millions of rows)."""
    import io

    from psycopg import sql

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    inner = sql.SQL(TRAINING_QUERY).format(split=sql.Literal(split), pct=sql.Literal(int(pct)))
    stmt = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT csv, HEADER true)").format(inner)
    buf = io.BytesIO()
    with db.connect() as conn, conn.cursor() as cur, cur.copy(stmt) as cp:
        for chunk in cp:
            buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf)


def load_split(split: str, pct: int | None = None, refresh: bool = False) -> pd.DataFrame:
    pct = config.USER_SAMPLE_PCT if pct is None else pct
    path = config.PROCESSED_DIR / f"{split}_pct{pct}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    t0 = time.perf_counter()
    df = export_split(split, pct)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  exported {split:<5} {len(df):>10,} rows, {df.user_id.nunique():>7,} users, "
          f"pos rate {df.label.mean():.3f}  ({time.perf_counter() - t0:.1f}s)")
    return df


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
