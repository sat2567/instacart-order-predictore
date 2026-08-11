"""All SQL the API runs. Pooled connections; parameterised queries only."""
from __future__ import annotations

import pandas as pd
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

CANDIDATES_SQL = """
SELECT c.*, p.product_name, a.aisle, d.department
FROM feat.candidates c
JOIN raw.products    p ON p.product_id    = c.product_id
JOIN raw.aisles      a ON a.aisle_id      = c.aisle_id
JOIN raw.departments d ON d.department_id = c.department_id
WHERE c.user_id = %s
"""

HISTORY_SQL = """
WITH recent AS (
    SELECT order_id, order_number, order_dow, order_hour_of_day, days_since_prior_order
    FROM raw.orders
    WHERE user_id = %(uid)s AND eval_set = 'prior'
    ORDER BY order_number DESC
    LIMIT %(n)s
)
SELECT r.order_number, r.order_dow, r.order_hour_of_day, r.days_since_prior_order,
       op.add_to_cart_order, op.product_id, p.product_name, op.reordered
FROM recent r
JOIN raw.order_products_prior op ON op.order_id = r.order_id
JOIN raw.products p             ON p.product_id = op.product_id
ORDER BY r.order_number DESC, op.add_to_cart_order
"""

SAMPLE_USERS_SQL = """
SELECT user_id, split FROM feat.user_split
WHERE split IN ('test', 'unlabeled')
ORDER BY random() LIMIT %s
"""


class Repository:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5):
        self.pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True,
                                   kwargs={"autocommit": True})

    def close(self):
        self.pool.close()

    def ping(self) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1

    def candidates(self, user_id: int) -> pd.DataFrame:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(CANDIDATES_SQL, (user_id,))
            cols = [d.name for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)

    def history(self, user_id: int, n_orders: int) -> list[dict]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(HISTORY_SQL, {"uid": user_id, "n": n_orders})
            return cur.fetchall()

    def sample_users(self, n: int) -> list[dict]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(SAMPLE_USERS_SQL, (n,))
            return cur.fetchall()
