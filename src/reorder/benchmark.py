"""EXPLAIN ANALYZE timings for the API's hot queries."""
from __future__ import annotations

import json
import statistics

from . import config, db


def explain(conn, sql: str, repeats: int = 3) -> dict:
    """Median execution time over `repeats` runs (first run warms the cache)."""
    times, plan_top = [], None
    for _ in range(repeats + 1):
        row = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}").fetchone()[0]
        plan = row[0] if isinstance(row, list) else json.loads(row)[0]
        times.append(plan["Execution Time"])
        plan_top = plan["Plan"]
    return {
        "ms": statistics.median(times[1:]),
        "plan": summarize(plan_top),
    }


def summarize(node: dict) -> list[str]:
    """Flatten the plan into the scan types used, e.g. 'Index Scan on orders'."""
    out = []
    def walk(n):
        if "Relation Name" in n:
            out.append(f"{n['Node Type']} on {n['Relation Name']}")
        for c in n.get("Plans", []):
            walk(c)
    walk(node)
    return out


def run_all() -> dict:
    results = {}
    with db.connect() as conn:
        for name, sql in db.named_queries(config.SQL_DIR / "benchmark_queries.sql"):
            results[name] = explain(conn, sql)
    return results


def compare(before: dict, after: dict) -> list[dict]:
    rows = []
    for name in before:
        b, a = before[name]["ms"], after[name]["ms"]
        rows.append({
            "query": name,
            "before_ms": round(b, 2),
            "after_ms": round(a, 3),
            "speedup": round(b / max(a, 1e-3), 1),
            "plan_before": before[name]["plan"],
            "plan_after": after[name]["plan"],
        })
    return rows


def serving_query_latency(user_id: int = 4242) -> dict:
    """Latency of the API's feature fetch on the candidates view."""
    with db.connect() as conn:
        return explain(conn, f"SELECT * FROM feat.candidates WHERE user_id = {int(user_id)}", repeats=5)
