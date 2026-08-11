"""Thin Postgres helpers (psycopg 3)."""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import psycopg

from . import config


@contextmanager
def connect(autocommit: bool = True):
    conn = psycopg.connect(config.DATABASE_URL, autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def wait_for_db(timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            with connect() as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError:
            if time.time() > deadline:
                raise
            time.sleep(2)


def run_sql_file(path: Path) -> None:
    """Execute a .sql file statement by statement, logging timings."""
    sql = Path(path).read_text()
    statements = [s.strip() for s in split_statements(sql) if s.strip()]
    with connect() as conn:
        for stmt in statements:
            head = " ".join(stmt.split())[:90]
            t0 = time.perf_counter()
            conn.execute(stmt)
            print(f"  {time.perf_counter() - t0:7.1f}s  {head}")


def split_statements(sql: str) -> list[str]:
    """Split on ';' at line ends, ignoring comment lines (no ';' in our literals)."""
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s for s in re.split(r";\s*$", "\n".join(lines), flags=re.M)]


def named_queries(path: Path) -> list[tuple[str, str]]:
    """Parse files written as '-- name: X' followed by one statement."""
    out, name, buf = [], None, []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"--\s*name:\s*(.+)", line)
        if m:
            if name:
                out.append((name, "\n".join(buf).strip().rstrip(";")))
            name, buf = m.group(1).strip(), []
        elif not line.strip().startswith("--"):
            buf.append(line)
    if name:
        out.append((name, "\n".join(buf).strip().rstrip(";")))
    return out


def fetch_df(query: str, params: tuple | dict | None = None) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
