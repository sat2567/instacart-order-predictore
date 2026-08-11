"""Download the raw CSVs and bulk-load them into Postgres."""
from __future__ import annotations

import json
import time
import urllib.request

from . import benchmark, config, db


def download(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.RAW_FILES:
        dest = config.RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"  have {name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        url = f"{config.DATA_BASE_URL}/{name}"
        print(f"  downloading {url}")
        tmp = dest.with_suffix(".part")
        req = urllib.request.Request(url, headers={"User-Agent": "instacart-reorder/1.0"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        print(f"  saved {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def is_loaded() -> bool:
    with db.connect() as conn:
        exists = conn.execute(
            "SELECT to_regclass('raw.order_products_prior') IS NOT NULL"
        ).fetchone()[0]
        if not exists:
            return False
        return conn.execute(
            "SELECT EXISTS (SELECT 1 FROM raw.order_products_prior)"
        ).fetchone()[0]


def copy_csv(conn, csv_path, table: str) -> int:
    t0 = time.perf_counter()
    with open(csv_path, "rb") as f, conn.cursor() as cur:
        with cur.copy(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)") as cp:
            while chunk := f.read(1 << 22):
                cp.write(chunk)
        n = cur.rowcount
    print(f"  {time.perf_counter() - t0:7.1f}s  COPY {n:>12,} rows -> {table}")
    return n


def load(force: bool = False) -> None:
    if is_loaded() and not force:
        print("  raw tables already loaded (use --force to reload)")
        return
    print("  creating schema")
    db.run_sql_file(config.SQL_DIR / "01_schema.sql")

    counts = {}
    with db.connect() as conn:
        for name, table in config.RAW_FILES.items():
            counts[table] = copy_csv(conn, config.RAW_DIR / name, table)
        conn.execute("ANALYZE")

    # Benchmark the API's lookups on bare heap tables, then again with keys.
    print("  benchmarking queries WITHOUT indexes")
    before = benchmark.run_all()
    print("  adding constraints + indexes")
    db.run_sql_file(config.SQL_DIR / "02_constraints_indexes.sql")
    print("  benchmarking queries WITH indexes")
    after = benchmark.run_all()

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"row_counts": counts, "benchmark": benchmark.compare(before, after)}
    (config.ARTIFACT_DIR / "sql_benchmark.json").write_text(json.dumps(report, indent=2))
    for row in report["benchmark"]:
        print(f"    {row['query']:<22} {row['before_ms']:>10.1f} ms -> "
              f"{row['after_ms']:>8.2f} ms  ({row['speedup']:,.0f}x)")
