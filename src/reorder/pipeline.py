"""End-to-end pipeline CLI.

    python -m reorder.pipeline all          # everything, idempotent
    python -m reorder.pipeline <step> [--force]

Steps: download -> load -> features -> analysis -> train-lgbm -> train-nn -> evaluate
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import benchmark, config, db, ingest, lgbm, nn
from .features import load_split, save_json
from .metrics import full_report, tune_threshold

ART = config.ARTIFACT_DIR


def step(name):
    def deco(fn):
        def wrapper(*a, **kw):
            print(f"\n=== {name} ===")
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            print(f"=== {name} done in {time.perf_counter() - t0:.1f}s ===")
            return out
        return wrapper
    return deco


@step("download")
def run_download(force=False):
    ingest.download(force)


@step("load raw tables")
def run_load(force=False):
    ingest.load(force)


def features_built() -> bool:
    with db.connect() as conn:
        return conn.execute("SELECT to_regclass('feat.candidates') IS NOT NULL").fetchone()[0]


@step("build SQL feature store")
def run_features(force=False):
    if features_built() and not force:
        print("  feature store exists (use --force to rebuild)")
        return
    db.run_sql_file(config.SQL_DIR / "03_features.sql")
    lat = benchmark.serving_query_latency()
    print(f"  serving query (one user's candidates): {lat['ms']:.2f} ms")
    path = ART / "sql_benchmark.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["serving_query"] = lat
    save_json(report, path)


@step("SQL analysis")
def run_analysis(force=False):
    out = ["# SQL analysis (auto-generated)\n"]
    for name, sql in db.named_queries(config.SQL_DIR / "analysis.sql"):
        df = db.fetch_df(sql)
        out.append(f"## {name}\n\n{df.to_markdown(index=False)}\n")
    (ART / "analysis.md").write_text("\n".join(out))
    print(f"  wrote {ART / 'analysis.md'}")


def vocab() -> dict:
    with db.connect() as conn:
        p, a, d = conn.execute(
            "SELECT (SELECT MAX(product_id) FROM raw.products),"
            "       (SELECT MAX(aisle_id) FROM raw.aisles),"
            "       (SELECT MAX(department_id) FROM raw.departments)").fetchone()
    return {"n_products": p + 1, "n_aisles": a + 1, "n_departments": d + 1}


@step("train LightGBM")
def run_train_lgbm(force=False):
    if (ART / "lgbm_model.txt").exists() and not force:
        print("  model exists (use --force to retrain)")
        return
    tr, va = load_split("train"), load_split("valid")
    model, info = lgbm.train(tr, va)
    info["sample_pct"] = config.USER_SAMPLE_PCT
    lgbm.save(model, info, ART)
    print(f"  valid: {info['valid']}")


@step("train neural net")
def run_train_nn(force=False):
    if (ART / "nn_model.pt").exists() and not force:
        print("  model exists (use --force to retrain)")
        return
    tr, va = load_split("train"), load_split("valid")
    model, scaler, info = nn.train(tr, va, vocab())
    info["sample_pct"] = config.USER_SAMPLE_PCT
    nn.save(model, scaler, vocab(), info, ART)


def score_all(df: pd.DataFrame) -> dict[str, np.ndarray]:
    model_nn, scaler, voc = nn.load(ART)
    return {
        "heuristic": df["up_order_rate"].to_numpy(),   # "share of past orders with this item"
        "lightgbm": lgbm.predict(lgbm.load(ART), df),
        "neural_net": nn.predict(model_nn, nn.make_inputs(df, scaler, voc)),
    }


@step("evaluate on held-out test users")
def run_evaluate(force=False):
    va, te = load_split("valid"), load_split("test")
    s_va, s_te = score_all(va), score_all(te)
    s_va["ensemble"] = (s_va["lightgbm"] + s_va["neural_net"]) / 2
    s_te["ensemble"] = (s_te["lightgbm"] + s_te["neural_net"]) / 2

    results = {}
    for name in s_te:
        thr, f1_va = tune_threshold(va[["user_id", "label"]].assign(prob=s_va[name]))
        rep = full_report(te[["user_id", "label"]].assign(prob=s_te[name]), thr, config.TOP_K)
        rep["valid_basket_f1"] = f1_va
        results[name] = rep

    summary = {
        "test_users": int(te.user_id.nunique()),
        "test_rows": int(len(te)),
        "positive_rate": float(te.label.mean()),
        "sample_pct": config.USER_SAMPLE_PCT,
        "models": results,
    }
    save_json(summary, ART / "metrics.json")

    cols = ["auc", "pr_auc", "logloss", "basket_f1", "threshold",
            "precision@10", "recall@10", "ndcg@10", "hit_rate@10"]
    table = pd.DataFrame(results).T[cols].astype(float).round(4)
    (ART / "metrics.md").write_text(
        f"Test users: {summary['test_users']:,} | candidate rows: {summary['test_rows']:,} | "
        f"positive rate: {summary['positive_rate']:.3f}\n\n" + table.to_markdown() + "\n")
    print(table.to_string())


STEPS = {
    "download": run_download,
    "load": run_load,
    "features": run_features,
    "analysis": run_analysis,
    "train-lgbm": run_train_lgbm,
    "train-nn": run_train_nn,
    "evaluate": run_evaluate,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--force", action="store_true", help="redo the step even if outputs exist")
    args = ap.parse_args()

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if args.step != "download":
        db.wait_for_db()
    if args.step == "all":
        for name, fn in STEPS.items():
            fn(force=args.force and name != "download")  # never re-download implicitly
    else:
        STEPS[args.step](force=args.force)


if __name__ == "__main__":
    main()
