"""LightGBM baseline — the bar the neural net has to clear."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import LGBM_COLS
from .metrics import pair_metrics

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "seed": config.SEED,
    "num_threads": config.TORCH_THREADS,
}


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[LGBM_COLS].copy()
    for c in ("aisle_id", "department_id"):
        x[c] = x[c].astype("int32")
    return x


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    dtr = lgb.Dataset(to_matrix(train_df), train_df["label"],
                      categorical_feature=["aisle_id", "department_id"], free_raw_data=True)
    dva = lgb.Dataset(to_matrix(valid_df), valid_df["label"], reference=dtr)
    model = lgb.train(
        PARAMS, dtr, num_boost_round=2000, valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    p = model.predict(to_matrix(valid_df), num_iteration=model.best_iteration)
    info = {"best_iteration": model.best_iteration,
            "valid": pair_metrics(valid_df["label"].to_numpy(), p)}
    return model, info


def predict(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return model.predict(to_matrix(df), num_iteration=model.best_iteration)


def save(model: lgb.Booster, info: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "lgbm_model.txt"), num_iteration=model.best_iteration)
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    imp = (imp / imp.sum()).sort_values(ascending=False).round(4)
    info["feature_importance_gain"] = imp.to_dict()
    (out_dir / "lgbm_training.json").write_text(json.dumps(info, indent=2))


def load(out_dir) -> lgb.Booster:
    return lgb.Booster(model_file=str(out_dir / "lgbm_model.txt"))
