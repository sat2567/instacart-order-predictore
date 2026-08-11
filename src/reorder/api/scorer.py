"""Loads trained artifacts once and scores candidate DataFrames."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import lgbm, nn

MODELS = ("neural_net", "lightgbm", "ensemble")


class Scorer:
    def __init__(self, artifact_dir: Path):
        self.dir = Path(artifact_dir)
        self.nn_model, self.scaler, self.vocab = nn.load(self.dir)
        self.lgbm_model = lgbm.load(self.dir)
        metrics_path = self.dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    def threshold(self, model: str) -> float:
        return float(self.metrics.get("models", {}).get(model, {}).get("threshold", 0.2))

    def score(self, df: pd.DataFrame, model: str = "ensemble") -> np.ndarray:
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if model == "lightgbm":
            return lgbm.predict(self.lgbm_model, df)
        p_nn = nn.predict(self.nn_model, nn.make_inputs(df, self.scaler, self.vocab))
        if model == "neural_net":
            return p_nn
        return (p_nn + lgbm.predict(self.lgbm_model, df)) / 2

    def info(self) -> dict:
        return {
            "models": list(MODELS),
            "default_model": "ensemble",
            "nn_params": int(sum(p.numel() for p in self.nn_model.parameters())),
            "lgbm_trees": int(self.lgbm_model.num_trees()),
            "test_metrics": self.metrics.get("models", {}),
            "test_users": self.metrics.get("test_users"),
        }
