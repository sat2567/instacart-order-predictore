"""PyTorch model: GRU over the item's purchase timeline + embeddings + MLP.

Plain idea: for each (user, product) we look at the user's last 20 orders as a
timeline — "did they buy this item in that order, and how many days apart
were the orders?" A GRU reads that timeline like a sentence and summarises the
rhythm (e.g. "buys it every 2nd order, last bought 3 orders ago"). That summary
is combined with learned product/aisle/department embeddings and the SQL
tabular features, and an MLP outputs P(reorder).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config
from .features import CAT_COLS, NUMERIC_COLS, SEQ_LEN, NumericScaler, decode_bitmask, gaps_matrix
from .metrics import pair_metrics


class ReorderNet(nn.Module):
    def __init__(self, n_products: int, n_aisles: int, n_departments: int,
                 n_numeric: int = len(NUMERIC_COLS), seq_features: int = 3,
                 gru_hidden: int = 32, emb_product: int = 16):
        super().__init__()
        self.hparams = dict(n_products=n_products, n_aisles=n_aisles, n_departments=n_departments,
                            n_numeric=n_numeric, seq_features=seq_features,
                            gru_hidden=gru_hidden, emb_product=emb_product)
        self.emb_product = nn.Embedding(n_products, emb_product)
        self.emb_aisle = nn.Embedding(n_aisles, 8)
        self.emb_dept = nn.Embedding(n_departments, 4)
        self.gru = nn.GRU(seq_features, gru_hidden, batch_first=True)
        self.num_mlp = nn.Sequential(nn.Linear(n_numeric, 64), nn.ReLU())
        d = gru_hidden + emb_product + 8 + 4 + 64
        self.head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, num, cats, seq):
        _, h = self.gru(seq)                      # h: (1, B, hidden) — last step summary
        z = torch.cat([
            h.squeeze(0),
            self.emb_product(cats[:, 0]),
            self.emb_aisle(cats[:, 1]),
            self.emb_dept(cats[:, 2]),
            self.num_mlp(num),
        ], dim=1)
        return self.head(z).squeeze(1)            # logits


# ---------------------------------------------------------------------
# Memory-light inputs: sequences are rebuilt per batch from a 64-bit mask
# and a per-user gap table instead of materialising (N, 20, 3) floats.
# ---------------------------------------------------------------------
@dataclass
class NNInputs:
    num: np.ndarray        # (N, n_numeric) float32
    cats: np.ndarray       # (N, 3) int64
    bitmask: np.ndarray    # (N,) int64
    ucode: np.ndarray      # (N,) row -> user index
    ugaps: np.ndarray      # (U, 20) float32
    uexists: np.ndarray    # (U, 20) float32
    y: np.ndarray | None = None

    def __len__(self):
        return len(self.num)

    def batch(self, idx: np.ndarray):
        bought = decode_bitmask(self.bitmask[idx])
        u = self.ucode[idx]
        seq = np.stack([bought, self.ugaps[u] / 30.0, self.uexists[u]], axis=-1)
        return (torch.from_numpy(self.num[idx]),
                torch.from_numpy(self.cats[idx]),
                torch.from_numpy(seq.astype(np.float32)))


def make_inputs(df: pd.DataFrame, scaler: NumericScaler, vocab: dict) -> NNInputs:
    codes, _ = pd.factorize(df["user_id"])
    first_row = pd.Series(np.arange(len(df))).groupby(codes).first().to_numpy()
    ugaps, uexists = gaps_matrix(df["gaps_last20"].iloc[first_row].reset_index(drop=True))
    cats = np.stack([
        np.clip(df["product_id"].to_numpy(), 0, vocab["n_products"] - 1),
        np.clip(df["aisle_id"].to_numpy(), 0, vocab["n_aisles"] - 1),
        np.clip(df["department_id"].to_numpy(), 0, vocab["n_departments"] - 1),
    ], axis=1).astype(np.int64)
    return NNInputs(
        num=scaler.transform(df),
        cats=cats,
        bitmask=df["up_bitmask"].to_numpy(dtype=np.int64),
        ucode=codes.astype(np.int64),
        ugaps=ugaps, uexists=uexists,
        y=df["label"].to_numpy(dtype=np.float32) if "label" in df else None,
    )


@torch.no_grad()
def predict(model: ReorderNet, inp: NNInputs, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    out = np.empty(len(inp), dtype=np.float32)
    for s in range(0, len(inp), batch_size):
        idx = np.arange(s, min(s + batch_size, len(inp)))
        out[idx] = torch.sigmoid(model(*inp.batch(idx))).numpy()
    return out


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, vocab: dict) -> tuple[ReorderNet, NumericScaler, dict]:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.set_num_threads(config.TORCH_THREADS)

    scaler = NumericScaler().fit(train_df)
    tr = make_inputs(train_df, scaler, vocab)
    va = make_inputs(valid_df, scaler, vocab)

    model = ReorderNet(vocab["n_products"], vocab["n_aisles"], vocab["n_departments"])
    opt = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.NN_EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}  "
          f"train rows: {len(tr):,}  valid rows: {len(va):,}")

    best, best_state, history, patience = np.inf, None, [], 0
    rng = np.random.default_rng(config.SEED)
    for epoch in range(1, config.NN_EPOCHS + 1):
        model.train()
        t0, perm, running = time.perf_counter(), rng.permutation(len(tr)), 0.0
        n_batches = int(np.ceil(len(tr) / config.NN_BATCH_SIZE))
        for b in range(n_batches):
            idx = perm[b * config.NN_BATCH_SIZE:(b + 1) * config.NN_BATCH_SIZE]
            num, cats, seq = tr.batch(idx)
            y = torch.from_numpy(tr.y[idx])
            loss = loss_fn(model(num, cats, seq), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()
        p = predict(model, va)
        m = pair_metrics(va.y, p)
        history.append({"epoch": epoch, "train_loss": running / n_batches, **m,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  epoch {epoch}  train_loss {running / n_batches:.4f}  "
              f"valid_logloss {m['logloss']:.4f}  valid_auc {m['auc']:.4f}  "
              f"({time.perf_counter() - t0:.0f}s)")
        if m["logloss"] < best - 1e-4:
            best, patience = m["logloss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                print("  early stop")
                break
    model.load_state_dict(best_state)
    return model, scaler, {"history": history, "best_valid_logloss": best}


def save(model: ReorderNet, scaler: NumericScaler, vocab: dict, info: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams}, out_dir / "nn_model.pt")
    (out_dir / "nn_preprocess.json").write_text(json.dumps(
        {"scaler": scaler.to_dict(), "vocab": vocab, "seq_len": SEQ_LEN, "cat_cols": CAT_COLS}, indent=2))
    (out_dir / "nn_training.json").write_text(json.dumps(info, indent=2))


def load(out_dir) -> tuple[ReorderNet, NumericScaler, dict]:
    ckpt = torch.load(out_dir / "nn_model.pt", map_location="cpu", weights_only=True)
    model = ReorderNet(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pre = json.loads((out_dir / "nn_preprocess.json").read_text())
    return model, NumericScaler.from_dict(pre["scaler"]), pre["vocab"]
