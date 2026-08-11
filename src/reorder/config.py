"""Central configuration, all overridable with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://instacart:instacart@localhost:5432/instacart"
)

SQL_DIR = Path(os.getenv("SQL_DIR", ROOT / "sql"))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))

# Public mirror of the Kaggle "Instacart Market Basket Analysis" files.
DATA_BASE_URL = os.getenv(
    "DATA_BASE_URL",
    "https://huggingface.co/datasets/attik/Instacart-Market-Basket-Analysis/resolve/main",
)
RAW_FILES = {
    # csv file name            -> target table
    "aisles.csv": "raw.aisles",
    "departments.csv": "raw.departments",
    "products.csv": "raw.products",
    "orders.csv": "raw.orders",
    "order_products__prior.csv": "raw.order_products_prior",
    "order_products__train.csv": "raw.order_products_train",
}

# Train on a % of users (by user_id % 100) to fit smaller machines.
# 100 = all ~131k labelled users (~8.5M candidate rows).
USER_SAMPLE_PCT = int(os.getenv("USER_SAMPLE_PCT", "100"))

SEQ_LEN = 20          # orders of history fed to the GRU
SEED = int(os.getenv("SEED", "42"))

# Neural-net training
NN_EPOCHS = int(os.getenv("NN_EPOCHS", "6"))
NN_BATCH_SIZE = int(os.getenv("NN_BATCH_SIZE", "4096"))
NN_LR = float(os.getenv("NN_LR", "2e-3"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 2)))

TOP_K = 10
