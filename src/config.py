"""Project configuration for reproducible stock-index signal backtests."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 42

START_DATE = "2013-01-01"
END_DATE = "2023-01-01"
TRAIN_END_DATE = "2019-12-31"

TRADING_DAYS_PER_YEAR = 252
TRANSACTION_COST = 0.001

TICKERS = {
    "S&P 500": "^GSPC",
    "Dow Jones Industrial Average": "^DJI",
    "NASDAQ Composite": "^IXIC",
    "Russell 2000": "^RUT",
    "FTSE 100": "^FTSE",
    "Cboe UK 100": "^BUK100P",
    "Cboe Volatility Index": "^VIX",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"


def set_global_seed(seed: int = SEED) -> None:
    """Set seeds for reproducible NumPy, Python, and optional TensorFlow code."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_directories() -> None:
    """Create runtime output folders if they are missing."""
    for directory in (DATA_DIR, CACHE_DIR, RESULTS_DIR, CHARTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


set_global_seed(SEED)
