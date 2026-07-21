#!/usr/bin/env python3
"""
Regenerate the rule-based indicator results table from cached price data.

Previously this table only existed as notebook output, so reproducing it meant
running notebooks by hand in the right order. Run:

    python3 -m src.run_indicator_backtests

Writes results/indicator_results.csv and prints a reconciliation summary.
"""

from __future__ import annotations

import re

import pandas as pd

from .backtest import apply_backtest, save_results_table, summarize_backtest
from .config import CACHE_DIR, RESULTS_DIR, TICKERS, TRANSACTION_COST, ensure_directories
from .signals import indicator_signals


def load_cached_data() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, symbol in TICKERS.items():
        stem = symbol.replace("^", "").replace(".", "_")
        path = CACHE_DIR / f"{stem}.csv"
        if not path.exists():
            path = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_") + ".csv")
        if path.exists():
            frames[name] = pd.read_csv(path, index_col=0, parse_dates=True)
    if not frames:
        raise FileNotFoundError(
            "No cached data found under data/cache/. Run notebooks/01_data.ipynb first."
        )
    return frames


def main() -> pd.DataFrame:
    ensure_directories()
    price_data = load_cached_data()
    print(f"Loaded {len(price_data)} cached indices from {CACHE_DIR}")

    rows = []
    for index_name, frame in price_data.items():
        for strategy_name, signal in indicator_signals(frame).items():
            result = apply_backtest(frame, signal, cost=TRANSACTION_COST)
            rows.append(summarize_backtest(index_name, strategy_name, result))

    table = save_results_table(rows, RESULTS_DIR / "indicator_results.csv")

    print(f"\n{len(table)} strategy/index runs written to results/indicator_results.csv")
    print("\nTrade counts (a table of all 1s is the position-latching signature):")
    print(table["n_trades"].describe()[["min", "50%", "max"]].to_string())

    beat = (table["vs_buy_and_hold"] > 0).sum()
    print(f"\nStrategies beating buy-and-hold net of {TRANSACTION_COST:.2%} costs: "
          f"{beat} of {len(table)}")
    print(f"Median excess return vs buy-and-hold: {table['vs_buy_and_hold'].median():.2%}")

    print("\nTop 5 by Sharpe:")
    print(table.sort_values("Sharpe", ascending=False)
              .head(5)[["index", "strategy", "net_return", "Sharpe", "n_trades",
                        "vs_buy_and_hold"]]
              .to_string(index=False))
    return table


if __name__ == "__main__":
    main()
