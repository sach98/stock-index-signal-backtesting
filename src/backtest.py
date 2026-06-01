"""Backtest utilities shared by the notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR, TRANSACTION_COST


def prepare_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized OHLCV frame with adjusted close and log returns."""
    if data.empty:
        raise ValueError("Cannot prepare an empty price frame.")

    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    if "adj_close" not in frame and "close" in frame:
        frame["adj_close"] = frame["close"]
    if "close" not in frame and "adj_close" in frame:
        frame["close"] = frame["adj_close"]

    required = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Price frame is missing required columns: {missing}")

    frame = frame.loc[:, ["open", "high", "low", "close", "adj_close", "volume"]]
    frame = frame.dropna(subset=["adj_close"]).sort_index()
    frame["returns"] = np.log(frame["adj_close"] / frame["adj_close"].shift(1))
    next_return = frame["returns"].shift(-1)
    frame["direction"] = (next_return > 0).astype(float)
    frame.loc[next_return.isna(), "direction"] = np.nan
    return frame.dropna(subset=["returns"])


def position_from_signal(signal: pd.Series) -> pd.Series:
    """Convert raw signal values to lagged market positions."""
    position = signal.replace(0, np.nan).ffill().fillna(0.0)
    return position.shift(1).fillna(0.0).clip(-1, 1)


def apply_backtest(
    frame: pd.DataFrame,
    signal: pd.Series,
    cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """Apply a long/flat/short signal with one-period lag and transaction costs."""
    result = frame.copy()
    aligned_signal = signal.reindex(result.index).fillna(0.0)
    result["signal"] = aligned_signal
    result["position"] = position_from_signal(aligned_signal)
    result["trade"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["gross_strategy_return"] = result["position"] * result["returns"]
    result["cost"] = result["trade"] * cost
    result["net_strategy_return"] = result["gross_strategy_return"] - result["cost"]
    result["buy_and_hold_return"] = result["returns"]
    return result


def cumulative_return(log_returns: pd.Series) -> float:
    """Convert a series of log returns into a simple cumulative return."""
    clean = log_returns.dropna()
    if clean.empty:
        return np.nan
    return float(np.exp(clean.sum()) - 1.0)


def annualized_volatility(log_returns: pd.Series) -> float:
    """Annualized volatility from daily log returns."""
    clean = log_returns.dropna()
    if clean.empty:
        return np.nan
    return float(clean.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(log_returns: pd.Series) -> float:
    """Annualized Sharpe ratio using a zero risk free rate."""
    clean = log_returns.dropna()
    std = clean.std(ddof=0)
    if clean.empty or std == 0 or np.isnan(std):
        return np.nan
    return float((clean.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(log_returns: pd.Series) -> float:
    """Maximum drawdown from daily log returns."""
    clean = log_returns.dropna()
    if clean.empty:
        return np.nan
    equity = np.exp(clean.cumsum())
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize_backtest(
    index_name: str,
    strategy_name: str,
    backtest_frame: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Summarize one strategy/index run for the results table."""
    gross = backtest_frame["gross_strategy_return"]
    net = backtest_frame["net_strategy_return"]
    buy_hold = backtest_frame["buy_and_hold_return"]
    net_return = cumulative_return(net)
    buy_hold_return = cumulative_return(buy_hold)

    return {
        "index": index_name,
        "strategy": strategy_name,
        "gross_return": cumulative_return(gross),
        "net_return": net_return,
        "ann_vol": annualized_volatility(net),
        "Sharpe": sharpe_ratio(net),
        "max_drawdown": max_drawdown(net),
        "n_trades": int(backtest_frame["trade"].sum()),
        "vs_buy_and_hold": net_return - buy_hold_return,
    }


def save_results_table(rows: list[dict[str, float | int | str]], path: str | None = None) -> pd.DataFrame:
    """Save a stable CSV results table from summary dictionaries."""
    columns = [
        "index",
        "strategy",
        "gross_return",
        "net_return",
        "ann_vol",
        "Sharpe",
        "max_drawdown",
        "n_trades",
        "vs_buy_and_hold",
    ]
    table = pd.DataFrame(rows, columns=columns)
    if path is not None:
        table.to_csv(path, index=False)
    return table
