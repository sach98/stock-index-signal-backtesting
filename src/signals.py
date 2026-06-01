"""Technical indicator and feature helpers for stock-index backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def exponential_moving_average(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = exponential_moving_average(close, fast)
    slow_ema = exponential_moving_average(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        },
        index=close.index,
    )


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    middle = simple_moving_average(close, window)
    rolling_std = close.rolling(window=window, min_periods=window).std()
    upper = middle + n_std * rolling_std
    lower = middle - n_std * rolling_std
    return pd.DataFrame(
        {
            "bb_lower": lower,
            "bb_middle": middle,
            "bb_upper": upper,
        },
        index=close.index,
    )


def momentum(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change(periods=window)


def indicator_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create lagged return and indicator features without target leakage."""
    close = frame["adj_close"]
    features = pd.DataFrame(index=frame.index)

    for lag in range(1, 6):
        features[f"return_lag_{lag}"] = frame["returns"].shift(lag)

    features["rsi_14"] = relative_strength_index(close, 14)
    features = features.join(macd(close))
    features = features.join(bollinger_bands(close))
    features["momentum_20"] = momentum(close, 20)
    features["sma_20_ratio"] = close / simple_moving_average(close, 20) - 1
    features["sma_50_ratio"] = close / simple_moving_average(close, 50) - 1
    features["volatility_20"] = frame["returns"].rolling(20, min_periods=20).std()
    return features


def indicator_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return rule-based strategy signals using only current/past data."""
    close = frame["adj_close"]
    macd_frame = macd(close)
    rsi = relative_strength_index(close)
    bands = bollinger_bands(close)
    mom = momentum(close)
    sma_fast = simple_moving_average(close, 20)
    sma_slow = simple_moving_average(close, 50)

    signals = {
        "MACD": np.where(macd_frame["macd"] > macd_frame["macd_signal"], 1, 0),
        "RSI": np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan)),
        "SMA crossover": np.where(sma_fast > sma_slow, 1, 0),
        "Bollinger Bands": np.where(close < bands["bb_lower"], 1, np.where(close > bands["bb_upper"], 0, np.nan)),
        "Momentum": np.where(mom > 0, 1, 0),
    }
    return {name: pd.Series(values, index=frame.index, name=name) for name, values in signals.items()}


def safe_ml_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Keep predictors while excluding labels, realized returns, and strategy outputs."""
    leakage_columns = {
        "direction",
        "returns",
        "strategy_return",
        "gross_strategy_return",
        "net_strategy_return",
        "buy_and_hold_return",
        "signal",
        "position",
        "trade",
        "cost",
    }
    return [column for column in frame.columns if column not in leakage_columns]
