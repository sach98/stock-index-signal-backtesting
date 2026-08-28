"""
Behavioural tests for the backtest engine.

The central test is TestPositionLatching. An earlier version of
position_from_signal forward-filled over zero, so once a strategy went long it
never exited. Every published result collapsed to buy-and-hold with a single
trade. These tests assert the exit actually happens.
"""

import unittest

import numpy as np
import pandas as pd

from src.backtest import (
    apply_backtest,
    cumulative_return,
    max_drawdown,
    position_from_signal,
    prepare_price_frame,
    sharpe_ratio,
    summarize_backtest,
)


def dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


class TestPositionLatching(unittest.TestCase):
    """Regression guard for the bug that voided every published result."""

    def test_zero_means_flat_and_is_not_carried_forward(self):
        # State-style signal: long, long, flat, flat.
        signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates(4))
        position = position_from_signal(signal)
        # Lagged by one day: [0, 1, 1, 0]
        self.assertEqual(list(position), [0.0, 1.0, 1.0, 0.0])

    def test_alternating_signal_produces_multiple_trades(self):
        signal = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=dates(6))
        position = position_from_signal(signal)
        trades = position.diff().abs().fillna(position.abs()).sum()
        self.assertGreater(
            trades, 1.0,
            msg="an alternating signal must trade more than once; a value of 1 "
                "means the position latched and the strategy is buy-and-hold",
        )

    def test_nan_means_hold(self):
        # Event-style signal: enter, hold, hold, exit.
        signal = pd.Series([1.0, np.nan, np.nan, 0.0], index=dates(4))
        position = position_from_signal(signal)
        self.assertEqual(list(position), [0.0, 1.0, 1.0, 1.0])

    def test_leading_nan_is_flat_not_forward_filled_backwards(self):
        signal = pd.Series([np.nan, np.nan, 1.0, 0.0], index=dates(4))
        position = position_from_signal(signal)
        self.assertEqual(list(position), [0.0, 0.0, 0.0, 1.0])

    def test_signal_is_lagged_by_one_period(self):
        """A signal computed from today's close cannot be traded today."""
        signal = pd.Series([1.0, 0.0, 0.0, 0.0], index=dates(4))
        position = position_from_signal(signal)
        self.assertEqual(position.iloc[0], 0.0)
        self.assertEqual(position.iloc[1], 1.0)


class TestApplyBacktest(unittest.TestCase):

    def setUp(self):
        n = 6
        idx = dates(n)
        price = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0, 104.0], index=idx)
        self.frame = pd.DataFrame({
            "open": price, "high": price, "low": price,
            "close": price, "adj_close": price, "volume": 1000.0,
        }, index=idx)
        self.frame = prepare_price_frame(self.frame)

    def alternating(self):
        """Alternate long/flat across however many rows survived preparation."""
        values = [1.0 if i % 2 == 0 else 0.0 for i in range(len(self.frame))]
        return pd.Series(values, index=self.frame.index)

    def test_flat_position_earns_nothing(self):
        signal = pd.Series(0.0, index=self.frame.index)
        result = apply_backtest(self.frame, signal, cost=0.0)
        self.assertAlmostEqual(result["gross_strategy_return"].sum(), 0.0, places=12)

    def test_always_long_matches_buy_and_hold_gross(self):
        signal = pd.Series(1.0, index=self.frame.index)
        result = apply_backtest(self.frame, signal, cost=0.0)
        # Position is lagged, so the first day is flat by construction.
        expected = result["returns"].iloc[1:].sum()
        self.assertAlmostEqual(result["gross_strategy_return"].sum(), expected, places=12)

    def test_costs_reduce_net_return(self):
        signal = self.alternating()
        free = apply_backtest(self.frame, signal, cost=0.0)
        costly = apply_backtest(self.frame, signal, cost=0.01)
        self.assertLess(costly["net_strategy_return"].sum(), free["net_strategy_return"].sum())

    def test_alternating_strategy_is_not_buy_and_hold(self):
        signal = self.alternating()
        result = apply_backtest(self.frame, signal, cost=0.0)
        summary = summarize_backtest("TEST", "alternating", result)
        self.assertGreater(
            summary["n_trades"], 1,
            msg="n_trades of 1 across a whole backtest is the latching signature",
        )

    def test_hold_semantics_survive_the_full_pipeline(self):
        """
        Event-driven rules (RSI, Bollinger) emit NaN on every day that is
        neither an entry nor an exit. apply_backtest must not flatten those
        NaNs to zero before position_from_signal sees them, or the strategy
        exits the market on every neutral day.
        """
        values = [np.nan] * len(self.frame)
        values[0] = 1.0          # entry, then hold for the rest of the window
        signal = pd.Series(values, index=self.frame.index)
        result = apply_backtest(self.frame, signal, cost=0.0)

        # After the one-day lag the position should be long for every
        # subsequent day, and should trade exactly once (the entry).
        self.assertEqual(list(result["position"])[1:], [1.0] * (len(self.frame) - 1))
        self.assertEqual(int(result["trade"].sum()), 1)


class TestMetrics(unittest.TestCase):

    def test_cumulative_return_of_known_log_returns(self):
        # Two 10% up days compound to 21%.
        r = pd.Series([np.log(1.1), np.log(1.1)])
        self.assertAlmostEqual(cumulative_return(r), 0.21, places=10)

    def test_max_drawdown_of_known_path(self):
        # 100 -> 120 -> 90: peak-to-trough is 25%.
        prices = pd.Series([100.0, 120.0, 90.0])
        r = np.log(prices / prices.shift(1)).dropna()
        self.assertAlmostEqual(max_drawdown(r), -0.25, places=10)

    def test_sharpe_of_constant_returns_is_not_finite_noise(self):
        r = pd.Series([0.001] * 50)
        self.assertTrue(np.isnan(sharpe_ratio(r)))

    def test_sharpe_is_nan_when_volatility_is_only_rounding_residue(self):
        # The constant-series test above only exercises this when numpy's
        # summation happens to leave a residue, which is architecture
        # dependent: it cancels to exactly 0.0 on arm64 and does not on
        # x86-64, so that test passed locally and failed in CI. This one
        # injects the residue directly, so the guard is checked everywhere.
        r = pd.Series([0.001] * 49 + [0.001 + 1e-18])
        self.assertGreater(r.std(ddof=0), 0.0)
        self.assertTrue(np.isnan(sharpe_ratio(r)))

    def test_sharpe_is_finite_for_genuine_volatility(self):
        rng = np.random.default_rng(20260829)
        r = pd.Series(rng.normal(0.0004, 0.01, 500))
        self.assertTrue(np.isfinite(sharpe_ratio(r)))

    def test_empty_series_returns_nan(self):
        empty = pd.Series(dtype=float)
        self.assertTrue(np.isnan(cumulative_return(empty)))
        self.assertTrue(np.isnan(max_drawdown(empty)))


class TestNoLookAhead(unittest.TestCase):
    """
    The direction label must describe the NEXT period, and must never be
    available to a position taken today.
    """

    def setUp(self):
        idx = dates(5)
        price = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0], index=idx)
        self.frame = prepare_price_frame(pd.DataFrame({
            "open": price, "high": price, "low": price,
            "close": price, "adj_close": price, "volume": 1000.0,
        }, index=idx))

    def test_direction_is_forward_looking_label(self):
        returns = self.frame["returns"]
        direction = self.frame["direction"]
        for i in range(len(self.frame) - 1):
            expected = 1.0 if returns.iloc[i + 1] > 0 else 0.0
            self.assertEqual(direction.iloc[i], expected)

    def test_final_direction_is_nan_because_next_return_is_unknown(self):
        self.assertTrue(np.isnan(self.frame["direction"].iloc[-1]))


if __name__ == '__main__':
    unittest.main()
