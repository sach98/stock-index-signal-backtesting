"""Tests for the deflated Sharpe implementation.

The formulae are easy to write down and easy to get subtly wrong, in ways that
all point the same direction: towards a strategy looking more significant than it
is. These check the properties that would catch each mistake.

The committed report is checked separately, so a rerun that moves the conclusion
fails here rather than silently changing what the README claims.
"""

from __future__ import annotations

import json
import math
import os
import unittest

import numpy as np
from scipy import stats

from src.deflated_sharpe import (
    EULER_MASCHERONI,
    TRADING_DAYS_PER_YEAR,
    deannualise,
    expected_max_sharpe,
    probabilistic_sharpe,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(BASE_DIR, "results", "deflated_sharpe.json")


class ExpectedMaxSharpeTests(unittest.TestCase):

    def test_the_bar_rises_with_the_number_of_trials(self):
        """The whole point: searching harder raises what counts as impressive."""
        v = 0.0035
        bars = [expected_max_sharpe(v, n) for n in (2, 10, 63, 1000)]
        self.assertEqual(bars, sorted(bars))
        self.assertLess(bars[0], bars[-1])

    def test_the_bar_rises_with_the_spread_of_results(self):
        # More dispersed trials produce a larger maximum by chance alone.
        self.assertLess(expected_max_sharpe(0.001, 63), expected_max_sharpe(0.01, 63))

    def test_zero_variance_gives_a_zero_bar(self):
        # If every trial returned the same Sharpe there is no selection to correct.
        self.assertEqual(expected_max_sharpe(0.0, 63), 0.0)

    def test_fewer_than_two_trials_is_refused(self):
        with self.assertRaises(ValueError):
            expected_max_sharpe(0.0035, 1)

    def test_matches_a_hand_evaluation_of_the_published_formula(self):
        v, n = 0.0035, 63
        z1 = stats.norm.ppf(1 - 1 / n)
        z2 = stats.norm.ppf(1 - 1 / (n * math.e))
        expected = math.sqrt(v) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
        self.assertAlmostEqual(expected_max_sharpe(v, n), expected, places=12)


class ProbabilisticSharpeTests(unittest.TestCase):

    def test_normal_returns_reduce_to_the_textbook_expression(self):
        """With normal moments and a zero benchmark, PSR is Phi(SR*sqrt(T-1))."""
        rng = np.random.default_rng(7)
        x = rng.normal(0.0005, 0.01, 3000)
        sr = float(x.mean() / x.std(ddof=0))
        got = probabilistic_sharpe(sr, 0.0, len(x), float(stats.skew(x)),
                                   float(stats.kurtosis(x, fisher=False)))
        want = float(stats.norm.cdf(sr * math.sqrt(len(x) - 1)))
        self.assertAlmostEqual(got, want, places=2)

    def test_negative_skew_and_fat_tails_lower_the_result(self):
        """The correction that gets skipped, and the direction it must move.

        A strategy with occasional large losses has a less trustworthy Sharpe
        than a normal one with the same mean and variance. If this ever came out
        higher, the correction would be inverted and every fat-tailed strategy
        would be flattered.
        """
        rng = np.random.default_rng(11)
        y = np.concatenate([rng.normal(0.0008, 0.008, 2900),
                            rng.normal(-0.05, 0.02, 100)])
        sr = float(y.mean() / y.std(ddof=0))
        as_normal = probabilistic_sharpe(sr, 0.0, len(y), 0.0, 3.0)
        with_moments = probabilistic_sharpe(sr, 0.0, len(y), float(stats.skew(y)),
                                            float(stats.kurtosis(y, fisher=False)))
        self.assertLess(with_moments, as_normal)

    def test_a_higher_benchmark_lowers_the_result(self):
        a = probabilistic_sharpe(0.05, 0.00, 2500, 0.0, 3.0)
        b = probabilistic_sharpe(0.05, 0.04, 2500, 0.0, 3.0)
        self.assertLess(b, a)

    def test_a_longer_sample_sharpens_the_verdict(self):
        short = probabilistic_sharpe(0.05, 0.0, 100, 0.0, 3.0)
        long = probabilistic_sharpe(0.05, 0.0, 5000, 0.0, 3.0)
        self.assertLess(short, long)

    def test_an_undefined_variance_returns_nan_rather_than_a_number(self):
        # Extreme moments can drive the estimator variance non-positive. Returning
        # a probability there would be inventing one.
        self.assertTrue(math.isnan(probabilistic_sharpe(5.0, 0.0, 2500, 3.0, 3.0)))

    def test_too_few_observations_returns_nan(self):
        self.assertTrue(math.isnan(probabilistic_sharpe(0.05, 0.0, 1, 0.0, 3.0)))


class UnitsTests(unittest.TestCase):
    """The mistake that would make everything look significant."""

    def test_deannualising_undoes_the_annualisation(self):
        self.assertAlmostEqual(deannualise(0.5 * math.sqrt(TRADING_DAYS_PER_YEAR)), 0.5, places=12)

    def test_feeding_an_annualised_sharpe_would_inflate_the_result(self):
        # Not a behaviour to rely on, a demonstration of why the conversion
        # exists: the same strategy looks certain if the units are wrong.
        daily = 0.036
        honest = probabilistic_sharpe(daily, 0.0, 2500, 0.0, 3.0)
        wrong = probabilistic_sharpe(daily * math.sqrt(TRADING_DAYS_PER_YEAR), 0.0, 2500, 0.0, 3.0)
        self.assertLess(honest, wrong)


@unittest.skipUnless(os.path.exists(REPORT), "results/deflated_sharpe.json not present")
class CommittedReportTests(unittest.TestCase):
    """The published conclusion, pinned."""

    @classmethod
    def setUpClass(cls):
        with open(REPORT, encoding="utf-8") as fh:
            cls.report = json.load(fh)

    def test_the_trial_count_is_every_trial_that_was_compared(self):
        # Not just the reproducible ones: selection bias comes from the size of
        # the search, not from how much of it can be recomputed.
        r = self.report
        self.assertEqual(r["trials_compared"], 63)
        self.assertEqual(r["trials_with_a_reproducible_return_series"]
                         + r["trials_without_one"], 63)

    def test_nothing_survives_the_correction(self):
        self.assertEqual(self.report["survivors_at_95pct"], 0)

    def test_nothing_survives_even_at_the_generous_trial_count(self):
        """The robustness check that makes the conclusion worth stating.

        Correlated trials lower the bar. If the result only held at the
        conservative N it would be fair to call it an artefact of
        over-correcting, so it is recomputed at the effective N too.
        """
        sens = self.report["sensitivity_at_effective_n"]
        self.assertEqual(sens["survivors_at_95pct"], 0)
        self.assertLess(sens["expected_max_sharpe_annualised"],
                        self.report["expected_max_sharpe_annualised"])

    def test_the_trials_are_correlated_which_is_why_the_sensitivity_exists(self):
        c = self.report["correlation"]
        self.assertGreater(c["mean_pairwise_correlation"], 0.0)
        self.assertLess(c["effective_trials"], self.report["trials_compared"])


if __name__ == "__main__":
    unittest.main()
