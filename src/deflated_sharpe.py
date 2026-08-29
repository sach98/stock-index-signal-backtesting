#!/usr/bin/env python3
"""Deflated Sharpe ratio: what survives having tried 63 things.

The results table reports 63 strategy-index Sharpe ratios and 13 of them beat
buy-and-hold. Some of that is the expected yield of searching 63 times. This
module asks which, if any, survives the search.

Two corrections, both from Bailey and Lopez de Prado (2014), and the second is
the one usually skipped:

1. SELECTION BIAS. Under a null where no strategy has skill, the maximum Sharpe
   across N trials is still positive and grows with N. The expected maximum is

       SR* = sqrt(V) * [ (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) ]

   with g the Euler-Mascheroni constant, V the variance of the Sharpe ratios
   across trials and Z the inverse normal CDF. That is the bar a real strategy
   has to clear, not zero.

2. NON-NORMALITY. Daily strategy returns are skewed and fat tailed, which makes
   a Sharpe ratio less trustworthy than its size suggests. The probabilistic
   Sharpe ratio corrects for it:

       PSR(SR*) = Phi[ (SR - SR*) * sqrt(T - 1)
                       / sqrt(1 - g3*SR + ((g4 - 1)/4)*SR^2) ]

   with g3 the skewness, g4 the NON-excess kurtosis and T the sample length.
   DSR is PSR evaluated at the SR* above. Reporting only correction 1 and
   calling it a deflated Sharpe is a common and material overstatement.

## Units, which is the easy way to get this wrong

`backtest.sharpe_ratio` returns an ANNUALISED figure. Both formulae above are in
per-period units, so everything here de-annualises by sqrt(252) first. Feeding an
annualised Sharpe into the PSR expression inflates it enormously and would make
every strategy look significant.

## What this covers, and what it does not

N is 63, every trial that was run and compared, because the selection bias comes
from the size of the search and not from how much of it is reproducible here.

Per-strategy DSR is computed only for the 35 indicator strategies, whose return
series this module can regenerate from cached prices. The 21 machine-learning and
7 combined-signal rows come from notebooks and their return series are not
committed, so their skewness, kurtosis and sample length are unavailable. They
count toward N and they do not get a DSR. That gap is reported rather than
quietly dropped.

Trials here are correlated: the same seven indices appear under nine strategies.
Correlation reduces the EFFECTIVE number of independent trials, which lowers the
bar, so using N = 63 is the conservative direction. The realised average pairwise
correlation and an effective-N estimate are both reported so the size of that
conservatism is visible rather than assumed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import apply_backtest
from .config import RESULTS_DIR
from .run_indicator_backtests import load_cached_data
from .signals import indicator_signals

TRADING_DAYS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe across `n_trials` under the null.

    This is the bar, and it is not zero. With no skill anywhere, searching more
    widely still produces a larger best result.
    """
    if n_trials < 2:
        raise ValueError("the expected maximum is undefined for fewer than 2 trials")
    if sharpe_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - g) * z1 + g * z2)


def probabilistic_sharpe(observed_sharpe: float, benchmark_sharpe: float,
                         n_obs: int, skew: float, kurtosis: float) -> float:
    """P(true Sharpe > benchmark), corrected for skew and fat tails.

    All Sharpe arguments are PER PERIOD. `kurtosis` is non-excess, so a normal
    distribution is 3 and not 0.
    """
    if n_obs < 2:
        return float("nan")
    denom_sq = (1.0
                - skew * observed_sharpe
                + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2)
    if denom_sq <= 0:
        # The correction is only defined where the estimator's variance is
        # positive. Returning nan says "undefined here" rather than inventing a
        # probability.
        return float("nan")
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deannualise(annual_sharpe: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return annual_sharpe / math.sqrt(periods_per_year)


def strategy_return_series() -> dict[tuple[str, str], pd.Series]:
    """Net return series per (index, strategy) for the reproducible trials."""
    series = {}
    for index_name, frame in load_cached_data().items():
        for strategy_name, signal in indicator_signals(frame).items():
            result = apply_backtest(frame, signal)
            series[(index_name, strategy_name)] = result["net_strategy_return"].dropna()
    return series


def effective_trials(series: dict[tuple[str, str], pd.Series], n_trials: int) -> dict:
    """Estimate independent trials from the realised correlation between them.

    N_eff = N / (1 + (N - 1) * rho_bar), the standard adjustment for equicorrelated
    tests. It is an approximation and it is reported beside the raw N rather than
    replacing it, because the raw N is the conservative choice.
    """
    frame = pd.DataFrame({f"{i}|{s}": v for (i, s), v in series.items()}).dropna()
    corr = frame.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    rho_bar = float(np.mean(off_diagonal))
    denom = 1.0 + (n_trials - 1) * rho_bar
    n_eff = n_trials / denom if denom > 0 else float(n_trials)
    return {
        "mean_pairwise_correlation": round(rho_bar, 4),
        "effective_trials": round(float(n_eff), 2),
        "pairs_measured": int(len(off_diagonal) // 2),
    }


def run(results_path: Path | None = None) -> dict:
    results_path = results_path or Path(RESULTS_DIR) / "results_table.csv"
    table = pd.read_csv(results_path)
    n_trials = len(table)

    # Variance of the per-period Sharpe across every trial that was compared.
    daily_sharpes = table["Sharpe"].astype(float).map(deannualise)
    sharpe_variance = float(np.var(daily_sharpes.to_numpy(), ddof=1))
    threshold = expected_max_sharpe(sharpe_variance, n_trials)

    series = strategy_return_series()
    correlation = effective_trials(series, n_trials)
    rows = []
    for (index_name, strategy_name), returns in sorted(series.items()):
        arr = returns.to_numpy()
        sd = arr.std(ddof=0)
        observed = float(arr.mean() / sd) if sd > 0 else float("nan")
        dsr = probabilistic_sharpe(
            observed, threshold, len(arr),
            float(stats.skew(arr)), float(stats.kurtosis(arr, fisher=False)),
        )
        rows.append({
            "index": index_name,
            "strategy": strategy_name,
            "sharpe_annual": round(observed * math.sqrt(TRADING_DAYS_PER_YEAR), 4),
            "sharpe_daily": round(observed, 6),
            "n_obs": len(arr),
            "skew": round(float(stats.skew(arr)), 4),
            "kurtosis": round(float(stats.kurtosis(arr, fisher=False)), 4),
            "deflated_sharpe": round(dsr, 4) if dsr == dsr else None,
            "survives_at_95pct": bool(dsr == dsr and dsr > 0.95),
        })

    evaluated = [r for r in rows if r["deflated_sharpe"] is not None]

    # Sensitivity to the trial count. Correlated trials mean the effective number
    # of independent tests is far below 63, which LOWERS the bar. Reporting the
    # result only at the conservative N would invite the objection that the
    # conclusion is an artefact of over-correcting, so it is recomputed at the
    # generous end too. A conclusion that holds at both ends is worth more than
    # one that holds at the end that flatters it.
    n_eff = max(2, int(round(correlation["effective_trials"])))
    lenient_threshold = expected_max_sharpe(sharpe_variance, n_eff)
    lenient_survivors = 0
    for r in evaluated:
        d = probabilistic_sharpe(r["sharpe_daily"], lenient_threshold,
                                 r["n_obs"], r["skew"], r["kurtosis"])
        if d == d and d > 0.95:
            lenient_survivors += 1

    return {
        "trials_compared": n_trials,
        "trials_with_a_reproducible_return_series": len(rows),
        "trials_without_one": n_trials - len(rows),
        "sharpe_variance_per_period": round(sharpe_variance, 8),
        "expected_max_sharpe_per_period": round(threshold, 6),
        "expected_max_sharpe_annualised": round(threshold * math.sqrt(TRADING_DAYS_PER_YEAR), 4),
        "correlation": correlation,
        "survivors_at_95pct": sum(r["survives_at_95pct"] for r in evaluated),
        "sensitivity_at_effective_n": {
            "n_used": n_eff,
            "expected_max_sharpe_annualised": round(
                lenient_threshold * math.sqrt(TRADING_DAYS_PER_YEAR), 4),
            "survivors_at_95pct": lenient_survivors,
        },
        "evaluated": len(evaluated),
        "strategies": rows,
    }


def main() -> int:
    report = run()
    out_dir = Path(RESULTS_DIR)
    (out_dir / "deflated_sharpe.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(report["strategies"]).to_csv(out_dir / "deflated_sharpe.csv", index=False)

    c = report["correlation"]
    print(f"trials compared                : {report['trials_compared']}")
    print(f"  with a reproducible series   : {report['trials_with_a_reproducible_return_series']}")
    print(f"  without one (notebooks)      : {report['trials_without_one']}")
    print(f"mean pairwise correlation      : {c['mean_pairwise_correlation']}")
    print(f"effective independent trials   : {c['effective_trials']}")
    print(f"expected max Sharpe under null : {report['expected_max_sharpe_annualised']} annualised")
    sens = report["sensitivity_at_effective_n"]
    print(f"survivors at DSR > 0.95        : {report['survivors_at_95pct']} of {report['evaluated']}")
    print(f"  same, at effective N={sens['n_used']:<9}: {sens['survivors_at_95pct']} of {report['evaluated']} "
          f"(bar {sens['expected_max_sharpe_annualised']} annualised)")
    best = max((r for r in report["strategies"] if r["deflated_sharpe"] is not None),
               key=lambda r: r["deflated_sharpe"], default=None)
    if best:
        print(f"best                           : {best['strategy']} on {best['index']}, "
              f"annual Sharpe {best['sharpe_annual']}, DSR {best['deflated_sharpe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
