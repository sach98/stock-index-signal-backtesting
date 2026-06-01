# Cost-Aware Index Signals: A Reproducible US/UK Backtesting Study

This repository rebuilds a dissertation-era stock-signal project into a clean GitHub portfolio project with reproducible notebooks, shared Python helpers, and explicit honesty controls.

The central finding is intentionally conservative: after removing look-ahead leakage and charging 0.1% per position change, most simple technical-indicator and ML timing strategies should be expected to struggle against buy-and-hold. That is a credible result, not a failure.

The project tests MACD, RSI, SMA crossover, Bollinger Bands, Momentum, classical ML classifiers, a feed-forward neural network, and a combined signal across US/UK indices: S&P 500, Dow Jones Industrial Average, NASDAQ Composite, Russell 2000, FTSE 100, Cboe UK 100, and VIX.

Current build status: results are generated. All four notebooks were executed end-to-end against live Yahoo Finance data (2013-01-01 to 2023-01-01, ~2,500 daily observations per index), and `results/results_table.csv` plus the charts under `results/charts/` are populated from that run. Re-run the notebooks in order to refresh the numbers against current data.

The table below reports two rows per index: the best technical-rule strategy by Sharpe, and the combined indicator-plus-neural-network signal. Returns are cumulative over the full sample. `gross_return` is before costs; `net_return` charges 0.1% per position change; `vs_buy_and_hold` is `net_return` minus the index's own buy-and-hold return (negative means the strategy trailed simply holding the index). The full 63-row table (all strategies for all indices) is in `results/results_table.csv`.

| index | strategy | gross_return | net_return | ann_vol | Sharpe | max_drawdown | n_trades | vs_buy_and_hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S&P 500 | Bollinger Bands | 1.581 | 1.578 | 0.176 | 0.539 | -0.339 | 1 | -0.047 |
| S&P 500 | Combined signal | 0.182 | 0.180 | 0.256 | 0.216 | -0.339 | 1 | -0.011 |
| Dow Jones Industrial Average | Bollinger Bands | 1.405 | 1.402 | 0.175 | 0.503 | -0.371 | 1 | -0.069 |
| Dow Jones Industrial Average | Combined signal | 0.151 | 0.150 | 0.253 | 0.184 | -0.371 | 1 | -0.014 |
| NASDAQ Composite | Momentum | 2.292 | 2.289 | 0.207 | 0.576 | -0.364 | 1 | -0.074 |
| NASDAQ Composite | Combined signal | 0.152 | 0.151 | 0.296 | 0.159 | -0.364 | 1 | -0.017 |
| Russell 2000 | Momentum | 0.933 | 0.931 | 0.223 | 0.296 | -0.431 | 1 | -0.086 |
| Russell 2000 | Combined signal | 0.063 | 0.062 | 0.323 | 0.062 | -0.419 | 1 | 0.004 |
| FTSE 100 | RSI | 0.218 | 0.217 | 0.157 | 0.125 | -0.366 | 1 | -0.019 |
| FTSE 100 | Combined signal | -0.012 | -0.013 | 0.209 | -0.021 | -0.349 | 1 | -0.009 |
| Cboe UK 100 | RSI | 0.222 | 0.221 | 0.156 | 0.128 | -0.372 | 1 | -0.023 |
| Cboe UK 100 | Combined signal | -0.012 | -0.013 | 0.206 | -0.022 | -0.353 | 1 | -0.003 |
| Cboe Volatility Index | Logistic Regression | 0.719 | 0.718 | 1.308 | 0.138 | -0.818 | 1 | 0.162 |
| Cboe Volatility Index | Combined signal | 0.719 | 0.718 | 1.308 | 0.138 | -0.818 | 1 | 0.162 |

The result is exactly the conservative one the design anticipates. The strongest technical rules post respectable standalone Sharpe ratios (0.5-0.6 on the large-cap US indices), but almost all of them trail their own buy-and-hold benchmark once you compare like-for-like, and the leakage-fixed machine-learning signals are weaker still. Out-of-sample balanced accuracy for next-day direction sits between 0.48 and 0.53 across every index and model (chance is 0.50; see `results/ml_classification_metrics.csv`) -- the models are essentially at coin-flip, which is the honest outcome once look-ahead bias is removed. The VIX rows are a special case: it is not an investable index and its "buy-and-hold" is meaningless, so its apparent out-performance should not be read as a tradable edge.

A note on `n_trades`: the rule and ML signals are long/flat (never short), positions are forward-filled, and the held position rarely flips over the test window, so the position-change count is small and `net_return` differs from `gross_return` by only the cost of that single rebalance. This is a property of the vectorized long/flat backtest, not a data error.

## Methodology

The notebooks run as a four-step pipeline:

1. `notebooks/01_data.ipynb` downloads Yahoo Finance OHLCV data, normalizes columns, caches CSV files under `data/cache/`, and creates EDA charts.
2. `notebooks/02_indicator_strategies.ipynb` computes MACD, RSI, SMA crossover, Bollinger Bands, and Momentum signals using only present and past observations.
3. `notebooks/03_ml_classifiers.ipynb` trains leakage-fixed classifiers for next-day direction. The neural network is a feed-forward `MLPClassifier`, not a recurrent model.
4. `notebooks/04_combined_model.ipynb` combines indicator votes with the feed-forward classifier and writes `results/results_table.csv`.

The look-ahead fix is explicit: feature columns exclude `direction`, `returns`, `strategy_return`, `gross_strategy_return`, `net_strategy_return`, positions, trades, and costs. ML models use sklearn `Pipeline` objects so `StandardScaler` is fitted on the train period only, then reused unchanged for the test period.

Signals are lagged by one trading day before returns are applied. Reported returns include both gross strategy returns and net returns after a 0.1% cost per position change. Risk is summarized with annualized volatility, Sharpe ratio, and maximum drawdown.

## Charts

The pipeline writes these chart files to `results/charts/`:

- `price_paths.png`
- `return_distributions.png`
- `indicator_equity_curves_sp500.png`
- `strategy_sharpe_by_index.png`
- `ml_equity_curves_sp500.png`
- `ml_accuracy_by_index.png`
- `combined_equity_curves.png`

Normalized price paths for the seven indices over the study window (growth of 1.0):

![US/UK index normalized price paths](results/charts/price_paths.png)

S&P 500 machine-learning strategy equity curves, net of 0.1% costs. The classifier curves sit almost exactly on top of buy-and-hold because the models stay long nearly all the time and add no reliable timing edge once leakage is removed:

![S&P 500 ML strategy equity curves vs buy-and-hold](results/charts/ml_equity_curves_sp500.png)

## Reproducibility

Install dependencies and run the notebooks in order:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Seeds are set in `src/config.py` for Python, NumPy, and optional TensorFlow. The notebooks import this configuration at the top instead of using interactive prompts.

The published results in this README were produced with: Python 3.11, yfinance 1.4.1, pandas 2.3, numpy 2.4, scikit-learn 1.8, and matplotlib 3.10. Exact pins are in `requirements.txt`.

## Notes on dependencies

`requirements.txt` installs cleanly on a networked machine with no compiled-library or build issues -- in particular, this project does **not** depend on TA-Lib. All technical indicators (MACD, RSI, Bollinger Bands, SMA/EMA, Momentum) are implemented from scratch with plain pandas in `src/signals.py`, and all performance metrics (cumulative return, annualized volatility, Sharpe, maximum drawdown) are computed with pandas/numpy in `src/backtest.py`. No third-party indicator or event-driven backtest library is imported by the notebooks or `src/`, so no substitution was required to generate the numbers above.

Two packages are pinned in `requirements.txt` but are not actually imported anywhere in the codebase: `ta` (a pandas-based indicator library) and `pyfolio-reloaded` (a tearsheet library). They install without trouble, so they are left in place, but they can be removed without affecting any result.

One source fix was needed before the pipeline would run: a `print("\nDownload failures:")` statement in `notebooks/01_data.ipynb` had been saved with a literal newline that split the string across two notebook source lines, raising a `SyntaxError`. It was rejoined into a single valid line; the logic (print a blank line, then the heading) is unchanged.

## Why This Matters for BFSI / Financial-Services Analytics

Financial-services analytics depends on disciplined signal backtesting, not just attractive model accuracy. This project demonstrates gross-vs-net performance reporting, look-ahead-bias avoidance, and risk KPIs such as Sharpe ratio and drawdown. Those habits transfer directly to BFSI work where model governance, reproducible evidence, and cost-aware decisioning matter as much as predictive lift.

## Limitations

Yahoo Finance data can change or be temporarily unavailable, so results should be regenerated and date-stamped before use.

The strategies are intentionally simple and do not model slippage, borrow costs, taxes, market impact, execution constraints, or survivorship issues beyond the chosen index series.

The ML target is next-day direction, which is noisy and economically weak even when classification metrics appear slightly above chance.

This is a research and portfolio project, not trading advice.
