# Business Requirements Document (BRD)
## Quantitative Signal Backtesting & Asset-Liability Risk Model

**Document Control**
- **Author:** Sachin Sharma (Lead Business Analyst)
- **Status:** Approved / Baseline
- **Target Audience:** Investment Risk Committee, ALM Actuarial Team, Model Governance Officers

---

## 1. Business Objective & Context
In Life & Pensions and Asset-Liability Management (ALM), insurers hold multi-billion pound investment portfolios backing long-term policyholder liabilities. Trading strategies or signal models evaluated for capital deployment must withstand rigorous stress testing, realistic friction costs, and strict model risk controls.

The purpose of this project is to specify and test quantitative signal timing models across 7 key equity and volatility indices (S&P 500, DJIA, NASDAQ, Russell 2000, FTSE 100, Cboe UK 100, VIX) to evaluate whether technical indicator and ML-based timing strategies offer true alpha over buy-and-hold benchmarks after accounting for transaction costs and eliminating look-ahead bias.

---

## 2. Business Requirements & Functional Specifications

### FR-01: Data Ingestion & Data Hygiene
- **Data Range:** 10-year historical daily OHLCV equity index data (2013-01-01 to 2023-01-01, ~2,500 trading days per index).
- **Integrity Rule:** Zero forward-looking leakage. All technical indicators (RSI, Moving Averages, Bollinger Bands, Momentum) must be calculated using strictly past information up to day $t-1$ when predicting day $t$.

### FR-02: Signal Strategy Logic & Controls
- **Technical Indicator Rules:**
  - *Bollinger Bands:* Long when price closes below lower band; Exit/Neutral when closing above middle band.
  - *RSI:* Long when RSI < 30 (oversold); Exit when RSI > 70.
  - *SMA Crossover:* Long when 20-day SMA > 50-day SMA; Short/Neutral otherwise.
- **Machine Learning Models:** Logistic Regression, Random Forest, Feed-Forward Neural Network trained on sliding window train/test splits.
- **Combined Signal Rule:** Ensemble voting requiring consensus across indicator signals and ML classification predictions.

### FR-03: Friction Cost & Realistic Execution Constraints
- **Transaction Cost Rule:** Charge a flat **0.10% (10 bps) fee** per position change (buy/sell/rebalance) to mirror real institutional broker execution fees and bid-ask spreads.
- **Net vs. Gross Return Rule:** Report both Gross Return and Net Return ($Net = Gross - \sum Transaction Costs$).

### FR-04: Model Performance Metrics & Governance Reporting
- **Risk Metrics Required:**
  - Cumulative Net Return
  - Annualized Volatility ($\sigma_{ann}$)
  - Sharpe Ratio ($S = \frac{R_p - R_f}{\sigma_p}$) using 0% risk-free baseline.
  - Maximum Drawdown (peak-to-trough decline).
  - Benchmark Comparative Alpha ($vs\_buy\_and\_hold = Net\_Return - Benchmark\_Return$).

---

## 3. Governance & Model Risk Compliance

- **Model Risk Assessment (SR 11-7 / PRA SS1/23 Standard):**
  - Out-of-sample balanced classification accuracy for next-day direction across all indices sits between 0.48 and 0.53 (effectively coin-flip).
  - *Governance Conclusion:* The baseline ML signals fail to demonstrate statistically significant predictive edge after cost friction. Recommendation is **NOT to deploy active signal timing for ALM capital allocation**; passive buy-and-hold or index hedging remains the compliant baseline.
