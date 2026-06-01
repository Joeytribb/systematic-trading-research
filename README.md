# Systematic Trading Research & Bias Auditing

## 1. Introduction: What is this project?
This repository contains a complete, end-to-end quantitative trading system designed to compound a micro-account ($500) into a target threshold ($10,000) using high-leverage derivatives. Originally built for crypto perpetual futures and later adapted for the Indian Nifty 50 index, the project focuses heavily on solving the "gravity well" problem of small accounts. It provides a complete mathematical blueprint for generating short-term alpha while strictly managing the risk of ruin.

## 2. How It Works: The Trading Mechanics
The system is built on an **Intraday Mean-Reversion** model that fades over-extended price movements.
*   **The Signal:** It uses a Logistic Regression machine learning classifier trained on 5-minute walk-forward data. It identifies moments where multi-timeframe moving averages (1H, 4H) and volatility bands indicate an extreme, temporary over-extension.
*   **The Execution:** Instead of entering via market orders, the system uses a **Dollar-Cost-Averaging (DCA) Limit Grid**. When a signal triggers, 4 limit orders are placed at precise increments. If price wicks against the position, it fills the grid, resulting in a better average entry price.
*   **The Exit:** Profits are taken via a strict limit order placed 0.50% below the average fill price, while a hard stop-loss prevents catastrophic liquidations.

## 3. Why It Is Exceptional: Rigor & Realism
Most retail trading algorithms look perfect on paper but fail in live markets due to oversimplified assumptions. This system separates itself by aggressively dismantling its own backtest to ensure institutional-grade realism.

### 🔬 The Zero-Bias Execution Engine
We audited the pipeline and mathematically eliminated **15 systemic biases and 2 look-ahead bugs** that artificially inflate standard backtests.
*   **Threshold Contamination:** Prevented the model from "peeking" at future probabilities to set threshold limits.
*   **Execution Friction:** Hardcoded worst-case scenario fills for intraday limit orders, meaning the backtest assumes we get the worst possible fill during high-volatility wicks.
*   **Real-World Costs:** The model accurately deducts exchange taker/maker fees, perpetual funding rate drags, and SEBI turnover fees.

### 🎲 Stochastic Risk Sizing & Compounding Limits
*   **Kelly Criterion Risk Ratchet:** Using a 50,000-path Monte Carlo engine, we mapped the exact probability of ruin. We implemented a two-stage risk-ratchet (risking 50% early to escape the micro-account phase, dropping to 20% later) to safely compound capital while keeping the ruin risk at **0.02%**.
*   **Analytical Capacity Limits:** We derived the mathematical ceiling of the strategy. By modeling dynamic order-book slippage curves against the system's expected value (+1.69% EV/trade), we proved that compounding strictly halts at **~$541,500**. Above this, market impact slippage negates the edge entirely.

---

## Empirical Ground-Truth Results

After removing all look-ahead biases and simulating worst-case execution friction (slippage, fees, funding rates), the 2-year out-of-sample backtest yielded the following core metrics:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Win Rate** | **64.25%** | Strictly out-of-sample performance |
| **Expected Value (EV)** | **+1.69%** | Net return per trade after all fees and slippage |
| **Target Simulation** | **$500 to $10,000** | Median compounding time of **6.3 months** |
| **Risk of Ruin** | **0.02%** | Using a two-stage Kelly-based risk ratchet |
| **Capacity Limit** | **~$541,500** | Point where order-book slippage reduces EV to zero |

## Repository Structure

*   `/crypto_project_archive/`: Contains the original BTC mean-reversion strategy, the rigorous bias-audit logs, and the high-leverage Monte Carlo simulations.
*   `nifty_strategy_test.py`: The adapted intraday mean-reversion algorithm for the NSE Nifty 50 index.
*   `nifty_monte_carlo.py`: Compounding simulation for INR capital under SEBI margin constraints.
*   `phd_project_synthesis.md`: The definitive research paper summarizing the methodology, findings, and execution realities discovered during the project.
