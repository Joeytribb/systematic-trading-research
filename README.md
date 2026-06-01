# Systematic Trading Research & Bias Auditing

This repository contains the quantitative research, backtesting infrastructure, and Monte Carlo simulations for an intraday mean-reversion algorithmic trading system. The project focuses heavily on **market microstructure**, **execution friction**, and **eliminating look-ahead biases** often found in retail backtests.

## Project Scope & Objective

The initial goal was to evaluate the feasibility of high-frequency compounding (e.g., scaling a hyper-small account using leveraged derivatives). Rather than accepting theoretical returns, this project systematically disassembled the backtest to account for real-world trading constraints.

The infrastructure was initially developed using highly volatile assets (Crypto/BTC Perpetual Futures) and later adapted for traditional domestic indices (NSE Nifty 50) to comply with regional regulatory frameworks and margin limits.

## Key Research Components

### 1. The Zero-Bias Execution Engine
The core of this repository is the robust bias-auditing pipeline. Initial naive backtests showed unrealistic >90% win rates with exponential compounding. We implemented a rigorous audit to destroy these illusions by identifying and fixing 15 distinct systemic biases and look-ahead bugs, most notably:

*   **Threshold Contamination [LA1]:** Prevented algorithms from knowing if a price limit was hit later in a candle before the signal was generated.
*   **Gap Execution [LA2]:** Enforced realistic fill logic, assuming the worst-case fill price when limit orders triggered intra-candle.
*   **Execution Friction:** Hardcoded exchange fees (e.g., 0.04% taker / 0.02% maker on Binance, SEBI turnover fees on NSE) and modeled order-book slippage curves for scaling capacity.
*   **Funding Rate Drag:** Modeled the exact drag of perpetual futures funding rates on highly leveraged positions.

### 2. Capital Compounding & Ruin Simulation
To test the resilience of the strategy, we built a Monte Carlo engine that simulates thousands of equity curves:
*   Modeled **Kelly Criterion** approximations to determine optimal leverage vs. risk-of-ruin.
*   Tested the physical limits of compounding, calculating the mathematical ceiling where order-book slippage eliminates the edge (identified at ~$541k for the crypto model).
*   Simulated multi-asset concurrency to model how simultaneous trading signals block capital allocation.

### 3. Market Adaptation (Crypto vs. Nifty 50)
The system architecture was abstracted to apply to different asset classes. We successfully adapted the crypto-native DCA strategy to the Indian domestic market (Nifty 50 futures), adjusting for:
*   Significantly lower volatility environments (requiring 5x tighter grid parameters).
*   Lower allowable leverage (SEBI 12% margin rules).
*   Strict Intraday limitations (hard square-off at 15:15 IST to prevent overnight gap risks).

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

## Conclusion

This project serves as a demonstration of institutional-grade quantitative research methodologies: skeptical data analysis, rigorous out-of-sample walk-forward validation, and an obsession with modeling real-world execution costs.
