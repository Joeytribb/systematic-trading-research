# Crypto Options & Futures Compounding Project Archive

This archive contains all codebase files, research artifacts, and simulation scripts developed during the crypto phase of the $10 to $10,000 compounding challenge.

## Directory Structure

*   `code/`: All primary execution models, walk-forward scanners, leverage sweepers, and simulation engines.
*   `artifacts/`: Markdown research logs detailing biases, pipeline audits, implementation plans, and final results.
*   `scratch/`: Early explorative and temporary Monte Carlo / testing scripts.

## Key Files & Purpose

### Scanners & Optimizers (`code/`)
*   `win_rate_optimizer.py`: Short scanner walk-forward validator (Logistic Regression, 2yr BTC data).
*   `long_rate_optimizer.py`: Long scanner walk-forward validator (Logistic Regression, 2yr BTC data).
*   `combined_scanner.py`: Simulates simultaneous Short + Long trading with a single shared concurrency lock to measure real non-overlapping joint trade frequency.
*   `multi_asset_scanner.py`: Multi-asset scan engine with same-candle worst-case fill logic.
*   `leverage_sweep.py`: Sweeps leverage levels from 25x to 125x for a $10 starting account, modeling gap liquidation risk.
*   `speedup_levers.py`: sweeps 4 independent levers (frequency, model win rate, Kelly risk sizing) to accelerate growth.

### Final Simulations & Synthesis (`code/` & `artifacts/`)
*   `simulate_500_final.py`: Zero-bias final Monte Carlo simulation for $500 starting capital with $50 ruin threshold.
*   `phd_project_synthesis.md` / `artifacts/phd_project_synthesis.md`: Complete academic-level project report prepared for a potential PhD advisor.

### Brain Research Logs (`artifacts/`)
*   `bias_audit.md` & `bias_audit_pass2.md`: Detailed logs documenting all 15 biases and 2 look-ahead bugs identified in the original pipeline.
*   `walkthrough.md`: A summary of the audit findings, fixes, and final calibrated parameters.

## Summary of Findings & Calibrated Ground Truths

After correcting for all 15 backtest biases and both look-ahead bugs, the verified parameters for the combined strategy on a 2-year BTC dataset are:
*   **Win Rate:** 64.25%
*   **Average Win:** +10.31% (on 25x margin, net of fees and funding)
*   **Average Loss:** -13.80% (on 25x margin, net of fees and funding)
*   **Expected Value (EV):** +1.69% per trade
*   **Combined Frequency:** 110 trades / month (multi-asset scaled)
*   **Ruin Risk ($500 Start, $50 Ruin):** 0.02% using a Two-Stage (50% -> 20% risk ratchet at $2,000).
*   **Median Time to $10k:** 6.3 months.
*   **Slippage Capacity Cap:** ~$541,522 (at which point order book slippage eliminates trading edge).
