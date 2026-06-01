"""
nifty_monte_carlo.py  -  Indian Market Compounding Simulation (INR 50,000 -> INR 10 Lakhs)
========================================================================================
Runs 50,000 Monte Carlo paths using the calibrated out-of-sample parameters from Nifty 50:
  - Win Rate: 70.49%
  - Average Win PnL: +0.98% (on margin, 8x leverage, net of fees)
  - Average Loss PnL: -1.25% (on margin, 8x leverage, net of fees)
  - Frequency: 41 trades / month
  - Ruin Threshold: INR 5,000 (10% of starting capital)
"""

import numpy as np
import time

WIN_RATE         = 0.7049
AVG_WIN_PNL      = 0.0098   # +0.98%
AVG_LOSS_PNL     = -0.0125  # -1.25%
TRADES_PER_MONTH = 41

RUNS   = 50_000
START  = 50000.0
TARGET = 1000000.0  # 10 Lakhs
RUIN   = 5000.0     # 90% drawdown

def run_mc(r1, r2, thresh):
    bals   = np.full(RUNS, START)
    locked = np.zeros(RUNS, dtype=bool)
    trades = np.zeros(RUNS, dtype=int)
    active = np.ones(RUNS,  dtype=bool)
    ok     = np.zeros(RUNS, dtype=bool)
    ruin   = np.zeros(RUNS, dtype=bool)

    # Max 100k steps
    for _ in range(100_000):
        if not active.any(): break
        locked |= (bals >= thresh)
        risk    = np.where(locked, r2, r1)
        wins    = np.random.rand(RUNS) < WIN_RATE
        ret     = np.where(wins, AVG_WIN_PNL, AVG_LOSS_PNL)
        bals    = np.where(active, bals + bals * risk * ret, bals)
        trades  = np.where(active, trades + 1, trades)
        nr      = active & (bals <= RUIN)
        ns      = active & (bals >= TARGET)
        ruin[nr] = True; ok[ns] = True
        active  &= ~nr & ~ns

    s  = ok.sum()/RUNS*100
    r  = ruin.sum()/RUNS*100
    tok = trades[ok]
    am = tok.mean()/TRADES_PER_MONTH   if tok.size else 999
    mm = np.median(tok)/TRADES_PER_MONTH if tok.size else 999
    return s, r, am, mm

if __name__ == "__main__":
    np.random.seed(42)
    t0 = time.time()

    print("=" * 70)
    print("  NIFTY 50 MONTE CARLO SIMULATION (INR 50,000 -> INR 1,000,000)")
    print(f"  WR: {WIN_RATE*100:.2f}%  |  Win: {AVG_WIN_PNL*100:+.2f}%  |  Loss: {AVG_LOSS_PNL*100:+.2f}%")
    print(f"  Trades/Month: {TRADES_PER_MONTH}  |  Ruin threshold: INR {RUIN:,.0f}  |  {RUNS:,} runs")
    print("=" * 70)

    configs = [
        ("10% flat risk (conservative)",            0.10, 0.10, 9999999),
        ("20% flat risk",                           0.20, 0.20, 9999999),
        ("30% flat risk",                           0.30, 0.30, 9999999),
        ("50% flat risk",                           0.50, 0.50, 9999999),
        ("Two-Stage: 50%->10% ratchet at 2 Lakhs",  0.50, 0.10, 200000),
        ("Two-Stage: 50%->20% ratchet at 2 Lakhs",  0.50, 0.20, 200000),
        ("Two-Stage: 30%->10% ratchet at 2 Lakhs",  0.30, 0.10, 200000),
    ]

    print(f"\n{'Scenario':<42} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9} {'Med(mo)':>9}")
    print("-" * 72)
    for label, r1, r2, thresh in configs:
        s, r, a, m = run_mc(r1, r2, thresh)
        print(f"{label:<42} {s:>8.1f}% {r:>5.1f}% {a:>9.1f} {m:>9.1f}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
