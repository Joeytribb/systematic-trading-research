"""
speed_analysis.py
=================
Finds what levers (trades/month, risk %, win rate) are needed to
hit $10k from $50 in under 12 months.
"""
import numpy as np
import time

# Current calibrated params
WIN_RATE     = 0.6265
AVG_WIN_PNL  = 0.1020
AVG_LOSS_PNL = -0.1345
RUNS         = 30_000

TARGET       = 10_000.0
START        = 50.0
TARGET_MONTHS = 12.0

def monte_carlo(win_rate, avg_win, avg_loss, trades_per_month,
                stage1_risk, stage2_risk=0.10, stage2_threshold=200.0):
    bals          = np.full(RUNS, START)
    stage2_locked = np.zeros(RUNS, dtype=bool)
    trades        = np.zeros(RUNS, dtype=int)
    active        = np.ones(RUNS, dtype=bool)
    success       = np.zeros(RUNS, dtype=bool)
    ruined        = np.zeros(RUNS, dtype=bool)

    for _ in range(25_000):
        if not active.any():
            break
        stage2_locked |= (bals >= stage2_threshold)
        risk  = np.where(stage2_locked, stage2_risk, stage1_risk)
        wins  = np.random.rand(RUNS) < win_rate
        ret   = np.where(wins, avg_win, avg_loss)
        bals  = np.where(active, bals + bals * risk * ret, bals)
        trades = np.where(active, trades + 1, trades)
        nr    = active & (bals <= 1.0)
        ns    = active & (bals >= TARGET)
        ruined[nr]  = True
        success[ns] = True
        active &= ~nr & ~ns

    s_rate = success.sum() / RUNS * 100
    r_rate = ruined.sum() / RUNS * 100
    t_ok   = trades[success]
    avg_mo = t_ok.mean() / trades_per_month if t_ok.size > 0 else 999
    med_mo = np.median(t_ok) / trades_per_month if t_ok.size > 0 else 999
    return s_rate, r_rate, avg_mo, med_mo

if __name__ == "__main__":
    np.random.seed(42)
    t0 = time.time()

    # ── 1. What trades/month is needed at 50% risk? ───────────────────────────
    print("=" * 70)
    print("LEVER 1: Vary trades/month  (50% risk, WR=62.65%, current params)")
    print("=" * 70)
    print(f"{'Trades/mo':>10} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9} {'Med(mo)':>9}")
    for tpm in [20, 30, 40, 60, 80, 100, 130, 160]:
        s, r, a, m = monte_carlo(WIN_RATE, AVG_WIN_PNL, AVG_LOSS_PNL,
                                 tpm, stage1_risk=0.50)
        marker = " <-- TARGET" if a <= TARGET_MONTHS else ""
        print(f"{tpm:>10} {s:>8.1f}% {r:>5.1f}% {a:>9.1f} {m:>9.1f}{marker}")

    # ── 2. What risk % is needed at 20 trades/month? ─────────────────────────
    print("\n" + "=" * 70)
    print("LEVER 2: Vary Stage-1 risk  (20 trades/mo, WR=62.65%, current params)")
    print("=" * 70)
    print(f"{'Stage1 Risk':>12} {'Stage2 Risk':>12} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9}")
    for r1, r2 in [(0.50, 0.10), (0.60, 0.10), (0.70, 0.10),
                   (0.80, 0.10), (0.90, 0.10), (1.00, 0.10),
                   (0.80, 0.20), (1.00, 0.20)]:
        s, r, a, m = monte_carlo(WIN_RATE, AVG_WIN_PNL, AVG_LOSS_PNL,
                                 20, stage1_risk=r1, stage2_risk=r2)
        marker = " <-- TARGET" if a <= TARGET_MONTHS else ""
        print(f"{r1*100:>11.0f}% {r2*100:>11.0f}% {s:>8.1f}% {r:>5.1f}% {a:>9.1f}{marker}")

    # ── 3. Top 3% vs Top 5% thresholds ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("LEVER 3: Better threshold (from optimizer output)")
    print("=" * 70)
    # From win_rate_optimizer.py output:
    # Top 5%: 35.5 signals/mo BTC, WR=64.3%, AvgPnL=+1.72%  -> need win/loss breakdown
    # Top 3%: 22.5 signals/mo BTC, WR=64.2%, AvgPnL=+1.71%  -> similar
    # Use EV to derive win/loss:  wr*W + (1-wr)*L = avg_pnl
    # For Top 3%: 0.642*W + 0.358*L = 0.0171  -> assuming W/L ratio same as Top2%
    # W/L ratio from Top 2%: 0.1020 / 0.1345 = 0.758
    # So L = -W/0.758; 0.642*W - 0.358*W/0.758 = 0.0171
    # 0.642W - 0.4724W = 0.0171  -> 0.1696W = 0.0171 -> W = 0.1009
    # L = -0.1009/0.758 = -0.1331

    configs = [
        # (label, wr, win_pnl, loss_pnl, btc_tpm, multi_mult, r1)
        ("Top 2% (current baseline)", 0.6265, 0.1020, -0.1345, 15.5, 1.30, 0.50),
        ("Top 3% threshold",          0.6420, 0.1009, -0.1331, 22.5, 1.30, 0.50),
        ("Top 5% threshold",          0.6430, 0.1050, -0.1382, 35.5, 1.30, 0.50),
        ("Top 5% + Long strategy",    0.6430, 0.1050, -0.1382, 35.5, 2.60, 0.50),
        ("Top 3% + Long strategy",    0.6420, 0.1009, -0.1331, 22.5, 2.60, 0.50),
        ("Top 5% + Long + 70% risk",  0.6430, 0.1050, -0.1382, 35.5, 2.60, 0.70),
        ("Top 5% + Long + 80% risk",  0.6430, 0.1050, -0.1382, 35.5, 2.60, 0.80),
    ]

    print(f"{'Scenario':<32} {'Trades/mo':>10} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9}")
    print("-" * 70)
    for label, wr, wp, lp, btc_tpm, mult, r1 in configs:
        tpm = int(btc_tpm * mult)
        s, r, a, m = monte_carlo(wr, wp, lp, tpm, stage1_risk=r1)
        marker = " <<" if a <= TARGET_MONTHS else ""
        print(f"{label:<32} {tpm:>10} {s:>8.1f}% {r:>5.1f}% {a:>9.1f}{marker}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
