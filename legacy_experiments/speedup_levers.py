"""
speedup_levers.py  -  What Can Actually Speed Up $10 -> $10k?
==============================================================
4 orthogonal levers (independent of leverage):

  1. FREQUENCY   - Add more assets (5 -> 10 -> 20 assets)
  2. WIN RATE    - Upgrade model: LogReg -> XGBoost/ensemble
  3. KELLY RISK  - Use closer-to-optimal risk sizing (math shows Kelly > 50%)
  4. COMPOUNDING - Per-trade compounding vs batched (already done)

Current baseline (verified, all biases fixed):
  WR=64.25%  Win=+10.31%  Loss=-13.80%  110 trades/month  25x leverage

Frequency note:
  5 assets gives 1.30x multiplier on BTC signals.
  Each additional similar asset adds ~0.20-0.25x (diminishing returns due to correlation).
  Estimated with 10 assets: ~1.80x  |  With 20 assets: ~2.80x
"""

import numpy as np, time

# ── BASELINE PARAMS ──────────────────────────────────────────────────────────
WIN_RATE     = 0.6425
AVG_WIN      = 0.1031
AVG_LOSS     = -0.1380
BASE_TPM     = 110       # trades/month at 5 assets, 25x leverage
LEV          = 25        # baseline leverage
RUNS         = 50_000
START        = 10.0
TARGET       = 10_000.0
RUIN         = 1.0

def kelly_fraction(wr, w, l):
    """
    Kelly criterion: optimal risk fraction for geometric growth.
    f* = (WR/|L| - (1-WR)/W) / (1/|L| + 1/W) ... simplified form.
    But since our losses are capped (not ruin-on-single-loss), the Kelly
    formula for fixed-fraction betting is:
        f* = (WR * W - (1-WR) * |L|) / (W * |L|)
    """
    return (wr * w - (1-wr) * abs(l)) / (w * abs(l))

def run(wr, w, l, tpm, r1, r2, thresh):
    bals   = np.full(RUNS, START)
    locked = np.zeros(RUNS, dtype=bool)
    trades = np.zeros(RUNS, dtype=int)
    active = np.ones(RUNS,  dtype=bool)
    ok     = np.zeros(RUNS, dtype=bool)
    ruin   = np.zeros(RUNS, dtype=bool)

    for _ in range(100_000):
        if not active.any(): break
        locked |= (bals >= thresh)
        risk = np.where(locked, r2, r1)
        wins = np.random.rand(RUNS) < wr
        ret  = np.where(wins, w, l)
        bals   = np.where(active, bals + bals * risk * ret, bals)
        trades = np.where(active, trades + 1, trades)
        nr = active & (bals <= RUIN)
        ns = active & (bals >= TARGET)
        ruin[nr]=True; ok[ns]=True
        active &= ~nr & ~ns

    s   = ok.sum()/RUNS*100
    r   = ruin.sum()/RUNS*100
    tok = trades[ok]
    am  = tok.mean()/tpm    if tok.size else 999
    mm  = np.median(tok)/tpm if tok.size else 999
    return s, r, am, mm

if __name__ == "__main__":
    np.random.seed(42)
    t0 = time.time()

    kf = kelly_fraction(WIN_RATE, AVG_WIN, abs(AVG_LOSS))
    print(f"Kelly fraction at baseline params: {kf*100:.1f}%")
    print(f"(i.e., Kelly says risk {kf*100:.1f}% of balance per trade as margin)")
    print(f"Current: 50%. Kelly-optimal: {min(kf, 1.0)*100:.0f}%\n")

    ev = WIN_RATE * AVG_WIN + (1 - WIN_RATE) * AVG_LOSS
    print(f"Base EV/trade: {ev*100:+.2f}%  at 50% risk = {ev*0.5*100:+.2f}%/trade on account\n")

    # ── LEVER 1: Frequency (more assets) ─────────────────────────────────────
    print("=" * 64)
    print("LEVER 1: FREQUENCY — Add more tradeable assets")
    print(f"         (BTC signals: 84.8/mo, 1.30x = 110/mo at 5 assets)")
    print("=" * 64)
    print(f"{'Assets':>8} {'Trades/mo':>10} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9}")
    asset_configs = [
        (5,  110),
        (8,  158),   # 84.8 * 1.86x estimate
        (10, 195),   # 84.8 * 2.30x estimate
        (15, 270),   # 84.8 * 3.18x estimate
        (20, 339),   # 84.8 * 4.00x estimate
    ]
    for na, tpm in asset_configs:
        s, r, a, m = run(WIN_RATE, AVG_WIN, AVG_LOSS, tpm, 0.50, 0.10, 200)
        flag = " *** < 12mo" if a <= 12 else ""
        print(f"{na:>8} {tpm:>10} {s:>8.1f}% {r:>5.1f}% {a:>9.1f}{flag}")

    # ── LEVER 2: Win Rate (better model) ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("LEVER 2: WIN RATE — Upgrade LR -> XGBoost/Ensemble")
    print(f"         (Current WR: {WIN_RATE*100:.2f}%  |  110 trades/month)")
    print("=" * 64)
    print(f"{'WR':>8} {'EV/trade':>10} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9}")
    for wr in [0.6425, 0.66, 0.68, 0.70, 0.72]:
        ev_t = wr * AVG_WIN + (1-wr) * AVG_LOSS
        s, r, a, m = run(wr, AVG_WIN, AVG_LOSS, BASE_TPM, 0.50, 0.10, 200)
        flag = " *** < 12mo" if a <= 12 else ""
        print(f"{wr*100:>7.2f}% {ev_t*100:>+9.2f}% {s:>8.1f}% {r:>5.1f}% {a:>9.1f}{flag}")

    # ── LEVER 3: Risk Sizing (Kelly-optimal) ─────────────────────────────────
    print("\n" + "=" * 64)
    print("LEVER 3: RISK SIZING — Kelly-optimal vs conservative")
    print(f"         (Kelly says {kf*100:.1f}%  |  110 trades/month)")
    print("=" * 64)
    print(f"{'Stage1 Risk':>12} {'Stage2 Risk':>12} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9}")
    risk_configs = [
        (0.30, 0.05),
        (0.50, 0.10),   # current
        (0.70, 0.10),
        (0.85, 0.15),
        (0.99, 0.20),   # near-Kelly for stage 1
        (1.19, 0.20),   # full Kelly (theoretical, can't exceed 1.0 in practice)
    ]
    for r1, r2 in risk_configs:
        actual_r1 = min(r1, 1.0)
        s, r, a, m = run(WIN_RATE, AVG_WIN, AVG_LOSS, BASE_TPM, actual_r1, r2, 200)
        label = "  (current)" if r1==0.50 else ("  (Kelly-cap)" if r1 >= 1.0 else "")
        print(f"{actual_r1*100:>11.0f}% {r2*100:>11.0f}%  {s:>8.1f}% {r:>5.1f}% {a:>9.1f}{label}")

    # ── LEVER 4: ALL THREE COMBINED ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print("LEVER 4: COMBINED — Best realistic combination")
    print("=" * 64)
    combos = [
        # label,         wr,    w,    l,        tpm, r1,   r2,  thresh
        ("Baseline (current)",  0.6425,0.1031,-0.1380, 110, 0.50, 0.10, 200),
        ("+ 10 assets",         0.6425,0.1031,-0.1380, 195, 0.50, 0.10, 200),
        ("+ 15 assets",         0.6425,0.1031,-0.1380, 270, 0.50, 0.10, 200),
        ("+ XGBoost WR 68%",    0.6800,0.1031,-0.1380, 110, 0.50, 0.10, 200),
        ("+ XGBoost WR 70%",    0.7000,0.1031,-0.1380, 110, 0.50, 0.10, 200),
        ("10 assets + WR 68%",  0.6800,0.1031,-0.1380, 195, 0.50, 0.10, 200),
        ("15 assets + WR 68%",  0.6800,0.1031,-0.1380, 270, 0.50, 0.10, 200),
        ("15 assets + WR 70%",  0.7000,0.1031,-0.1380, 270, 0.50, 0.10, 200),
        ("20 assets + WR 70%",  0.7000,0.1031,-0.1380, 339, 0.50, 0.10, 200),
    ]
    print(f"{'Scenario':<28} {'Trades/mo':>10} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9}")
    print("-" * 64)
    for label,wr,w,l,tpm,r1,r2,thresh in combos:
        s,r,a,m = run(wr,w,l,tpm,r1,r2,thresh)
        flag = " *** UNDER 12mo!" if a<=12 else (" ** < 18mo" if a<=18 else "")
        print(f"{label:<28} {tpm:>10} {s:>8.1f}% {r:>5.1f}% {a:>9.1f}{flag}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
