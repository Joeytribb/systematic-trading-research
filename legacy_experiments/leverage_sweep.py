"""
leverage_sweep.py  -  $10 Start, Maximum Leverage Analysis
===========================================================
Sweeps leverage from 25x to 125x (Binance max for BTC).

Key derivation:
  From the 2-year backtest at 25x leverage:
    Avg Win PnL  = +10.31% on margin
    Avg Loss PnL = -13.80% on margin

  Decompose into gross price moves (independent of leverage):
    fee = 0.04% per side, 2 sides per trade
    win_gross_move  = (10.31% + 2*0.04%*25) / 25 = 12.31% / 25 = 0.4924%
    loss_gross_move = (13.80% - 2*0.04%*25) / 25 = 11.80% / 25 = 0.4720%

  At leverage L:
    win_pnl(L)  = win_gross_move  * L - 2*fee*L = (0.004924 - 0.0008) * L = 0.004124 * L
    loss_pnl(L) = -(loss_gross_move * L + 2*fee*L) = -(0.004720 + 0.0008) * L = -0.005520 * L

Liquidation constraint:
  Binance BTC maintenance margin rate = 0.5%
  Liquidation at adverse price move = (1/L) * (1 - 0.005)
  SL at 0.75% adverse move

  Max safe leverage = 1 / (0.0075 + 0.005) = 1 / 0.01275 = ~78x
  (above this, liquidation can fire BEFORE SL)
  BUT: at 125x, liq at 0.8% vs SL at 0.75% - SL fires first in theory,
       but requires perfect execution with zero gap slippage.
"""

import numpy as np, time

WIN_RATE  = 0.6425
FEE       = 0.0004
WIN_PRICE_MOVE  = 0.004924   # gross from backtest at 25x
LOSS_PRICE_MOVE = 0.004720   # gross from backtest at 25x

RUNS    = 50_000
START   = 10.0
TARGET  = 10_000.0
RUIN    = 1.0     # below $1 = effectively wiped from $10 start

def run(lev, r1=0.50, r2=0.10, thresh=200.0, gap_pct=0.0):
    """
    gap_pct: fraction of LOSING trades where the SL gaps and liquidation
             fires instead (realistic at very high leverage in volatile markets).
             Liquidation = full margin lost = -100% on that trade's margin.
    """
    wp = (WIN_PRICE_MOVE  - 2*FEE) * lev    # net win on margin
    lp = -(LOSS_PRICE_MOVE + 2*FEE) * lev   # net loss on margin (normal SL)
    ev = WIN_RATE*wp + (1-WIN_RATE)*lp

    bals   = np.full(RUNS, START)
    locked = np.zeros(RUNS, dtype=bool)
    trades = np.zeros(RUNS, dtype=int)
    active = np.ones(RUNS,  dtype=bool)
    ok     = np.zeros(RUNS, dtype=bool)
    ruin   = np.zeros(RUNS, dtype=bool)

    for _ in range(80_000):
        if not active.any(): break
        locked |= (bals >= thresh)
        risk  = np.where(locked, r2, r1)

        wins  = np.random.rand(RUNS) < WIN_RATE
        # Gap risk: some fraction of losses are full liquidations (-100% of margin)
        is_gap = (~wins) & (np.random.rand(RUNS) < gap_pct)
        ret = np.where(wins, wp, np.where(is_gap, -1.0, lp))

        bals   = np.where(active, bals + bals * risk * ret, bals)
        trades = np.where(active, trades + 1, trades)
        nr = active & (bals <= RUIN)
        ns = active & (bals >= TARGET)
        ruin[nr]=True; ok[ns]=True
        active &= ~nr & ~ns

    s  = ok.sum()/RUNS*100
    r  = ruin.sum()/RUNS*100
    tok = trades[ok]
    am = tok.mean()/TRADES_PER_MONTH(lev)   if tok.size else 999
    mm = np.median(tok)/TRADES_PER_MONTH(lev) if tok.size else 999
    return dict(lev=lev, ev=ev*100, wp=wp*100, lp=lp*100,
                success=s, ruin=r, avg_mo=am, med_mo=mm)

def TRADES_PER_MONTH(lev):
    # Frequency doesn't change with leverage (it's a signal frequency)
    # But at very high leverage, some signals may be skipped if margin
    # is insufficient for the DCA grid (min notional constraint).
    # At $10, 50% risk = $5 margin. Min BTC notional = $100.
    # Max leverage before margin too small: $100/$5 = 20x min.
    # We're always above that, so frequency is constant.
    return 110

if __name__ == "__main__":
    np.random.seed(42)
    t0 = time.time()

    leverages = [25, 50, 75, 100, 125]
    liq_price = {lev: round(1/lev * (1-0.005) * 100, 3) for lev in leverages}

    print("=" * 72)
    print("  $10 START — LEVERAGE SWEEP  (Two-Stage 50% risk until $200, then 10%)")
    print("  WR: 64.25%  |  110 trades/month  |  50,000 runs")
    print("=" * 72)
    print(f"\nLiquidation distance (Binance BTC, 0.5% maint. margin):")
    for lev, dist in liq_price.items():
        sl_ok = "  SL fires FIRST" if dist > 0.75 else "  ** LIQ BEFORE SL **"
        print(f"  {lev:>3}x leverage: liq at {dist:.3f}% adverse move{sl_ok}")

    print(f"\n{'Lev':>5} {'EV/trade':>9} {'Avg Win':>9} {'Avg Loss':>10} "
          f"{'Success':>9} {'Ruin':>7} {'Avg(mo)':>9} {'Med(mo)':>9}")
    print("-" * 72)

    # No gap risk (perfect SL execution)
    print("\n--- SCENARIO A: Perfect SL execution (no gap risk) ---")
    for lev in leverages:
        r = run(lev, r1=0.50, r2=0.10, thresh=200.0, gap_pct=0.0)
        flag = " <<< FAST" if r['avg_mo'] <= 3 else ""
        print(f"{lev:>4}x  {r['ev']:>+8.2f}%  {r['wp']:>+8.2f}%  {r['lp']:>+9.2f}%"
              f"  {r['success']:>8.1f}%  {r['ruin']:>6.1f}%  {r['avg_mo']:>9.1f}"
              f"  {r['med_mo']:>9.1f}{flag}")

    # Realistic gap risk (5% of losing trades gap through SL at 100x+)
    print("\n--- SCENARIO B: Realistic gap risk (5% of losses = full liquidation) ---")
    gap_rates = {25: 0.00, 50: 0.01, 75: 0.02, 100: 0.05, 125: 0.10}
    for lev in leverages:
        r = run(lev, r1=0.50, r2=0.10, thresh=200.0, gap_pct=gap_rates[lev])
        flag = " <<< FAST" if r['avg_mo'] <= 3 else ""
        print(f"{lev:>4}x  {r['ev']:>+8.2f}%  gap={gap_rates[lev]*100:.0f}%"
              f"  {r['success']:>8.1f}%  {r['ruin']:>6.1f}%  {r['avg_mo']:>9.1f}"
              f"  {r['med_mo']:>9.1f}{flag}")

    # Best risk sizing at each leverage
    print("\n--- SCENARIO C: Optimised risk sizing per leverage level ---")
    risk_by_lev = {25: (0.50, 0.10, 200), 50: (0.30, 0.10, 200),
                   75: (0.20, 0.05, 200), 100: (0.15, 0.05, 200),
                   125: (0.10, 0.05, 200)}
    for lev in leverages:
        r1, r2, thresh = risk_by_lev[lev]
        r = run(lev, r1=r1, r2=r2, thresh=thresh, gap_pct=gap_rates[lev])
        print(f"{lev:>4}x  risk={r1*100:.0f}%->{r2*100:.0f}%  "
              f"{r['success']:>8.1f}%  {r['ruin']:>6.1f}%"
              f"  {r['avg_mo']:>9.1f}  {r['med_mo']:>9.1f}")

    # Milestone for best realistic combo
    print("\n--- MILESTONES: 50x leverage, 30%->10% ratchet at $200 ---")
    for tgt_label, tgt_val in [("$10->$100",100),("$10->$500",500),
                                ("$10->$1k",1000),("$10->$10k",10000)]:
        bals  = np.full(RUNS, START)
        locked= np.zeros(RUNS, dtype=bool)
        tr    = np.zeros(RUNS, dtype=int)
        act   = np.ones(RUNS, dtype=bool)
        ok    = np.zeros(RUNS, dtype=bool)
        rn    = np.zeros(RUNS, dtype=bool)
        lev   = 50
        wp_m  = (WIN_PRICE_MOVE  - 2*FEE) * lev
        lp_m  = -(LOSS_PRICE_MOVE + 2*FEE) * lev
        for _ in range(80_000):
            if not act.any(): break
            locked |= (bals >= 200)
            risk = np.where(locked, 0.10, 0.30)
            wins = np.random.rand(RUNS) < WIN_RATE
            is_gap = (~wins) & (np.random.rand(RUNS) < 0.01)
            ret  = np.where(wins, wp_m, np.where(is_gap, -1.0, lp_m))
            bals = np.where(act, bals + bals*risk*ret, bals)
            tr   = np.where(act, tr+1, tr)
            nr   = act & (bals <= RUIN); ns = act & (bals >= tgt_val)
            rn[nr]=True; ok[ns]=True; act &= ~nr & ~ns
        tok = tr[ok]
        am  = tok.mean()/TRADES_PER_MONTH(lev)   if tok.size else 999
        mm  = np.median(tok)/TRADES_PER_MONTH(lev) if tok.size else 999
        print(f"  {tgt_label}: avg {am:.1f} mo  (med {mm:.1f})  | Ruin: {rn.sum()/RUNS*100:.2f}%")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
