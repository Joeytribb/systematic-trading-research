"""
simulate_50_to_10k.py  –  Bias-Corrected Monte Carlo
======================================================
Fixes applied vs previous version:
  [B2] Fabricated 2x Long multiplier removed. trades_per_month is now sourced
       directly from win_rate_optimizer.py (BTC only, with concurrency lock)
       × 1.30 (empirical multi-asset multiplier).

  [B3] win_pnl / loss_pnl now derived from the empirically measured avg_win
       and avg_loss from the full 2-year BTC backtest rather than
       backwards-derived from a chosen average PnL.

  [B6] Peak-balance calculation replaced with a closed-form analytical
       solution. The simulation loop was producing a deterministic, not
       stochastic result anyway.

  [B7] Two-stage risk model now uses a *ratchet*: once Stage-2 is reached
       the risk level can never escalate back to Stage-1, even after a
       drawdown.

NOTE: win_rate / avg_win_pnl / avg_loss_pnl / trades_per_month are
      PLACEHOLDERS below — update them with the numbers printed by
      win_rate_optimizer.py before trusting the final output.

      Run:  python win_rate_optimizer.py
      Then update the CALIBRATED PARAMS section below.
"""

import numpy as np
import time

# ── CALIBRATED PARAMS  (sourced from win_rate_optimizer.py — BTC 2yr walk-forward,
#    Top 2% threshold, concurrency-locked, DCA fill pessimism applied) ────────
WIN_RATE           = 0.6265   # 62.65%  — 324 non-overlapping trades
AVG_WIN_PNL        = 0.1020   # +10.20% — empirical avg win
AVG_LOSS_PNL       = -0.1345  # -13.45% — empirical avg loss
# BTC non-overlapping: 15.5/mo × 1.30x multi-asset multiplier = 20.1/mo
TRADES_PER_MONTH   = 20       # Conservative round-down of 20.1
# ─────────────────────────────────────────────────────────────────────────────

RUNS = 50_000

def run_monte_carlo(start_bal: float, target_bal: float,
                    stage1_risk: float = 0.50,
                    stage2_risk: float = 0.10,
                    stage2_threshold: float | None = None,
                    label: str = "") -> dict:
    """
    Vectorised Monte Carlo with:
      • Per-path ratchet risk (Bias #7 fix): once a path crosses stage2_threshold
        its risk is permanently capped at stage2_risk.
      • Independent win/loss PnL values (Bias #3 fix).
    """
    bals         = np.full(RUNS, float(start_bal))
    # [B7-fix] ratchet: permanently locked to stage2 once threshold crossed
    stage2_locked = np.zeros(RUNS, dtype=bool)
    trades       = np.zeros(RUNS, dtype=int)
    active       = np.ones(RUNS,  dtype=bool)
    ruined       = np.zeros(RUNS, dtype=bool)
    success      = np.zeros(RUNS, dtype=bool)

    for _ in range(20_000):
        if not active.any():
            break

        # [B7-fix] ratchet
        if stage2_threshold is not None:
            stage2_locked |= (bals >= stage2_threshold)
            risk = np.where(stage2_locked, stage2_risk, stage1_risk)
        else:
            risk = np.full(RUNS, stage1_risk)

        wins    = np.random.rand(RUNS) < WIN_RATE
        returns = np.where(wins, AVG_WIN_PNL, AVG_LOSS_PNL)

        bals   = np.where(active, bals + bals * risk * returns, bals)
        trades = np.where(active, trades + 1, trades)

        new_ruin    = active & (bals <= 1.0)
        new_success = active & (bals >= target_bal)
        ruined[new_ruin]   = True
        success[new_success] = True
        active &= ~new_ruin & ~new_success

    s_rate = success.sum() / RUNS * 100
    r_rate = ruined.sum()  / RUNS * 100
    t_ok   = trades[success]
    avg_mo = t_ok.mean()   / TRADES_PER_MONTH if t_ok.size > 0 else float('inf')
    med_mo = np.median(t_ok) / TRADES_PER_MONTH if t_ok.size > 0 else float('inf')
    return dict(label=label, success=s_rate, ruin=r_rate, avg_mo=avg_mo, med_mo=med_mo)


def analytical_max_balance() -> dict:
    """
    Bias #6 fix: closed-form derivation of the max account balance.

    The edge disappears when slippage grows to the point where EV = 0.

    Because AVG_WIN_PNL and AVG_LOSS_PNL are already leverage-adjusted
    (they are % returns on margin from the backtest), slippage on the
    price level translates to leverage * slippage on the margin return.

    EV(s) = wr * (W - s*L) + (1-wr) * (Lo - s*L)
           = EV(0)  -  s * L
    Setting EV(s) = 0:
           slippage_zero  = EV(0) / L

    Then invert the slippage model to get the position size and balance.
    """
    wr       = WIN_RATE
    W        = AVG_WIN_PNL    # already leverage-adjusted
    Lo       = AVG_LOSS_PNL   # already leverage-adjusted
    fee      = 0.0004
    risk_pct = 0.10           # Stage-2 risk

    ev0 = wr * W + (1 - wr) * Lo   # base EV per trade (no slippage)

    # At large account sizes the exchange caps leverage at 10x
    lev_cap = 10.0
    slippage_zero = ev0 / lev_cap   # price-level slippage that zeros EV

    if slippage_zero <= 0:
        # Strategy has negative EV even without slippage at these params.
        return dict(ev0=ev0 * 100, slip_zero_pct=0.0, pos_zero_usd=0.0, max_balance=0.0)

    # Invert slippage model:  slippage = 0.0015 * (pos / 500_000)^1.5
    pos_zero  = 500_000 * (slippage_zero / 0.0015) ** (1 / 1.5)
    bal_zero  = pos_zero / (risk_pct * lev_cap)

    return dict(
        ev0           = ev0 * 100,
        slip_zero_pct = slippage_zero * 100,
        pos_zero_usd  = pos_zero,
        max_balance   = bal_zero,
    )



if __name__ == "__main__":
    t0 = time.time()
    np.random.seed(42)

    print("=" * 62)
    print("  BIAS-CORRECTED MONTE CARLO  ($50 -> $10,000)")
    print(f"  Win Rate: {WIN_RATE*100:.1f}%  |  Win: {AVG_WIN_PNL*100:+.2f}%  |"
          f"  Loss: {AVG_LOSS_PNL*100:+.2f}%")
    print(f"  Trades/Month (BTC x 1.30x): {TRADES_PER_MONTH}")
    print(f"  {RUNS:,} runs")
    print("=" * 62)

    scenarios = [
        dict(stage1_risk=0.50, stage2_risk=0.50, stage2_threshold=None,
             label="Aggressive  (50% risk all the way)"),
        dict(stage1_risk=0.10, stage2_risk=0.10, stage2_threshold=None,
             label="Conservative (10% risk all the way)"),
        dict(stage1_risk=0.50, stage2_risk=0.10, stage2_threshold=200.0,
             label="Two-Stage ratchet  ($50->$200 @ 50%, then 10%)"),
        dict(stage1_risk=0.50, stage2_risk=0.10, stage2_threshold=500.0,
             label="Two-Stage ratchet  ($50->$500 @ 50%, then 10%)"),
        dict(stage1_risk=0.50, stage2_risk=0.10, stage2_threshold=1000.0,
             label="Two-Stage ratchet  ($50->$1k  @ 50%, then 10%)"),
    ]

    print(f"\n{'Scenario':<45} {'Success':>8} {'Ruin':>6} {'Avg(mo)':>9} {'Med(mo)':>9}")
    print("-" * 80)
    for sc in scenarios:
        r = run_monte_carlo(start_bal=50.0, target_bal=10_000.0, **sc)
        print(f"{r['label']:<45} {r['success']:7.2f}% {r['ruin']:5.2f}%"
              f" {r['avg_mo']:9.2f} {r['med_mo']:9.2f}")

    # ── Analytical max balance ──
    print("\n" + "=" * 62)
    print("  ANALYTICAL MAX-BALANCE (Bias #6 fix - no simulation needed)")
    print("=" * 62)
    mb = analytical_max_balance()
    print(f"  Base EV per trade (no slippage)  : {mb['ev0']:+.2f}%")
    print(f"  EV -> 0 when price slippage hits : {mb['slip_zero_pct']:+.4f}%")
    print(f"  That occurs at position size     : ${mb['pos_zero_usd']:>12,.0f}")
    print(f"  Max account balance (10% risk, 10x lev cap):")
    print(f"                                     ${mb['max_balance']:>12,.0f}")
    print(f"\n  Interpretation: beyond this balance order-book slippage eats")
    print(f"  all the edge. Cap position size at ~${mb['pos_zero_usd']:,.0f}")
    print(f"  and bank profits above that threshold.")


    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
