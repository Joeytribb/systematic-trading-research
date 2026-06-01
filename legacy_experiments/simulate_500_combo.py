"""
simulate_500_combo.py
=====================
Best combination: Top 5% + Long scanner + various risk levels, starting at $500
Trades/month: 92 (BTC 35.5 x 1.30 multi-asset x 2.0 for Long+Short)
WR: 64.3%, Avg Win: +10.5%, Avg Loss: -13.82%
"""
import numpy as np, time

# Top-5% params (from optimizer output), symmetric Long+Short
WIN_RATE   = 0.643
AVG_WIN    =  0.1050
AVG_LOSS   = -0.1382
TPM        = 92      # trades per month

RUNS       = 50_000
TARGET     = 10_000.0

def mc(start, target, r1, r2, thresh, tpm=TPM, wr=WIN_RATE, w=AVG_WIN, l=AVG_LOSS):
    bals   = np.full(RUNS, float(start))
    locked = np.zeros(RUNS, dtype=bool)
    trades = np.zeros(RUNS, dtype=int)
    active = np.ones(RUNS,  dtype=bool)
    ok     = np.zeros(RUNS, dtype=bool)
    ruin   = np.zeros(RUNS, dtype=bool)

    for _ in range(20_000):
        if not active.any(): break
        locked |= bals >= thresh
        risk = np.where(locked, r2, r1)
        wins = np.random.rand(RUNS) < wr
        ret  = np.where(wins, w, l)
        bals  = np.where(active, bals + bals * risk * ret, bals)
        trades= np.where(active, trades + 1, trades)
        nr = active & (bals <= 10.0)   # ruin = below $10
        ns = active & (bals >= target)
        ruin[nr]=True; ok[ns]=True
        active &= ~nr & ~ns

    sr = ok.sum()/RUNS*100
    rr = ruin.sum()/RUNS*100
    tok = trades[ok]
    am = tok.mean()/tpm   if tok.size else 999
    mm = np.median(tok)/tpm if tok.size else 999
    return sr, rr, am, mm

if __name__=="__main__":
    np.random.seed(42)
    t0=time.time()
    print("="*72)
    print(f"  STARTING CAPITAL: $500  |  TARGET: $10,000  |  {RUNS:,} runs")
    print(f"  Strategy: Top 5% Short + Long scanner (mirror)")
    print(f"  WR: {WIN_RATE*100:.1f}%  Win: {AVG_WIN*100:+.1f}%  Loss: {AVG_LOSS*100:+.1f}%  Trades/mo: {TPM}")
    print("="*72)

    configs = [
        # label, r1, r2, threshold
        ("10% risk flat (most conservative)",         0.10, 0.10, 9999),
        ("20% risk flat",                             0.20, 0.20, 9999),
        ("30% risk flat",                             0.30, 0.30, 9999),
        ("50% risk flat",                             0.50, 0.50, 9999),
        ("Two-Stage: 50% -> 10% at $2k",              0.50, 0.10, 2000),
        ("Two-Stage: 50% -> 10% at $3k",              0.50, 0.10, 3000),
        ("Two-Stage: 50% -> 20% at $2k",              0.50, 0.20, 2000),
        ("Two-Stage: 30% -> 10% at $2k",              0.30, 0.10, 2000),
    ]

    print(f"\n{'Scenario':<42} {'Success':>9} {'Ruin':>6} {'Avg(mo)':>9} {'Med(mo)':>9}")
    print("-"*72)
    for label,r1,r2,thresh in configs:
        s,r,a,m = mc(500, TARGET, r1, r2, thresh)
        flag = "  <<< UNDER 12 MONTHS" if a<=12 else ("  <<< UNDER 18 MONTHS" if a<=18 else "")
        print(f"{label:<42} {s:>8.1f}% {r:>5.1f}% {a:>9.1f} {m:>9.1f}{flag}")

    # Also show milestones for the best scenario
    print("\n--- MILESTONE BREAKDOWN (50% flat risk, 92 trades/mo, $500 start) ---")
    for milestone in [1000, 2000, 5000, 10000]:
        s,r,a,m = mc(500, milestone, 0.50, 0.50, 9999)
        print(f"  $500 -> ${milestone:>6,}: {a:5.1f} months avg  ({m:5.1f} median)  |  Ruin: {r:.2f}%")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
