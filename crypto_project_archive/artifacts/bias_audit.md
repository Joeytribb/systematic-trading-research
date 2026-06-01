# Bias Audit Report

A ruthless audit of every assumption in the simulation pipeline.
All 7 biases are rated by severity and given concrete fix recommendations.

---

## 🔴 CRITICAL Biases

### Bias #1 — Short Test Window (Only 2.3 Days of Live Out-of-Sample Data)

**Location:** `multi_asset_scanner.py` → `main()` → aligned `common_idx`

**The Bug:**
The Multi-Asset engine aligns all 5 assets to the **intersection** of their test indices.
Because the 5 CSVs end on different dates, the intersection shrinks to only **2.3 days** of data.
```python
# Aligned Out-Of-Sample Period: 674 periods (~2.3 days)
```
The entire **Win Rate (69.2%), Avg PnL (+3.60%), and trade count (13 trades)** used for
the Monte Carlo are derived from only **13 trades in 2.3 days**. 

Statistically, a 13-trade sample has a **95% confidence interval of ±27%** around the true win rate.
This means the "true" Win Rate could realistically be anywhere from **42% to 96%** — completely
uninformative. All downstream projections (time to $10k, max balance) depend on this tiny window.

> [!CAUTION]
> **Impact on Monte Carlo:** The `win_rate=0.645`, `win_pnl=0.12`, and `loss_pnl=-0.1732`
> fed into `simulate_50_to_10k.py` are derived from 13 trades. The real figures could
> easily be far worse. We should need at least **200+ out-of-sample trades** to establish
> any statistical confidence.

**Fix:** Extend the individual altcoin CSV datasets or use walk-forward testing that generates enough out-of-sample trades across the full BTC dataset (which has 2 years of data) and apply the altcoin multiplier as a separate fixed scalar.

---

### Bias #2 — Extrapolation Error: 2.3-Day Frequency Projected as Monthly Rate

**Location:** `walkthrough.md` and Monte Carlo parameters

**The Bug:**
The model observed **13 trades in 2.3 days** and linearly extrapolated to:
```
~169 Shorts/month × 2 (adding Longs) = ~338 trades/month
```
But:
- There is **zero evidence of Long trades** anywhere in the scanner code. The scanner only runs `sim_dca_grid_short`. The 2× multiplier for "Longs" was fabricated.
- 2.3 days is not enough to capture **weekend volume dry-ups**, **news events**, or **market regime changes** that reduce signal frequency by 50-80%.
- The actual frequency in the full BTC-only walk-forward (`win_rate_optimizer.py`) was **~17 signals/month at Top 2%** threshold — **20× lower** than the 338 projected.

> [!CAUTION]
> **Impact:** The time-to-$10k estimates of **2.9 months** are almost certainly wrong by a factor of 5-10x. A more realistic frequency is 17–34 non-overlapping trades/month (BTC only × 2 for multi-asset boost), giving a **realistic timeline of 15–30 months**.

**Fix:** Remove the fabricated 2× Long multiplier. Run the full walk-forward over the complete 2-year BTC dataset and multiply the BTC frequency by the empirically measured 1.30x multi-asset multiplier (from the 13-trade test).

---

## 🟠 HIGH Biases

### Bias #3 — Optimistic Win Return Assumption in Peak Balance Simulation

**Location:** `simulate_50_to_10k.py` → `simulate_peak_balance_aligned()`

**The Bug:**
The slippage model uses a base price movement of `0.0056` for wins and `0.00613` for losses:
```python
win_return = (0.0056 - slippage) * leverage - (fee_pct * leverage * 2)
loss_return = -(0.00613 + slippage) * leverage - (fee_pct * leverage * 2)
```
But these numbers are **backward-derived** from the overall average PnL to force the math to
work, not measured directly. The actual DCA grid engine uses:
- **TP at 0.5% below avg entry** → win price move = `0.005`
- **SL at 0.75% above signal price** → loss price move can be **up to** `0.0075`

If we plug in the correct DCA mechanics (`0.005` win, `0.0075` loss), the expected value per trade at 25x leverage becomes:
```
EV = 0.645 × (0.005×25 - 0.02) + 0.355 × (-(0.0075×25) - 0.02)
EV = 0.645 × 0.1050 + 0.355 × (-0.2075)
EV = 0.0677 - 0.0737 = -0.0060  (NEGATIVE!)
```
At the correct TP/SL ratio, the strategy has a **negative expected value** at 25x leverage without relying on the DCA grid averaging down to improve the average entry!

> [!WARNING]
> **Impact:** The whole edge depends on whether the DCA grid averaging (buying more as price wicks up) actually reduces the average entry enough. The 13-trade sample isn't large enough to confirm this definitively.

**Fix:** Use the exact win/loss PnL values from the full-dataset `win_rate_optimizer.py` backtest (not the multi-asset scanner), and plug those directly into the Monte Carlo. The walkthrough reports `Avg PnL: +1.59%` per trade — this is the number to use, not the reverse-engineered decomposition.

---

### Bias #4 — Threshold Selection Bias (Top 2% is Cherry-Picked)

**Location:** `multi_asset_scanner.py` line 145, `win_rate_optimizer.py` line 145

**The Bug:**
```python
thr = np.percentile(probs, 98.0)  # Top 2% Sweet Spot
```
The "Top 2%" threshold was chosen **because it backtested well**. This is classic **threshold
selection bias** (a form of overfitting). The optimizer ran 5 thresholds (Top 1%–5%) and
picked the best one. In live trading, the probability distribution of the model's output will
shift slightly each week. A threshold tuned to Top 2% on historical data will not stay
at Top 2% in production.

> [!WARNING]
> **Impact:** The win rate will likely decay from 64.5% toward the base rate (~55%) over 3-6 months
> as the model encounters unseen market regimes. The actual out-of-sample degradation is unknown
> because the out-of-sample window was only 2.3 days.

**Fix:** Use proper time-series cross-validation (walk-forward with multiple expanding windows),
pick the threshold that gives the **most stable** win rate across all windows, not the highest.

---

## 🟡 MEDIUM Biases

### Bias #5 — DCA Fill Optimism: All 4 Levels Filled on Same Candle

**Location:** `multi_asset_scanner.py` / `win_rate_optimizer.py` → `sim_dca_grid_short()`

**The Bug:**
```python
for lvl in levels:
    if lvl not in fills and high >= lvl:
        fills.append(lvl)
```
On a single 1-minute candle where `high >= S0*1.0045`, **all 4 DCA levels get filled
simultaneously**. In reality, a market order at `S0*1.0015` is a different order than one at
`S0*1.0045`. A sharp wick that hits all 4 levels in one candle means the **actual average
fill will be worse** (closer to the spike top), not exactly at the 4 predefined prices.

> [!NOTE]
> **Impact:** Small but real. Overestimates the benefit of DCA grid averaging by ~0.1-0.2%
> on wick trades. This means win returns are slightly optimistic and loss amounts are slightly understated.

**Fix:** Add a `if len(fills) > 1: avg_entry = (fills[0]*3 + fills[-1]) / 4` worst-case fill
simulation, or add small random slippage per fill.

---

### Bias #6 — Survivorship Bias in the Peak Balance Simulation

**Location:** `simulate_50_to_10k.py` → `simulate_peak_balance_aligned()`

**The Bug:**
The simulation terminates a path as soon as `EV <= 0` and records the **current balance as the
peak**. But in reality, a trader doesn't instantly know their EV has crossed zero. They keep
trading. The simulation is also **vectorized across all paths simultaneously** — meaning all 20,000
paths hit the same leverage tier boundary at roughly the same balance, creating an artificially
tight peak distribution (all paths cluster near $283k).

> [!NOTE]
> **Impact:** The peak balance figure of **$283,500** is very deterministic — it's really just
> the mathematical EV-zero crossing point, not a Monte Carlo result. The "simulation" adds no
> real value here; it's just solving an equation numerically.

**Fix:** This is better answered analytically than via simulation. The true limit is simply:
`max_position = balance_where_EV_becomes_zero`. No Monte Carlo needed for this question.

---

### Bias #7 — Two-Stage Risk Model Has No Drawdown Guard

**Location:** `simulate_50_to_10k.py` → `run_simulation_vectorized()`

**The Bug:**
```python
risk = np.where(bals < stage2_threshold, risk_pct, stage2_risk)
```
The two-stage model switches from 50% risk to 10% risk when balance crosses `$200` (or `$500`,
`$1000`). But if the balance **drops back below** the threshold (e.g., from $250 back to $180),
the simulation **reverts to 50% risk again**. This is the correct Kelly behavior but it means
a drawdown from $500 → $199 would flip back to aggressive sizing, potentially liquidating a
recovered account that was previously "safe."

There is no maximum drawdown guard, no trailing stop on risk, and no "once Stage 2, always Stage 2" lock.

> [!NOTE]
> **Impact:** The 0.02-0.03% ruin rate is slightly understated because of this re-escalation
> behavior. In practice, a disciplined trader would **never** go back to 50% risk after
> reaching $200+ and suffering a drawdown. They'd keep using 10% risk.

**Fix:** Add a `max_risk_ever = np.minimum(current_risk, past_risk)` ratchet mechanism to prevent
risk escalation after Stage 2 is reached.

---

## Summary Table

| # | Bias | Severity | Affected Metric | Direction of Inflation |
|:--|:-----|:---------|:----------------|:----------------------|
| 1 | 2.3-Day Test Window (13 trades) | 🔴 CRITICAL | Win Rate, Avg PnL | Upward (unknown magnitude) |
| 2 | Fabricated 2× Long Multiplier | 🔴 CRITICAL | Trades/Month, Time to $10k | Trades 2×-20× overestimated |
| 3 | Backward-Derived Win/Loss PnL | 🟠 HIGH | Max Balance, EV | Upward bias on edge |
| 4 | Top 2% Threshold Selection Bias | 🟠 HIGH | Win Rate (live decay) | +5-10% optimistic WR |
| 5 | DCA Fill Optimism | 🟡 MEDIUM | Avg Entry on Wicks | Win PnL ~0.1-0.2% high |
| 6 | Survivorship in Peak Simulation | 🟡 MEDIUM | Max Balance figure | Deterministic, not stochastic |
| 7 | No Drawdown Guard on Risk Stage | 🟡 MEDIUM | Risk of Ruin | Slightly underestimated |

---

## Realistic Adjusted Expectations

Correcting for **Bias #1** (confidence intervals) and **Bias #2** (fabricated frequency):

| Metric | Current (Optimistic) | Corrected (Conservative) |
|:-------|:---------------------|:------------------------|
| Win Rate (live) | 64.5% | ~55-60% (expected decay) |
| Trades / Month | 338 | ~22-34 (BTC 17 × 1.3x multiplier) |
| Time to $10k (50% risk) | 2.9 months | **12-18 months** |
| Time to $10k (two-stage) | 6 months | **18-30 months** |
| Max Account Balance | ~$283,500 | **$283,500** (unchanged, this is math) |
