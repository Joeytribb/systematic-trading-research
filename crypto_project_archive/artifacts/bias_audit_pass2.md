# Bias Audit — Pass 2 (Long Scanner + Combined Pipeline)

8 new biases found after introducing the Long scanner and the $500 combo simulation.

---

## 🔴 CRITICAL

### Bias #8 — Long Scanner Trained on 2.5 Months vs BTC's 2 Years

**Location:** `long_scanner.py` → `process_asset()` line 145-146

**The Bug:**
```python
start_date = pd.to_datetime('2026-03-01')
end_date   = pd.to_datetime('2026-05-15')
```
The Short scanner (`win_rate_optimizer.py`) uses **2 years of BTC data** with a proper
walk-forward validator and 324 non-overlapping out-of-sample trades.

The Long scanner uses the **same 2.5-month window** as the short scanner's altcoin
alignment fix — with a **50/50 static split** (not walk-forward). That gives:
- Training data: ~5.5 weeks (not enough to capture even one full monthly cycle)
- Out-of-sample test: ~5.5 weeks
- Executed trades in test: **14** (same tiny sample problem as Bias #1)

The Long scanner's 64.3% win rate has the **same ±26% confidence interval** as
before. The "verified" Long win rate is statistically meaningless.

> [!CAUTION]
> **Impact:** The entire 92 trades/month and 4.5-month timeline for the $500
> scenario rests on the assumption that Long = Short in terms of win rate. This
> has NOT been verified. We have 14 trades of evidence.

---

### Bias #9 — Combined Frequency: Long + Short Share a Concurrency Lock

**Location:** `simulate_500_combo.py` line 14 — `TPM = 92`

**The Bug:**
```python
TPM = 92  # trades per month
# (BTC 35.5 x 1.30 multi-asset x 2.0 for Long+Short)
```
The 2.0x multiplier for "Long + Short" assumes the two strategies are **completely
additive** — i.e., every Long signal fires independently of every Short signal.

In reality, the combined engine uses a **single shared concurrency lock**. When a
Short trade is active (2-hour window), ALL Long signals during those 2 hours are
blocked, and vice versa. Both strategies respond to the SAME volatility events
(e.g., a sharp move triggers both a Short signal on one asset and a Long signal on
another). Those signals will cluster at the same time and block each other.

**Mathematical Impact:**
- Short trades: ~46/month, each holding ~2 hours → 46 × 2 = 92 hours locked/month
- Long trades: ~46/month → another 92 hours locked if independent
- Total: 184 hours/month out of 720 = **25.6% of time locked**
- Signals blocked by the lock: ~25.6% of 92 = ~24 trades/month blocked
- **True combined frequency: ~68-75 trades/month, NOT 92**

> [!CAUTION]
> **Impact:** The 92 trades/month figure is overstated by ~20-25%.
> Re-running with 70 trades/month: 4.5 months → ~6 months.

---

## 🟠 HIGH

### Bias #10 — Market Regime Asymmetry (Crypto is NOT Symmetric)

**Location:** Fundamental assumption in `long_scanner.py`

**The Bug:**
The Long scanner assumes the LONG edge is symmetric to the SHORT edge. But crypto
markets have a well-documented **asymmetric volatility structure**:
- **Crashes are fast and sharp** (wick down hard in seconds)
- **Recoveries are slow and choppy** (grind up over hours/days)

This means:
- SHORT strategy (fading rallies): works well because pumps reverse fast → TP hits quickly
- LONG strategy (fading drops): works less reliably because drops often continue ("catching
  a falling knife") before recovering

The 2.3-day test window happened to show 64.3% win rate for Longs. But this window
may have been a low-volatility, choppy market where drops did quickly reverse.
In a trending bear market, the Long strategy would have a dramatically lower win rate.

> [!WARNING]
> **Impact:** In adverse market conditions (sustained downtrend), the Long scanner's
> win rate could fall from 64.3% to below 50%, turning it into a money-losing strategy.

---

### Bias #11 — Top 5% is LESS Stable than Top 3% (By Our Own Metric)

**Location:** `speed_analysis.py` line 97-98 and `simulate_500_combo.py` line 11

**The Bug:**
The optimizer's own stability analysis showed:
```
Top 3%: rolling WR std = 3.8%  (most stable)
Top 5%: rolling WR std = 5.2%  (less stable)
Top 2%: rolling WR std = 3.9%
```
But in `simulate_500_combo.py`, we used Top 5% parameters because it gives more
trades/month. We explicitly overrode our own stability criterion in favour of speed.

> [!WARNING]
> **Impact:** Top 5% produces more signals but with greater variance. In live trading,
> the win rate at Top 5% is more likely to degrade than at Top 3%. The "12-month"
> scenario is built on the LESS stable threshold.

---

### Bias #12 — 5-Minute Signal-to-Entry Price Gap (Execution Slippage)

**Location:** Both scanners — `entry_time = current_time + pd.Timedelta(minutes=5)`

**The Bug:**
The model fires a signal at bar `T`. Entry is placed at `T+5` (next bar open).
The simulation uses `S0 = df.loc[current_time, 'close']` — the CLOSE price of bar T.
But actual entry happens at the OPEN of bar T+5, which can differ significantly.

For a volatile crypto asset after a sharp move (exactly the scenario that triggers
the signal), the next 5-minute open can easily be 0.1-0.3% away from the signal
close. At 25x leverage this is 2.5-7.5% on margin before the trade even begins.

> [!WARNING]
> **Impact:** This gap systematically worsens BOTH win returns and loss magnitudes.
> Not modelled anywhere in the pipeline. Estimated impact: -0.5 to -1.0% on avg PnL.

---

## 🟡 MEDIUM

### Bias #13 — Funding Rate Not Modelled

**Location:** Both DCA grid simulations

**The Bug:**
Perpetual futures charge a funding rate every 8 hours (typically 0.01-0.03% per
funding period). For our 2-hour trades, each trade has a ~25% chance of crossing
a funding payment window.

- Expected funding cost per trade: 0.01% × (2/8) × 25x = 0.0625% on margin
- On 92 trades/month: 92 × 0.0625% = 5.75% drag on margin per month
- At 10% risk: actual account drag = 0.575% per month

> [!NOTE]
> **Impact:** Small but consistent ~0.5-0.6% monthly drag not included in any
> simulation. Over 12 months: ~7% lower final balance than simulated.

---

### Bias #14 — Ruin Threshold Changed Silently Between Scripts

**Location:** `simulate_500_combo.py` line 35 vs `simulate_50_to_10k.py`

**The Bug:**
```python
# simulate_500_combo.py:
nr = active & (bals <= 10.0)   # ruin = below $10

# simulate_50_to_10k.py:
new_ruin = active & (bals <= 1.0)   # ruin = below $1
```
The ruin threshold silently changed from $1 to $10 between scripts. This makes the
$500-start simulation slightly more conservative (good), but it's inconsistent and
not documented. More importantly — if starting from $500, ruin should arguably
be defined as "below $50" (lost 90%) rather than "below $10" (lost 98%).

> [!NOTE]
> **Impact:** The 0.0% ruin rate shown is real, but it's measuring ruin defined
> as "below $10" not "below $50" (a 90% drawdown from starting capital). The 90%
> drawdown probability would be higher than 0%.

---

### Bias #15 — Compounding Assumes Infinite Divisibility (No Min Contract Size)

**Location:** Both Monte Carlo scripts — `bals + bals * risk * ret`

**The Bug:**
The simulation multiplies balance × risk and treats the result as perfectly divisible.
In practice, exchanges have minimum contract sizes:
- BTC perp: min notional $5-$10
- DOGE/LINK perp: min 1 contract = typically 1 DOGE (~$0.15) or 1 LINK (~$10)

When your account is $500 and you're risking 20% ($100 margin at 25x = $2,500 notional),
this is fine. But as the account grows to $5,000+ at 10% risk ($500 margin = $5,000
notional), contract granularity is no longer an issue.

**Real problem**: On the way UP from $500 to $1,000, if the account hits e.g. $503.47,
the simulation uses $100.69 margin (20% of $503.47), but the exchange rounds to the
nearest contract. This introduces a tiny but consistent underperformance.

> [!NOTE]
> **Impact:** Negligible at $500+. Less than 0.1% drag on performance.

---

## Summary of All Biases (Both Passes)

| # | Bias | Severity | Status | Direction |
|:--|:-----|:---------|:-------|:----------|
| 1 | 2.3-day test window (13 Short trades) | CRITICAL | Fixed (BTC 2yr used) | Inflates WR |
| 2 | Fabricated 2x Long multiplier | CRITICAL | Fixed (removed) | Inflates freq |
| 3 | Backward-derived win/loss PnL | HIGH | Fixed | Inflates edge |
| 4 | Top 2% threshold cherry-picked | HIGH | Fixed (stability added) | Inflates WR |
| 5 | DCA fill optimism (same-candle) | MEDIUM | Fixed | Inflates wins |
| 6 | Simulation-based peak balance | MEDIUM | Fixed (analytical) | Deterministic |
| 7 | No ratchet on risk staging | MEDIUM | Fixed | Understates ruin |
| **8** | **Long scanner: 14 trades, 2.5mo data** | **CRITICAL** | **NOT Fixed** | **Inflates WR** |
| **9** | **Shared lock: freq ≠ Short + Long** | **CRITICAL** | **NOT Fixed** | **Inflates freq ~25%** |
| **10** | **Crypto asymmetry (drops ≠ bounces)** | **HIGH** | **NOT Fixed** | **Inflates Long WR** |
| **11** | **Top 5% less stable than Top 3%** | **HIGH** | **NOT Fixed** | **Inflates stability** |
| **12** | **5-min signal-to-entry price gap** | **HIGH** | **NOT Fixed** | **Worsens avg PnL** |
| **13** | **Funding rate not modelled** | **MEDIUM** | **NOT Fixed** | **~7% over 12mo** |
| **14** | **Ruin threshold changed silently** | **MEDIUM** | **NOT Fixed** | **Understates ruin** |
| **15** | **No minimum contract size** | **LOW** | **NOT Fixed** | **Negligible** |

---

## Corrected Realistic Estimate for $500 Start

Accounting for the unfixed critical biases:

| Metric | Simulated | Realistic (bias-adjusted) |
|:-------|:----------|:--------------------------|
| Long WR | 64.3% | **Unknown (14 trades only)** |
| Combined trades/month | 92 | **~68-75** (shared lock) |
| Time to $10k (20% risk) | 9.7 months | **~13-16 months** |
| Time to $10k (50% risk) | 4.5 months | **~6-8 months** |

The 12-month target is still achievable but requires the Long scanner's edge to hold.
**The only way to verify the Long scanner edge is to run it on 2 years of BTC data
with a walk-forward validator** — exactly as was done for the Short scanner.
