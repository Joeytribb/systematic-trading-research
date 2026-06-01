# Implementation Plan: Zero-Bias Pipeline

## Objective
Fix all 8 remaining biases (#8-#15) and perform a rigorous look-ahead audit.

---

## Look-Ahead Audit Findings

### CONFIRMED CLEAN (no future data used)
| Location | Check | Result |
|:---------|:------|:-------|
| MTF 1H/4H SMA | `label='right', closed='right'` + `ffill` | CLEAN — bar at T uses SMA ending at T |
| StandardScaler | Fit on train only, transform on test | CLEAN |
| Walk-forward (Short) | `model.fit(X[:s-H], y[:s-H])` | CLEAN — last H rows excluded from training targets |
| Rolling features (BB, RSI, MACD, vel) | Standard pandas rolling, backward-looking | CLEAN |
| Target variable construction | `shift(-H)` is intentional labeling, never used as a feature | CLEAN |

### ⚠️  SUBTLE LOOK-AHEAD CONFIRMED: `multi_asset_scanner.py` + `long_scanner.py`

```python
# Line 158, multi_asset_scanner.py — STILL WRONG:
thr = np.percentile(probs, 98.0)   # Top 2% Sweet Spot

# Line 176, long_scanner.py — ALSO WRONG:
thr = np.percentile(probs, 95.0)   # Top 5% threshold
```

`np.percentile(probs, 98.0)` computes the threshold using the **entire test set's probability
distribution** — including the last bar of the test window. This means the threshold
is computed by looking at all future probabilities in the test set before deciding what
"Top 2%" means. In live trading, you don't know what the 98th percentile of future
probabilities will be — you can only know what it was historically.

**Correct fix:** Compute the threshold from the **training set probabilities only**, then
apply it to the test set. The threshold from training represents what the model
"expected" the 98th percentile to be, without seeing test data.

### ⚠️  LOOK-AHEAD IN `long_scanner.py`: Static split vs Walk-Forward

```python
# long_scanner.py lines 161-173
split = int(len(df) * 0.5)
train = df.iloc[:split]
test  = df.iloc[split:len(df)-H]
X_tr = train[FCOLS].values
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)   # CLEAN
X_te = sc.transform(X_te)       # CLEAN
```

The static split is not look-ahead per se, but the model has only 2.5 months of data
total (split to 5.5 weeks train, 5.5 weeks test). This isn't look-ahead but is
**Bias #8** (too little data). The Long scanner MUST use 2 years of BTC data +
walk-forward identical to the Short scanner.

### ⚠️  ENTRY PRICE LOOK-AHEAD: Signal close used as entry price

```python
# All scanners:
S0 = df.loc[current_time, 'close']   # signal bar close
entry_time = current_time + pd.Timedelta(minutes=5)
window = raw.loc[entry_time : max_end_time]
# window starts 5 minutes AFTER S0 was observed.
# The actual market price at entry_time can differ from S0.
```

This is Bias #12. Using `S0 = signal_close` as the DCA entry reference when
the actual entry is 5 minutes later introduces systematic optimism. Fix: use
the OPEN of the first candle in `window` as S0.

---

## Files to Create / Modify

### [NEW] `long_rate_optimizer.py`  (Bias #8 fix)
Mirror of `win_rate_optimizer.py` for LONG trades.
- Uses full 2-year BTC dataset
- Walk-forward LR (identical INIT=24000, STEP=8000)
- Target: `high.rolling(H).max().shift(-H) >= close * 1.005`
- DCA grid LONG with worst-case fill
- Outputs calibrated WR, avg_win, avg_loss, trades/month

### [MODIFY] `win_rate_optimizer.py`  (Look-ahead threshold fix)
Fix the percentile threshold computation:
```python
# WRONG (uses test set distribution):
thr = np.percentile(probs, pct)  # probs is the TEST set

# CORRECT (compute threshold from training set probabilities):
train_probs = preds[INIT:wi]   # or use in-sample predictions
thr = np.percentile(train_probs, pct)
```

### [MODIFY] `multi_asset_scanner.py`  (Look-ahead threshold + entry price)
1. Fix threshold: compute from training-period probabilities, not test
2. Fix entry price: `S0 = window.iloc[0]['open']` instead of signal bar close
3. Add funding rate: `-0.01% per 8 hours` prorated per trade duration

### [MODIFY] `long_scanner.py`  (Full rebuild)
Replace static split + short data window with:
- 2-year BTC walk-forward (same as `long_rate_optimizer.py`)
- Threshold from training probabilities only
- Entry price from first candle open

### [NEW] `combined_scanner.py`  (Bias #9 fix — measure real combined frequency)
Run both Short and Long signals simultaneously on BTC 2yr data with a
**single shared concurrency lock**. Measures the true non-overlapping
combined frequency directly, rather than estimating it analytically.

### [MODIFY] `simulate_500_combo.py`  (Bias #9, #13, #14 fixes)
1. Use the real combined trades/month from `combined_scanner.py` output
2. Add funding rate drag: `-0.0025% per trade` (0.01% per 8h × 2h average)
3. Standardize ruin threshold to $50 (10% of $500 start = meaningful ruin)

---

## Verification Plan

After all fixes, run in sequence:
1. `python long_rate_optimizer.py` → get calibrated Long params
2. `python combined_scanner.py` → get real combined frequency  
3. `python simulate_500_combo.py` → final corrected Monte Carlo

Expected outcome: 
- Combined frequency: **~55-70 trades/month** (less than the 92 assumed)
- Long WR: statistically meaningful from 200+ trades
- Timeline to $10k from $500: **honest range, not inflated**
