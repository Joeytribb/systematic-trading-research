"""
long_rate_optimizer.py  -  Long-Side Walk-Forward Optimizer
=============================================================
Bias #8 fix: uses the FULL 2-year BTC dataset with proper walk-forward
validation (identical structure to win_rate_optimizer.py) instead of
the 2.5-month static-split used by long_scanner.py.

Look-ahead fixes applied:
  [LA1] Threshold computed from TRAINING probabilities, not test set.
  [LA2] Entry price = first candle OPEN after signal (not signal-bar close).
  [B13] Funding rate deducted per trade (-0.01%/8h prorated to hold time).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── FUNDING RATE ──────────────────────────────────────────────────────────────
FUNDING_RATE_PER_8H = 0.0001   # 0.01% per 8-hour window (typical Binance)

def rsi(s, p=14):
    d = s.diff()
    g =  d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df, raw_1m):
    ma  = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    ub  = ma + 2*std; lb = ma - 2*std; bw = ub - lb
    df['pct_b']      = (df['close'] - lb) / (bw + 1e-10)
    df['band_width'] = bw / (ma + 1e-10)
    df['rsi']        = rsi(df['close'])
    e12  = df['close'].ewm(span=12, adjust=False).mean()
    e26  = df['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / (df['close'] + 1e-10)
    df['macd_hist'] = (macd - sig) / (df['close'] + 1e-10)
    df['vel1'] = df['close'].pct_change(1)
    df['vel3'] = df['close'].pct_change(3)
    cr = df['high'] - df['low']
    df['uw'] = (df['high'] - df[['open','close']].max(axis=1)) / (cr + 1e-10)
    df['lw'] = (df[['open','close']].min(axis=1) - df['low'])  / (cr + 1e-10)
    vol_ma = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (vol_ma + 1e-10)
    # [LA1-fix] MTF: label='right', closed='right' + ffill  (already correct)
    raw_1h = raw_1m['close'].resample('1h', label='right', closed='right').last().dropna()
    raw_4h = raw_1m['close'].resample('4h', label='right', closed='right').last().dropna()
    sma_1h = raw_1h.rolling(50).mean()
    sma_4h = raw_4h.rolling(50).mean()
    df['sma_1h']      = sma_1h.reindex(df.index, method='ffill')
    df['sma_4h']      = sma_4h.reindex(df.index, method='ffill')
    df['mtf_1h_dist'] = (df['close'] - df['sma_1h']) / (df['sma_1h'] + 1e-10)
    df['mtf_4h_dist'] = (df['close'] - df['sma_4h']) / (df['sma_4h'] + 1e-10)
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist',
         'vel1','vel3','uw','lw','vol_ratio','mtf_1h_dist','mtf_4h_dist']

def walk_forward_lr(df, target_col, H=24):
    """Identical walk-forward to win_rate_optimizer. Returns (preds, wi, train_preds)."""
    X = df[FCOLS].values
    y = df[target_col].values
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
    preds       = np.zeros(len(X))
    train_preds = np.zeros(len(X))  # in-sample predictions (for threshold computation)
    INIT, STEP  = 24000, 8000

    if len(X) < INIT + H:
        sc = StandardScaler()
        model.fit(sc.fit_transform(X), y)
        p = model.predict_proba(sc.transform(X))[:,1]
        return p, 0, p

    for s in range(INIT, len(X)-H, STEP):
        e  = min(s + STEP, len(X) - H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H])
        Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
        # [LA1-fix] Also store training-period probabilities for threshold
        train_preds[s-H-STEP:s-H] = model.predict_proba(
            sc.transform(X[max(0,s-H-STEP):s-H])
        )[:,1]

    return preds, INIT, train_preds

def sim_dca_grid_long(window, S0, leverage=25.0, fee=0.0004):
    """
    DCA grid LONG — worst-case fill applied (Bias #5 fix).
    [LA2] S0 is passed in as the first candle OPEN (not signal-bar close).
    [B13] Returns (pnl, hold_minutes) so caller can deduct funding.
    """
    levels   = [S0, S0*0.9985, S0*0.997, S0*0.9955]
    fills    = []
    sl_price = S0 * 0.9925
    avg_entry = None; tp_price = None
    entry_time = window.index[0]

    for idx, row in window.iterrows():
        high, low = row['high'], row['low']

        newly_filled = []
        for lvl in levels:
            if lvl not in fills and low <= lvl:
                fills.append(lvl); newly_filled.append(lvl)

        if len(newly_filled) > 1:
            worst = min(newly_filled)
            base  = [f for f in fills if f not in newly_filled]
            fills = base + [worst] * len(newly_filled)

        if fills:
            avg_entry = sum(fills) / len(fills)
            tp_price  = avg_entry * 1.005

        if not fills:
            if high >= S0:
                fills.append(S0); avg_entry = S0; tp_price = avg_entry * 1.005

        if not fills: continue

        if low <= sl_price:
            pct = (sl_price - avg_entry) / avg_entry
            hold_m = (idx - entry_time).total_seconds() / 60
            raw_pnl = pct * leverage - fee * leverage * 2
            funding = FUNDING_RATE_PER_8H * (hold_m / 480) * leverage
            return raw_pnl - funding, idx

        if high >= tp_price:
            pct = (tp_price - avg_entry) / avg_entry
            hold_m = (idx - entry_time).total_seconds() / 60
            raw_pnl = pct * leverage - fee * leverage * 2
            funding = FUNDING_RATE_PER_8H * (hold_m / 480) * leverage
            return raw_pnl - funding, idx

    if not fills: return 0.0, window.index[-1]
    pct = (window.iloc[-1]['close'] - avg_entry) / avg_entry
    hold_m = (window.index[-1] - entry_time).total_seconds() / 60
    raw_pnl = pct * leverage - fee * leverage * 2
    funding = FUNDING_RATE_PER_8H * (hold_m / 480) * leverage
    return raw_pnl - funding, window.index[-1]

def main():
    print("Loading 2 Years of BTC data...")
    raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    raw.columns = [c.lower() for c in raw.columns]
    if pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='coerce')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=730)]

    print("Building 5m features...")
    df = raw.resample('5min').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna()
    H = 24
    # LONG target: did HIGH rise 0.5% within next H bars?
    df['target_long'] = (
        df['high'].rolling(H, 1).max().shift(-H) >= df['close'] * 1.005
    ).astype(int)

    df = add_features(df, raw)

    print("Training Long Walk-Forward LR...")
    preds, wi, train_preds = walk_forward_lr(df, 'target_long', H)
    test_idx   = df.index[wi : len(df)-H]
    test_probs = preds[wi : len(df)-H]
    # [LA1-fix] Training-period probabilities for threshold (no test data contamination)
    tr_probs   = train_preds[: wi]

    thresholds_to_test = [95.0, 96.0, 97.0, 98.0, 99.0]

    print("\n--- LONG SCANNER LEADERBOARD (with Concurrency Lock, LA-clean threshold) ---")
    print(f"{'Threshold':12} {'Signals/mo':>12} {'Win Rate':>10} {'Avg PnL':>10} {'Stability':>10}")

    best_threshold = None; best_stability = float('inf')

    for pct in thresholds_to_test:
        # [LA1-fix] Threshold from TRAINING probabilities, not test set
        thr = np.percentile(tr_probs[tr_probs > 0], pct)
        signal_indices = test_idx[test_probs >= thr]

        lock_until = signal_indices[0] - pd.Timedelta(minutes=1)
        valid = []
        for idx in signal_indices:
            if idx < lock_until: continue
            entry_time   = idx + pd.Timedelta(minutes=5)
            end_time     = entry_time + pd.Timedelta(minutes=120)
            window = raw.loc[entry_time : end_time]
            if len(window) < 2: continue

            # [LA2-fix] Use first candle OPEN as entry reference, not signal bar close
            S0 = window.iloc[0]['open']
            r, exit_time = sim_dca_grid_long(window, S0, leverage=25.0)
            if r != 0.0:
                valid.append(max(r, -1.0))
                lock_until = exit_time

        arr  = np.array(valid)
        wr   = (arr > 0).mean() * 100 if len(arr) > 0 else 0
        apnl = arr.mean() * 100       if len(arr) > 0 else 0

        win_arr = (arr > 0).astype(float)
        if len(win_arr) >= 50:
            stab = pd.Series(win_arr).rolling(50).mean().dropna().std() * 100
        else:
            stab = float('nan')

        mo     = len(test_idx) * 5 / (60 * 24 * 30.44)
        sig_mo = len(arr) / mo
        top_str  = f"Top {100 - pct:.0f}%"
        stab_str = f"{stab:.1f}%" if not np.isnan(stab) else "n/a"
        print(f"{top_str:12} {sig_mo:12.1f} {wr:10.1f}% {apnl:+10.2f}% {stab_str:>10}")

        if not np.isnan(stab) and stab < best_stability:
            best_stability = stab; best_threshold = 100 - pct

    print(f"\n[LA1] Most stable Long threshold: Top {best_threshold}%")

    # ── Export MC params for the most stable threshold ────────────────────────
    ref_pct = 100 - best_threshold
    thr_ref = np.percentile(tr_probs[tr_probs > 0], ref_pct)
    sig_ref = test_idx[test_probs >= thr_ref]

    lock_until = sig_ref[0] - pd.Timedelta(minutes=1)
    mc = []
    for idx in sig_ref:
        if idx < lock_until: continue
        entry_time = idx + pd.Timedelta(minutes=5)
        end_time   = entry_time + pd.Timedelta(minutes=120)
        window = raw.loc[entry_time : end_time]
        if len(window) < 2: continue
        S0 = window.iloc[0]['open']   # [LA2-fix]
        r, exit_time = sim_dca_grid_long(window, S0, leverage=25.0)
        if r != 0.0:
            mc.append(max(r, -1.0)); lock_until = exit_time

    mc  = np.array(mc)
    mo  = len(test_idx) * 5 / (60 * 24 * 30.44)
    print(f"\n--- CALIBRATED LONG MC PARAMS (BTC 2yr, Top {best_threshold}%, lock+LA-clean) ---")
    print(f"Total non-overlapping trades : {len(mc)}")
    print(f"Trades/Month (BTC only)      : {len(mc)/mo:.1f}")
    print(f"Win Rate                     : {(mc>0).mean()*100:.2f}%")
    print(f"Avg PnL per trade            : {mc.mean()*100:+.2f}%")
    wins   = mc[mc > 0]; losses = mc[mc < 0]
    if len(wins):   print(f"Avg Win PnL                  : {wins.mean()*100:+.2f}%")
    if len(losses): print(f"Avg Loss PnL                 : {losses.mean()*100:+.2f}%")
    print(f"Std Dev                      : {mc.std()*100:.2f}%")
    print(f"\nApply 1.30x multi-asset mult -> {len(mc)/mo*1.30:.1f} trades/month")

if __name__ == '__main__':
    main()
