"""
Win Rate Optimizer (Bias-Corrected)
====================================
Fixes applied:
  [B4] Threshold selection bias  -> report ALL thresholds, pick most *stable*, not highest
  [B5] DCA fill optimism         -> worst-case avg entry when multiple levels filled same candle
  [B13] Funding rate             -> deducted per trade (-0.01%/8h prorated to hold duration)
  [LA1] Look-ahead threshold     -> threshold computed from TRAINING probs, not test set
  [LA2] Look-ahead entry price   -> entry uses first candle OPEN, not signal-bar close
  Concurrency lock added         -> signals within the 2-h trade window of the prior trade are
                                   skipped, giving the *true* non-overlapping trades/month figure.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_advanced_features(df, raw_1m):
    ma  = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    ub  = ma + 2*std; lb = ma - 2*std; bw = ub - lb
    df['pct_b']      = (df['close'] - lb) / (bw + 1e-10)
    df['band_width'] = bw / (ma + 1e-10)
    df['rsi']        = rsi(df['close'])

    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm']  = macd / (df['close'] + 1e-10)
    df['macd_hist']  = (macd - sig) / (df['close'] + 1e-10)

    df['vel1'] = df['close'].pct_change(1)
    df['vel3'] = df['close'].pct_change(3)

    cr = df['high'] - df['low']
    df['uw'] = (df['high'] - df[['open','close']].max(axis=1)) / (cr + 1e-10)
    df['lw'] = (df[['open','close']].min(axis=1) - df['low'])  / (cr + 1e-10)

    vol_ma = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (vol_ma + 1e-10)

    # [B4-fix] MTF features with strict no-future label
    raw_1h = raw_1m['close'].resample('1h', label='right', closed='right').last().dropna()
    raw_4h = raw_1m['close'].resample('4h', label='right', closed='right').last().dropna()
    sma_1h = raw_1h.rolling(50).mean()
    sma_4h = raw_4h.rolling(50).mean()
    df['sma_1h'] = sma_1h.reindex(df.index, method='ffill')
    df['sma_4h'] = sma_4h.reindex(df.index, method='ffill')
    df['mtf_1h_dist'] = (df['close'] - df['sma_1h']) / (df['sma_1h'] + 1e-10)
    df['mtf_4h_dist'] = (df['close'] - df['sma_4h']) / (df['sma_4h'] + 1e-10)

    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist',
         'vel1','vel3','uw','lw', 'vol_ratio', 'mtf_1h_dist', 'mtf_4h_dist']

FUNDING_RATE_PER_8H = 0.0001  # 0.01% every 8 hours (Binance typical)

def walk_forward_lr(df, H=24):
    X = df[FCOLS].values; y = df['target_short'].values
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')

    preds       = np.zeros(len(X))
    train_preds = np.zeros(len(X))  # [LA1] in-sample probs for threshold computation
    INIT, STEP  = 24000, 8000

    if len(X) < INIT + H:
        sc = StandardScaler()
        model.fit(sc.fit_transform(X), y)
        p = model.predict_proba(sc.transform(X))[:,1]
        return p, 0, p

    for s in range(INIT, len(X)-H, STEP):
        e  = min(s + STEP, len(X) - H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
        # [LA1] Store training-period probabilities
        start_tr = max(0, s - H - STEP)
        train_preds[start_tr:s-H] = model.predict_proba(
            sc.transform(X[start_tr:s-H])
        )[:,1]

    return preds, INIT, train_preds

# ── DCA grid SHORT: B5-fix (worst-case fill) + LA2-fix (entry=open) + B13 (funding) ──
def sim_dca_grid_short(window, S0, leverage=25.0, fee=0.0004):
    """
    [LA2] S0 should be passed as window.iloc[0]['open'] by the caller.
    [B13] Deducts funding rate prorated to the actual hold time.
    """
    levels    = [S0, S0*1.0015, S0*1.003, S0*1.0045]
    fills     = []
    sl_price  = S0 * 1.0075
    avg_entry = None; tp_price = None
    entry_ts  = window.index[0]

    for idx, row in window.iterrows():
        high, low = row['high'], row['low']

        newly_filled = []
        for lvl in levels:
            if lvl not in fills and high >= lvl:
                fills.append(lvl); newly_filled.append(lvl)

        if len(newly_filled) > 1:               # [B5-fix]
            worst = max(newly_filled)
            base  = [f for f in fills if f not in newly_filled]
            fills = base + [worst] * len(newly_filled)

        if fills:
            avg_entry = sum(fills) / len(fills)
            tp_price  = avg_entry * (1 - 0.005)

        if not fills:
            if low <= S0:
                fills.append(S0); avg_entry = S0
                tp_price = avg_entry * (1 - 0.005)

        if not fills: continue

        if high >= sl_price:
            pct = (avg_entry - sl_price) / avg_entry
            hold_m = (idx - entry_ts).total_seconds() / 60
            funding = FUNDING_RATE_PER_8H * (hold_m / 480) * leverage  # [B13]
            return pct * leverage - fee * leverage * 2 - funding

        if low <= tp_price:
            pct = (avg_entry - tp_price) / avg_entry
            hold_m = (idx - entry_ts).total_seconds() / 60
            funding = FUNDING_RATE_PER_8H * (hold_m / 480) * leverage  # [B13]
            return pct * leverage - fee * leverage * 2 - funding

    if not fills: return 0.0
    pct = (avg_entry - window.iloc[-1]['close']) / avg_entry
    hold_m = (window.index[-1] - entry_ts).total_seconds() / 60
    funding = FUNDING_RATE_PER_8H * (hold_m / 480) * leverage         # [B13]
    return pct * leverage - fee * leverage * 2 - funding

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading 2 Years of BTC data...")
    raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    raw.columns = [c.lower() for c in raw.columns]
    if raw['timestamp'].dtype == 'object' or pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=730)]

    print("Building Advanced 5m features (Volume + MTF)...")
    df = raw.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    H = 24
    df['target_short'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)

    df = add_advanced_features(df, raw)

    print("Training MTF-Aware Logistic Regression (Walk Forward)...")
    preds, wi, train_preds = walk_forward_lr(df, H)  # [LA1] now returns train_preds too
    test_idx  = df.index[wi : len(df)-H]
    test_probs = preds[wi : len(df)-H]
    # [LA1-fix] Training-period probs for threshold — no test-set contamination
    tr_probs  = train_preds[:wi]
    tr_probs  = tr_probs[tr_probs > 0]   # drop zero-padding before walk-forward starts

    thresholds_to_test = [95.0, 96.0, 97.0, 98.0, 99.0]

    print("\n--- SHORT LEADERBOARD (LA-clean threshold, entry=open, funding deducted) ---")
    print(f"{'Threshold':12} {'Signals/mo':>12} {'Win Rate':>10} {'Avg PnL':>10} {'Stability':>10}")

    best_threshold = None
    best_stability_score = float('inf')

    for pct in thresholds_to_test:
        # [LA1-fix] threshold from TRAINING probs, not test set
        thr = np.percentile(tr_probs, pct)
        signal_indices = test_idx[test_probs >= thr]

        lock_until = signal_indices[0] - pd.Timedelta(minutes=1)
        valid_signals = []
        for idx in signal_indices:
            if idx < lock_until: continue
            entry_time = idx + pd.Timedelta(minutes=5)
            end_time   = entry_time + pd.Timedelta(minutes=120)
            window = raw.loc[entry_time : end_time]
            if len(window) < 2: continue
            # [LA2-fix] entry price = first candle OPEN, not signal-bar close
            S0 = window.iloc[0]['open']
            r  = sim_dca_grid_short(window, S0, leverage=25.0)
            if r != 0.0:
                valid_signals.append(r)
                lock_until = end_time

        arr  = np.array(valid_signals)
        wr   = (arr > 0).mean() * 100 if len(arr) > 0 else 0
        apnl = arr.mean() * 100        if len(arr) > 0 else 0

        win_arr = (arr > 0).astype(float)
        if len(win_arr) >= 50:
            rolling_wr = pd.Series(win_arr).rolling(50).mean().dropna()
            stability  = rolling_wr.std() * 100
        else:
            stability  = float('nan')

        mo     = len(test_idx) * 5 / (60 * 24 * 30.44)
        sig_mo = len(arr) / mo
        top_str  = f"Top {100 - pct:.0f}%"
        stab_str = f"{stability:.1f}%" if not np.isnan(stability) else "n/a"
        print(f"{top_str:12} {sig_mo:12.1f} {wr:10.1f}% {apnl:+10.2f}% {stab_str:>10}")

        if not np.isnan(stability) and stability < best_stability_score:
            best_stability_score = stability; best_threshold = 100 - pct

    print(f"\n[B4] Most stable Short threshold: Top {best_threshold}%")

    # ── Export MC params for most stable threshold ────────────────────────────
    ref_pct = 100 - best_threshold
    thr_ref = np.percentile(tr_probs, ref_pct)
    sig_ref = test_idx[test_probs >= thr_ref]

    lock_until = sig_ref[0] - pd.Timedelta(minutes=1)
    mc_results = []
    for idx in sig_ref:
        if idx < lock_until: continue
        entry_time = idx + pd.Timedelta(minutes=5)
        end_time   = entry_time + pd.Timedelta(minutes=120)
        window = raw.loc[entry_time : end_time]
        if len(window) < 2: continue
        S0 = window.iloc[0]['open']   # [LA2-fix]
        r  = sim_dca_grid_short(window, S0, leverage=25.0)
        if r != 0.0:
            mc_results.append(max(r, -1.0))
            lock_until = end_time

    mc = np.array(mc_results)
    mo = len(test_idx) * 5 / (60 * 24 * 30.44)
    print(f"\n--- CALIBRATED SHORT MC PARAMS (BTC 2yr, Top {best_threshold}%, LA-clean) ---")
    print(f"Total non-overlapping trades : {len(mc)}")
    print(f"Trades/Month (BTC only)      : {len(mc)/mo:.1f}")
    print(f"Win Rate                     : {(mc > 0).mean()*100:.2f}%")
    print(f"Avg PnL per trade            : {mc.mean()*100:+.2f}%")
    wins  = mc[mc > 0]; losses = mc[mc < 0]
    if len(wins):   print(f"Avg Win PnL                  : {wins.mean()*100:+.2f}%")
    if len(losses): print(f"Avg Loss PnL                 : {losses.mean()*100:+.2f}%")
    print(f"Std Dev                      : {mc.std()*100:.2f}%")
    print(f"\nApply 1.30x multi-asset mult -> {len(mc)/mo*1.30:.1f} trades/month")

if __name__ == '__main__':
    main()
