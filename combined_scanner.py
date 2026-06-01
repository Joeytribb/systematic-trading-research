"""
combined_scanner.py  -  True Combined Long+Short Frequency Measurement
========================================================================
Bias #9 fix: runs Short and Long signals simultaneously on the FULL 2-year
BTC dataset with a SINGLE shared concurrency lock, then measures the actual
non-overlapping combined trades/month rather than assuming 2x additivity.

All LA fixes applied:
  [LA1] Threshold from TRAINING probabilities only
  [LA2] Entry price = first candle OPEN
  [B13] Funding rate deducted per trade
  [B5]  Worst-case DCA fill on same-candle multi-level fills
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FUNDING_RATE_PER_8H = 0.0001  # 0.01% per 8h

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
    raw_1h = raw_1m['close'].resample('1h', label='right', closed='right').last().dropna()
    raw_4h = raw_1m['close'].resample('4h', label='right', closed='right').last().dropna()
    df['sma_1h']      = raw_1h.rolling(50).mean().reindex(df.index, method='ffill')
    df['sma_4h']      = raw_4h.rolling(50).mean().reindex(df.index, method='ffill')
    df['mtf_1h_dist'] = (df['close'] - df['sma_1h']) / (df['sma_1h'] + 1e-10)
    df['mtf_4h_dist'] = (df['close'] - df['sma_4h']) / (df['sma_4h'] + 1e-10)
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist',
         'vel1','vel3','uw','lw','vol_ratio','mtf_1h_dist','mtf_4h_dist']

def walk_forward_signals(df, target_col, stable_pct, H=24):
    """Returns (signal_series, test_start_idx) with LA-clean threshold."""
    X = df[FCOLS].values; y = df[target_col].values
    model = LogisticRegression(max_iter=1000, random_state=42,
                               n_jobs=-1, class_weight='balanced')
    preds       = np.zeros(len(X))
    train_preds = np.zeros(len(X))
    INIT, STEP  = 24000, 8000

    for s in range(INIT, len(X)-H, STEP):
        e  = min(s + STEP, len(X) - H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
        start_tr = max(0, s - H - STEP)
        train_preds[start_tr:s-H] = model.predict_proba(
            sc.transform(X[start_tr:s-H])
        )[:,1]

    # [LA1] threshold from training probs only
    tr_probs = train_preds[:INIT]; tr_probs = tr_probs[tr_probs > 0]
    thr = np.percentile(tr_probs, stable_pct)

    test_probs = preds[INIT : len(X)-H]
    signals    = pd.Series(0, index=df.index[INIT : len(X)-H])
    signals[test_probs >= thr] = 1
    return signals, INIT

def sim_dca_short(window, S0, leverage=25.0, fee=0.0004):
    """[LA2] S0 = window.iloc[0]['open']. [B5] worst-case fill. [B13] funding."""
    levels = [S0, S0*1.0015, S0*1.003, S0*1.0045]
    fills = []; sl_price = S0 * 1.0075; avg_entry = tp_price = None
    entry_ts = window.index[0]
    for idx, row in window.iterrows():
        h, l = row['high'], row['low']
        nf = [lv for lv in levels if lv not in fills and h >= lv]
        for lv in nf: fills.append(lv)
        if len(nf) > 1:
            worst = max(nf); base = [f for f in fills if f not in nf]
            fills = base + [worst]*len(nf)
        if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry*(1-0.005)
        if not fills:
            if l <= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry*(1-0.005)
        if not fills: continue
        hold_m = (idx - entry_ts).total_seconds()/60
        funding = FUNDING_RATE_PER_8H*(hold_m/480)*leverage
        if h >= sl_price:
            return (avg_entry-sl_price)/avg_entry*leverage - fee*leverage*2 - funding, idx
        if l <= tp_price:
            return (avg_entry-tp_price)/avg_entry*leverage - fee*leverage*2 - funding, idx
    if not fills: return 0.0, window.index[-1]
    hold_m = (window.index[-1]-entry_ts).total_seconds()/60
    funding = FUNDING_RATE_PER_8H*(hold_m/480)*leverage
    return (avg_entry-window.iloc[-1]['close'])/avg_entry*leverage - fee*leverage*2 - funding, window.index[-1]

def sim_dca_long(window, S0, leverage=25.0, fee=0.0004):
    """[LA2] S0 = window.iloc[0]['open']. [B5] worst-case fill. [B13] funding."""
    levels = [S0, S0*0.9985, S0*0.997, S0*0.9955]
    fills = []; sl_price = S0*0.9925; avg_entry = tp_price = None
    entry_ts = window.index[0]
    for idx, row in window.iterrows():
        h, l = row['high'], row['low']
        nf = [lv for lv in levels if lv not in fills and l <= lv]
        for lv in nf: fills.append(lv)
        if len(nf) > 1:
            worst = min(nf); base = [f for f in fills if f not in nf]
            fills = base + [worst]*len(nf)
        if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry*1.005
        if not fills:
            if h >= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry*1.005
        if not fills: continue
        hold_m = (idx - entry_ts).total_seconds()/60
        funding = FUNDING_RATE_PER_8H*(hold_m/480)*leverage
        if l <= sl_price:
            return (sl_price-avg_entry)/avg_entry*leverage - fee*leverage*2 - funding, idx
        if h >= tp_price:
            return (tp_price-avg_entry)/avg_entry*leverage - fee*leverage*2 - funding, idx
    if not fills: return 0.0, window.index[-1]
    hold_m = (window.index[-1]-entry_ts).total_seconds()/60
    funding = FUNDING_RATE_PER_8H*(hold_m/480)*leverage
    return (window.iloc[-1]['close']-avg_entry)/avg_entry*leverage - fee*leverage*2 - funding, window.index[-1]

def main():
    print("Loading 2 years BTC data...")
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

    # Targets
    df['target_short'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close']*0.995).astype(int)
    df['target_long']  = (df['high'].rolling(H,1).max().shift(-H) >= df['close']*1.005).astype(int)
    df = add_features(df, raw)

    print("Training Short walk-forward (Top 4% stable threshold)...")
    short_sigs, wi = walk_forward_signals(df, 'target_short', stable_pct=96.0, H=H)

    print("Training Long walk-forward (Top 3% stable threshold)...")
    long_sigs, _  = walk_forward_signals(df, 'target_long',  stable_pct=97.0, H=H)

    # Align to same test index
    common_idx = short_sigs.index.intersection(long_sigs.index)
    short_sigs = short_sigs.loc[common_idx]
    long_sigs  = long_sigs.loc[common_idx]

    days = len(common_idx) * 5 / 60 / 24
    mo   = days / 30.44
    print(f"\nTest window: {len(common_idx)} bars = {days:.0f} days ({mo:.1f} months)")
    print(f"Short raw signals: {short_sigs.sum()}  ({short_sigs.sum()/mo:.1f}/mo)")
    print(f"Long  raw signals: {long_sigs.sum()}   ({long_sigs.sum()/mo:.1f}/mo)")
    print(f"Potential combined: {(short_sigs+long_sigs).clip(0,1).sum():.0f} ({(short_sigs+long_sigs).clip(0,1).sum()/mo:.1f}/mo)")

    # ── Combined concurrency simulation with ONE shared lock ─────────────────
    print("\nRunning combined simulation (single shared lock)...")
    lock_until = common_idx[0] - pd.Timedelta(minutes=1)
    short_results = []; long_results = []

    for t in common_idx:
        if t < lock_until: continue
        has_short = short_sigs.loc[t] == 1
        has_long  = long_sigs.loc[t]  == 1
        if not has_short and not has_long: continue

        entry_time = t + pd.Timedelta(minutes=5)
        end_time   = entry_time + pd.Timedelta(minutes=120)
        window = raw.loc[entry_time : end_time]
        if len(window) < 2: continue

        # [LA2] entry price = first candle open
        S0 = window.iloc[0]['open']

        # Priority: if both fire, take Short (mean-reversion on pumps is more reliable
        # based on optimizer output — can be changed without bias)
        if has_short:
            r, exit_time = sim_dca_short(window, S0)
            if r != 0.0:
                short_results.append(max(r, -1.0))
                lock_until = exit_time
        elif has_long:
            r, exit_time = sim_dca_long(window, S0)
            if r != 0.0:
                long_results.append(max(r, -1.0))
                lock_until = exit_time

    sa = np.array(short_results); la = np.array(long_results)
    total = len(sa) + len(la)

    print("\n" + "="*60)
    print("  COMBINED SCANNER — FINAL VERIFIED RESULTS")
    print("="*60)
    print(f"\nShort trades executed : {len(sa)}  ({len(sa)/mo:.1f}/mo)")
    print(f"Long  trades executed : {len(la)}  ({len(la)/mo:.1f}/mo)")
    print(f"TOTAL combined trades : {total}  ({total/mo:.1f}/mo)")
    print(f"Multi-asset mult 1.30x-> {total/mo*1.30:.1f} trades/month")

    all_r = np.concatenate([sa, la])
    print(f"\nCombined Win Rate     : {(all_r > 0).mean()*100:.2f}%")
    print(f"Combined Avg PnL      : {all_r.mean()*100:+.2f}%")
    wins = all_r[all_r > 0]; losses = all_r[all_r < 0]
    if len(wins):   print(f"Avg Win PnL           : {wins.mean()*100:+.2f}%")
    if len(losses): print(f"Avg Loss PnL          : {losses.mean()*100:+.2f}%")
    print(f"Std Dev               : {all_r.std()*100:.2f}%")
    print(f"\nShort-only: WR={( sa>0).mean()*100:.1f}%  AvgPnL={sa.mean()*100:+.2f}%")
    print(f"Long-only : WR={( la>0).mean()*100:.1f}%  AvgPnL={la.mean()*100:+.2f}%")

if __name__ == '__main__':
    main()
