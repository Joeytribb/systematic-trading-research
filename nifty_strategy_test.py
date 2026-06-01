"""
nifty_strategy_test.py  -  Walk-Forward Validation on Nifty 50 Index
=====================================================================
Tweaks the 5m mean-reversion DCA grid strategy to match the Indian market:
  1. Market Hours: Trades only between 09:30 and 14:15 IST.
  2. Intraday Square-Off: Positions closed at 15:15 IST to prevent overnight gap risk.
  3. No Volume: Replaces volume ratio with rolling range ratio (ATR-equivalent).
  4. Lower Fees: 0.02% round-trip transaction costs (standard Indian tax/STT).
  5. Recalibrated Grid: TP = 0.15%, SL = 0.25%, DCA spacing = 0.05% (5x smaller than crypto).
  6. Leverage: 8x (standard SEBI margin requirements for index futures).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FEES_ROUND_TRIP = 0.0002  # 0.02% (STT, GST, Exchange charges, Broker clearing)
LEVERAGE = 8.0            # SEBI standard (10-12% SPAN+Exposure margin)

def rsi(s, p=14):
    d = s.diff()
    g =  d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df):
    # Bollinger Bands
    ma  = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    ub = ma + 2*std; lb = ma - 2*std; bw = ub - lb
    df['pct_b']      = (df['close'] - lb) / (bw + 1e-10)
    df['band_width'] = bw / (ma + 1e-10)
    
    # RSI
    df['rsi'] = rsi(df['close'])
    
    # MACD
    e12  = df['close'].ewm(span=12, adjust=False).mean()
    e26  = df['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / (df['close'] + 1e-10)
    df['macd_hist'] = (macd - sig) / (df['close'] + 1e-10)
    
    # Velocity
    df['vel1'] = df['close'].pct_change(1)
    df['vel3'] = df['close'].pct_change(3)
    
    # Wicks
    cr = df['high'] - df['low']
    df['uw'] = (df['high'] - df[['open','close']].max(axis=1)) / (cr + 1e-10)
    df['lw'] = (df[['open','close']].min(axis=1) - df['low'])  / (cr + 1e-10)
    
    # Replaces volume ratio with price range ratio (volatility metric)
    rng = df['high'] - df['low']
    rng_ma = rng.rolling(20).mean()
    df['range_ratio'] = rng / (rng_ma + 1e-10)
    
    # MTF SMA distances (computed from index itself)
    raw_15m = df['close'].resample('15min', label='right', closed='right').last().dropna()
    raw_1h = df['close'].resample('1h', label='right', closed='right').last().dropna()
    df['sma_15m']     = raw_15m.rolling(20).mean().reindex(df.index, method='ffill')
    df['sma_1h']      = raw_1h.rolling(20).mean().reindex(df.index, method='ffill')
    df['mtf_15m_dist'] = (df['close'] - df['sma_15m']) / (df['sma_15m'] + 1e-10)
    df['mtf_1h_dist'] = (df['close'] - df['sma_1h']) / (df['sma_1h'] + 1e-10)
    
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist',
         'vel1','vel3','uw','lw','range_ratio','mtf_15m_dist','mtf_1h_dist']

def walk_forward_signals(df, target_col, stable_pct, H=12):
    """Walk-forward splits with LA-clean training threshold."""
    X = df[FCOLS].values; y = df[target_col].values
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
    
    preds = np.zeros(len(X))
    train_preds = np.zeros(len(X))
    
    # Small test dataset (4280 rows), so we use a smaller start split
    INIT = 2000
    STEP = 400
    
    for s in range(INIT, len(X)-H, STEP):
        e = min(s + STEP, len(X) - H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
        
        start_tr = max(0, s - H - STEP)
        train_preds[start_tr:s-H] = model.predict_proba(sc.transform(X[start_tr:s-H]))[:,1]
        
    tr_probs = train_preds[:INIT]; tr_probs = tr_probs[tr_probs > 0]
    thr = np.percentile(tr_probs, stable_pct)
    
    test_probs = preds[INIT : len(X)-H]
    signals = pd.Series(0, index=df.index[INIT : len(X)-H])
    signals[test_probs >= thr] = 1
    return signals, INIT

def sim_dca_short(window, S0):
    """Nifty short DCA execution with intraday square-off at 15:15 IST."""
    levels = [S0, S0*1.0005, S0*1.0010, S0*1.0015] # 0.05% spacing
    sl_price = S0 * 1.0025                          # 0.25% stop loss
    fills = []; avg_entry = tp_price = None
    
    for idx, row in window.iterrows():
        # Intraday square-off at 15:15 IST
        if idx.time() >= pd.to_datetime('15:15:00').time():
            if not fills: return 0.0, idx
            # Close at current market close price
            return (avg_entry - row['close'])/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
        h, l = row['high'], row['low']
        nf = [lv for lv in levels if lv not in fills and h >= lv]
        for lv in nf: fills.append(lv)
        
        # Worst-case fill adjustment
        if len(nf) > 1:
            worst = max(nf); base = [f for f in fills if f not in nf]
            fills = base + [worst]*len(nf)
            
        if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry*(1-0.0015) # 0.15% TP
        if not fills:
            if l <= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry*(1-0.0015)
        if not fills: continue
        
        if h >= sl_price:
            return (avg_entry - sl_price)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
        if l <= tp_price:
            return (avg_entry - tp_price)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
    if not fills: return 0.0, window.index[-1]
    return (avg_entry - window.iloc[-1]['close'])/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, window.index[-1]

def sim_dca_long(window, S0):
    """Nifty long DCA execution with intraday square-off at 15:15 IST."""
    levels = [S0, S0*0.9995, S0*0.9990, S0*0.9985]  # 0.05% spacing
    sl_price = S0 * 0.9975                          # 0.25% stop loss
    fills = []; avg_entry = tp_price = None
    
    for idx, row in window.iterrows():
        # Intraday square-off at 15:15 IST
        if idx.time() >= pd.to_datetime('15:15:00').time():
            if not fills: return 0.0, idx
            return (row['close'] - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
        h, l = row['high'], row['low']
        nf = [lv for lv in levels if lv not in fills and l <= lv]
        for lv in nf: fills.append(lv)
        
        # Worst-case fill adjustment
        if len(nf) > 1:
            worst = min(nf); base = [f for f in fills if f not in nf]
            fills = base + [worst]*len(nf)
            
        if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry*1.0015 # 0.15% TP
        if not fills:
            if h >= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry*1.0015
        if not fills: continue
        
        if l <= sl_price:
            return (sl_price - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
        if h >= tp_price:
            return (tp_price - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
    if not fills: return 0.0, window.index[-1]
    return (window.iloc[-1]['close'] - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, window.index[-1]

def main():
    print("Loading Nifty 50 5m dataset...")
    df = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/NIFTY50-5m.csv')
    df.columns = [c.lower() for c in df.columns]
    
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Asia/Kolkata')
    df.set_index('datetime', inplace=True); df.sort_index(inplace=True)
    
    # S&P 500 equivalent: check if market is open and restrict target computing
    H = 12  # 1 hour prediction window
    
    print("Calculating Nifty features and forward labels...")
    df = add_features(df)
    
    # Target definition (Hits 0.15% move before 15:15 same day)
    target_short_list = [0] * len(df)
    target_long_list = [0] * len(df)
    
    # Compute labels correctly respecting time limitations (only target same-day intraday)
    for i in range(len(df) - H):
        curr_time = df.index[i]
        # Only trade during peak market hours
        if curr_time.time() < pd.to_datetime('09:30:00').time() or curr_time.time() > pd.to_datetime('14:15:00').time():
            continue
            
        # Get windows for today
        end_idx = i + H
        sub_df = df.iloc[i+1 : end_idx+1]
        
        # Keep only times <= 15:15 on same day
        sub_df = sub_df[sub_df.index.date == curr_time.date()]
        sub_df = sub_df[sub_df.index.time <= pd.to_datetime('15:15:00').time()]
        if len(sub_df) == 0:
            continue
            
        close_val = df.iloc[i]['close']
        if sub_df['low'].min() <= close_val * 0.9985:
            target_short_list[i] = 1
        if sub_df['high'].max() >= close_val * 1.0015:
            target_long_list[i] = 1
            
    df['target_short'] = target_short_list
    df['target_long'] = target_long_list

    print("Running Short walk-forward (Top 4% stable threshold)...")
    short_sigs, INIT = walk_forward_signals(df, 'target_short', stable_pct=96.0, H=H)
    
    print("Running Long walk-forward (Top 4% stable threshold)...")
    long_sigs, _ = walk_forward_signals(df, 'target_long', stable_pct=96.0, H=H)
    
    common_idx = short_sigs.index.intersection(long_sigs.index)
    short_sigs = short_sigs.loc[common_idx]
    long_sigs = long_sigs.loc[common_idx]
    
    # Concurrency Lock Simulation
    print("\nRunning joint simulation under single shared lock...")
    lock_until = common_idx[0] - pd.Timedelta(minutes=1)
    short_results = []; long_results = []
    
    for t in common_idx:
        # Only trade between 09:30 and 14:15 IST
        if t.time() < pd.to_datetime('09:30:00').time() or t.time() > pd.to_datetime('14:15:00').time():
            continue
        if t < lock_until:
            continue
            
        has_short = short_sigs.loc[t] == 1
        has_long = long_sigs.loc[t] == 1
        if not has_short and not has_long:
            continue
            
        entry_time = t + pd.Timedelta(minutes=5)
        # Scan window up to 15:20 IST on same day
        end_time = entry_time + pd.Timedelta(minutes=60)
        window = df.loc[entry_time : end_time]
        window = window[window.index.date == entry_time.date()]
        if len(window) < 2:
            continue
            
        S0 = window.iloc[0]['open']
        
        # Priority: Short
        if has_short:
            r, exit_time = sim_dca_short(window, S0)
            if r != 0.0:
                short_results.append(r)
                lock_until = exit_time
        elif has_long:
            r, exit_time = sim_dca_long(window, S0)
            if r != 0.0:
                long_results.append(r)
                lock_until = exit_time

    sa = np.array(short_results); la = np.array(long_results)
    total = len(sa) + len(la)
    
    # Calculate timeframe
    days = (common_idx[-1] - common_idx[0]).days
    trading_days = days * (5/7) # approximate trading days
    mo = trading_days / 20.0
    
    print("\n" + "="*60)
    print("  NIFTY 50 TRADING RESULTS — INTRADAY DCA GRID (8x Leverage)")
    print("="*60)
    print(f"Test period           : {trading_days:.1f} trading days ({mo:.1f} months)")
    print(f"Short trades executed : {len(sa)} ({len(sa)/mo:.1f}/mo)")
    print(f"Long trades executed  : {len(la)} ({len(la)/mo:.1f}/mo)")
    print(f"TOTAL combined trades : {total} ({total/mo:.1f}/mo)")
    
    all_r = np.concatenate([sa, la])
    if len(all_r) == 0:
        print("No trades executed.")
        return
        
    print(f"\nCombined Win Rate     : {(all_r > 0).mean()*100:.2f}%")
    print(f"Combined Avg PnL      : {all_r.mean()*100:+.2f}%")
    wins = all_r[all_r > 0]; losses = all_r[all_r < 0]
    if len(wins):   print(f"Avg Win PnL           : {wins.mean()*100:+.2f}%")
    if len(losses): print(f"Avg Loss PnL          : {losses.mean()*100:+.2f}%")
    print(f"Expected Value / Trade: {all_r.mean()*100:+.2f}% (incl. transaction charges)")
    
if __name__ == '__main__':
    main()
