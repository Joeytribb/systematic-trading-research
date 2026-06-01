"""
Multi-Asset Concurrency Engine
==============================
Scans BTC, ETH, SOL, LINK, and DOGE simultaneously using the 
optimized MTF/Volume Logistic Regression model to measure the 
absolute multiplier on trade frequency with a strict Concurrency Lock.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['BTC-1m.csv', 'ETHUSDT-1m.csv', 'SOLUSDT-1m.csv', 'LINKUSDT-1m.csv', 'DOGEUSDT-1m.csv']
NAMES = ['BTC', 'ETH', 'SOL', 'LINK', 'DOGE']
DATA_DIR = 'c:/Users/onepiece/Documents/_Garage/Ohhv2/data/'
DAYS_TO_LOAD = 180

# ── FEATURE PIPELINE ─────────────────────────────────────────────────────────

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

# ── DCA GRID EXECUTION ENGINE ────────────────────────────────────────────────

def sim_dca_grid_short(window, S0, leverage=25.0, fee=0.0004):
    levels = [S0, S0*1.0015, S0*1.003, S0*1.0045]
    fills = []
    sl_price = S0 * 1.0075
    avg_entry = None; tp_price = None

    for idx, row in window.iterrows():
        high, low = row['high'], row['low']

        # [B5-fix] Track which levels get newly filled this candle
        newly_filled = []
        for lvl in levels:
            if lvl not in fills and high >= lvl:
                fills.append(lvl)
                newly_filled.append(lvl)

        # [B5-fix] Worst-case fill: if a violent wick fills multiple levels
        # in one candle, all those fills execute at the wick top, not the
        # idealized ladder prices. Replace same-candle fills with worst price.
        if len(newly_filled) > 1:
            worst = max(newly_filled)
            base  = [f for f in fills if f not in newly_filled]
            fills = base + [worst] * len(newly_filled)

        if fills:
            avg_entry = sum(fills) / len(fills)
            tp_price  = avg_entry * (1 - 0.005)

        if not fills:
            if low <= S0:
                fills.append(S0)
                avg_entry = S0
                tp_price  = avg_entry * (1 - 0.005)

        if not fills: continue

        if high >= sl_price:
            pct_change = (avg_entry - sl_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2), idx

        if low <= tp_price:
            pct_change = (avg_entry - tp_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2), idx

    if not fills: return 0.0, window.index[-1]
    pct_change = (avg_entry - window.iloc[-1]['close']) / avg_entry
    return pct_change * leverage - (fee * leverage * 2), window.index[-1]

# ── MAIN ENGINE ──────────────────────────────────────────────────────────────

def process_asset(filename):
    print(f"Loading & Processing {filename}...")
    raw = pd.read_csv(DATA_DIR + filename)
    raw.columns = [c.lower() for c in raw.columns]
    if raw['timestamp'].dtype == 'object' or pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    
    # Hardcoded absolute dates to guarantee perfectly aligned overlap
    start_date = pd.to_datetime('2026-03-01')
    end_date = pd.to_datetime('2026-05-15')
    raw = raw[(raw.index >= start_date) & (raw.index < end_date)]
    
    df = raw.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    H = 24
    df['target_short'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)
    
    df = add_advanced_features(df, raw)
    
    # Simple Train/Test split for speed (First 50% Train, Last 50% Test)
    split_idx = int(len(df) * 0.5)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:len(df)-H]
    
    X_tr = train_df[FCOLS].values; y_tr = train_df['target_short'].values
    X_te = test_df[FCOLS].values
    
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_te)[:,1]
    thr = np.percentile(probs, 98.0) # Top 2% Sweet Spot
    
    signals = pd.Series(0, index=test_df.index)
    signals[probs >= thr] = 1
    
    return raw, df, signals, test_df.index

def main():
    print("=== INITIALIZING MULTI-ASSET ENGINE ===")
    asset_data = {}
    
    # Process all assets
    for name, f in zip(NAMES, ASSETS):
        try:
            raw, df, signals, test_idx = process_asset(f)
            asset_data[name] = {'raw': raw, 'df': df, 'signals': signals}
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            
    active_names = list(asset_data.keys())
    if not active_names: return
    
    # Create aligned timeline from intersection of all test indices
    print("\nAligning Timelines...")
    common_idx = asset_data[active_names[0]]['signals'].index
    for name in active_names[1:]:
        common_idx = common_idx.intersection(asset_data[name]['signals'].index)
        
    print(f"Aligned Out-Of-Sample Period: {len(common_idx)} periods (~{len(common_idx)*5/60/24:.1f} days)")
    
    # Build Signal Matrix
    signal_matrix = pd.DataFrame(index=common_idx)
    for name in active_names:
        signal_matrix[name] = asset_data[name]['signals'].loc[common_idx]
        
    # Baseline: BTC Only (No overlap worries)
    btc_signals = signal_matrix['BTC'].sum()
    print(f"\nBaseline (BTC Only): {btc_signals} Total Signals")
    
    # ── CONCURRENCY SIMULATOR ──
    print("Running Event-Driven Concurrency Simulator...")
    
    lock_until = common_idx[0] - pd.Timedelta(minutes=1)
    results = []
    
    for current_time in common_idx:
        if current_time < lock_until:
            continue
            
        # Check for signals across all assets at this specific minute
        row = signal_matrix.loc[current_time]
        fired_assets = row[row == 1].index.tolist()
        
        if not fired_assets:
            continue
            
        # If multiple fire simultaneously, pick the first one (arbitrary tie-breaker)
        chosen_asset = fired_assets[0]
        
        # Execute Trade
        raw = asset_data[chosen_asset]['raw']
        df = asset_data[chosen_asset]['df']
        
        entry_time = current_time + pd.Timedelta(minutes=5)
        max_end_time = entry_time + pd.Timedelta(minutes=120)
        window = raw.loc[entry_time : max_end_time]
        
        if len(window) < 2: continue
        
        S0 = df.loc[current_time, 'close']
        r, exit_time = sim_dca_grid_short(window, S0, leverage=25.0)
        
        if r != 0.0:
            results.append(max(r, -1.0))
            # Lock the account until the trade officially closes
            lock_until = exit_time
            
    arr = np.array(results)
    total_trades = len(arr)
    wr = (arr > 0).mean() * 100 if total_trades > 0 else 0
    apnl = arr.mean() * 100 if total_trades > 0 else 0
    
    multiplier = total_trades / btc_signals if btc_signals > 0 else 0
    
    print("\n=== MULTI-ASSET SCANNER RESULTS ===")
    print(f"Total Executed Trades: {total_trades}")
    print(f"Signal Multiplier:     {multiplier:.2f}x vs BTC Alone")
    print(f"Win Rate:              {wr:.1f}%")
    print(f"Average PnL:           {apnl:+.2f}%")

if __name__ == '__main__':
    main()
