import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FUNDING_RATE_PER_8H = 0.0001
FEE = 0.0004
LEVERAGE = 25.0
H = 60
DATA_DIR = 'c:/Users/onepiece/Documents/_Garage/Ohhv2/data'

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df):
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    ma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    ub = ma + 2*std; lb = ma - 2*std; bw = ub - lb
    df['pct_b'] = (df['close'] - lb) / (bw + 1e-10)
    df['band_width'] = bw / (ma + 1e-10)
    df['rsi'] = rsi(df['close'])
    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / (df['close'] + 1e-10)
    df['macd_hist'] = (macd - sig) / (df['close'] + 1e-10)
    df['vel1'] = df['close'].pct_change(1)
    df['vel3'] = df['close'].pct_change(3)
    cr = df['high'] - df['low']
    df['uw'] = (df['high'] - df[['open','close']].max(axis=1)) / (cr + 1e-10)
    df['lw'] = (df[['open','close']].min(axis=1) - df['low']) / (cr + 1e-10)
    vol_ma = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (vol_ma + 1e-10)
    
    raw_1h = df['close'].resample('1h', label='right', closed='right').last().dropna()
    raw_4h = df['close'].resample('4h', label='right', closed='right').last().dropna()
    df['sma_1h'] = raw_1h.rolling(50).mean().reindex(df.index, method='ffill')
    df['sma_4h'] = raw_4h.rolling(50).mean().reindex(df.index, method='ffill')
    df['mtf_1h_dist'] = (df['close'] - df['sma_1h']) / (df['sma_1h'] + 1e-10)
    df['mtf_4h_dist'] = (df['close'] - df['sma_4h']) / (df['sma_4h'] + 1e-10)
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist','vel1','vel3','uw','lw','vol_ratio','mtf_1h_dist','mtf_4h_dist']

def compute_labels(df_part, direction):
    """Vectorized label computation — ~100x faster than Python loop."""
    close = df_part['close'].values
    high  = df_part['high'].values
    low   = df_part['low'].values
    n = len(df_part)
    target = np.zeros(n, dtype=int)
    # Rolling future max/min using numpy stride tricks
    for i in range(n - H):
        if direction == 'short':
            if low[i+1:i+H+1].min() <= close[i] * 0.9985:
                target[i] = 1
        else:
            if high[i+1:i+H+1].max() >= close[i] * 1.0015:
                target[i] = 1
    return target

def sim_dca_generic(window, S0, direction, tp_pct, sl_pct, dca_spacing):
    if direction == 'short':
        levels = [S0 * (1 + dca_spacing * i) for i in range(4)]
        sl_price = S0 * (1 + sl_pct)
        tp_mult = 1 - tp_pct
    else:
        levels = [S0 * (1 - dca_spacing * i) for i in range(4)]
        sl_price = S0 * (1 - sl_pct)
        tp_mult = 1 + tp_pct

    fills = []; avg_entry = tp_price = None
    entry_ts = window.index[0]

    for idx, row in window.iterrows():
        h, l = row['high'], row['low']
        
        if direction == 'short':
            nf = [lv for lv in levels if lv not in fills and h >= lv]
            for lv in nf: fills.append(lv)
            if len(nf) > 1:
                worst = max(nf); base = [f for f in fills if f not in nf]
                fills = base + [worst]*len(nf)
            if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry * tp_mult
            if not fills:
                if l <= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry * tp_mult
            if not fills: continue
            
            hold_m = (idx - entry_ts).total_seconds()/60
            funding = FUNDING_RATE_PER_8H*(hold_m/480)*LEVERAGE
            if h >= sl_price: return (avg_entry-sl_price)/avg_entry*LEVERAGE - FEE*LEVERAGE*2 - funding, idx
            if l <= tp_price: return (avg_entry-tp_price)/avg_entry*LEVERAGE - FEE*LEVERAGE*2 - funding, idx
        else:
            nf = [lv for lv in levels if lv not in fills and l <= lv]
            for lv in nf: fills.append(lv)
            if len(nf) > 1:
                worst = min(nf); base = [f for f in fills if f not in nf]
                fills = base + [worst]*len(nf)
            if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry * tp_mult
            if not fills:
                if h >= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry * tp_mult
            if not fills: continue
            
            hold_m = (idx - entry_ts).total_seconds()/60
            funding = FUNDING_RATE_PER_8H*(hold_m/480)*LEVERAGE
            if l <= sl_price: return (sl_price-avg_entry)/avg_entry*LEVERAGE - FEE*LEVERAGE*2 - funding, idx
            if h >= tp_price: return (tp_price-avg_entry)/avg_entry*LEVERAGE - FEE*LEVERAGE*2 - funding, idx

    if not fills: return 0.0, window.index[-1]
    hold_m = (window.index[-1]-entry_ts).total_seconds()/60
    funding = FUNDING_RATE_PER_8H*(hold_m/480)*LEVERAGE
    if direction == 'short':
        return (avg_entry-window.iloc[-1]['close'])/avg_entry*LEVERAGE - FEE*LEVERAGE*2 - funding, window.index[-1]
    else:
        return (window.iloc[-1]['close']-avg_entry)/avg_entry*LEVERAGE - FEE*LEVERAGE*2 - funding, window.index[-1]

def main():
    # FIX C10: Use long 1m.csv files (~6 months) instead of 30-day files
    coins = ['ADA','ATOM','AVAX','BNB','DOGE','DOT','ETH','LINK','LTC','SHIB','SOL','TRX','UNI','XRP']
    train_dfs = []
    val_dfs = []

    # FIX C11: Load BTC for correlation crash filter
    print("Loading BTC for crash filter...")
    btc_path = f'{DATA_DIR}/BTC-1m.csv'
    btc_raw = pd.read_csv(btc_path, usecols=['Timestamp','Close'])
    btc_raw.columns = ['timestamp','close']
    btc_raw['datetime'] = pd.to_datetime(btc_raw['timestamp'], unit='s')
    btc_raw.set_index('datetime', inplace=True)
    btc_raw = btc_raw[btc_raw.index >= pd.Timestamp('2024-01-01')].copy()
    btc_close = btc_raw['close'].astype(float)
    # 10-minute rolling return on BTC
    btc_ret_10m = btc_close.pct_change(10)  # 10 x 1-min candles
    
    print("Loading and computing features for 14 Altcoins (using 6-month 1m.csv files)...")
    for coin in coins:
        # FIX C10: prefer long 1m.csv over 30d files
        fpath_long  = f'{DATA_DIR}/{coin}USDT-1m.csv'
        fpath_short = f'{DATA_DIR}/{coin}USDT-30d-1m.csv'
        fpath = fpath_long if os.path.exists(fpath_long) else fpath_short
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        df.columns = [c.lower() for c in df.columns]
        
        # Use first 3 weeks as Train, last 2 weeks as Val
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else: continue
            
        df.set_index('datetime', inplace=True)
        df = add_features(df)
        df['coin'] = coin

        # 30 days train → ~60 days validation (maximum possible with 90-day files)
        split_point = df.index[0] + pd.Timedelta(days=30)
        train_df = df[df.index < split_point].copy()
        val_df = df[df.index >= split_point].copy()
        
        if len(train_df) > 5000 and len(val_df) > 5000:
            train_df['target_short'] = compute_labels(train_df, 'short')
            train_df['target_long'] = compute_labels(train_df, 'long')
            train_dfs.append(train_df)
            val_dfs.append(val_df)

    if not train_dfs:
        print("No valid train data found.")
        return

    full_train = pd.concat(train_dfs)
    full_val = pd.concat(val_dfs)
    
    print(f"Unified Train Set: {len(full_train)} rows.")
    print(f"Unified Val Set:   {len(full_val)} rows.")

    # Pre-sort val for consistent ordering
    full_val = full_val.sort_index()

    sc = StandardScaler()
    X_tr = sc.fit_transform(full_train[FCOLS].values)
    X_val = sc.transform(full_val[FCOLS].values)

    print("Training unified altcoin models...")
    model_s = LogisticRegression(class_weight='balanced', max_iter=500, n_jobs=-1)
    model_s.fit(X_tr, full_train['target_short'].values)

    model_l = LogisticRegression(class_weight='balanced', max_iter=500, n_jobs=-1)
    model_l.fit(X_tr, full_train['target_long'].values)

    tr_probs_s = model_s.predict_proba(X_tr)[:,1]
    tr_probs_l = model_l.predict_proba(X_tr)[:,1]
    thr_s = np.percentile(tr_probs_s[tr_probs_s > 0], 96.0)
    thr_l = np.percentile(tr_probs_l[tr_probs_l > 0], 97.0)

    full_val['prob_s'] = model_s.predict_proba(X_val)[:,1]
    full_val['prob_l'] = model_l.predict_proba(X_val)[:,1]

    # Build pre-grouped dict AFTER prob columns are assigned — O(1) lookups in sim loop
    val_by_time = {ts: grp for ts, grp in full_val.groupby(level=0)}
    
    # Altcoin Grids (wider due to volatility)
    configs = [
        (0.0060, 0.0100, 0.0020, "Tight (TP=0.6% SL=1.0%)"),
        (0.0075, 0.0125, 0.0025, "Mid (TP=0.75% SL=1.25%)"),
        (0.0100, 0.0150, 0.0030, "Wide (TP=1.0% SL=1.5%)"),
        (0.0125, 0.0200, 0.0040, "Extra (TP=1.25% SL=2.0%)"),
        (0.0150, 0.0250, 0.0050, "Huge (TP=1.5% SL=2.5%)")
    ]
    
    print("\nRunning Altcoin Basket Concurrency Simulation...")
    print("-" * 80)
    print(f"{'Config':<30} {'Trades':>6} {'Trades/Mo':>10} {'WinRate':>8} {'EV/Trade':>10}")
    print("-" * 80)
    
    days = (full_val.index[-1] - full_val.index[0]).total_seconds() / 86400
    mo = days / 30.4
    
    for tp, sl, sp, label in configs:
        lock_until = full_val.index[0] - pd.Timedelta(minutes=1)
        results = []
        
        times = list(val_by_time.keys())
        times.sort()

        for t_ts in times:
            if t_ts < lock_until: continue

            # FIX C11: BTC Crash Filter — O(log n) with .asof()
            btc_crash = False
            btc_10m_ret = btc_ret_10m.asof(t_ts)
            if pd.notna(btc_10m_ret) and btc_10m_ret < -0.015:
                btc_crash = True

            # Get all coins at this timestamp from pre-grouped dict
            slice_df = val_by_time[t_ts]
            
            # Find the strongest signal
            best_coin = None
            best_dir = None
            max_prob = 0
            
            for idx, row in slice_df.iterrows():
                if row['prob_s'] >= thr_s and row['prob_s'] > max_prob:
                    max_prob = row['prob_s']
                    best_dir = 'short'
                    best_coin = row['coin']
                # FIX C11: Block LONG signals during BTC crash
                if not btc_crash and row['prob_l'] >= thr_l and row['prob_l'] > max_prob:
                    max_prob = row['prob_l']
                    best_dir = 'long'
                    best_coin = row['coin']
                    
            if best_coin:
                entry_time = t_ts + pd.Timedelta(minutes=1)
                end_time = entry_time + pd.Timedelta(minutes=120)
                # Use pre-grouped coin data
                coin_df = full_val[full_val['coin'] == best_coin]
                window = coin_df.loc[entry_time : end_time]
                
                if len(window) < 2: continue
                S0 = window.iloc[0]['open']
                r, exit_time = sim_dca_generic(window, S0, best_dir, tp, sl, sp)
                if r != 0.0:
                    results.append(r)
                    lock_until = exit_time
                    
        arr = np.array(results)
        if len(arr) == 0: continue
        
        print(f"{label:<30} {len(arr):>6} {len(arr)/mo:>10.1f} {(arr>0).mean()*100:>7.2f}% {arr.mean()*100:>+9.4f}%")

if __name__ == '__main__':
    main()
