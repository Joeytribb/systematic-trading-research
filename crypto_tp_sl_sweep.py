import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Constants ---
FUNDING_RATE_PER_8H = 0.0001
FEE = 0.0004
LEVERAGE = 25.0
H = 60  # 60 minutes lookahead for targets

# --- Feature Engineering ---
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df):
    print("Computing features...")
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
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


def compute_labels(df_part, direction):
    print(f"Computing labels for {direction}...")
    target = [0] * len(df_part)
    idx_arr = df_part.index
    for i in range(len(df_part) - H):
        sub = df_part.iloc[i+1:i+H+1]
        close_val = df_part.iloc[i]['close']
        if direction == 'short' and sub['low'].min() <= close_val * 0.9985:
            target[i] = 1
        elif direction == 'long' and sub['high'].max() >= close_val * 1.0015:
            target[i] = 1
    return target

def main():
    # Load full dataset, filter to 2021+, then take 300k rows
    df_raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    df_raw = df_raw[df_raw['Timestamp'] >= 1609459200]  # Jan 1 2021
    df = df_raw.head(300000).copy()
    
    df.columns = [c.lower() for c in df.columns]
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    df = add_features(df)
    
    # Split into Train (first half) and Val (second half)
    split_idx = len(df) // 2
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    train_df['target_short'] = compute_labels(train_df, 'short')
    val_df['target_short'] = compute_labels(val_df, 'short')
    train_df['target_long'] = compute_labels(train_df, 'long')
    val_df['target_long'] = compute_labels(val_df, 'long')
    
    print(f"Training models on {len(train_df)} rows...")
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_df[FCOLS].values)
    X_val = sc.transform(val_df[FCOLS].values)
    
    model_s = LogisticRegression(class_weight='balanced', max_iter=500, n_jobs=-1)
    model_s.fit(X_tr, train_df['target_short'].values)
    
    model_l = LogisticRegression(class_weight='balanced', max_iter=500, n_jobs=-1)
    model_l.fit(X_tr, train_df['target_long'].values)
    
    # Get thresholds from train
    tr_probs_s = model_s.predict_proba(X_tr)[:,1]
    tr_probs_l = model_l.predict_proba(X_tr)[:,1]
    thr_s = np.percentile(tr_probs_s[tr_probs_s > 0], 96.0) # Top 4%
    thr_l = np.percentile(tr_probs_l[tr_probs_l > 0], 97.0) # Top 3%
    
    # Predict on Val
    val_df['prob_s'] = model_s.predict_proba(X_val)[:,1]
    val_df['prob_l'] = model_l.predict_proba(X_val)[:,1]
    
    configs = [
        # (tp_pct, sl_pct, dca_spacing, label)
        (0.0030, 0.0050, 0.0010, "Tight (TP=0.3% SL=0.5%)"),
        (0.0045, 0.0075, 0.0015, "Baseline (TP=0.45% SL=0.75%)"),
        (0.0060, 0.0100, 0.0020, "Mid (TP=0.6% SL=1.0%)"),
        (0.0075, 0.0125, 0.0025, "Wide (TP=0.75% SL=1.25%)"),
        (0.0100, 0.0150, 0.0030, "Extra Wide (TP=1.0% SL=1.5%)")
    ]
    
    print("\nRunning TP/SL Sweep on Validation Set (First 3 Months Out-of-Sample)")
    print("-" * 80)
    print(f"{'Config':<30} {'Trades':>6} {'WinRate':>8} {'EV/Trade':>10} {'Monthly Ret':>12}")
    print("-" * 80)
    
    for tp, sl, sp, label in configs:
        lock_until = val_df.index[0] - pd.Timedelta(minutes=1)
        results = []
        
        for i in range(len(val_df) - 60): # leave room for trade window
            t_ts = val_df.index[i]
            if t_ts < lock_until: continue
            
            ps = val_df.iloc[i]['prob_s']
            pl = val_df.iloc[i]['prob_l']
            
            if ps >= thr_s:
                entry_time = t_ts + pd.Timedelta(minutes=1)
                end_time = entry_time + pd.Timedelta(minutes=120)
                window = val_df.loc[entry_time : end_time]
                if len(window) < 2: continue
                S0 = window.iloc[0]['open']
                r, exit_time = sim_dca_generic(window, S0, 'short', tp, sl, sp)
                if r != 0.0:
                    results.append(r)
                    lock_until = exit_time
            elif pl >= thr_l:
                entry_time = t_ts + pd.Timedelta(minutes=1)
                end_time = entry_time + pd.Timedelta(minutes=120)
                window = val_df.loc[entry_time : end_time]
                if len(window) < 2: continue
                S0 = window.iloc[0]['open']
                r, exit_time = sim_dca_generic(window, S0, 'long', tp, sl, sp)
                if r != 0.0:
                    results.append(r)
                    lock_until = exit_time
                    
        arr = np.array(results)
        if len(arr) == 0: continue
        
        days = (val_df.index[-1] - val_df.index[0]).total_seconds() / 86400
        mo = days / 30.4
        monthly_ret = ((1 + arr.mean()) ** (len(arr)/mo) - 1) * 100
        print(f"{label:<30} {len(arr):>6} {(arr>0).mean()*100:>7.2f}% {arr.mean()*100:>+9.4f}% {monthly_ret:>+11.2f}%")

if __name__ == '__main__':
    main()
