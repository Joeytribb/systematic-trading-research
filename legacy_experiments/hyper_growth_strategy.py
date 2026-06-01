"""
Hyper Growth Strategy
=====================
Integrating Dual-Directional Hunting (Long + Short), Adaptive Volatility Leverage, 
and Binance Funding Rate Skimming over 2 years of data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── INDICATORS ────────────────────────────────────────────────────────────────
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def calc_atr(df, p=14):
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low']  - df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def add_features(df):
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
    
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist',
         'vel1','vel3','uw','lw']

def walk_forward_lr(df, H=24, target_col='target_short'):
    X = df[FCOLS].values; y = df[target_col].values
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    
    preds = np.zeros(len(X))
    INIT, STEP = 24000, 8000 # Larger chunks for 2 years of data
    
    if len(X) < INIT + H:
        sc = StandardScaler()
        model.fit(sc.fit_transform(X), y)
        preds = model.predict_proba(sc.transform(X))[:,1]
        return preds, 0
        
    for s in range(INIT, len(X)-H, STEP):
        e = min(s+STEP, len(X)-H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
            
    return preds, INIT

# ── DCA GRID EXECUTION ENGINES ────────────────────────────────────────────────

def determine_leverage(atr_val, atr_thresholds):
    if pd.isna(atr_val): return 25.0
    p25, p75 = atr_thresholds
    if atr_val <= p25: return 50.0  # Low vol, high lev
    elif atr_val >= p75: return 10.0 # High vol, low lev
    return 25.0 # Normal

def get_funding_bonus(S0, sma200, is_short):
    # Proxy: If price > 200 SMA, market is bullish, Longs pay Shorts.
    # We assume a fixed +0.01% bonus for trading with the funding rate.
    if pd.isna(sma200): return 0.0
    if is_short and S0 > sma200:
        return 0.0001
    if not is_short and S0 < sma200:
        return 0.0001
    return 0.0

def sim_dca_grid_short(window, S0, leverage, fee=0.0004, funding_bonus=0.0):
    levels = [S0, S0*1.0015, S0*1.003, S0*1.0045]
    fills = []
    sl_price = S0 * 1.0075 
    avg_entry = None; tp_price = None
    
    for _, row in window.iterrows():
        high, low = row['high'], row['low']
        
        for lvl in levels:
            if lvl not in fills and high >= lvl:
                fills.append(lvl)
                avg_entry = sum(fills) / len(fills)
                tp_price = avg_entry * (1 - 0.005)
                
        if not fills:
            if low <= S0:
                fills.append(S0)
                avg_entry = S0
                tp_price = avg_entry * (1 - 0.005)
        
        if not fills: continue
        
        if high >= sl_price:
            pct_change = (avg_entry - sl_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2) + (funding_bonus * leverage)
            
        if low <= tp_price:
            pct_change = (avg_entry - tp_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2) + (funding_bonus * leverage)
            
    if not fills: return 0.0
    pct_change = (avg_entry - window.iloc[-1]['close']) / avg_entry
    return pct_change * leverage - (fee * leverage * 2) + (funding_bonus * leverage)

def sim_dca_grid_long(window, S0, leverage, fee=0.0004, funding_bonus=0.0):
    levels = [S0, S0*0.9985, S0*0.997, S0*0.9955]
    fills = []
    sl_price = S0 * 0.9925 
    avg_entry = None; tp_price = None
    
    for _, row in window.iterrows():
        high, low = row['high'], row['low']
        
        for lvl in levels:
            if lvl not in fills and low <= lvl:
                fills.append(lvl)
                avg_entry = sum(fills) / len(fills)
                tp_price = avg_entry * (1 + 0.005)
                
        if not fills:
            if high >= S0:
                fills.append(S0)
                avg_entry = S0
                tp_price = avg_entry * (1 + 0.005)
        
        if not fills: continue
        
        if low <= sl_price:
            pct_change = (sl_price - avg_entry) / avg_entry
            return pct_change * leverage - (fee * leverage * 2) + (funding_bonus * leverage)
            
        if high >= tp_price:
            pct_change = (tp_price - avg_entry) / avg_entry
            return pct_change * leverage - (fee * leverage * 2) + (funding_bonus * leverage)
            
    if not fills: return 0.0
    pct_change = (window.iloc[-1]['close'] - avg_entry) / avg_entry
    return pct_change * leverage - (fee * leverage * 2) + (funding_bonus * leverage)

def main():
    print("Loading 2 Years of BTC data...")
    raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    raw.columns = [c.lower() for c in raw.columns]
    if raw['timestamp'].dtype == 'object' or pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=730)] # 2 YEARS
    
    print("Calculating ATR and SMA200 for Meta-Mechanics...")
    raw['atr_1h'] = calc_atr(raw, 60) # 60 min ATR
    
    print("Building 5m features...")
    df = raw.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df['sma200'] = df['close'].rolling(200).mean()
    
    H = 24
    df['target_short'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)
    df['target_long'] = (df['high'].rolling(H,1).max().shift(-H) >= df['close'] * 1.005).astype(int)
    df = add_features(df)
    
    # Calculate ATR thresholds on the 5m dataframe downsampled from raw
    df['atr_1h'] = raw['atr_1h'].resample('5min').last()
    atr_p25 = df['atr_1h'].quantile(0.25)
    atr_p75 = df['atr_1h'].quantile(0.75)
    atr_thresholds = (atr_p25, atr_p75)
    
    print(f"ATR Volatility Bounds: Low < {atr_p25:.2f}, High > {atr_p75:.2f}")
    
    print("\nTraining Logistic Regression (SHORTS)...")
    preds_short, wi = walk_forward_lr(df, H, 'target_short')
    
    print("Training Logistic Regression (LONGS)...")
    preds_long, _ = walk_forward_lr(df, H, 'target_long')
    
    test_idx = df.index[wi:len(df)-H]
    probs_short = preds_short[wi:len(df)-H]
    probs_long = preds_long[wi:len(df)-H]
    
    thr_short = np.percentile(probs_short, 95.0)
    thr_long = np.percentile(probs_long, 95.0)
    
    print(f"\nEvaluating Top 5% Signals (Out of Sample: {len(test_idx)} periods)")
    
    results_short = []
    results_long = []
    
    for idx in test_idx:
        is_short = probs_short[test_idx.get_loc(idx)] >= thr_short
        is_long = probs_long[test_idx.get_loc(idx)] >= thr_long
        
        if not is_short and not is_long: continue
        
        entry_time = idx + pd.Timedelta(minutes=5)
        end_time = entry_time + pd.Timedelta(minutes=120)
        window = raw.loc[entry_time : end_time]
        
        if len(window) < 2: continue
        
        S0 = df.loc[idx, 'close']
        atr_val = df.loc[idx, 'atr_1h']
        sma200 = df.loc[idx, 'sma200']
        
        leverage = determine_leverage(atr_val, atr_thresholds)
        
        if is_short:
            fb = get_funding_bonus(S0, sma200, True)
            r = sim_dca_grid_short(window, S0, leverage, funding_bonus=fb)
            if r != 0.0: results_short.append(max(r, -1.0))
            
        if is_long:
            fb = get_funding_bonus(S0, sma200, False)
            r = sim_dca_grid_long(window, S0, leverage, funding_bonus=fb)
            if r != 0.0: results_long.append(max(r, -1.0))
            
    arr_s = np.array(results_short)
    arr_l = np.array(results_long)
    arr_all = np.concatenate([arr_s, arr_l])
    
    wr_s = (arr_s > 0).mean() * 100 if len(arr_s) > 0 else 0
    wr_l = (arr_l > 0).mean() * 100 if len(arr_l) > 0 else 0
    wr_all = (arr_all > 0).mean() * 100
    
    apnl_s = arr_s.mean() * 100 if len(arr_s) > 0 else 0
    apnl_l = arr_l.mean() * 100 if len(arr_l) > 0 else 0
    apnl_all = arr_all.mean() * 100
    
    mo = len(test_idx) * 5 / (60 * 24 * 30.44) # Exact months in test period
    
    print("\n--- HYPER GROWTH RESULTS (2 YEARS) ---")
    print(f"SHORTS | Signals/mo: {len(arr_s)/mo:5.1f} | Win Rate: {wr_s:5.1f}% | Avg PnL: {apnl_s:+6.2f}%")
    print(f"LONGS  | Signals/mo: {len(arr_l)/mo:5.1f} | Win Rate: {wr_l:5.1f}% | Avg PnL: {apnl_l:+6.2f}%")
    print("-" * 65)
    print(f"TOTAL  | Signals/mo: {len(arr_all)/mo:5.1f} | Win Rate: {wr_all:5.1f}% | Avg PnL: {apnl_all:+6.2f}%")
    
    if apnl_all > 0:
        import math
        comp_rate = 1 + (apnl_all / 100)
        n_trades = math.log(1000) / math.log(comp_rate)
        # Assuming we can only take 1 trade at a time, we cap effective trades per month to roughly 300 to account for overlap
        effective_trades_mo = min(len(arr_all)/mo, 300) 
        months = n_trades / effective_trades_mo
        print(f"\n-> Projected Time to $10k (assuming 300 trades max/mo capacity): {months:.1f} Months")

if __name__ == '__main__':
    main()
