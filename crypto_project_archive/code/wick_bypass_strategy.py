"""
Wick Bypass Strategy Simulation
===============================
Testing 4 execution methods to survive the noise before the drop.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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

def walk_forward_xgb(df, H=24):
    X = df[FCOLS].values; y = df['target'].values
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    preds = np.zeros(len(X))
    INIT, STEP = 12000, 4000
    if len(X) < INIT + H:
        model.fit(X, y)
        return model.predict_proba(X)[:,1], 0
    for s in range(INIT, len(X)-H, STEP):
        e = min(s+STEP, len(X)-H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
    return preds, INIT

# ── EXECUTION ENGINES ────────────────────────────────────────────────────────

def sim_delayed_entry(window, S0, leverage=50.0, fee=0.0004):
    trigger_price = S0 * (1 - 0.001) # 0.1% drop
    entry_price = None
    
    for _, row in window.iterrows():
        high, low = row['high'], row['low']
        
        if entry_price is None:
            if low <= trigger_price:
                entry_price = trigger_price
                sl_price = entry_price * (1 + 0.003)
                tp_price = entry_price * (1 - 0.005)
                # Did we hit SL instantly in the same candle? (worst case)
                if high >= sl_price:
                    return -0.003 * leverage - (fee * leverage * 2)
                if low <= tp_price:
                    return 0.005 * leverage - (fee * leverage * 2)
            continue
            
        if high >= sl_price:
            return -0.003 * leverage - (fee * leverage * 2)
        if low <= tp_price:
            return 0.005 * leverage - (fee * leverage * 2)
            
    if entry_price is None:
        return 0.0 # Trade never triggered
        
    final_price = window.iloc[-1]['close']
    pct_change = (entry_price - final_price) / entry_price
    return pct_change * leverage - (fee * leverage * 2)


def sim_dca_grid(window, S0, leverage=25.0, fee=0.0004):
    # Enter 25% at S0, S0*1.0015, S0*1.003, S0*1.0045
    levels = [S0, S0*1.0015, S0*1.003, S0*1.0045]
    fills = []
    
    sl_price = S0 * 1.0075 # Hard stop at 0.75% above S0
    avg_entry = None
    tp_price = None
    
    for _, row in window.iterrows():
        high, low = row['high'], row['low']
        
        # Check for new fills
        for lvl in levels:
            if lvl not in fills and high >= lvl:
                fills.append(lvl)
                avg_entry = sum(fills) / len(fills)
                tp_price = avg_entry * (1 - 0.005) # Target 0.5% from avg entry
                
        if not fills:
            # We assume S0 is hit on candle open, so fills should always have at least S0
            # Just in case:
            if low <= S0:
                fills.append(S0)
                avg_entry = S0
                tp_price = avg_entry * (1 - 0.005)
        
        if not fills: continue
        
        if high >= sl_price:
            pct_change = (avg_entry - sl_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2)
            
        if low <= tp_price:
            pct_change = (avg_entry - tp_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2)
            
    if not fills: return 0.0
    final_price = window.iloc[-1]['close']
    pct_change = (avg_entry - final_price) / avg_entry
    return pct_change * leverage - (fee * leverage * 2)


def sim_wide_stop(window, S0, leverage=10.0, fee=0.0004):
    sl_price = S0 * (1 + 0.015)
    tp_price = S0 * (1 - 0.005)
    
    for _, row in window.iterrows():
        high, low = row['high'], row['low']
        if high >= sl_price:
            return -0.015 * leverage - (fee * leverage * 2)
        if low <= tp_price:
            return 0.005 * leverage - (fee * leverage * 2)
            
    final_price = window.iloc[-1]['close']
    pct_change = (S0 - final_price) / S0
    return pct_change * leverage - (fee * leverage * 2)


def sim_atr_stop(window, S0, atr_val, leverage=20.0, fee=0.0004):
    sl_price = S0 + (2.0 * atr_val)
    sl_pct = (sl_price - S0) / S0
    tp_price = S0 * (1 - 0.005)
    
    for _, row in window.iterrows():
        high, low = row['high'], row['low']
        if high >= sl_price:
            return -sl_pct * leverage - (fee * leverage * 2)
        if low <= tp_price:
            return 0.005 * leverage - (fee * leverage * 2)
            
    final_price = window.iloc[-1]['close']
    pct_change = (S0 - final_price) / S0
    return pct_change * leverage - (fee * leverage * 2)


def main():
    print("Loading BTC data...")
    raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    raw.columns = [c.lower() for c in raw.columns]
    if raw['timestamp'].dtype == 'object' or pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=180)]
    
    print("Calculating 1m ATR...")
    raw['atr'] = calc_atr(raw, 14)
    
    print("Building 5m features...")
    df = raw.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    H = 24
    df['target'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)
    df = add_features(df)
    
    print("Training XGBoost...")
    preds, wi = walk_forward_xgb(df, H)
    test = df.iloc[wi:len(df)-H].copy()
    test['prob'] = preds[wi:len(df)-H]
    
    print("\n--- SWEET SPOT GRID SEARCH (DCA Grid) ---")
    for p_val in [99.9, 99.5, 99.0, 98.0, 95.0, 90.0]:
        thr = np.percentile(test['prob'], p_val)
        pool = test[test['prob'] >= thr]
        
        results = []
        for idx, row in pool.iterrows():
            entry_time = idx + pd.Timedelta(minutes=5)
            end_time = entry_time + pd.Timedelta(minutes=120)
            window = raw.loc[entry_time : end_time]
            
            if len(window) < 2: continue
            
            S0 = row['close']
            r_dca = sim_dca_grid(window, S0)
            results.append(max(r_dca, -1.0))
            
        if not results: continue
        arr = np.array(results)
        wr = (arr > 0).mean() * 100
        apnl = arr.mean() * 100
        trades_per_mo = len(arr) / 6.0
        
        if apnl > 0:
            import math
            comp_rate = 1 + (apnl / 100)
            n_trades = math.log(1000) / math.log(comp_rate)
            months = n_trades / trades_per_mo if trades_per_mo > 0 else float('inf')
        else:
            months = float('inf')
            
        top_pct = 100 - p_val
        print(f"Top {top_pct:4.1f}% | Thr: {thr:.4f} | Signals/mo: {trades_per_mo:5.1f} | WR: {wr:5.1f}% | PnL: {apnl:+6.2f}% | Months to 10k: {months:.1f}")

if __name__ == '__main__':
    main()
