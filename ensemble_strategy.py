"""
Ensemble ML Strategy
====================
Testing multiple ML models (XGBoost, LightGBM, Random Forest, Logistic Regression)
and a Soft Voting Ensemble to maximize the Win Rate of the DCA Grid.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
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

def walk_forward_ensemble(df, H=24):
    X = df[FCOLS].values; y = df['target'].values
    
    # Define Models
    models = {
        'XGB': xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'LGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        'LR': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    }
    
    preds_dict = {name: np.zeros(len(X)) for name in models.keys()}
    INIT, STEP = 12000, 4000
    
    if len(X) < INIT + H:
        # Fallback if not enough data
        for name, m in models.items():
            sc = StandardScaler()
            m.fit(sc.fit_transform(X), y)
            preds_dict[name] = m.predict_proba(sc.transform(X))[:,1]
        return preds_dict, 0
        
    for s in range(INIT, len(X)-H, STEP):
        e = min(s+STEP, len(X)-H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        
        for name, m in models.items():
            m.fit(Xtr, y[:s-H])
            preds_dict[name][s:e] = m.predict_proba(Xte)[:,1]
            
    # Add Ensemble (Average of Tree Models: XGB, LGBM, RF)
    preds_dict['Ensemble'] = (preds_dict['XGB'] + preds_dict['LGBM'] + preds_dict['RF']) / 3.0
    
    return preds_dict, INIT

# ── DCA GRID EXECUTION ENGINE ────────────────────────────────────────────────

def sim_dca_grid(window, S0, leverage=25.0, fee=0.0004):
    levels = [S0, S0*1.0015, S0*1.003, S0*1.0045]
    fills = []
    
    sl_price = S0 * 1.0075 
    avg_entry = None
    tp_price = None
    
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
            return pct_change * leverage - (fee * leverage * 2)
            
        if low <= tp_price:
            pct_change = (avg_entry - tp_price) / avg_entry
            return pct_change * leverage - (fee * leverage * 2)
            
    if not fills: return 0.0
    final_price = window.iloc[-1]['close']
    pct_change = (avg_entry - final_price) / avg_entry
    return pct_change * leverage - (fee * leverage * 2)


def main():
    print("Loading BTC data...")
    raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    raw.columns = [c.lower() for c in raw.columns]
    if raw['timestamp'].dtype == 'object' or pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=180)]
    
    print("Building 5m features...")
    df = raw.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    H = 24
    df['target'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)
    df = add_features(df)
    
    print("Training Models (Walk Forward)...")
    preds_dict, wi = walk_forward_ensemble(df, H)
    
    test_idx = df.index[wi:len(df)-H]
    
    print("\n--- MODEL LEADERBOARD (TOP 5% SIGNALS) ---")
    
    for name, preds in preds_dict.items():
        probs = preds[wi:len(df)-H]
        thr = np.percentile(probs, 95.0) # Top 5% threshold
        
        # Get signal indices
        signal_mask = probs >= thr
        signal_indices = test_idx[signal_mask]
        
        results = []
        for idx in signal_indices:
            entry_time = idx + pd.Timedelta(minutes=5)
            end_time = entry_time + pd.Timedelta(minutes=120)
            window = raw.loc[entry_time : end_time]
            
            if len(window) < 2: continue
            
            S0 = df.loc[idx, 'close']
            r_dca = sim_dca_grid(window, S0)
            results.append(max(r_dca, -1.0))
            
        if not results: continue
        arr = np.array(results)
        wr = (arr > 0).mean() * 100
        apnl = arr.mean() * 100
        signals_mo = len(arr) / 6.0
        
        print(f"{name:8} | Signals/mo: {signals_mo:5.1f} | Win Rate: {wr:5.1f}% | Avg PnL: {apnl:+6.2f}%")

if __name__ == '__main__':
    main()
