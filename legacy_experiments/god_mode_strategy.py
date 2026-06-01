"""
God Mode Acceleration Strategy
==============================
Testing Multi-Asset, Micro-Timeframe (5m), High-Leverage Futures, and All-In Pyramiding.
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
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    preds = np.zeros(len(X))
    INIT, STEP = 12000, 4000 # Since 5m, more candles
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

def sim_futures_trade(df_raw, entry_time, entry_price, sl_pct, tp_pct, H_minutes=120, leverage=50.0, fee=0.0004):
    """
    Simulate a short futures trade.
    """
    end_time = entry_time + pd.Timedelta(minutes=H_minutes)
    window = df_raw.loc[entry_time : end_time]
    if len(window) < 2: return None

    S0 = entry_price
    sl_price = S0 * (1 + sl_pct)
    tp_price = S0 * (1 - tp_pct)
    
    # Iterate through 1m candles
    for _, row in window.iterrows():
        high = row['high']
        low = row['low']
        
        # Did we hit SL first? (Pessimistic: assume high hit before low)
        if high >= sl_price:
            pnl_pct = -sl_pct * leverage - (fee * leverage * 2) # Enter & Exit fee
            return pnl_pct
            
        # Did we hit TP?
        if low <= tp_price:
            pnl_pct = tp_pct * leverage - (fee * leverage * 2)
            return pnl_pct
            
    # Time expiration
    final_price = window.iloc[-1]['close']
    pct_change = (S0 - final_price) / S0 # Short profit
    pnl_pct = pct_change * leverage - (fee * leverage * 2)
    return pnl_pct

def load_and_prep(coin_path):
    print(f"Loading {coin_path}...")
    raw = pd.read_csv(coin_path)
    raw.columns = [c.lower() for c in raw.columns]
    if raw['timestamp'].dtype == 'object' or pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    # Get last 6 months to avoid massive memory blowup on 3 coins
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=180)]
    
    df = raw.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    H = 24 # 2 hours
    df['target'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)
    df = add_features(df)
    return raw, df

def simulate_pyramiding(pnl_arr, prob_arr, target=10000.0):
    """ All-in pyramiding """
    bal = 10.0; trades = 0
    for pnl in pnl_arr:
        if bal >= target: break
        if bal <= 0: break
        # Risk 100% of balance
        bal += bal * pnl
        trades += 1
    return bal, trades

def main():
    coins = [
        'c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv',
        'c:/Users/onepiece/Documents/_Garage/Ohhv2/data/ETHUSDT-1m.csv',
        'c:/Users/onepiece/Documents/_Garage/Ohhv2/data/SOLUSDT-1m.csv'
    ]
    
    all_pnls = []
    
    for coin in coins:
        try:
            raw, df = load_and_prep(coin)
            H = 24
            preds, wi = walk_forward_xgb(df, H)
            test = df.iloc[wi:len(df)-H].copy()
            test['prob'] = preds[wi:len(df)-H]
            
            # Extract top 1% signals
            thr = np.percentile(test['prob'], 99)
            pool = test[test['prob'] >= thr]
            
            print(f"{coin.split('/')[-1]} - Top 1% threshold: {thr:.4f}, Signals: {len(pool)}")
            
            for idx, row in pool.iterrows():
                # Futures Trade: 0.3% Stop Loss, 0.5% Take Profit, 50x Lev
                p = sim_futures_trade(raw, idx + pd.Timedelta(minutes=5), row['close'], 
                                      sl_pct=0.003, tp_pct=0.005, H_minutes=120, leverage=50.0)
                if p is not None:
                    # Clip max loss to -1.0 (-100%) so we don't get negative balances
                    p = max(p, -1.0)
                    all_pnls.append((p, row['prob']))
        except Exception as e:
            print(f"Failed on {coin}: {e}")
            
    if not all_pnls:
        print("No signals found.")
        return
        
    pnl_arr = np.array([x[0] for x in all_pnls])
    prob_arr = np.array([x[1] for x in all_pnls])
    
    win_rate = (pnl_arr > 0).mean() * 100
    avg_pnl = pnl_arr.mean() * 100
    
    print("\n--- God Mode Aggregated Results (6 Months) ---")
    print(f"Total High-Conviction Signals: {len(pnl_arr)}")
    print(f"Monthly Signals: {len(pnl_arr) / 6.0:.1f}")
    print(f"Futures Win Rate: {win_rate:.1f}%")
    print(f"Avg PnL Per Trade: {avg_pnl:+.1f}%")
    
    print("\n--- Pyramiding Simulation ---")
    ok = 0
    runs = 10000
    trades_to_win = []
    for _ in range(runs):
        idx = np.random.permutation(len(pnl_arr))
        bal, t = simulate_pyramiding(pnl_arr[idx], prob_arr[idx])
        if bal >= 10000.0:
            ok += 1
            trades_to_win.append(t)
            
    print(f"100% Risk Pyramiding Success Rate: {ok/runs*100:.2f}%")
    if ok > 0:
        print(f"Avg trades to reach $10k: {np.mean(trades_to_win):.1f}")
    else:
        print("Zero success paths found. Account blew up every time.")

if __name__ == '__main__':
    main()
