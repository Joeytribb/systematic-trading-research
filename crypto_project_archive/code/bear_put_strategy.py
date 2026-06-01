"""
Final 10k Challenge Strategy
================================
Optimal Configuration:
- Model: Walk-forward Logistic Regression
- Threshold: Probability >= 0.90
- Strategy: Bear Put Debit Spread
- Buy Strike: -1.5% OTM
- Sell Strike: -4.0% OTM
- Expiry: 4 hours
- Exit: Take Profit limit order at 70% of max spread value
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import itertools
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
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low']  - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr_norm']   = tr.rolling(14).mean() / (df['close'] + 1e-10)
    df['vel1'] = df['close'].pct_change(1)
    df['vel3'] = df['close'].pct_change(3)
    df['vel5'] = df['close'].pct_change(5)
    cr = df['high'] - df['low']
    df['uw'] = (df['high'] - df[['open','close']].max(axis=1)) / (cr + 1e-10)
    df['lw'] = (df[['open','close']].min(axis=1) - df['low'])  / (cr + 1e-10)
    vm = df['volume'].rolling(20).mean()
    df['vol_surge'] = df['volume'] / (vm + 1e-10)
    e50  = df['close'].ewm(span=50,  adjust=False).mean()
    e200 = df['close'].ewm(span=200, adjust=False).mean()
    df['d50']  = (df['close'] - e50)  / (e50  + 1e-10)
    df['d200'] = (df['close'] - e200) / (e200 + 1e-10)
    h = df.index.hour; dw = df.index.dayofweek
    df['h_sin'] = np.sin(2*np.pi*h/24); df['h_cos'] = np.cos(2*np.pi*h/24)
    df['d_sin'] = np.sin(2*np.pi*dw/7); df['d_cos'] = np.cos(2*np.pi*dw/7)
    lr = np.log(df['close'] / df['close'].shift())
    df['vol_ann'] = (lr.rolling(96).std() * np.sqrt(35040)).clip(0.15, 1.50)
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist','atr_norm',
         'vel1','vel3','vel5','uw','lw','vol_surge','d50','d200',
         'h_sin','h_cos','d_sin','d_cos']

def walk_forward_lr(df, H=16):
    X = df[FCOLS].values; y = df['target'].values
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    preds = np.zeros(len(X))
    INIT, STEP = 17500, 2900
    for s in range(INIT, len(X)-H, STEP):
        e = min(s+STEP, len(X)-H)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:s-H]); Xte = sc.transform(X[s:e])
        model.fit(Xtr, y[:s-H])
        preds[s:e] = model.predict_proba(Xte)[:,1]
    return preds, INIT

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def sim_put_spread(df_raw, entry_time, entry_close, vol, k_high_pct, k_low_pct, H_minutes=240, r=0.05, tp_frac=0.70):
    end_time = entry_time + pd.Timedelta(minutes=H_minutes)
    window = df_raw.loc[entry_time : end_time]
    if len(window) < 5: return None

    S0 = entry_close
    K_high = S0 * (1 - k_high_pct)
    K_low  = S0 * (1 - k_low_pct)
    T_full = H_minutes / 525600.0

    p_high_entry = bs_put(S0, K_high, T_full, r, vol)
    p_low_entry  = bs_put(S0, K_low,  T_full, r, vol)
    net_debit    = p_high_entry - p_low_entry

    if net_debit <= 0: return None
    max_profit = (K_high - K_low) - net_debit
    if max_profit <= 0: return None

    closes = window['close'].values
    for j in range(1, len(closes)):
        S_curr = closes[j]
        T_curr = max((H_minutes - j) / 525600.0, 1e-6)
        spread_val = bs_put(S_curr, K_high, T_curr, r, vol) - bs_put(S_curr, K_low, T_curr, r, vol)
        if (spread_val - net_debit) >= tp_frac * max_profit:
            return (spread_val - net_debit) / net_debit

    S_exp = closes[-1]
    final_val = max(0.0, K_high - S_exp) - max(0.0, K_low - S_exp)
    return (final_val - net_debit) / net_debit

def simulate(pnl_arr, prob_arr, s1_thr, s1_risk, s2_thr, s2_risk, target=1000.0):
    bal = 10.0; s1n = s2n = 0
    for pnl, prob in zip(pnl_arr, prob_arr):
        if bal >= target: break
        if bal < 1.0: bal = 0.0; break
        if bal < 100.0:
            if prob >= s1_thr:
                m = min(max(bal * s1_risk, 1.0), bal)
                bal += m * pnl; s1n += 1
        else:
            if prob >= s2_thr:
                m = min(max(bal * s2_risk, 1.0), bal)
                bal += m * pnl; s2n += 1
    return bal, s1n, s2n

def main():
    print("Running optimal Bear Put strategy backtest...", flush=True)
    raw = pd.read_csv('c:/Users/onepiece/Documents/_Garage/Ohhv2/data/BTC-1m.csv')
    raw.columns = [c.lower() for c in raw.columns]
    raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='ignore')
    if raw['timestamp'].dtype == 'object': raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw.set_index('timestamp', inplace=True); raw.sort_index(inplace=True)
    raw = raw[raw.index >= raw.index.max() - pd.Timedelta(days=730)]

    df = raw.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    H = 16
    df['target'] = (df['low'].rolling(H,1).min().shift(-H) <= df['close'] * 0.995).astype(int)
    df = add_features(df)
    preds, wi = walk_forward_lr(df, H)
    
    test = df.iloc[wi:len(df)-H].copy()
    test['prob'] = preds[wi:len(df)-H]
    
    THR = 0.90
    pool = test[test['prob'] >= THR]
    pnls = []
    
    for idx, row in pool.iterrows():
        p = sim_put_spread(raw, idx + pd.Timedelta(minutes=15), row['close'], row['vol_ann'], 0.015, 0.040, 240, 0.05, 0.70)
        if p is not None: pnls.append((p, row['prob']))
            
    pnl_arr, prob_arr = np.array([x[0] for x in pnls]), np.array([x[1] for x in pnls])
    
    print("\n--- Final Results ---")
    print(f"Total Signals: {len(pnls)}")
    print(f"Win Rate: {(pnl_arr > 0).mean()*100:.1f}%")
    print(f"Avg PnL per trade: {pnl_arr.mean()*100:+.1f}%")
    
    ok = 0; s1_avg = []; s2_avg = []
    runs = 1000
    tile_pnl, tile_prob = np.tile(pnl_arr, 8), np.tile(prob_arr, 8)
    
    for _ in range(runs):
        idx = np.random.permutation(len(tile_pnl))
        bal, s1, s2 = simulate(tile_pnl[idx], tile_prob[idx], 0.95, 0.50, 0.90, 0.10, target=1000.0)
        if bal >= 1000:
            ok += 1; s1_avg.append(s1); s2_avg.append(s2)
            
    print("\n--- Compounding Results ($10 -> $1K) ---")
    print(f"Success Probability: {ok/runs*100:.1f}%")
    if ok > 0:
        print(f"Average Trades to Goal: {np.mean(s1_avg) + np.mean(s2_avg):.0f}")

if __name__ == '__main__':
    main()
