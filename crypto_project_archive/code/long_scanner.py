"""
long_scanner.py  -  Multi-Asset LONG Concurrency Engine
=========================================================
Mirror image of multi_asset_scanner.py (short strategy).

Edge: after a sharp 5-minute DROP, price statistically bounces
      back up at least 0.5% within the next 2 hours.

Model:  same MTF Logistic Regression but target_long =
        'did price rise 0.5% within the next 24 bars?'

Execution: DCA Grid LONG
  - Buy at signal price S0
  - Scale in at S0*0.9985, S0*0.997, S0*0.9955 (DCA on further dips)
  - Take-Profit at avg_entry * 1.005  (+0.5% above avg entry)
  - Stop-Loss   at S0 * 0.9925       (-0.75% below signal price)
  - Leverage: 25x
  - Fee: 0.04% per side
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ASSETS   = ['BTC-1m.csv', 'ETHUSDT-1m.csv', 'SOLUSDT-1m.csv',
            'LINKUSDT-1m.csv', 'DOGEUSDT-1m.csv']
NAMES    = ['BTC', 'ETH', 'SOL', 'LINK', 'DOGE']
DATA_DIR = 'c:/Users/onepiece/Documents/_Garage/Ohhv2/data/'

# ── FEATURE PIPELINE (identical to short scanner) ────────────────────────────

def rsi(s, p=14):
    d = s.diff()
    g =  d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df, raw_1m):
    ma  = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    ub  = ma + 2*std; lb = ma - 2*std; bw = ub - lb
    df['pct_b']      = (df['close'] - lb)  / (bw + 1e-10)
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
    sma_1h = raw_1h.rolling(50).mean()
    sma_4h = raw_4h.rolling(50).mean()
    df['sma_1h']      = sma_1h.reindex(df.index, method='ffill')
    df['sma_4h']      = sma_4h.reindex(df.index, method='ffill')
    df['mtf_1h_dist'] = (df['close'] - df['sma_1h']) / (df['sma_1h'] + 1e-10)
    df['mtf_4h_dist'] = (df['close'] - df['sma_4h']) / (df['sma_4h'] + 1e-10)

    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist',
         'vel1','vel3','uw','lw','vol_ratio','mtf_1h_dist','mtf_4h_dist']

# ── DCA GRID LONG EXECUTION ──────────────────────────────────────────────────

def sim_dca_grid_long(window, S0, leverage=25.0, fee=0.0004):
    """
    Mirror of sim_dca_grid_short but for LONG trades.
    Fills as price DROPS through DCA levels (averaging down).
    [B5-fix] same-candle multi-level fills use the worst (lowest) price.
    """
    levels   = [S0, S0*0.9985, S0*0.997, S0*0.9955]
    fills    = []
    sl_price = S0 * 0.9925        # -0.75% hard stop
    avg_entry = None; tp_price = None

    for idx, row in window.iterrows():
        high, low = row['high'], row['low']

        newly_filled = []
        for lvl in levels:
            if lvl not in fills and low <= lvl:
                fills.append(lvl)
                newly_filled.append(lvl)

        # [B5-fix] worst-case fill = lowest price on same-candle multi-fills
        if len(newly_filled) > 1:
            worst = min(newly_filled)
            base  = [f for f in fills if f not in newly_filled]
            fills = base + [worst] * len(newly_filled)

        if fills:
            avg_entry = sum(fills) / len(fills)
            tp_price  = avg_entry * 1.005

        if not fills:
            if high >= S0:
                fills.append(S0)
                avg_entry = S0
                tp_price  = avg_entry * 1.005

        if not fills: continue

        # Stop-Loss hit (price drops below SL)
        if low <= sl_price:
            pct = (sl_price - avg_entry) / avg_entry
            return pct * leverage - fee * leverage * 2, idx

        # Take-Profit hit (price rises above TP)
        if high >= tp_price:
            pct = (tp_price - avg_entry) / avg_entry
            return pct * leverage - fee * leverage * 2, idx

    if not fills: return 0.0, window.index[-1]
    pct = (window.iloc[-1]['close'] - avg_entry) / avg_entry
    return pct * leverage - fee * leverage * 2, window.index[-1]

# ── PROCESS ASSET ────────────────────────────────────────────────────────────

def process_asset(filename):
    print(f"  Loading {filename}...")
    raw = pd.read_csv(DATA_DIR + filename)
    raw.columns = [c.lower() for c in raw.columns]
    if pd.api.types.is_numeric_dtype(raw['timestamp']):
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], unit='s', errors='coerce')
    raw.set_index('timestamp', inplace=True)
    raw.sort_index(inplace=True)

    start_date = pd.to_datetime('2026-03-01')
    end_date   = pd.to_datetime('2026-05-15')
    raw = raw[(raw.index >= start_date) & (raw.index < end_date)]

    df = raw.resample('5min').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna()

    H = 24
    # LONG target: did price RISE 0.5% within next H bars?
    df['target_long'] = (
        df['high'].rolling(H, 1).max().shift(-H) >= df['close'] * 1.005
    ).astype(int)

    df = add_features(df, raw)

    split = int(len(df) * 0.5)
    train = df.iloc[:split]
    test  = df.iloc[split:len(df)-H]

    X_tr = train[FCOLS].values; y_tr = train['target_long'].values
    X_te = test[FCOLS].values

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)[:, 1]
    thr   = np.percentile(probs, 95.0)   # Top 5% threshold

    signals = pd.Series(0, index=test.index)
    signals[probs >= thr] = 1
    return raw, df, signals, test.index

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=== LONG SCANNER: Multi-Asset Concurrency Engine ===\n")
    asset_data = {}
    for name, f in zip(NAMES, ASSETS):
        try:
            raw, df, signals, test_idx = process_asset(f)
            asset_data[name] = {'raw': raw, 'df': df, 'signals': signals}
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    active_names = list(asset_data.keys())
    if not active_names: return

    print("\nAligning timelines...")
    common_idx = asset_data[active_names[0]]['signals'].index
    for name in active_names[1:]:
        common_idx = common_idx.intersection(asset_data[name]['signals'].index)

    days = len(common_idx) * 5 / 60 / 24
    print(f"Aligned window: {len(common_idx)} periods (~{days:.1f} days)")

    signal_matrix = pd.DataFrame(index=common_idx)
    for name in active_names:
        signal_matrix[name] = asset_data[name]['signals'].loc[common_idx]

    btc_signals = signal_matrix['BTC'].sum()
    print(f"BTC-only signals in window: {btc_signals}")

    # Event-driven concurrency simulator
    lock_until = common_idx[0] - pd.Timedelta(minutes=1)
    results    = []

    for current_time in common_idx:
        if current_time < lock_until:
            continue
        row = signal_matrix.loc[current_time]
        fired = row[row == 1].index.tolist()
        if not fired: continue

        chosen = fired[0]
        raw    = asset_data[chosen]['raw']
        df     = asset_data[chosen]['df']

        entry_time   = current_time + pd.Timedelta(minutes=5)
        max_end_time = entry_time   + pd.Timedelta(minutes=120)
        window = raw.loc[entry_time : max_end_time]
        if len(window) < 2: continue

        S0 = df.loc[current_time, 'close']
        r, exit_time = sim_dca_grid_long(window, S0, leverage=25.0)

        if r != 0.0:
            results.append(max(r, -1.0))
            lock_until = exit_time

    arr   = np.array(results)
    n     = len(arr)
    wr    = (arr > 0).mean() * 100 if n else 0
    apnl  = arr.mean() * 100       if n else 0
    mult  = n / btc_signals         if btc_signals else 0
    mo    = days / 30.44

    print("\n=== LONG SCANNER RESULTS ===")
    print(f"Total trades executed:  {n}")
    print(f"Multi-asset multiplier: {mult:.2f}x vs BTC alone")
    print(f"Win Rate:               {wr:.1f}%")
    print(f"Average PnL:            {apnl:+.2f}%")
    print(f"Trades/Month (extrap.): {n/mo:.0f}")

    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    if len(wins):   print(f"Avg Win PnL:            {wins.mean()*100:+.2f}%")
    if len(losses): print(f"Avg Loss PnL:           {losses.mean()*100:+.2f}%")

if __name__ == '__main__':
    main()
