"""
tp_sl_sweep.py - Grid Search for Optimal TP/SL on Nifty 50 Equities
Tests multiple TP/SL combinations using pre-loaded signals from the unified ML model.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FEES_ROUND_TRIP = 0.0004
LEVERAGE = 5.0

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df):
    if len(df) < 50: return pd.DataFrame()
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
    df['lw'] = (df[['open','close']].min(axis=1) - df['low'])  / (cr + 1e-10)
    rng = df['high'] - df['low']
    df['range_ratio'] = rng / (rng.rolling(20).mean() + 1e-10)
    df['sma_15m_approx'] = df['close'].rolling(60).mean()
    df['mtf_1h_dist'] = (df['close'] - df['sma_15m_approx']) / (df['sma_15m_approx'] + 1e-10)
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist','vel1','vel3','uw','lw','range_ratio','mtf_1h_dist']

def process_ticker(filepath):
    ticker = os.path.basename(filepath).replace('-5m.csv', '')
    df = pd.read_csv(filepath, header=[0,1], index_col=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Kolkata')
    df.index.name = 'datetime'
    if 'adj close' in df.columns:
        df.drop(columns=['adj close'], inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    df = add_features(df)
    df['ticker'] = ticker
    return df

def sim_trade(window, S0, direction, tp_pct, sl_pct, dca_spacing):
    """Generic DCA sim for any TP/SL/spacing."""
    if direction == 'short':
        levels = [S0 * (1 + dca_spacing * i) for i in range(4)]
        sl_price = S0 * (1 + sl_pct)
        tp_mult = 1 - tp_pct
    else:
        levels = [S0 * (1 - dca_spacing * i) for i in range(4)]
        sl_price = S0 * (1 - sl_pct)
        tp_mult = 1 + tp_pct

    fills = []; avg_entry = tp_price = None

    for idx, row in window.iterrows():
        if idx.time() >= pd.to_datetime('15:15:00').time():
            if not fills: return 0.0
            pnl_raw = (avg_entry - row['close']) if direction=='short' else (row['close'] - avg_entry)
            return pnl_raw / avg_entry * LEVERAGE - FEES_ROUND_TRIP * LEVERAGE

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
            if h >= sl_price:
                return (avg_entry - sl_price)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE
            if l <= tp_price:
                return (avg_entry - tp_price)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE
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
            if l <= sl_price:
                return (sl_price - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE
            if h >= tp_price:
                return (tp_price - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE

    if not fills: return 0.0
    pnl_raw = (avg_entry - window.iloc[-1]['close']) if direction=='short' else (window.iloc[-1]['close'] - avg_entry)
    return pnl_raw / avg_entry * LEVERAGE - FEES_ROUND_TRIP * LEVERAGE

def run_sweep(test_df, signal_times, dict_dfs, tp_pct, sl_pct, dca_spacing):
    """Run the full concurrency-lock simulation for one TP/SL config."""
    lock_until = signal_times[0] - pd.Timedelta(minutes=1)
    results = []

    for t in signal_times:
        t_ts = pd.Timestamp(t)
        if t_ts.time() < pd.to_datetime('09:30:00').time() or t_ts.time() > pd.to_datetime('14:15:00').time():
            continue
        if t_ts < lock_until:
            continue

        current_rows = test_df.loc[[t_ts]] if t_ts in test_df.index else pd.DataFrame()
        if len(current_rows) == 0: continue

        shorts = current_rows[current_rows['sig_short'] == 1]
        longs  = current_rows[current_rows['sig_long']  == 1]

        if len(shorts) == 0 and len(longs) == 0:
            continue

        if len(shorts) > 0:
            best_ticker = shorts.sort_values('prob_short', ascending=False).iloc[0]['ticker']
            direction = 'short'
        else:
            best_ticker = longs.sort_values('prob_long', ascending=False).iloc[0]['ticker']
            direction = 'long'

        entry_time = t_ts + pd.Timedelta(minutes=5)
        end_time   = entry_time + pd.Timedelta(minutes=90)

        ticker_df = dict_dfs.get(best_ticker)
        if ticker_df is None: continue

        window = ticker_df.loc[entry_time : end_time]
        window = window[window.index.date == entry_time.date()]
        if len(window) < 2: continue

        S0 = window.iloc[0]['open']
        r = sim_trade(window, S0, direction, tp_pct, sl_pct, dca_spacing)
        if r != 0.0:
            results.append(r)
            lock_until = window.index[-1]

    arr = np.array(results)
    if len(arr) == 0:
        return {'trades': 0, 'win_rate': 0, 'ev': 0, 'monthly_ret': 0}

    win_rate = (arr > 0).mean() * 100
    ev = arr.mean() * 100
    monthly_ret = ((1 + arr.mean()) ** len(arr) - 1) * 100
    return {'trades': len(arr), 'win_rate': round(win_rate, 2), 'ev': round(ev, 4), 'monthly_ret': round(monthly_ret, 2)}

def main():
    print("Loading 47 Nifty 50 equities...")
    data_dir = "c:/Users/onepiece/Documents/_Garage/Ohhv2/data/nifty50_equities"
    files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_dfs = []
    for f in files:
        df = process_ticker(f)
        if len(df) > 0:
            all_dfs.append(df)

    master_df = pd.concat(all_dfs)
    master_df.sort_index(inplace=True)

    unique_dates = np.unique(master_df.index.date)
    split_date   = unique_dates[len(unique_dates)//2]

    train_df = master_df[master_df.index.date < split_date]
    test_df  = master_df[master_df.index.date >= split_date]

    # Compute labels for train_df
    print("Computing labels...")
    H = 12
    for df_part, name in [(train_df, 'train'), (test_df, 'test')]:
        ts_list = [0] * len(df_part)
        tl_list = [0] * len(df_part)
        idx_arr = df_part.index
        for i in range(len(df_part) - H):
            curr_time = idx_arr[i]
            if curr_time.time() < pd.to_datetime('09:30:00').time() or curr_time.time() > pd.to_datetime('14:15:00').time():
                continue
            sub = df_part.iloc[i+1:i+H+1]
            sub = sub[sub.index.date == curr_time.date()]
            sub = sub[sub.index.time <= pd.to_datetime('15:15:00').time()]
            if len(sub) == 0: continue
            close_val = df_part.iloc[i]['close']
            if sub['low'].min() <= close_val * 0.9985:
                ts_list[i] = 1
            if sub['high'].max() >= close_val * 1.0015:
                tl_list[i] = 1
        if name == 'train':
            train_df = train_df.copy()
            train_df['target_short'] = ts_list
            train_df['target_long']  = tl_list
        else:
            test_df = test_df.copy()
            test_df['target_short'] = ts_list
            test_df['target_long']  = tl_list

    print(f"Training unified models on {len(train_df)} rows...")
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[FCOLS].values)

    model_s = LogisticRegression(class_weight='balanced', max_iter=500)
    model_s.fit(X_train, train_df['target_short'].values)

    model_l = LogisticRegression(class_weight='balanced', max_iter=500)
    model_l.fit(X_train, train_df['target_long'].values)

    tr_probs_s = model_s.predict_proba(X_train)[:,1]
    tr_probs_l = model_l.predict_proba(X_train)[:,1]
    thr_s = np.percentile(tr_probs_s[tr_probs_s > 0], 97.0)
    thr_l = np.percentile(tr_probs_l[tr_probs_l > 0], 97.0)

    X_test = sc.transform(test_df[FCOLS].values)
    test_df = test_df.copy()
    test_df['prob_short'] = model_s.predict_proba(X_test)[:,1]
    test_df['prob_long']  = model_l.predict_proba(X_test)[:,1]
    test_df['sig_short']  = (test_df['prob_short'] >= thr_s).astype(int)
    test_df['sig_long']   = (test_df['prob_long']  >= thr_l).astype(int)

    dict_dfs     = {ticker: grp for ticker, grp in test_df.groupby('ticker')}
    signal_times = np.unique(test_df.index)

    # TP/SL combinations to sweep
    configs = [
        # (tp_pct,  sl_pct,  dca_spacing, label)
        (0.0015, 0.0025, 0.0005, "TP=0.15% SL=0.25% [Baseline]"),
        (0.0020, 0.0035, 0.0007, "TP=0.20% SL=0.35%"),
        (0.0025, 0.0040, 0.0008, "TP=0.25% SL=0.40%"),
        (0.0030, 0.0050, 0.0010, "TP=0.30% SL=0.50%"),
        (0.0040, 0.0060, 0.0013, "TP=0.40% SL=0.60%"),
        (0.0050, 0.0075, 0.0015, "TP=0.50% SL=0.75%"),
        (0.0075, 0.0100, 0.0020, "TP=0.75% SL=1.00%"),
        (0.0100, 0.0150, 0.0025, "TP=1.00% SL=1.50%"),
    ]

    print("\nRunning TP/SL parameter sweep...\n")
    print(f"{'Config':<40} {'Trades':>6} {'WinRate':>8} {'EV/Trade':>10} {'Monthly Return':>15}")
    print("-" * 85)

    best = None
    best_ev = -999

    for tp, sl, sp, label in configs:
        res = run_sweep(test_df, signal_times, dict_dfs, tp, sl, sp)
        monthly_ret = res['monthly_ret']
        print(f"{label:<40} {res['trades']:>6} {res['win_rate']:>7.2f}% {res['ev']:>+9.4f}% {monthly_ret:>+14.2f}%")
        if res['ev'] > best_ev and res['trades'] > 30:
            best_ev = res['ev']
            best = (label, res)

    print("-" * 85)
    if best:
        print(f"\n>>> BEST CONFIG: {best[0]}")
        print(f"    Trades: {best[1]['trades']} | Win Rate: {best[1]['win_rate']}% | EV: {best[1]['ev']:+.4f}% | Monthly Return: {best[1]['monthly_ret']:+.2f}%")

if __name__ == '__main__':
    main()
