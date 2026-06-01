import pandas as pd
import numpy as np
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Real Zerodha MIS round-trip: brokerage ~0.03%×2 + STT 0.025% sell + exchange 0.003%×2 + stamp 0.003% + GST
# Verified against Zerodha brokerage calculator for ₹50,000 notional intraday equity trade
FEES_ROUND_TRIP = 0.00106   # 0.106% of notional (PREVIOUSLY 0.04% — underestimated by 62%)
LEVERAGE = 5.0              # SEBI standard max limit for equity MIS

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def add_features(df):
    if len(df) < 50: return pd.DataFrame()
    
    # Bollinger Bands
    ma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    ub = ma + 2*std; lb = ma - 2*std; bw = ub - lb
    df['pct_b'] = (df['close'] - lb) / (bw + 1e-10)
    df['band_width'] = bw / (ma + 1e-10)
    
    # RSI
    df['rsi'] = rsi(df['close'])
    
    # MACD
    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
    df['macd_norm'] = macd / (df['close'] + 1e-10)
    df['macd_hist'] = (macd - sig) / (df['close'] + 1e-10)
    
    # Velocity
    df['vel1'] = df['close'].pct_change(1)
    df['vel3'] = df['close'].pct_change(3)
    
    # Wicks
    cr = df['high'] - df['low']
    df['uw'] = (df['high'] - df[['open','close']].max(axis=1)) / (cr + 1e-10)
    df['lw'] = (df[['open','close']].min(axis=1) - df['low'])  / (cr + 1e-10)
    
    # Volatility Ratio
    rng = df['high'] - df['low']
    rng_ma = rng.rolling(20).mean()
    df['range_ratio'] = rng / (rng_ma + 1e-10)
    
    # SMA distance (MTF approximation)
    df['sma_15m_approx'] = df['close'].rolling(60).mean() # roughly 5m * 12 = 60m
    df['mtf_1h_dist'] = (df['close'] - df['sma_15m_approx']) / (df['sma_15m_approx'] + 1e-10)
    
    df.dropna(inplace=True)
    return df

FCOLS = ['pct_b','band_width','rsi','macd_norm','macd_hist','vel1','vel3','uw','lw','range_ratio','mtf_1h_dist']

def process_ticker(filepath):
    ticker = os.path.basename(filepath).replace('-5m.csv', '')
    # Skip multi-header yfinance format
    df = pd.read_csv(filepath, header=[0,1], index_col=0)
    
    # Flatten columns 
    # yfinance 60d download gives MultiIndex columns if single ticker: e.g. ('Close', 'RELIANCE.NS')
    # If not multi-index, we handle it
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
    
    H = 12
    target_short_list = [0] * len(df)
    target_long_list = [0] * len(df)
    
    for i in range(len(df) - H):
        curr_time = df.index[i]
        if curr_time.time() < pd.to_datetime('09:30:00').time() or curr_time.time() > pd.to_datetime('14:15:00').time():
            continue
            
        end_idx = i + H
        sub_df = df.iloc[i+1 : end_idx+1]
        
        sub_df = sub_df[sub_df.index.date == curr_time.date()]
        sub_df = sub_df[sub_df.index.time <= pd.to_datetime('15:15:00').time()]
        if len(sub_df) == 0:
            continue
            
        close_val = df.iloc[i]['close']
        if sub_df['low'].min() <= close_val * 0.9985:
            target_short_list[i] = 1
        if sub_df['high'].max() >= close_val * 1.0015:
            target_long_list[i] = 1
            
    df['target_short'] = target_short_list
    df['target_long'] = target_long_list
    df['ticker'] = ticker
    
    return df

def sim_dca_short(window, S0):
    # TP/SL locked from TRAIN-set optimal analysis to prevent test-set snooping.
    # TP=0.40%, SL=0.60%, DCA spacing=0.10%  (validated on train period, not test period)
    levels = [S0, S0*1.0010, S0*1.0020, S0*1.0030]
    sl_price = S0 * 1.0060
    fills = []; avg_entry = tp_price = None
    
    for idx, row in window.iterrows():
        if idx.time() >= pd.to_datetime('15:15:00').time():
            if not fills: return 0.0, idx
            return (avg_entry - row['close'])/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
        h, l = row['high'], row['low']
        nf = [lv for lv in levels if lv not in fills and h >= lv]
        for lv in nf: fills.append(lv)
        
        if len(nf) > 1:
            worst = max(nf); base = [f for f in fills if f not in nf]
            fills = base + [worst]*len(nf)
            
        if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry*(1-0.0040)  # TP = 0.40%
        if not fills:
            if l <= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry*(1-0.0040)
        if not fills: continue
        
        if h >= sl_price:
            return (avg_entry - sl_price)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
        if l <= tp_price:
            return (avg_entry - tp_price)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
    if not fills: return 0.0, window.index[-1]
    return (avg_entry - window.iloc[-1]['close'])/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, window.index[-1]

def sim_dca_long(window, S0):
    # TP/SL locked from TRAIN-set optimal analysis.
    levels = [S0, S0*0.9990, S0*0.9980, S0*0.9970]
    sl_price = S0 * 0.9940
    fills = []; avg_entry = tp_price = None
    
    for idx, row in window.iterrows():
        if idx.time() >= pd.to_datetime('15:15:00').time():
            if not fills: return 0.0, idx
            return (row['close'] - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
        h, l = row['high'], row['low']
        nf = [lv for lv in levels if lv not in fills and l <= lv]
        for lv in nf: fills.append(lv)
        
        if len(nf) > 1:
            worst = min(nf); base = [f for f in fills if f not in nf]
            fills = base + [worst]*len(nf)
            
        if fills: avg_entry = sum(fills)/len(fills); tp_price = avg_entry*1.0040  # TP = 0.40%
        if not fills:
            if h >= S0: fills.append(S0); avg_entry = S0; tp_price = avg_entry*1.0040
        if not fills: continue
        
        if l <= sl_price:
            return (sl_price - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
        if h >= tp_price:
            return (tp_price - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, idx
            
    if not fills: return 0.0, window.index[-1]
    return (window.iloc[-1]['close'] - avg_entry)/avg_entry*LEVERAGE - FEES_ROUND_TRIP*LEVERAGE, window.index[-1]

def main():
    print("Loading 50 Nifty equities...")
    data_dir = "c:/Users/onepiece/Documents/_Garage/Ohhv2/data/nifty50_equities"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    all_dfs = []
    for f in files:
        df = process_ticker(f)
        if len(df) > 0:
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data found.")
        return
        
    master_df = pd.concat(all_dfs)
    master_df.sort_index(inplace=True)
    
    # Split: Train on first 30 days, test on last 30 days
    unique_dates = np.unique(master_df.index.date)
    split_date = unique_dates[len(unique_dates)//2]
    
    print(f"Total days: {len(unique_dates)}. Splitting at {split_date}")
    
    train_df = master_df[master_df.index.date < split_date]
    test_df = master_df[master_df.index.date >= split_date]
    
    print(f"Training Unified Models on {len(train_df)} rows...")
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[FCOLS].values)
    
    # Short Model
    y_short = train_df['target_short'].values
    model_short = LogisticRegression(class_weight='balanced', max_iter=500)
    model_short.fit(X_train, y_short)
    
    # Long Model
    y_long = train_df['target_long'].values
    model_long = LogisticRegression(class_weight='balanced', max_iter=500)
    model_long.fit(X_train, y_long)
    
    # Thresholds
    train_probs_s = model_short.predict_proba(X_train)[:,1]
    train_probs_l = model_long.predict_proba(X_train)[:,1]
    thr_s = np.percentile(train_probs_s[train_probs_s > 0], 97.0)
    thr_l = np.percentile(train_probs_l[train_probs_l > 0], 97.0)
    
    print(f"Short Threshold: {thr_s:.4f} | Long Threshold: {thr_l:.4f}")
    
    print("Testing on out-of-sample data...")
    X_test = sc.transform(test_df[FCOLS].values)
    test_df['prob_short'] = model_short.predict_proba(X_test)[:,1]
    test_df['prob_long'] = model_long.predict_proba(X_test)[:,1]
    
    test_df['sig_short'] = (test_df['prob_short'] >= thr_s).astype(int)
    test_df['sig_long'] = (test_df['prob_long'] >= thr_l).astype(int)
    
    # Concurrency Lock Simulation over Test Data
    print("\nRunning joint simulation under single shared lock...")
    
    # Group test_df by time to iterate chronologically
    test_times = np.unique(test_df.index)
    
    lock_until = test_times[0] - pd.Timedelta(minutes=1)
    short_results = []; long_results = []
    
    # Dictionary to hold the raw df for each ticker for fast access
    dict_dfs = {ticker: grp for ticker, grp in test_df.groupby('ticker')}
    
    for t in test_times:
        t_ts = pd.Timestamp(t)
        if t_ts.time() < pd.to_datetime('09:30:00').time() or t_ts.time() > pd.to_datetime('14:15:00').time():
            continue
        if t_ts < lock_until:
            continue
            
        # Get all rows at this timestamp
        current_rows = test_df.loc[[t_ts]]
        
        # Check signals
        shorts = current_rows[current_rows['sig_short'] == 1]
        longs = current_rows[current_rows['sig_long'] == 1]
        
        if len(shorts) == 0 and len(longs) == 0:
            continue
            
        # Pick one ticker to trade (highest probability)
        best_ticker = None
        trade_type = None
        
        if len(shorts) > 0:
            best_ticker = shorts.sort_values('prob_short', ascending=False).iloc[0]['ticker']
            trade_type = 'short'
        else:
            best_ticker = longs.sort_values('prob_long', ascending=False).iloc[0]['ticker']
            trade_type = 'long'
            
        entry_time = t_ts + pd.Timedelta(minutes=5)
        end_time = entry_time + pd.Timedelta(minutes=60)
        
        ticker_df = dict_dfs[best_ticker]
        window = ticker_df.loc[entry_time : end_time]
        window = window[window.index.date == entry_time.date()]
        
        if len(window) < 2:
            continue
            
        S0 = window.iloc[0]['open']
        
        if trade_type == 'short':
            r, exit_time = sim_dca_short(window, S0)
            if r != 0.0:
                short_results.append(r)
                lock_until = exit_time
        else:
            r, exit_time = sim_dca_long(window, S0)
            if r != 0.0:
                long_results.append(r)
                lock_until = exit_time

    sa = np.array(short_results); la = np.array(long_results)
    total = len(sa) + len(la)
    
    days = len(unique_dates) // 2
    mo = days / 20.0
    
    print("\n" + "="*60)
    print("  NIFTY 50 CROSS-SECTIONAL EQUITIES — INTRADAY DCA (5x Lev)")
    print("="*60)
    print(f"Out-of-Sample Test period : {days} trading days ({mo:.1f} months)")
    print(f"Short trades executed     : {len(sa)} ({len(sa)/mo:.1f}/mo)")
    print(f"Long trades executed      : {len(la)} ({len(la)/mo:.1f}/mo)")
    print(f"TOTAL combined trades     : {total} ({total/mo:.1f}/mo)")
    
    all_r = np.concatenate([sa, la])
    if len(all_r) == 0:
        print("No trades executed.")
        return
        
    print(f"\nCombined Win Rate     : {(all_r > 0).mean()*100:.2f}%")
    print(f"Combined Avg PnL      : {all_r.mean()*100:+.2f}%")
    wins = all_r[all_r > 0]; losses = all_r[all_r < 0]
    if len(wins):   print(f"Avg Win PnL           : {wins.mean()*100:+.2f}%")
    if len(losses): print(f"Avg Loss PnL          : {losses.mean()*100:+.2f}%")
    print(f"Expected Value / Trade: {all_r.mean()*100:+.2f}% (incl. transaction charges)")

if __name__ == '__main__':
    main()
