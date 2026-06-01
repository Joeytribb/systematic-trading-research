import sys
import subprocess

# Ensure yfinance is installed
try:
    import yfinance as yf
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf
import pandas as pd

import os

data_dir = r"c:\Users\onepiece\Documents\_Garage\Ohhv2\data"
os.makedirs(data_dir, exist_ok=True)
dest_path = os.path.join(data_dir, "NIFTY50-5m.csv")

print("Downloading 5m historical data for Nifty 50 (^NSEI) from Yahoo Finance...")
# yfinance allows max 59 days of 5m data
ticker = "^NSEI"
df = yf.download(tickers=ticker, period="60d", interval="5m")

if df.empty:
    print("Error: Downloaded dataframe is empty. Trying daily data or checking ticker.")
    # Try backup ticker or daily
    df = yf.download(tickers=ticker, period="1y", interval="1d")
    print(f"Downloaded daily data instead. Shape: {df.shape}")
else:
    print(f"Successfully downloaded 5m data. Shape: {df.shape}")

# Flatten columns if multi-indexed (yfinance v2 sometimes has multi-index columns)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Save
df.to_csv(dest_path)
print(f"Saved Nifty data to {dest_path}")
print("Sample data:")
print(df.head())
