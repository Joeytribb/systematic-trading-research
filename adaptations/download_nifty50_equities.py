import yfinance as yf
import os
import pandas as pd
import time

# Current top Nifty 50 constituents
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "M&M.NS", "NTPC.NS", "KOTAKBANK.NS", "TITAN.NS", "ONGC.NS",
    "ULTRACEMCO.NS", "AXISBANK.NS", "COALIND.NS", "POWERGRID.NS", "ADANIENT.NS",
    "BAJAJFINSV.NS", "ASIANPAINT.NS", "GRASIM.NS", "JSWSTEEL.NS", "VEDL.NS",
    "NESTLEIND.NS", "WIPRO.NS", "DRREDDY.NS", "HINDALCO.NS", "TATASTEEL.NS",
    "CIPLA.NS", "SBILIFE.NS", "TECHM.NS", "BRITANNIA.NS", "APOLLOHOSP.NS",
    "EICHERMOT.NS", "DIVISLAB.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS", "INDUSINDBK.NS",
    "TATACONSUM.NS", "BPCL.NS", "HEROMOTOCO.NS", "UPL.NS"
]

DATA_DIR = "c:/Users/onepiece/Documents/_Garage/Ohhv2/data/nifty50_equities"

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Downloading 60 days of 5m data for {len(NIFTY50_TICKERS)} Nifty 50 constituents...")
    
    success = 0
    for ticker in NIFTY50_TICKERS:
        filepath = os.path.join(DATA_DIR, f"{ticker.replace('.NS', '')}-5m.csv")
        print(f"Fetching {ticker}...", end=" ")
        
        try:
            # yfinance max intraday limit is 60d
            data = yf.download(ticker, interval="5m", period="60d", progress=False)
            if len(data) > 0:
                data.to_csv(filepath)
                print(f"Saved {len(data)} rows.")
                success += 1
            else:
                print("No data found.")
            time.sleep(0.5) # Basic rate limiting
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"Successfully downloaded {success}/{len(NIFTY50_TICKERS)} tickers.")

if __name__ == "__main__":
    main()
