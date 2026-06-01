import pandas as pd
import numpy as np

# Simulate the mathematical expectation of the DCA Grid at different leverages
# Assuming Win Rate = 62.4% (from Top 5% threshold in previous test)
# Average gross profit from market movement (before fees/leverage):
# When winning: avg drop from avg entry = 0.5%
# When losing: avg rise from avg entry = 0.525%
# Fees: 0.04% per side (0.0004)

def simulate_leverage(leverage, win_rate=0.624, runs=10000):
    fee_pct = 0.0004
    
    # Net profit percentage per winning trade
    win_pnl = (0.005 * leverage) - (fee_pct * leverage * 2)
    # Net loss percentage per losing trade
    loss_pnl = -(0.00525 * leverage) - (fee_pct * leverage * 2)
    
    # Calculate average expected value
    ev = (win_rate * win_pnl) + ((1 - win_rate) * loss_pnl)
    
    # Simulate Pyramiding
    ok = 0
    months_list = []
    trades_per_month = 330 / 3.0 # Approx 110 unique non-overlapping trades a month
    
    for _ in range(runs):
        bal = 10.0
        trades = 0
        while bal < 10000 and bal > 1.0 and trades < 1000:
            trades += 1
            if np.random.rand() < win_rate:
                bal += bal * win_pnl
            else:
                bal += bal * loss_pnl
                
        if bal >= 10000:
            ok += 1
            months_list.append(trades / trades_per_month)
            
    success_rate = ok / runs * 100
    avg_months = np.mean(months_list) if ok > 0 else float('inf')
    
    print(f"Lev: {int(leverage)}x | Win PnL: +{win_pnl*100:.1f}% | Loss PnL: {loss_pnl*100:.1f}% | EV: {ev*100:+.2f}% | Success to $10k: {success_rate:.1f}% | Avg Months: {avg_months:.1f}")

print("Testing DCA Grid at Higher Leverages (100% Risk per trade):")
for lev in [10, 25, 50, 75, 100, 125]:
    simulate_leverage(lev)
