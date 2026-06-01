import numpy as np
import pandas as pd
import math

# Parameters from the 2-Year Hyper Growth Backtest
win_rate = 0.628
target_1k = 1000.0
target_10k = 10000.0
trades_per_month = 300
runs = 10000

# Calculate exact Win/Loss PnL needed to yield +0.56% EV at 62.8% WR
# Let's assume a standard 25x leverage DCA Grid:
# Win: we make 0.5% * 25 - fees = ~ +12.0%
# Loss: EV equation -> 0.628 * 0.12 + 0.372 * L = +0.0056
# 0.07536 + 0.372*L = 0.0056 -> 0.372*L = -0.06976 -> L = -18.75%
# This perfectly aligns with our 0.75% stop loss at 25x leverage!
win_pnl = 0.12
loss_pnl = -0.1875

def run_monte_carlo():
    success_1k = 0
    success_10k = 0
    ruin = 0
    
    months_to_1k = []
    months_to_10k = []
    
    print(f"Running Monte Carlo Stress Test ({runs} iterations)...")
    print(f"Win Rate: {win_rate*100:.1f}%, Win PnL: +{win_pnl*100:.1f}%, Loss PnL: {loss_pnl*100:.1f}%, EV: +0.56%")
    
    for _ in range(runs):
        bal = 10.0
        trades = 0
        hit_1k = False
        
        while bal > 1.0 and bal < target_10k and trades < 5000:
            trades += 1
            if np.random.rand() < win_rate:
                bal += (bal * 0.5) * win_pnl
            else:
                bal += (bal * 0.5) * loss_pnl
                
            if bal >= target_1k and not hit_1k:
                hit_1k = True
                months_to_1k.append(trades / trades_per_month)
                success_1k += 1
                
        if bal >= target_10k:
            success_10k += 1
            months_to_10k.append(trades / trades_per_month)
        elif bal <= 1.0:
            ruin += 1
            
    print("\n--- MONTE CARLO RESULTS ---")
    print(f"Risk of Ruin (Account < $1): {ruin / runs * 100:.1f}%")
    
    if success_1k > 0:
        print(f"Probability of hitting $1,000:  {success_1k / runs * 100:.1f}%")
        print(f"Average time to hit $1,000:   {np.mean(months_to_1k):.1f} Months (Median: {np.median(months_to_1k):.1f})")
    
    if success_10k > 0:
        print(f"Probability of hitting $10,000: {success_10k / runs * 100:.1f}%")
        print(f"Average time to hit $10,000:  {np.mean(months_to_10k):.1f} Months (Median: {np.median(months_to_10k):.1f})")
        
if __name__ == "__main__":
    run_monte_carlo()
