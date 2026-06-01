import numpy as np

# Multi-Asset Strategy Parameters
win_rate = 0.644
win_pnl = 0.12     # Approx +12% return on margin per winning trade
loss_pnl = -0.1732 # Approx -17.3% return on margin per losing trade (EV = +1.56%)
trades_per_month = 256
runs = 20000

def sim(risk):
    ruin, success, months = 0, 0, []
    
    for _ in range(runs):
        bal = 10.0
        trades = 0
        
        while bal > 1.0 and bal < 10000.0 and trades < 10000:
            trades += 1
            if np.random.rand() < win_rate:
                bal += (bal * risk) * win_pnl
            else:
                bal += (bal * risk) * loss_pnl
                
        if bal >= 10000.0:
            success += 1
            months.append(trades / trades_per_month)
        elif bal <= 1.0:
            ruin += 1
            
    avg_mo = np.mean(months) if months else 0.0
    med_mo = np.median(months) if months else 0.0
    
    print(f"--- RESULTS FOR {int(risk*100)}% RISK PER TRADE ---")
    print(f"Risk of Ruin (Account < $1): {ruin / runs * 100:.1f}%")
    print(f"Success Probability ($10k):  {success / runs * 100:.1f}%")
    if success > 0:
        print(f"Time to Target:              {avg_mo:.1f} Months (Median: {med_mo:.1f} Months)\n")

if __name__ == "__main__":
    print("Running Multi-Asset Monte Carlo ($10 to $10,000) - 20,000 Iterations...\n")
    sim(1.0) # 100% Risk
    sim(0.5) # 50% Risk
