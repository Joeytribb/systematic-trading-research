import numpy as np

win_rate = 0.628; win_pnl = 0.12; loss_pnl = -0.1875
runs = 10000; trades_per_month = 300

def sim(risk):
    ruin, success, months = 0, 0, []
    for _ in range(runs):
        bal = 100.0; trades = 0
        while bal > 1.0 and bal < 10000.0 and trades < 5000:
            trades += 1
            if np.random.rand() < win_rate: bal += (bal * risk) * win_pnl
            else: bal += (bal * risk) * loss_pnl
        if bal >= 10000.0:
            success += 1
            months.append(trades / trades_per_month)
        elif bal <= 1.0: ruin += 1
    
    avg_mo = np.mean(months) if months else 0.0
    print(f'Risk {int(risk*100)}% | Ruin: {ruin/runs*100:.1f}% | Success to $10k: {success/runs*100:.1f}% | Avg Months: {avg_mo:.1f}')

print("Starting Balance: $100")
sim(1.0)
sim(0.5)
