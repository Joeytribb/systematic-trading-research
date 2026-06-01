import numpy as np

runs = 100000
target = 1000.0
success = 0
trades_to_goal = []

prob_win = 0.229
win_mult = 4.457
loss_mult = -0.988

for _ in range(runs):
    bal = 10.0
    trades = 0
    while bal < target and bal > 1.0 and trades < 500:
        trades += 1
        risk_pct = 0.50 if bal < 100.0 else 0.10
        bet = bal * risk_pct
        
        is_win = np.random.rand() < prob_win
        if is_win:
            bal += bet * win_mult
        else:
            bal += bet * loss_mult
            
    if bal >= target:
        success += 1
        trades_to_goal.append(trades)

success_rate = success / runs * 100
avg_trades = np.mean(trades_to_goal) if success > 0 else 0
print(f"Success Rate to 1k: {success_rate:.1f}%")
print(f"Average Trades to 1k: {avg_trades:.1f}")
