import numpy as np

runs = 10000
target = 10000.0

p1_win_rate = 0.388
p1_win_mult = 3.925
p1_loss_mult = -0.983
p1_signals_per_mo = 2.8

# Standard Phase 2
p2_win_rate = 0.229
p2_win_mult = 4.457
p2_loss_mult = -0.988
p2_signals_per_mo = 9.0

print('R1%\tR2%\tSuccess%\tAvgMonths')
for r1 in [0.5, 0.75, 0.9, 0.95]:
    for r2 in [0.2, 0.4, 0.6, 0.8, 0.9]:
        ok = 0
        months_list = []
        for _ in range(runs):
            bal = 10.0
            months = 0.0
            while bal < target and bal > 1.0 and months < 120:
                if bal < 100.0:
                    bet = bal * r1
                    if np.random.rand() < p1_win_rate:
                        bal += bet * p1_win_mult
                    else:
                        bal += bet * p1_loss_mult
                    months += 1.0 / p1_signals_per_mo
                else:
                    bet = bal * r2
                    if np.random.rand() < p2_win_rate:
                        bal += bet * p2_win_mult
                    else:
                        bal += bet * p2_loss_mult
                    months += 1.0 / p2_signals_per_mo
            if bal >= target:
                ok += 1
                months_list.append(months)
        if ok > 0:
            avg_months = np.mean(months_list)
            success = ok / runs * 100
            if avg_months < 3.0:
                print(f'{int(r1*100)}\t{int(r2*100)}\t{success:.2f}\t\t{avg_months:.2f}')
