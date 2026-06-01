import numpy as np

runs = 100000
target = 10000.0
success = 0
trades_to_goal = []

# Phase 1: Market Order (prob >= 0.95 pool)
# Stats from task-116: WR 38.8%, Avg Win 3.92x, Avg Loss -0.98x
p1_win_rate = 0.388
p1_win_mult = 3.925
p1_loss_mult = -0.983
p1_signals_per_mo = 2.8

# Phase 2: Limit Order +0.5% (prob >= 0.90 pool)
# Stats from task-308: WR 18.2%, Avg Win 7.55x, Avg Loss -1.0x (approx)
p2_win_rate = 0.182
p2_win_mult = 7.55
p2_loss_mult = -1.0
p2_signals_per_mo = 187 / 24.0 # 7.8 signals/month

# We will also test Phase 2 with standard Market Order for comparison
# Phase 2 Standard: WR 22.9%, Avg Win 4.45x, Avg Loss -0.98x
p2std_win_rate = 0.229
p2std_win_mult = 4.457
p2std_loss_mult = -0.988
p2std_signals_per_mo = 9.0

def run_sim(use_limit_in_p2):
    ok = 0
    months_list = []
    
    for _ in range(runs):
        bal = 10.0
        months = 0.0
        
        while bal < target and bal > 1.0 and months < 120:
            if bal < 100.0:
                # Phase 1
                risk_pct = 0.50
                bet = bal * risk_pct
                if np.random.rand() < p1_win_rate:
                    bal += bet * p1_win_mult
                else:
                    bal += bet * p1_loss_mult
                months += 1.0 / p1_signals_per_mo
            else:
                # Phase 2
                risk_pct = 0.10
                bet = bal * risk_pct
                if use_limit_in_p2:
                    if np.random.rand() < p2_win_rate:
                        bal += bet * p2_win_mult
                    else:
                        bal += bet * p2_loss_mult
                    months += 1.0 / p2_signals_per_mo
                else:
                    if np.random.rand() < p2std_win_rate:
                        bal += bet * p2std_win_mult
                    else:
                        bal += bet * p2std_loss_mult
                    months += 1.0 / p2std_signals_per_mo
                    
        if bal >= target:
            ok += 1
            months_list.append(months)
            
    success_rate = ok / runs * 100
    avg_months = np.mean(months_list) if ok > 0 else 0
    return success_rate, avg_months

print("--- Standard Strategy (Market Order Phase 2) ---")
sr_std, mo_std = run_sim(False)
print(f"Success Rate: {sr_std:.1f}%")
print(f"Average Time: {mo_std:.1f} months")

print("\n--- Hybrid Strategy (Limit Order Phase 2) ---")
sr_hyb, mo_hyb = run_sim(True)
print(f"Success Rate: {sr_hyb:.1f}%")
print(f"Average Time: {mo_hyb:.1f} months")
