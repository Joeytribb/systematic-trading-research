import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters based on the audited multi-asset performance
# Purified Win Rate: 64.5%
# Frequency: ~338 trades per month
win_rate = 0.645
win_pnl = 0.12      # +12.0% return on margin per winning trade
loss_pnl = -0.1732  # -17.32% return on margin per losing trade
trades_per_month = 338
runs = 20000

def run_simulation(start_bal, target_bal, risk_pct, two_stage=False, stage2_threshold=500.0, stage2_risk=0.10):
    ruin_count = 0
    success_count = 0
    trade_counts = []
    
    for _ in range(runs):
        bal = start_bal
        trades = 0
        
        while bal > 1.0 and bal < target_bal and trades < 15000:
            trades += 1
            # Determine current risk
            current_risk = risk_pct
            if two_stage and bal >= stage2_threshold:
                current_risk = stage2_risk
                
            if np.random.rand() < win_rate:
                bal += (bal * current_risk) * win_pnl
            else:
                bal += (bal * current_risk) * loss_pnl
                
        if bal >= target_bal:
            success_count += 1
            trade_counts.append(trades)
        elif bal <= 1.0:
            ruin_count += 1
            
    success_rate = success_count / runs * 100
    ruin_rate = ruin_count / runs * 100
    
    if success_count > 0:
        avg_trades = np.mean(trade_counts)
        med_trades = np.median(trade_counts)
        avg_months = avg_trades / trades_per_month
        med_months = med_trades / trades_per_month
    else:
        avg_months, med_months = float('inf'), float('inf')
        
    return success_rate, ruin_rate, avg_months, med_months

# Let's also model the Slippage and Leverage Limits to find the Max Account Balance.
# As position size increases, the exchange enforces lower leverage tiers:
# Tier 1: Position <= $50,000 -> 25x max
# Tier 2: Position <= $250,000 -> 20x max
# Tier 3: Position <= $1,000,000 -> 10x max
# Tier 4: Position <= $5,000,000 -> 5x max
# Tier 5: Position > $5,000,000 -> 2x max (or less)
#
# Additionally, slippage increases as a function of position size:
# Slippage = base_slippage + coefficient * (position_size / daily_volume)
# Let's model this. If we limit our position size to a maximum dollar amount (e.g., $250,000 to keep slippage low), 
# or if we let the balance grow and scale leverage/slippage, what is the ceiling?

def simulate_max_balance():
    # Model a single trajectory with slippage and exchange limits
    # Starting at $50, using 10% risk, 25x base leverage.
    # We will simulate 1000 trajectories to see where they plateau or get ruined.
    np.random.seed(42)
    trajectories = 500
    max_trades = 20000
    
    final_balances = []
    
    for _ in range(trajectories):
        bal = 50.0
        trades = 0
        while bal > 1.0 and trades < max_trades:
            trades += 1
            # Current risk percentage (stage 1/2)
            risk = 0.50 if bal < 200.0 else 0.10
            
            # Exchange leverage and position limit rules
            # We assume we want to risk a portion of balance, but leverage is capped by the exchange:
            margin = bal * risk
            
            # Position size if we used full 25x leverage
            desired_pos = margin * 25.0
            
            # Leverage limits based on position size (typical Binance/Bybit rules)
            if desired_pos <= 50000:
                leverage = 25.0
            elif desired_pos <= 250000:
                leverage = 20.0
            elif desired_pos <= 1000000:
                leverage = 10.0
            elif desired_pos <= 5000000:
                leverage = 5.0
            else:
                leverage = 2.0
                
            pos_size = margin * leverage
            
            # Slippage model: Altcoin liquidity limit
            # Average daily volume across our altcoins SOL, LINK, DOGE is around $150M.
            # 5-minute bar volume is around $500,000.
            # If our position size is a significant fraction of 5-min volume, slippage increases.
            # For BTC/ETH, volume is higher, but we pick the worst-case (altcoins).
            # Slippage adds to loss and subtracts from win.
            # Base slippage = 0.05% of price (0.0005)
            # Market impact = 0.001 * (pos_size / 500000)**2
            market_impact = 0.0005 + 0.001 * ((pos_size / 300000.0) ** 1.5)
            # Cap market impact to 2% max (beyond which we are just market moving)
            market_impact = min(market_impact, 0.02)
            
            # Net PnL calculation with leverage-adjusted returns and slippage
            # Win: price drops 0.50% -> return on margin = (0.005 - slippage) * leverage - fee * leverage * 2
            # Loss: price rises 0.525% -> return on margin = -(0.00525 + slippage) * leverage - fee * leverage * 2
            fee_pct = 0.0004
            
            win_return = (0.005 - market_impact) * leverage - (fee_pct * leverage * 2)
            loss_return = -(0.00525 + market_impact) * leverage - (fee_pct * leverage * 2)
            
            # If win_return <= 0, we can no longer grow
            if win_return <= loss_return or win_return < 0:
                # We have hit the liquidity wall!
                break
                
            if np.random.rand() < win_rate:
                bal += margin * win_return
            else:
                bal += margin * loss_return
                
        final_balances.append(bal)
        
    return final_balances

if __name__ == "__main__":
    print("=== MONTE CARLO ANALYSIS: STARTING FROM $50 ===")
    
    # 1. 50% risk all the way
    s, r, a, m = run_simulation(50.0, 10000.0, 0.50)
    print(f"50% Risk: Success: {s:.2f}%, Ruin: {r:.2f}%, Avg Time: {a:.2f} months, Median Time: {m:.2f} months")
    
    # 2. 10% risk all the way
    s, r, a, m = run_simulation(50.0, 10000.0, 0.10)
    print(f"10% Risk: Success: {s:.2f}%, Ruin: {r:.2f}%, Avg Time: {a:.2f} months, Median Time: {m:.2f} months")
    
    # 3. Two-Stage (50% up to $200, then 10% risk)
    s, r, a, m = run_simulation(50.0, 10000.0, 0.50, two_stage=True, stage2_threshold=200.0, stage2_risk=0.10)
    print(f"Two-Stage ($200 threshold): Success: {s:.2f}%, Ruin: {r:.2f}%, Avg Time: {a:.2f} months, Median Time: {m:.2f} months")
    
    # 4. Two-Stage (50% up to $500, then 10% risk)
    s, r, a, m = run_simulation(50.0, 10000.0, 0.50, two_stage=True, stage2_threshold=500.0, stage2_risk=0.10)
    print(f"Two-Stage ($500 threshold): Success: {s:.2f}%, Ruin: {r:.2f}%, Avg Time: {a:.2f} months, Median Time: {m:.2f} months")
    
    # 5. Two-Stage (50% up to $1000, then 10% risk)
    s, r, a, m = run_simulation(50.0, 10000.0, 0.50, two_stage=True, stage2_threshold=1000.0, stage2_risk=0.10)
    print(f"Two-Stage ($1000 threshold): Success: {s:.2f}%, Ruin: {r:.2f}%, Avg Time: {a:.2f} months, Median Time: {m:.2f} months")

    print("\n=== RUNNING LIQUIDITY & EXCHANGE LIMIT SIMULATION ===")
    max_bals = simulate_max_balance()
    print(f"Average Max Balance reached: ${np.mean(max_bals):,.2f}")
    print(f"Median Max Balance reached:  ${np.median(max_bals):,.2f}")
    print(f"90th Percentile Max Balance: ${np.percentile(max_bals, 90):,.2f}")
    print(f"Maximum Balance achieved in simulations: ${np.max(max_bals):,.2f}")
