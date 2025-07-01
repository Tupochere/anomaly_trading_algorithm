import pandas as pd
import numpy as np
from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def simulate_prop_firm_challenge(df, algorithm, initial_balance=100_000, daily_loss_limit=-5000, total_drawdown_limit=-10000):
    """
    Simulate a prop firm challenge with daily loss limits and total drawdown constraints.
    
    Args:
        df: DataFrame with trading data including 'date' column
        algorithm: Instance of AdvancedTradingAlgorithmPropV2
        initial_balance: Starting capital for the challenge
        daily_loss_limit: Maximum daily loss allowed (negative value)
        total_drawdown_limit: Maximum total drawdown allowed (negative value)
    
    Returns:
        final_equity: Final equity after simulation
    """
    equity = initial_balance
    daily_pnl = 0
    high_watermark = initial_balance
    max_drawdown = 0

    for day in df['date'].unique():
        if algorithm.stop_trading:
            print("Challenge ended due to total drawdown.")
            break

        day_df = df[df['date'] == day]
        algorithm.skip_today = False
        algorithm.daily_pnl = 0.0

        # Placeholder: Replace with your per-day strategy execution
        day_pnl = np.random.uniform(-3000, 5000)  # Fake PnL for example
        algorithm.daily_pnl += day_pnl
        equity += day_pnl

        if equity > high_watermark:
            high_watermark = equity

        drawdown = equity - high_watermark
        if drawdown < max_drawdown:
            max_drawdown = drawdown

        algorithm.update_risk_limits(equity, algorithm.daily_pnl, daily_loss_limit, total_drawdown_limit)

        if algorithm.skip_today:
            print(f"Day {day}: Daily loss limit hit. Skip next day trades.")
            continue

        if algorithm.stop_trading:
            print("Total drawdown limit hit. Stopping challenge.")
            break

    print(f"Final equity: ${equity:.2f}")
    print(f"Max drawdown: ${max_drawdown:.2f}")
    return equity
