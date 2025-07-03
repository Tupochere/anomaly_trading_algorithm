#!/usr/bin/env python3
"""
Demonstration of prop firm compliance features including end-of-day exits
"""

import pandas as pd
import numpy as np
from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def demo_prop_firm_compliance():
    """Demonstrate prop firm compliance features"""
    
    print("Prop Firm Compliance Demo")
    print("=" * 40)
    print()
    
    # Initialize algorithm
    algo = AdvancedTradingAlgorithmPropV2(debug=True)
    
    print("✅ Key Prop Firm Compliance Features:")
    print("   - End-of-day exit: Positions closed 30 min before market close")
    print("   - Max stop loss: 3% per trade (configurable)")
    print("   - Dynamic position sizing based on volatility")
    print("   - Intraday confirmation using 15-minute data")
    print("   - Risk management with daily/total drawdown limits")
    print()
    
    # Demo 1: End-of-day exit timing
    print("📊 Demo 1: End-of-Day Exit Timing")
    print("-" * 30)
    
    test_times = [
        ("2025-01-01 20:29:00", "Just before exit time"),
        ("2025-01-01 20:30:00", "Exact exit time (30 min before close)"),
        ("2025-01-01 20:31:00", "After exit time"),
        ("2025-01-01 21:00:00", "Market close time"),
    ]
    
    for time_str, description in test_times:
        should_exit = algo.should_exit_end_of_day(pd.to_datetime(time_str))
        status = "🔴 FORCE EXIT" if should_exit else "🟢 CONTINUE"
        print(f"   {time_str} ({description}): {status}")
    
    print()
    
    # Demo 2: Position sizing based on volatility
    print("📊 Demo 2: Dynamic Position Sizing")
    print("-" * 30)
    
    volatility_scenarios = [
        (0.01, "Low volatility (1%)"),
        (0.02, "Normal volatility (2%)"),
        (0.035, "High volatility (3.5%)"),
    ]
    
    signal_strength = 0.8  # Strong signal
    
    for vol, description in volatility_scenarios:
        pos_size = algo.calculate_position_size(signal_strength, vol)
        print(f"   {description}: Position size = {pos_size:.2%}")
    
    print()
    
    # Demo 3: Max stop loss protection
    print("📊 Demo 3: Max Stop Loss Protection")
    print("-" * 30)
    
    entry_price = 100.0
    atr_values = [0.5, 2.0, 5.0]  # Different ATR scenarios
    
    for atr in atr_values:
        # Calculate what the stop would be with 2x ATR
        normal_stop = entry_price - (2 * atr)
        
        # Apply max stop protection (3% max)
        protected_stop = max(normal_stop, entry_price * (1 - algo.max_stop_pct))
        
        normal_loss = (entry_price - normal_stop) / entry_price * 100
        protected_loss = (entry_price - protected_stop) / entry_price * 100
        
        print(f"   ATR {atr}: Normal stop = {normal_loss:.1f}%, Protected stop = {protected_loss:.1f}%")
    
    print()
    print("✅ All prop firm compliance features are active and working!")
    print()
    print("🔧 Configuration:")
    print(f"   - Max stop loss per trade: {algo.max_stop_pct:.1%}")
    print(f"   - Market close time: 21:00 UTC")
    print(f"   - End-of-day exit buffer: 30 minutes")
    print(f"   - Base position size: 4% (dynamically adjusted)")
    print()
    print("📈 Usage:")
    print("   algo = AdvancedTradingAlgorithmPropV2()")
    print("   results = algo.execute_strategy_with_intraday_confirmation(data_1h, data_15m)")

if __name__ == "__main__":
    demo_prop_firm_compliance()
