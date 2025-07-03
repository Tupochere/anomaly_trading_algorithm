#!/usr/bin/env python3
"""
Test script to demonstrate end-of-day exit functionality
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def create_test_data_with_timestamps():
    """Create test data with specific timestamps including end-of-day times"""
    
    # Create hourly data spanning across multiple days including end-of-day times
    start_date = '2025-01-01 08:00:00'  # Start at 8 AM
    periods = 50  # About 2-3 days of hourly data
    
    dates = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Generate synthetic price data
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.normal(0, 0.01, periods)
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    }, index=dates)
    
    return data

def test_end_of_day_exit():
    """Test the end-of-day exit functionality"""
    
    print("Testing End-of-Day Exit Functionality")
    print("=" * 50)
    
    # Create algorithm instance
    algo = AdvancedTradingAlgorithmPropV2(debug=True)
    
    # Test individual should_exit_end_of_day method
    print("\n1. Testing should_exit_end_of_day method:")
    print("-" * 40)
    
    test_times = [
        "2025-01-01 19:00:00",  # 7 PM - should NOT exit
        "2025-01-01 20:00:00",  # 8 PM - should NOT exit
        "2025-01-01 20:30:00",  # 8:30 PM - should EXIT (30 min before close)
        "2025-01-01 20:45:00",  # 8:45 PM - should EXIT
        "2025-01-01 21:00:00",  # 9 PM - should EXIT (market close)
        "2025-01-01 22:00:00",  # 10 PM - should EXIT
    ]
    
    for time_str in test_times:
        timestamp = pd.to_datetime(time_str)
        should_exit = algo.should_exit_end_of_day(timestamp)
        print(f"  {time_str}: {'EXIT' if should_exit else 'HOLD'}")
    
    print("\n2. Testing with different time formats:")
    print("-" * 40)
    
    # Test with different time formats
    test_formats = [
        pd.Timestamp("2025-01-01 20:30:00"),  # Pandas Timestamp
        "2025-01-01 20:30:00",                # String
        pd.to_datetime("2025-01-01 20:30:00") # Datetime
    ]
    
    for time_obj in test_formats:
        should_exit = algo.should_exit_end_of_day(time_obj)
        print(f"  {type(time_obj).__name__}: {'EXIT' if should_exit else 'HOLD'}")
    
    print("\n3. Testing strategy execution with end-of-day exits:")
    print("-" * 40)
    
    # Create test data
    data = create_test_data_with_timestamps()
    print(f"Created test data with {len(data)} bars")
    print(f"Time range: {data.index[0]} to {data.index[-1]}")
    
    # Execute strategy
    try:
        results = algo.execute_strategy_with_intraday_confirmation(data, None)
        print(f"Strategy executed successfully with {len(results)} results")
        
        # Check for end-of-day exits
        end_of_day_exits = results[results['action'] == 'EXIT_END_OF_DAY']
        print(f"End-of-day exits found: {len(end_of_day_exits)}")
        
        if len(end_of_day_exits) > 0:
            print("End-of-day exit details:")
            for idx, exit_row in end_of_day_exits.iterrows():
                print(f"  {exit_row['timestamp']}: Price {exit_row['close']:.2f}")
        
        # Show action summary
        action_counts = results['action'].value_counts()
        print(f"\nAction summary:")
        for action, count in action_counts.items():
            print(f"  {action}: {count}")
        
        # Check trades that were closed by end-of-day exits
        eod_trades = [trade for trade in algo.trades if trade['exit_reason'] == 'EXIT_END_OF_DAY']
        print(f"\nTrades closed by end-of-day: {len(eod_trades)}")
        
        for i, trade in enumerate(eod_trades):
            print(f"  Trade {i+1}: Entry={trade['entry_price']:.2f}, Exit={trade['exit_price']:.2f}, "
                  f"P&L={trade['pnl_pct']:.2%}, Duration={trade['duration']} bars")
        
    except Exception as e:
        print(f"Error during strategy execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing edge cases:")
    print("-" * 40)
    
    # Test edge cases
    edge_cases = [
        None,           # None input
        "invalid",      # Invalid string
        12345,          # Number
    ]
    
    for case in edge_cases:
        try:
            result = algo.should_exit_end_of_day(case)
            print(f"  {case}: {result}")
        except Exception as e:
            print(f"  {case}: Error - {e}")
    
    print("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_end_of_day_exit()
