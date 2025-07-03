#!/usr/bin/env python3
"""
Test script to demonstrate intraday confirmation functionality
"""

import pandas as pd
import numpy as np
from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def create_sample_data(num_bars=100, timeframe='1H'):
    """Create sample OHLCV data for testing"""
    
    # Create date range
    if timeframe == '1H':
        dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='H')
    elif timeframe == '15T':
        dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='15T')
    else:
        dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='H')
    
    # Generate synthetic price data
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.normal(0, 0.02, num_bars)
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, num_bars)
    }, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def test_intraday_confirmation():
    """Test the intraday confirmation functionality"""
    
    print("Testing Intraday Confirmation Functionality")
    print("=" * 50)
    
    # Create sample data
    data_1h = create_sample_data(num_bars=100, timeframe='1H')
    data_15m = create_sample_data(num_bars=400, timeframe='15T')  # 4x more bars for 15-minute
    
    # Initialize algorithm
    algo = AdvancedTradingAlgorithmPropV2(debug=True)
    
    # Test individual confirmation method
    print("\n1. Testing individual confirmation method:")
    print("-" * 40)
    
    # Add some indicators to 15-minute data
    data_15m_with_indicators = algo.calculate_indicators(data_15m)
    
    # Test long confirmation
    test_idx = data_15m_with_indicators.index[50]  # Use a timestamp from the middle
    long_confirmed = algo.get_intraday_confirmation(data_15m_with_indicators, test_idx, 1)
    print(f"Long confirmation at {test_idx}: {long_confirmed}")
    
    # Test short confirmation
    short_confirmed = algo.get_intraday_confirmation(data_15m_with_indicators, test_idx, -1)
    print(f"Short confirmation at {test_idx}: {short_confirmed}")
    
    # Test with integer index
    long_confirmed_int = algo.get_intraday_confirmation(data_15m_with_indicators, 50, 1)
    print(f"Long confirmation at index 50: {long_confirmed_int}")
    
    print("\n2. Testing strategy execution with intraday confirmation:")
    print("-" * 40)
    
    # Execute strategy with intraday confirmation
    try:
        results = algo.execute_strategy_with_intraday_confirmation(data_1h, data_15m)
        print(f"Strategy executed successfully with {len(results)} results")
        
        # Show some statistics
        if len(results) > 0:
            actions = results['action'].value_counts()
            print(f"Actions taken: {actions.to_dict()}")
            
            # Show any trades
            if len(algo.trades) > 0:
                print(f"Trades executed: {len(algo.trades)}")
                for i, trade in enumerate(algo.trades[:3]):  # Show first 3 trades
                    print(f"  Trade {i+1}: {trade['pnl_pct']:.2%} PnL, {trade['exit_reason']}")
        
    except Exception as e:
        print(f"Error in strategy execution: {e}")
    
    print("\n3. Testing error handling:")
    print("-" * 40)
    
    # Test with invalid index
    invalid_confirmed = algo.get_intraday_confirmation(data_15m_with_indicators, 999999, 1)
    print(f"Confirmation with invalid index: {invalid_confirmed}")
    
    # Test with missing indicators
    empty_data = pd.DataFrame({'close': [100, 101, 102]})
    missing_confirmed = algo.get_intraday_confirmation(empty_data, 0, 1)
    print(f"Confirmation with missing indicators: {missing_confirmed}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_intraday_confirmation()
