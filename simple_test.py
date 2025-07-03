#!/usr/bin/env python3
"""
Simple test for intraday confirmation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Starting test...")
    
    from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2
    print("✓ Import successful")
    
    # Create algorithm instance
    algo = AdvancedTradingAlgorithmPropV2(debug=True)
    print("✓ Algorithm instance created")
    
    # Test intraday confirmation with minimal data
    import pandas as pd
    import numpy as np
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'RSI': [60.0],
        'EMA_12': [100.5],
        'EMA_26': [100.0]
    })
    print("✓ Test data created")
    
    # Test long confirmation
    result_long = algo.get_intraday_confirmation(test_data, 0, 1)
    print(f"✓ Long confirmation test: {result_long}")
    
    # Test short confirmation  
    result_short = algo.get_intraday_confirmation(test_data, 0, -1)
    print(f"✓ Short confirmation test: {result_short}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
