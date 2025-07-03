#!/usr/bin/env python3
"""
Test script for enhanced debug logging in update_risk_limits() method
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def test_enhanced_risk_logging():
    """Test the enhanced debug logging in update_risk_limits method"""
    
    print("🔍 Testing Enhanced Debug Logging in update_risk_limits()")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Normal Trading Conditions",
            "equity": 105000,
            "daily_pnl": 1500,
            "high_watermark": 100000,
            "expected": "No violations"
        },
        {
            "name": "Daily Loss Limit Breach",
            "equity": 95000,
            "daily_pnl": -6000,  # Exceeds -5000 limit
            "high_watermark": 100000,
            "expected": "Daily loss violation"
        },
        {
            "name": "Total Drawdown Limit Breach",
            "equity": 89000,
            "daily_pnl": -1000,
            "high_watermark": 100000,  # 11% drawdown exceeds 10% limit
            "expected": "Total drawdown violation"
        },
        {
            "name": "Both Limits Breached",
            "equity": 85000,
            "daily_pnl": -8000,
            "high_watermark": 100000,
            "expected": "Daily loss violation (checked first)"
        },
        {
            "name": "Edge Case - Exactly at Limit",
            "equity": 95000,
            "daily_pnl": -5000,  # Exactly at limit
            "high_watermark": 100000,
            "expected": "Daily loss violation (at limit)"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        print("-" * 40)
        
        # Test with debug enabled
        print("🔍 With Debug Logging:")
        algo_debug = AdvancedTradingAlgorithmPropV2(debug=True)
        algo_debug.high_watermark = scenario['high_watermark']
        
        result = algo_debug.update_risk_limits(
            equity=scenario['equity'],
            daily_pnl=scenario['daily_pnl'],
            max_daily_loss=-5000,
            max_total_drawdown=-10000
        )
        
        print(f"Result: {'PASSED' if result else 'FAILED'}")
        print(f"Skip Today: {algo_debug.skip_today}")
        print(f"Stop Trading: {algo_debug.stop_trading}")
        
        # Test with debug disabled for comparison
        print("\n🔇 Without Debug Logging:")
        algo_no_debug = AdvancedTradingAlgorithmPropV2(debug=False)
        algo_no_debug.high_watermark = scenario['high_watermark']
        
        result_no_debug = algo_no_debug.update_risk_limits(
            equity=scenario['equity'],
            daily_pnl=scenario['daily_pnl'],
            max_daily_loss=-5000,
            max_total_drawdown=-10000
        )
        
        print(f"Result: {'PASSED' if result_no_debug else 'FAILED'}")
        print(f"Skip Today: {algo_no_debug.skip_today}")
        print(f"Stop Trading: {algo_no_debug.stop_trading}")

def test_progressive_risk_deterioration():
    """Test logging during progressive risk deterioration"""
    
    print(f"\n\n🌊 Testing Progressive Risk Deterioration")
    print("=" * 50)
    
    # Simulate account declining over time
    algo = AdvancedTradingAlgorithmPropV2(debug=True)
    algo.high_watermark = 100000
    
    # Simulate daily trading with declining performance
    daily_scenarios = [
        {"day": 1, "equity": 98000, "daily_pnl": -2000},
        {"day": 2, "equity": 96500, "daily_pnl": -1500},
        {"day": 3, "equity": 94800, "daily_pnl": -1700},
        {"day": 4, "equity": 92500, "daily_pnl": -2300},
        {"day": 5, "equity": 89000, "daily_pnl": -3500},  # Should trigger daily loss
        {"day": 6, "equity": 87000, "daily_pnl": -2000},  # Should trigger total drawdown
    ]
    
    for scenario in daily_scenarios:
        print(f"\n📅 Day {scenario['day']} Trading:")
        print("-" * 30)
        
        # Reset daily flags for new day (except stop_trading)
        if not algo.stop_trading:
            algo.skip_today = False
        
        result = algo.update_risk_limits(
            equity=scenario['equity'],
            daily_pnl=scenario['daily_pnl'],
            max_daily_loss=-5000,
            max_total_drawdown=-10000
        )
        
        if algo.stop_trading:
            print("🛑 Trading permanently stopped - simulation ended")
            break
        elif algo.skip_today:
            print("⏸️  Trading paused for today")

def test_percentage_based_logging():
    """Test percentage calculations in logging"""
    
    print(f"\n\n📊 Testing Percentage Calculations in Logging")
    print("=" * 50)
    
    # Test with different equity levels to verify percentage calculations
    test_cases = [
        {"equity": 100000, "daily_pnl": -3000, "desc": "3% daily loss"},
        {"equity": 50000, "daily_pnl": -2500, "desc": "5% daily loss"},
        {"equity": 200000, "daily_pnl": -8000, "desc": "4% daily loss"},
    ]
    
    for case in test_cases:
        print(f"\n💰 Testing {case['desc']}:")
        print("-" * 30)
        
        algo = AdvancedTradingAlgorithmPropV2(debug=True)
        algo.high_watermark = case['equity']
        
        algo.update_risk_limits(
            equity=case['equity'],
            daily_pnl=case['daily_pnl'],
            max_daily_loss=-5000,
            max_total_drawdown=-10000
        )

if __name__ == "__main__":
    test_enhanced_risk_logging()
    test_progressive_risk_deterioration()
    test_percentage_based_logging()
    
    print(f"\n" + "=" * 60)
    print("✅ Enhanced Debug Logging Tests Complete!")
    print("🔑 Key Features Tested:")
    print("  • Detailed equity and P&L logging")
    print("  • Percentage calculations for losses")
    print("  • Clear violation messages with thresholds")
    print("  • Debug flag toggling functionality")
    print("  • Progressive risk deterioration tracking")
    print("  • Both daily loss and total drawdown monitoring")
