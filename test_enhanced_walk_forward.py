#!/usr/bin/env python3
"""
Test script for enhanced walk-forward testing with prop firm compliance validation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def create_sample_data_for_testing():
    """Create extended sample data for walk-forward testing"""
    np.random.seed(42)
    
    # Create 1000 data points (approximately 6 months of hourly data)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    
    # Create realistic price movement with trend and volatility
    returns = np.random.normal(0.0002, 0.015, len(dates))  # Slightly positive drift
    
    # Add some regime changes
    regime_changes = [200, 400, 600, 800]
    for change_point in regime_changes:
        if change_point < len(returns):
            # Add volatility spike
            returns[change_point:change_point+50] *= 2.0
    
    prices = 100 * (1 + returns).cumprod()
    
    # Add some realistic volatility clustering
    volatility = np.random.uniform(0.98, 1.02, len(dates))
    prices *= volatility
    
    # Create OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.998, 1.002, len(dates)),
        'high': prices * np.random.uniform(1.000, 1.015, len(dates)),
        'low': prices * np.random.uniform(0.985, 1.000, len(dates)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    data.set_index('timestamp', inplace=True)
    return data

def test_enhanced_walk_forward():
    """Test the enhanced walk-forward testing with compliance validation"""
    
    print("🚀 Testing Enhanced Walk-Forward Analysis with Prop Firm Compliance")
    print("=" * 70)
    
    # Create sample data
    data = create_sample_data_for_testing()
    print(f"📊 Created test data: {len(data)} data points")
    print(f"📅 Period: {data.index[0]} to {data.index[-1]}")
    print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "FTMO Evaluation Phase",
            "account_phase": "evaluation",
            "compliance_firm": "FTMO",
            "initial_balance": 100000
        },
        {
            "name": "FTMO Funded Phase", 
            "account_phase": "funded",
            "compliance_firm": "FTMO",
            "initial_balance": 100000
        },
        {
            "name": "FXIFY Starter Account",
            "account_phase": "evaluation",
            "compliance_firm": "FXIFY",
            "initial_balance": 100000
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*50}")
        print(f"🏛️ Testing Scenario: {scenario['name']}")
        print(f"{'='*50}")
        
        # Initialize algorithm with scenario parameters
        algo = AdvancedTradingAlgorithmPropV2(
            account_phase=scenario['account_phase'],
            debug=False,
            lookback_period=100
        )
        
        # Run enhanced walk-forward test
        try:
            walk_results = algo.walk_forward_test(
                data,
                window_size_days=30,  # Smaller windows for demo
                step_size_days=15,    # More frequent testing
                compliance_firm=scenario['compliance_firm'],
                initial_balance=scenario['initial_balance']
            )
            
            if not walk_results.empty:
                print(f"\n📈 DETAILED RESULTS FOR {scenario['name']}:")
                print("-" * 50)
                
                # Show first few windows with details
                display_cols = ['window_number', 'total_return_pct', 'win_rate', 'num_trades', 
                               'compliance_passed', 'compliance_notes']
                
                print(walk_results[display_cols].head(10).to_string(index=False))
                
                # Calculate compliance statistics
                total_windows = len(walk_results)
                compliant_windows = walk_results['compliance_passed'].sum()
                compliance_rate = compliant_windows / total_windows * 100
                
                print(f"\n🎯 SCENARIO SUMMARY:")
                print(f"Compliance Rate: {compliant_windows}/{total_windows} ({compliance_rate:.1f}%)")
                print(f"Average Return: {walk_results['total_return_pct'].mean():.2f}%")
                print(f"Profitable Windows: {(walk_results['total_return_pct'] > 0).sum()}/{total_windows}")
                
                # Assessment
                if compliance_rate >= 80:
                    assessment = "🟢 EXCELLENT - Ready for deployment"
                elif compliance_rate >= 60:
                    assessment = "🟡 GOOD - Suitable with monitoring"
                elif compliance_rate >= 40:
                    assessment = "🟠 FAIR - Needs optimization"
                else:
                    assessment = "🔴 POOR - Not suitable"
                
                print(f"Assessment: {assessment}")
                
                # Save results
                filename = f"enhanced_walk_forward_{scenario['name'].lower().replace(' ', '_')}.csv"
                filepath = f"backtests/results/experiments/{filename}"
                walk_results.to_csv(filepath, index=False)
                print(f"📁 Results saved to: {filepath}")
                
            else:
                print("❌ No results generated")
                
        except Exception as e:
            print(f"❌ Error in scenario '{scenario['name']}': {e}")
            import traceback
            traceback.print_exc()

def test_compliance_edge_cases():
    """Test compliance validation with edge cases"""
    print(f"\n{'='*50}")
    print("🧪 Testing Compliance Edge Cases")
    print(f"{'='*50}")
    
    # Create data with extreme moves to trigger compliance violations
    np.random.seed(123)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
    
    # Create scenario with large drawdowns
    returns = np.random.normal(-0.001, 0.03, len(dates))  # Negative drift, high volatility
    
    # Add some extreme negative days
    extreme_loss_days = [50, 100, 150]
    for day in extreme_loss_days:
        if day < len(returns):
            returns[day] = -0.08  # 8% loss in one period
    
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    })
    data.set_index('timestamp', inplace=True)
    
    print(f"📊 Created stress test data with extreme moves")
    print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"📉 Total decline: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.1f}%")
    
    # Test with FTMO rules (should fail due to drawdowns)
    algo = AdvancedTradingAlgorithmPropV2(account_phase="evaluation", debug=False)
    
    try:
        stress_results = algo.walk_forward_test(
            data,
            window_size_days=20,
            step_size_days=10,
            compliance_firm="FTMO",
            initial_balance=100000
        )
        
        if not stress_results.empty:
            failed_windows = stress_results[~stress_results['compliance_passed']]
            print(f"\n📊 STRESS TEST RESULTS:")
            print(f"Failed Windows: {len(failed_windows)}/{len(stress_results)}")
            print(f"Common Failure Reasons:")
            
            # Analyze failure reasons
            for _, row in failed_windows.head(5).iterrows():
                print(f"  Window {row['window_number']}: {row['compliance_notes']}")
        
    except Exception as e:
        print(f"❌ Error in stress test: {e}")

if __name__ == "__main__":
    test_enhanced_walk_forward()
    test_compliance_edge_cases()
    
    print(f"\n{'='*70}")
    print("✅ Enhanced Walk-Forward Testing with Compliance Validation Complete!")
    print("🎯 Key Features Demonstrated:")
    print("  - Real-time compliance checking during walk-forward testing")
    print("  - FTMO and FXIFY rule validation")
    print("  - Account phase-aware position sizing compliance")
    print("  - Detailed violation reporting and analysis")
    print("  - Deployment readiness assessment")
    print("📁 All results saved to backtests/results/experiments/")
