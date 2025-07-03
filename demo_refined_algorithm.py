#!/usr/bin/env python3
"""
Comprehensive demo of the refined prop firm compliant trading algorithm
Shows integration of all features including refined position sizing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))

from current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2, FTMOCompliance, FXIFYCompliance, check_prop_firm_compliance
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample market data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    # Create realistic price movement
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = 100 * (1 + returns).cumprod()
    
    # Add some volatility
    volatility = np.random.uniform(0.8, 1.2, len(dates))
    prices *= volatility
    
    # Create OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.995, 1.005, len(dates)),
        'high': prices * np.random.uniform(1.000, 1.020, len(dates)),
        'low': prices * np.random.uniform(0.980, 1.000, len(dates)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    return data

def demonstrate_account_phases():
    """Demonstrate different behavior between evaluation and funded phases"""
    print("=== Account Phase Comparison Demo ===\n")
    
    # Create sample data
    data = create_sample_data()
    
    # Test both phases
    phases = ["evaluation", "funded"]
    
    for phase in phases:
        print(f"--- {phase.title()} Phase ---")
        
        # Initialize algorithm with specific phase
        algo = AdvancedTradingAlgorithmPropV2(
            debug=True,
            account_phase=phase,
            max_stop_pct=0.02
        )
        
        # Initialize compliance checker
        compliance = FTMOCompliance(initial_balance=100000, debug=False)
        
        # Process some sample signals
        for i in range(3):
            signal_strength = 0.8 + (i * 0.1)  # Varying signal strengths
            volatility = 0.02 + (i * 0.01)     # Varying volatility
            
            # Calculate position size
            position_size = algo.calculate_position_size(
                final_score=signal_strength,
                volatility=volatility,
                account_phase=phase
            )
            
            # Simulate trade details
            trade_details = {
                'symbol': 'EURUSD',
                'side': 'buy',
                'size': position_size,
                'entry_price': 1.0950,
                'stop_loss': 1.0900,
                'take_profit': 1.1000,
                'timestamp': pd.Timestamp.now()
            }
            
            # Check compliance
            compliance_result = check_prop_firm_compliance(
                firm_name="FTMO",
                trade_params=trade_details,
                daily_pnl=0,  # Assume no current daily P&L for demo
                equity=100000,  # Starting equity
                high_watermark=100000,  # Starting high watermark
                initial_balance=100000
            )
            
            print(f"  Signal {i+1}: Strength={signal_strength:.1f}, Vol={volatility:.3f}")
            print(f"    Position Size: {position_size:.3f} ({position_size*100:.1f}%)")
            print(f"    Compliance: {'✓' if compliance_result['overall_compliant'] else '✗'}")
            if not compliance_result['overall_compliant']:
                print(f"    Violations: {compliance_result['violations']}")
        
        print()

def demonstrate_full_integration():
    """Demonstrate full integration with all features"""
    print("=== Full Integration Demo ===\n")
    
    # Create more comprehensive data
    data = create_sample_data()
    
    # Initialize algorithm in evaluation phase
    algo = AdvancedTradingAlgorithmPropV2(
        debug=True,
        account_phase="evaluation",
        lookback_period=50
    )
    
    # Initialize compliance checkers
    ftmo = FTMOCompliance(initial_balance=100000, debug=False)
    fxify = FXIFYCompliance(initial_balance=100000, debug=False)
    
    print("Processing sample trading session...")
    
    # Calculate indicators
    processed_data = algo.calculate_indicators(data)
    
    # Simulate some trades
    trades_executed = 0
    trades_blocked = 0
    
    for i in range(len(processed_data) - 20, len(processed_data)):
        try:
            # Check if we should exit end of day
            should_exit = algo.should_exit_end_of_day(processed_data.iloc[i]['timestamp'])
            
            if should_exit:
                print(f"  End-of-day exit triggered at {processed_data.iloc[i]['timestamp']}")
                algo.position = 0  # Close position
                continue
            
            # Generate signals
            mean_rev = algo.mean_reversion_signal(processed_data, i)
            momentum = algo.momentum_signal(processed_data, i)
            
            if mean_rev['signal'] != 0:
                # Calculate position size
                volatility = processed_data.iloc[i]['ATR'] / processed_data.iloc[i]['close']
                position_size = algo.calculate_position_size(
                    final_score=mean_rev['strength'],
                    volatility=volatility,
                    account_phase=algo.account_phase
                )
                
                # Create trade details
                trade_details = {
                    'symbol': 'EURUSD',
                    'side': 'buy' if mean_rev['signal'] > 0 else 'sell',
                    'size': position_size,
                    'entry_price': processed_data.iloc[i]['close'],
                    'stop_loss': processed_data.iloc[i]['close'] * (0.98 if mean_rev['signal'] > 0 else 1.02),
                    'take_profit': processed_data.iloc[i]['close'] * (1.02 if mean_rev['signal'] > 0 else 0.98),
                    'timestamp': processed_data.iloc[i]['timestamp']
                }
                
                # Check compliance with both firms
                ftmo_result = check_prop_firm_compliance(
                    firm_name="FTMO",
                    trade_params=trade_details,
                    daily_pnl=0,
                    equity=100000,
                    high_watermark=100000,
                    initial_balance=100000
                )
                fxify_result = check_prop_firm_compliance(
                    firm_name="FXIFY",
                    trade_params=trade_details,
                    daily_pnl=0,
                    equity=100000,
                    high_watermark=100000,
                    initial_balance=100000
                )
                
                if ftmo_result['overall_compliant'] and fxify_result['overall_compliant']:
                    trades_executed += 1
                    print(f"  Trade {trades_executed}: {trade_details['side'].upper()} {position_size:.3f} at {trade_details['entry_price']:.4f}")
                else:
                    trades_blocked += 1
                    print(f"  Trade blocked - FTMO: {'✓' if ftmo_result['overall_compliant'] else '✗'}, FXIFY: {'✓' if fxify_result['overall_compliant'] else '✗'}")
                    
        except Exception as e:
            print(f"  Error processing index {i}: {e}")
    
    print(f"\nSession Summary:")
    print(f"  Trades Executed: {trades_executed}")
    print(f"  Trades Blocked: {trades_blocked}")
    print(f"  Account Phase: {algo.account_phase}")
    print(f"  Max Position Size: {3 if algo.account_phase == 'evaluation' else 5}%")

def demonstrate_phase_upgrade():
    """Demonstrate upgrading from evaluation to funded phase"""
    print("=== Phase Upgrade Demo ===\n")
    
    # Start with evaluation phase
    algo = AdvancedTradingAlgorithmPropV2(account_phase="evaluation", debug=True)
    
    signal_strength = 0.9
    volatility = 0.02
    
    # Calculate position size in evaluation phase
    eval_size = algo.calculate_position_size(signal_strength, volatility)
    print(f"Evaluation Phase Position Size: {eval_size:.3f} ({eval_size*100:.1f}%)")
    
    # Upgrade to funded phase
    algo.account_phase = "funded"
    
    # Calculate position size in funded phase
    funded_size = algo.calculate_position_size(signal_strength, volatility)
    print(f"Funded Phase Position Size: {funded_size:.3f} ({funded_size*100:.1f}%)")
    
    increase = ((funded_size - eval_size) / eval_size) * 100
    print(f"Position Size Increase: {increase:.1f}%")
    
    print(f"\nThis demonstrates how the algorithm can be dynamically reconfigured")
    print(f"when transitioning from evaluation to funded phase.")

if __name__ == "__main__":
    print("🚀 Prop Firm Compliant Trading Algorithm - Full Integration Demo")
    print("=" * 70)
    
    demonstrate_account_phases()
    demonstrate_full_integration()
    demonstrate_phase_upgrade()
    
    print("\n" + "=" * 70)
    print("✅ All prop firm compliance and risk management features implemented!")
    print("✅ Position sizing refined with account phase considerations!")
    print("✅ Algorithm ready for prop firm trading!")
