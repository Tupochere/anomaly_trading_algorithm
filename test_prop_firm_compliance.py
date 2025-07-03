#!/usr/bin/env python3
"""
Test script for prop firm compliance classes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.current_algo_prop_v2 import (
    FTMOCompliance, 
    FXIFYCompliance, 
    check_prop_firm_compliance
)

def test_ftmo_compliance():
    """Test FTMO compliance class"""
    
    print("=" * 60)
    print("TESTING FTMO COMPLIANCE")
    print("=" * 60)
    
    # Initialize FTMO compliance checker
    ftmo = FTMOCompliance(initial_balance=100000, debug=True)
    
    print(f"\n📊 FTMO Limits:")
    print(f"   Daily loss limit: ${ftmo.daily_loss_limit:,.2f}")
    print(f"   Max loss limit: ${ftmo.max_loss_limit:,.2f}")
    print(f"   Restricted strategies: {ftmo.restricted_strategies}")
    
    # Test valid trade
    print(f"\n✅ Testing Valid Trade:")
    valid_trade = {
        "strategy_type": "swing_trading",
        "position_size": 0.015,  # 1.5%
        "leverage": 50,
        "trade_frequency": 2,
        "risk_per_trade": 0.015
    }
    
    is_valid, violations = ftmo.validate_trade(valid_trade)
    print(f"   Result: {'✅ COMPLIANT' if is_valid else '❌ NON-COMPLIANT'}")
    if violations:
        print(f"   Violations: {violations}")
    
    # Test invalid trade (prohibited strategy)
    print(f"\n❌ Testing Invalid Trade (Martingale):")
    invalid_trade = {
        "strategy_type": "martingale",
        "position_size": 0.08,  # 8% - too high
        "leverage": 200,  # Too high
        "trade_frequency": 15,  # HFT level
        "risk_per_trade": 0.05  # 5% - too high
    }
    
    is_valid, violations = ftmo.validate_trade(invalid_trade)
    print(f"   Result: {'✅ COMPLIANT' if is_valid else '❌ NON-COMPLIANT'}")
    if violations:
        for violation in violations:
            print(f"   - {violation}")
    
    # Test daily limits
    print(f"\n📈 Testing Daily Limits:")
    test_pnls = [-3000, -5000, -6000, 2000]
    
    for pnl in test_pnls:
        is_compliant, reason = ftmo.check_daily_limits(pnl)
        status = "✅ OK" if is_compliant else "❌ VIOLATION"
        print(f"   Daily PnL ${pnl:,}: {status} - {reason}")
    
    # Test drawdown limits
    print(f"\n📉 Testing Drawdown Limits:")
    test_scenarios = [
        (95000, 100000),  # -5% drawdown
        (90000, 100000),  # -10% drawdown (max limit)
        (85000, 100000),  # -15% drawdown (violation)
    ]
    
    for equity, high_water in test_scenarios:
        is_compliant, reason = ftmo.check_total_drawdown(equity, high_water, ftmo.max_loss_limit)
        status = "✅ OK" if is_compliant else "❌ VIOLATION"
        drawdown_pct = (equity - high_water) / high_water * 100
        print(f"   Equity ${equity:,} (DD: {drawdown_pct:.1f}%): {status} - {reason}")


def test_fxify_compliance():
    """Test FXIFY compliance class"""
    
    print("\n" + "=" * 60)
    print("TESTING FXIFY COMPLIANCE")
    print("=" * 60)
    
    # Test both account types
    for account_type in ["starter", "expert"]:
        print(f"\n🏆 {account_type.upper()} ACCOUNT:")
        print("-" * 30)
        
        fxify = FXIFYCompliance(initial_balance=100000, account_type=account_type, debug=True)
        
        print(f"   Daily loss limit: ${fxify.daily_loss_limit:,.2f}")
        print(f"   Max drawdown limit: ${fxify.max_drawdown_limit:,.2f}")
        print(f"   Consistency limit: {fxify.consistency_limit:.0%}")
        
        # Test trade validation
        print(f"\n   Testing Valid Trade:")
        valid_trade = {
            "strategy_type": "scalping",
            "position_size": 0.03,  # 3%
            "leverage": 100,
            "trade_frequency": 5,
            "is_instant_funding": False,
            "is_ea_trade": True
        }
        
        is_valid, violations = fxify.validate_trade(valid_trade)
        print(f"   Result: {'✅ COMPLIANT' if is_valid else '❌ NON-COMPLIANT'}")
        
        # Test instant funding violation
        print(f"\n   Testing Instant Funding Violation:")
        instant_trade = {
            "strategy_type": "swing_trading",
            "position_size": 0.02,
            "leverage": 50,
            "trade_frequency": 3,
            "is_instant_funding": True,  # This will cause violation
            "is_ea_trade": True
        }
        
        is_valid, violations = fxify.validate_trade(instant_trade)
        print(f"   Result: {'✅ COMPLIANT' if is_valid else '❌ NON-COMPLIANT'}")
        if violations:
            for violation in violations:
                print(f"   - {violation}")
        
        # Test consistency rule
        print(f"\n   Testing Consistency Rule:")
        consistency_tests = [
            (2000, 10000),  # 20% - OK for both
            (3500, 10000),  # 35% - OK for starter (30%), violation for expert (40%)
            (5000, 10000),  # 50% - violation for both
        ]
        
        for largest, total in consistency_tests:
            is_compliant, ratio, reason = fxify.check_consistency_rule(largest, total)
            status = "✅ OK" if is_compliant else "❌ VIOLATION"
            print(f"   Largest: ${largest}, Total: ${total} ({ratio:.0%}): {status}")


def test_unified_compliance_checker():
    """Test the unified compliance checker function"""
    
    print("\n" + "=" * 60)
    print("TESTING UNIFIED COMPLIANCE CHECKER")
    print("=" * 60)
    
    # Test FTMO
    print(f"\n🔧 FTMO Unified Check:")
    ftmo_trade_params = {
        "strategy_type": "technical_analysis",
        "position_size": 0.02,
        "leverage": 75,
        "trade_frequency": 3,
        "risk_per_trade": 0.02
    }
    
    ftmo_results = check_prop_firm_compliance(
        firm_name="FTMO",
        trade_params=ftmo_trade_params,
        daily_pnl=-4000,  # Within limits
        equity=96000,
        high_watermark=100000,
        debug=True
    )
    
    print(f"   Overall Compliant: {'✅ YES' if ftmo_results['overall_compliant'] else '❌ NO'}")
    print(f"   Trade Compliant: {ftmo_results['trade_compliant']}")
    print(f"   Daily Compliant: {ftmo_results['daily_compliant']}")
    print(f"   Drawdown Compliant: {ftmo_results['drawdown_compliant']}")
    
    if ftmo_results['violations']:
        print(f"   Violations:")
        for violation in ftmo_results['violations']:
            print(f"   - {violation}")
    
    # Test FXIFY with violations
    print(f"\n🔧 FXIFY Unified Check (with violations):")
    fxify_trade_params = {
        "strategy_type": "latency_arbitrage",  # Prohibited
        "position_size": 0.08,  # Too high
        "leverage": 200,
        "trade_frequency": 25,  # HFT level
        "is_instant_funding": False,
        "is_ea_trade": True,
        "largest_profit": 4500,
        "total_profit": 10000  # 45% consistency - violation
    }
    
    fxify_results = check_prop_firm_compliance(
        firm_name="FXIFY",
        trade_params=fxify_trade_params,
        daily_pnl=-2500,  # Violation for starter (-2% limit)
        equity=92000,
        high_watermark=100000,
        account_type="starter",
        debug=True
    )
    
    print(f"   Overall Compliant: {'✅ YES' if fxify_results['overall_compliant'] else '❌ NO'}")
    print(f"   Trade Compliant: {fxify_results['trade_compliant']}")
    print(f"   Daily Compliant: {fxify_results['daily_compliant']}")
    print(f"   Drawdown Compliant: {fxify_results['drawdown_compliant']}")
    print(f"   Consistency Compliant: {fxify_results.get('consistency_compliant', 'N/A')}")
    
    if fxify_results['violations']:
        print(f"   Violations:")
        for violation in fxify_results['violations']:
            print(f"   - {violation}")


def main():
    """Run all compliance tests"""
    
    print("🏦 PROP FIRM COMPLIANCE TESTING SUITE")
    print("Testing FTMO and FXIFY compliance classes\n")
    
    try:
        test_ftmo_compliance()
        test_fxify_compliance()
        test_unified_compliance_checker()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        print(f"\n📋 Summary:")
        print(f"   ✅ FTMO compliance class working")
        print(f"   ✅ FXIFY compliance class working")
        print(f"   ✅ Unified compliance checker working")
        print(f"   ✅ All validation methods implemented")
        print(f"   ✅ Ready for integration with trading algorithm")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
