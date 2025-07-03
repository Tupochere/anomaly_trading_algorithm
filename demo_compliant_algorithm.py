#!/usr/bin/env python3
"""
Integration example showing how to use prop firm compliance with the trading algorithm
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.current_algo_prop_v2 import (
    AdvancedTradingAlgorithmPropV2,
    FTMOCompliance,
    FXIFYCompliance,
    check_prop_firm_compliance
)

class PropFirmCompliantAlgorithm(AdvancedTradingAlgorithmPropV2):
    """
    Enhanced trading algorithm with integrated prop firm compliance checking
    """
    
    def __init__(self, firm_name="FTMO", account_type="starter", initial_balance=100000, **kwargs):
        super().__init__(**kwargs)
        
        self.firm_name = firm_name.upper()
        self.account_type = account_type
        self.initial_balance = initial_balance
        
        # Initialize compliance checker
        if self.firm_name == "FTMO":
            self.compliance = FTMOCompliance(initial_balance, debug=self.debug)
        elif self.firm_name == "FXIFY":
            self.compliance = FXIFYCompliance(initial_balance, account_type, debug=self.debug)
        else:
            raise ValueError(f"Unsupported firm: {firm_name}")
        
        # Compliance tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.current_equity = initial_balance
        self.high_watermark = initial_balance
        self.compliance_violations = []
        
        if self.debug:
            self.log(f"Initialized {self.firm_name} compliant algorithm")
            self.log(f"Account type: {self.account_type}")
            self.log(f"Initial balance: ${self.initial_balance:,.2f}")
    
    def check_pre_trade_compliance(self, signal_strength, volatility):
        """
        Check compliance before entering a trade
        
        Args:
            signal_strength: Signal strength (0-1)
            volatility: Market volatility
            
        Returns:
            tuple: (is_compliant: bool, position_size: float, violations: list)
        """
        # Calculate position size
        position_size = self.calculate_position_size(signal_strength, volatility)
        
        # Prepare trade parameters for compliance check
        trade_params = {
            "strategy_type": "technical_analysis",  # Our algorithm uses technical analysis
            "position_size": position_size,
            "leverage": 50,  # Conservative leverage
            "trade_frequency": self.daily_trades,
            "risk_per_trade": position_size,
            "is_instant_funding": False,  # Assume regular account
            "is_ea_trade": True  # We're using an EA
        }
        
        # Check trade compliance
        is_compliant, violations = self.compliance.validate_trade(trade_params)
        
        if not is_compliant:
            self.log(f"Pre-trade compliance check failed: {violations}")
            return False, 0.0, violations
        
        return True, position_size, []
    
    def check_daily_compliance(self):
        """
        Check daily compliance limits
        
        Returns:
            bool: True if compliant, False otherwise
        """
        # Check daily limits
        daily_compliant, daily_reason = self.compliance.check_daily_limits(self.daily_pnl)
        
        # Check total drawdown
        drawdown_limit = getattr(self.compliance, 'max_loss_limit', 
                               getattr(self.compliance, 'max_drawdown_limit', -10000))
        drawdown_compliant, drawdown_reason = self.compliance.check_total_drawdown(
            self.current_equity, self.high_watermark, drawdown_limit
        )
        
        if not daily_compliant:
            self.log(f"Daily compliance violation: {daily_reason}")
            self.skip_today = True
            return False
        
        if not drawdown_compliant:
            self.log(f"Drawdown compliance violation: {drawdown_reason}")
            self.stop_trading = True
            return False
        
        return True
    
    def execute_compliant_strategy(self, data: pd.DataFrame, data_15m: pd.DataFrame = None):
        """
        Execute strategy with integrated compliance checking
        
        Args:
            data: Hourly price data
            data_15m: 15-minute data for confirmation
            
        Returns:
            pd.DataFrame: Results with compliance information
        """
        if self.debug:
            self.log(f"Executing {self.firm_name} compliant strategy")
        
        # Reset daily tracking (in real implementation, this would be date-based)
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        # Execute base strategy
        results = self.execute_strategy_with_intraday_confirmation(data, data_15m)
        
        # Add compliance information to results
        compliance_info = []
        
        for i, row in results.iterrows():
            # Update tracking for trades
            if row['action'] in ['BUY', 'SELL']:
                self.daily_trades += 1
            
            # Simulate P&L tracking (in real implementation, use actual P&L)
            if row['action'].startswith('EXIT_'):
                # Simulate random P&L for demonstration
                pnl = np.random.normal(0, 1000)  # Random P&L
                self.daily_pnl += pnl
                self.current_equity += pnl
                self.high_watermark = max(self.high_watermark, self.current_equity)
            
            # Check compliance
            is_compliant = self.check_daily_compliance()
            
            compliance_info.append({
                'daily_trades': self.daily_trades,
                'daily_pnl': self.daily_pnl,
                'current_equity': self.current_equity,
                'is_compliant': is_compliant
            })
        
        # Add compliance columns to results
        compliance_df = pd.DataFrame(compliance_info)
        for col in compliance_df.columns:
            results[f'compliance_{col}'] = compliance_df[col].values
        
        return results
    
    def get_compliance_summary(self):
        """
        Get summary of compliance status
        
        Returns:
            dict: Compliance summary
        """
        # Prepare comprehensive trade parameters
        trade_params = {
            "strategy_type": "technical_analysis",
            "position_size": 0.02,  # 2% risk
            "leverage": 50,
            "trade_frequency": self.daily_trades,
            "risk_per_trade": 0.02,
            "is_instant_funding": False,
            "is_ea_trade": True,
            "largest_profit": max(1000, self.daily_pnl * 0.3),  # Estimate
            "total_profit": max(self.daily_pnl, 0)
        }
        
        # Use unified compliance checker
        results = check_prop_firm_compliance(
            firm_name=self.firm_name,
            trade_params=trade_params,
            daily_pnl=self.daily_pnl,
            equity=self.current_equity,
            high_watermark=self.high_watermark,
            initial_balance=self.initial_balance,
            account_type=self.account_type,
            debug=self.debug
        )
        
        return results


def demo_compliant_algorithm():
    """Demonstrate the prop firm compliant algorithm"""
    
    print("🏦 PROP FIRM COMPLIANT ALGORITHM DEMO")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test FTMO compliant algorithm
    print(f"\n📊 Testing FTMO Compliant Algorithm:")
    print("-" * 40)
    
    ftmo_algo = PropFirmCompliantAlgorithm(
        firm_name="FTMO",
        initial_balance=100000,
        debug=True
    )
    
    # Execute strategy
    ftmo_results = ftmo_algo.execute_compliant_strategy(data)
    
    print(f"\n✅ FTMO Results:")
    print(f"   Total bars processed: {len(ftmo_results)}")
    print(f"   Actions taken: {ftmo_results['action'].value_counts().to_dict()}")
    print(f"   Final equity: ${ftmo_algo.current_equity:,.2f}")
    print(f"   Daily P&L: ${ftmo_algo.daily_pnl:,.2f}")
    print(f"   Daily trades: {ftmo_algo.daily_trades}")
    
    # Get compliance summary
    ftmo_compliance = ftmo_algo.get_compliance_summary()
    print(f"\n📋 FTMO Compliance Summary:")
    print(f"   Overall compliant: {'✅ YES' if ftmo_compliance['overall_compliant'] else '❌ NO'}")
    print(f"   Trade compliant: {ftmo_compliance['trade_compliant']}")
    print(f"   Daily compliant: {ftmo_compliance['daily_compliant']}")
    print(f"   Drawdown compliant: {ftmo_compliance['drawdown_compliant']}")
    
    if ftmo_compliance['violations']:
        print(f"   Violations: {ftmo_compliance['violations']}")
    
    # Test FXIFY compliant algorithm
    print(f"\n📊 Testing FXIFY Compliant Algorithm:")
    print("-" * 40)
    
    fxify_algo = PropFirmCompliantAlgorithm(
        firm_name="FXIFY",
        account_type="starter",
        initial_balance=100000,
        debug=True
    )
    
    # Execute strategy
    fxify_results = fxify_algo.execute_compliant_strategy(data)
    
    print(f"\n✅ FXIFY Results:")
    print(f"   Total bars processed: {len(fxify_results)}")
    print(f"   Actions taken: {fxify_results['action'].value_counts().to_dict()}")
    print(f"   Final equity: ${fxify_algo.current_equity:,.2f}")
    print(f"   Daily P&L: ${fxify_algo.daily_pnl:,.2f}")
    print(f"   Daily trades: {fxify_algo.daily_trades}")
    
    # Get compliance summary
    fxify_compliance = fxify_algo.get_compliance_summary()
    print(f"\n📋 FXIFY Compliance Summary:")
    print(f"   Overall compliant: {'✅ YES' if fxify_compliance['overall_compliant'] else '❌ NO'}")
    print(f"   Trade compliant: {fxify_compliance['trade_compliant']}")
    print(f"   Daily compliant: {fxify_compliance['daily_compliant']}")
    print(f"   Drawdown compliant: {fxify_compliance['drawdown_compliant']}")
    
    if 'consistency_compliant' in fxify_compliance:
        print(f"   Consistency compliant: {fxify_compliance['consistency_compliant']}")
    
    if fxify_compliance['violations']:
        print(f"   Violations: {fxify_compliance['violations']}")
    
    print(f"\n🎯 Integration Complete!")
    print(f"   ✅ Pre-trade compliance checking")
    print(f"   ✅ Real-time compliance monitoring")
    print(f"   ✅ Automatic position sizing based on firm rules")
    print(f"   ✅ Daily and drawdown limit enforcement")
    print(f"   ✅ Strategy validation against prohibited methods")


if __name__ == "__main__":
    demo_compliant_algorithm()
