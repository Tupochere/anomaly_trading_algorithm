# Product Requirements Document (PRD)
## Anomaly Adaptive Hybrid Trading Algorithm - Prop Firm Edition

---

## 1. Product Overview

### 1.1 Product Name
**Anomaly Adaptive Hybrid Trading Algorithm - Prop Firm Edition (AAHTA-PF)**

### 1.2 Objective
Develop a robust, adaptive, and risk-controlled quantitative trading algorithm specifically optimized for **proprietary firm funded trading accounts** that generates consistent returns while strictly adhering to prop firm risk management and drawdown constraints.

### 1.3 Target Market
- **Primary**: Prop firm funded accounts (FTMO, Fidelcrest, FXIFY, Funded Trading Plus)
- **Secondary**: Capital scaling opportunities for funded traders
- **Tertiary**: Professional trading desks seeking systematic strategies

### 1.4 Success Criteria
- Pass prop firm evaluation phases with consistent performance
- Maintain daily drawdown <5% and maximum drawdown <10%
- Achieve walk-forward validated performance across 5-year historical period
- Successfully deploy from Python development to MQL5 production environment

---

## 2. Current Implementation Analysis

### 2.1 Existing Codebase Foundation
Based on `current_algo_prop_v2.py`, the algorithm already implements:

#### Core Strategy Components (Implemented)
1. **Hybrid Signal Generation**
   - Mean-reversion signals using Z-Score, RSI, and Bollinger Bands
   - Momentum signals using EMA crossovers and MACD
   - Combined scoring system with regime-adaptive thresholds

2. **Advanced Risk Management**
   - Dynamic position sizing based on signal strength and volatility
   - ATR-based trailing stops with activation triggers
   - Maximum stop-loss cap (3% default, configurable)
   - Emergency risk controls for prop firm compliance

3. **Market Regime Detection**
   - Six regime classification system: STRONG_UPTREND, STRONG_DOWNTREND, TRENDING, RANGING, HIGH_VOLATILITY, NEUTRAL
   - ADX-based trend strength measurement
   - Bollinger Band width for volatility assessment

4. **Walk-Forward Testing Framework**
   - Rolling window validation with configurable periods
   - Performance metrics calculation (Sharpe, drawdown, win rate)
   - Out-of-sample testing methodology

### 2.2 Repository Structure Alignment
Your current structure supports the PRD objectives:

```
/prop-firm-trading-algorithm
├── strategies/
│   ├── current_algo_prop_v2.py          # Base implementation
│   └── versions/                        # Version history
├── backtests/
│   ├── engine.py                        # Backtesting framework
│   ├── metrics.py                       # Performance calculations
│   └── results/                         # Test outputs
├── notebooks/
│   ├── 00_baseline_establishment.ipynb  # Development environment
│   └── experiments/                     # Strategy variations
├── data/processed/                      # Clean market data
├── prop_tests/                          # Prop firm simulation
└── docs/                               # Documentation
```

### 2.3 Technical Architecture

#### Data Infrastructure (Current)
- **Data Format**: Processed CSV files with OHLCV data
- **Timeframe**: 1-hour bars (optimal for prop firm trading)
- **Historical Period**: January 2020 - December 2025
- **Instruments**: 8 symbols across equities, forex, and commodities

#### Algorithm Architecture (Current Implementation)
The `AdvancedTradingAlgorithmPropV2` class implements:

1. **Signal Processing Pipeline**
   ```python
   def generate_signal(data, idx) -> int  # Returns 1, -1, or 0
   def get_signal_with_strength(data, idx) -> Tuple[int, float]
   ```

2. **Position Management System**
   ```python
   def calculate_position_size(final_score, volatility, threshold=0.3)
   def calculate_stops(data, idx, signal, entry_price) -> Tuple[float, float]
   def should_exit(position, price, indicators, ...) -> Tuple[bool, str]
   ```

3. **Risk Control Framework**
   ```python
   def update_risk_limits(equity, daily_pnl, max_daily_loss, max_total_drawdown)
   def update_trailing_stop(data, idx)
   ```

---

## 3. Enhanced Development Roadmap

### 3.1 Phase 1: Baseline Optimization (Weeks 1-2)
Building on existing `current_algo_prop_v2.py`:

#### Immediate Enhancements
1. **Intraday Confirmation System**
   - Implement `get_intraday_confirmation()` method
   - 15-minute timeframe validation for entry signals
   - Higher frequency trade confirmation

2. **End-of-Day Risk Management**
   - Implement `should_exit_end_of_day()` logic
   - Automatic position closure before market close
   - Prop firm overnight risk compliance

3. **Enhanced Walk-Forward Testing**
   - Optimize existing `walk_forward_test()` method
   - Implement 125-day training windows with 60-day steps
   - Statistical significance testing for results

#### Position Sizing Refinements
Current implementation uses:
```python
def calculate_position_size(final_score, volatility, threshold=0.3, 
                          base_size=0.04, max_size=0.08, min_size=0.02)
```

**Enhancements needed:**
- Prop firm specific sizing limits
- Correlation-based exposure limits
- Dynamic sizing based on account equity

### 3.2 Phase 2: Prop Firm Compliance Integration (Weeks 3-4)

#### Firm-Specific Rule Implementation
Extend existing risk management to include:

1. **FTMO Compliance Module**
   ```python
   class FTMOCompliance:
       daily_loss_limit = 0.05
       max_loss_limit = 0.10
       prohibited_strategies = ['HFT', 'Arbitrage', 'Martingale']
   ```

2. **FXIFY Compliance Module**
   ```python
   class FXIFYCompliance:
       daily_loss_limit = 0.02  # Most restrictive
       max_loss_limit = 0.04
       special_rules = {'no_ea_instant_funding': True}
   ```

3. **Universal Risk Manager**
   ```python
   def check_prop_firm_compliance(firm_type, current_state) -> bool
   ```

#### Enhanced Risk Controls
Building on existing `update_risk_limits()`:
- Real-time P&L monitoring
- Automatic trading halt mechanisms
- Position size scaling based on drawdown

### 3.3 Phase 3: Performance Optimization (Weeks 5-6)

#### Signal Quality Improvements
Enhance existing signal generation:

1. **Multi-Timeframe Confirmation**
   - 1H primary signals + 15M confirmation
   - Improve existing `generate_signal()` method
   - Reduce false positives

2. **Regime-Adaptive Parameters**
   - Dynamic threshold adjustment based on market regime
   - Optimize existing `detect_market_regime()` logic
   - Improve strategy adaptability

3. **Signal Filtering Enhancement**
   - Volume confirmation integration
   - Market session awareness
   - Economic event filtering

### 3.4 Phase 4: Walk-Forward Validation (Weeks 7-8)

#### Comprehensive Testing Protocol
Utilize existing `walk_forward_test()` method with enhancements:

1. **Extended Historical Testing**
   - 2020-2025 complete validation
   - Monthly performance consistency
   - Regime-specific performance analysis

2. **Statistical Validation**
   - Monte Carlo simulation
   - Bootstrap confidence intervals
   - Overfitting detection

3. **Multi-Asset Validation**
   - Cross-asset performance consistency
   - Correlation impact analysis
   - Portfolio-level risk assessment

---

## 4. Technical Implementation Specifications

### 4.1 Core Algorithm Enhancements

#### Enhanced Signal Generation
Building on existing implementation:
```python
def generate_enhanced_signal(self, data: pd.DataFrame, idx: int) -> Dict:
    """
    Enhanced signal generation with multi-timeframe confirmation
    """
    # Use existing mean_reversion_signal() and momentum_signal()
    primary_signal = self.generate_signal(data, idx)
    
    # Add intraday confirmation
    if primary_signal != 0:
        confirmed = self.get_intraday_confirmation(data_15m, idx_15m)
        if not confirmed:
            return {"signal": 0, "reason": "intraday_rejection"}
    
    return {"signal": primary_signal, "strength": signal_strength}
```

#### Prop Firm Risk Integration
Extend existing risk management:
```python
def enhanced_risk_check(self, equity, daily_pnl, firm_rules):
    """
    Enhanced risk checking with firm-specific rules
    """
    # Use existing update_risk_limits() as base
    base_check = self.update_risk_limits(equity, daily_pnl)
    
    # Add firm-specific compliance
    firm_check = self.check_firm_specific_rules(firm_rules)
    
    return base_check and firm_check
```

### 4.2 Performance Optimization Targets

#### Current Baseline Performance (From existing results)
Based on your existing baseline results:
- **AAPL**: Strong performance with high returns
- **SPY**: Consistent moderate returns
- **Multi-asset**: Proven across different instruments

#### Target Improvements
1. **Risk-Adjusted Returns**: Improve Sharpe ratio >1.5
2. **Drawdown Control**: Maximum 8% in any 3-month period
3. **Win Rate**: Maintain >50% across all instruments
4. **Trade Frequency**: Optimize for prop firm evaluation periods

### 4.3 Walk-Forward Testing Enhancement

#### Existing Framework Optimization
Your current `walk_forward_test()` method needs:

1. **Parameter Optimization**
   ```python
   # Current: window_size_days=125, step_size_days=60
   # Proposed: Dynamic window sizing based on market volatility
   ```

2. **Performance Metrics Enhancement**
   ```python
   def enhanced_performance_metrics(self) -> Dict:
       # Extend existing get_performance_metrics()
       # Add prop firm specific metrics
       # Include regime-specific performance
   ```

3. **Statistical Validation**
   ```python
   def statistical_validation(self, results_df) -> Dict:
       # Statistical significance testing
       # Confidence interval calculation
       # Overfitting detection metrics
   ```

---

## 5. Prop Firm Compliance Framework

### 5.1 Implementation Strategy
Building on existing risk management:

#### Current Risk Controls (Implemented)
- Maximum stop-loss cap: 3% (configurable)
- Dynamic position sizing: 2-8% of equity
- Trailing stop system with ATR-based calculation
- Daily and total drawdown monitoring

#### Required Enhancements
1. **Firm-Specific Rule Classes**
   ```python
   class PropFirmRules:
       def __init__(self, firm_type):
           self.firm_type = firm_type
           self.load_rules()
       
       def validate_trade(self, trade_params) -> bool
       def check_daily_limits(self, pnl) -> bool
       def check_exposure_limits(self, positions) -> bool
   ```

2. **Real-Time Compliance Monitoring**
   ```python
   def real_time_compliance_check(self):
       # Integration with existing should_exit() method
       # Automatic position closure on rule violations
       # Risk alert system
   ```

### 5.2 Supported Prop Firms

#### FTMO Integration
- Daily loss: 5% of account balance
- Maximum loss: 10% of initial balance
- Strategy compliance: No HFT, arbitrage, or martingale
- **Implementation**: Extend existing `update_risk_limits()`

#### FXIFY Integration  
- Daily loss: 2-3% (most restrictive)
- Maximum loss: 4-5% of initial balance
- EA restrictions: No EAs on Instant Funding
- **Implementation**: Separate compliance class

#### Fidelcrest & Funded Trading Plus
- Similar to FTMO with minor variations
- **Implementation**: Unified compliance framework

---

## 6. Development Workflow & Testing

### 6.1 Branch Strategy
Create new development branch from your current main:
```bash
git checkout -b prop-firm-ea-development
```

### 6.2 Development Milestones

#### Week 1-2: Enhancement Implementation
- [ ] Enhance existing `current_algo_prop_v2.py`
- [ ] Implement intraday confirmation system
- [ ] Add end-of-day risk management
- [ ] Create prop firm compliance classes

#### Week 3-4: Testing Framework
- [ ] Enhance existing walk-forward testing
- [ ] Implement statistical validation
- [ ] Create prop firm simulation tests
- [ ] Multi-asset validation framework

#### Week 5-6: Optimization
- [ ] Parameter optimization using walk-forward results
- [ ] Signal quality improvements
- [ ] Risk management fine-tuning
- [ ] Performance consistency validation

#### Week 7-8: Final Validation
- [ ] Complete 2020-2025 walk-forward testing
- [ ] Statistical significance validation
- [ ] Prop firm compliance verification
- [ ] Documentation and deployment preparation

### 6.3 Testing Protocol

#### Walk-Forward Testing Enhancement
Optimize existing method:
```python
def enhanced_walk_forward_test(self, data, 
                             window_size_days=125, 
                             step_size_days=60,
                             validation_method='time_series_split'):
    """
    Enhanced walk-forward testing with statistical validation
    """
    # Use existing framework as base
    # Add statistical significance testing
    # Include regime-specific analysis
    # Generate comprehensive reports
```

#### Performance Validation
```python
def validate_prop_firm_readiness(self, results_df) -> Dict:
    """
    Validate algorithm readiness for prop firm deployment
    """
    # Check drawdown compliance across all test periods
    # Verify consistent performance metrics
    # Validate risk control effectiveness
    # Generate deployment recommendation
```

---

## 7. Success Metrics & Validation Criteria

### 7.1 Performance Benchmarks

#### Primary Success Criteria (Must Achieve)
- **Daily Drawdown**: <4% (most restrictive prop firm requirement)
- **Maximum Drawdown**: <8% in any 3-month test period  
- **Win Rate**: >45% across all instruments
- **Profit Factor**: >2.0 minimum
- **Walk-Forward Consistency**: Positive returns in >75% of test periods

#### Secondary Optimization Goals
- **Sharpe Ratio**: >1.5 across all test periods
- **Trade Frequency**: Optimal for prop firm evaluation periods
- **Multi-Asset Stability**: Consistent performance across all 8 instruments
- **Regime Adaptability**: Positive performance in all market regimes

### 7.2 Walk-Forward Validation Success
- [ ] 2020-2025 complete historical validation
- [ ] Statistical significance at 95% confidence level
- [ ] Consistent performance across market regimes
- [ ] Prop firm compliance in 100% of test periods

### 7.3 Deployment Readiness Criteria
- [ ] Python implementation validated across 5-year period
- [ ] MQL5 port matches Python performance within 3%
- [ ] Real-time execution testing successful
- [ ] Prop firm compliance verified across all supported firms

---

## 8. Next Steps & Implementation Plan

### 8.1 Immediate Actions (Next 1-2 Weeks)

#### Repository Preparation
1. **Create Development Branch**
   ```bash
   git checkout -b prop-firm-ea-development
   ```

2. **Enhance Base Algorithm**
   - Modify `strategies/current_algo_prop_v2.py`
   - Add intraday confirmation methods
   - Implement prop firm compliance classes

3. **Testing Framework Enhancement**
   - Optimize `backtests/engine.py` for prop firm simulation
   - Enhance `backtests/metrics.py` with additional KPIs
   - Create prop firm specific test scenarios

#### Development Environment Setup
1. **Jupyter Notebook Creation**
   - `notebooks/01_prop_firm_optimization.ipynb`
   - `notebooks/02_enhanced_walk_forward.ipynb` 
   - `notebooks/03_compliance_validation.ipynb`

2. **Testing Infrastructure**
   - Enhance `prop_tests/` with specific firm simulations
   - Create automated testing pipeline
   - Implement statistical validation framework

### 8.2 Implementation Priority Queue

#### High Priority (Weeks 1-2)
1. Implement intraday confirmation system
2. Add end-of-day risk management
3. Create prop firm compliance framework
4. Enhance walk-forward testing statistical validation

#### Medium Priority (Weeks 3-4)  
1. Multi-timeframe signal confirmation
2. Advanced regime detection optimization
3. Statistical significance testing implementation
4. Portfolio-level risk assessment

#### Low Priority (Weeks 5-6)
1. Economic calendar integration
2. Advanced correlation analysis
3. Machine learning signal enhancement
4. Real-time monitoring dashboard

### 8.3 Success Validation Checkpoints

#### Checkpoint 1 (Week 2)
- [ ] Enhanced algorithm implementation complete
- [ ] Basic prop firm compliance integrated
- [ ] Initial walk-forward testing results

#### Checkpoint 2 (Week 4)
- [ ] Complete statistical validation framework
- [ ] Multi-asset testing results
- [ ] Prop firm simulation successful

#### Checkpoint 3 (Week 6)
- [ ] Full 2020-2025 walk-forward validation complete
- [ ] Performance optimization finalized
- [ ] Deployment readiness assessment

#### Final Validation (Week 8)
- [ ] MQL5 port development initiated
- [ ] Production deployment plan finalized
- [ ] Comprehensive documentation complete

---

## 9. Risk Management & Compliance Specifications

### 9.1 Enhanced Risk Control Framework

#### Building on Existing Implementation
Your current `should_exit()` method provides:
- Hard stop-loss at -5%
- Take-profit at +10%
- Trailing stops using ATR
- Momentum-based exits
- Maximum duration limits

#### Required Enhancements
1. **Prop Firm Specific Controls**
   ```python
   def prop_firm_risk_check(self, firm_type, current_state):
       """
       Firm-specific risk validation
       """
       if firm_type == "FXIFY":
           return self.fxify_compliance_check(current_state)
       elif firm_type == "FTMO":
           return self.ftmo_compliance_check(current_state)
       # ... other firms
   ```

2. **Real-Time Risk Monitoring**
   ```python
   def real_time_risk_monitor(self):
       """
       Continuous risk monitoring during trading
       """
       # Monitor existing daily_pnl and total_drawdown
       # Add real-time compliance checking
       # Implement automatic position closure
   ```

### 9.2 Position Sizing Optimization

#### Current Implementation Analysis
Your `calculate_position_size()` method uses:
- Base size: 4% of equity
- Maximum size: 8% of equity  
- Minimum size: 2% of equity
- Volatility adjustment based on ATR

#### Prop Firm Optimizations
1. **Conservative Sizing for Evaluation**
   - Reduce base size to 2% during evaluation phases
   - Maximum size capped at 5% for prop firm accounts
   - Dynamic adjustment based on current drawdown

2. **Risk-Adjusted Sizing**
   ```python
   def prop_firm_position_size(self, signal_strength, volatility, account_phase):
       """
       Prop firm optimized position sizing
       """
       if account_phase == "evaluation":
           max_size = 0.03  # 3% maximum during evaluation
       else:
           max_size = 0.05  # 5% maximum during funded phase
       
       # Use existing volatility adjustment logic
       return min(calculated_size, max_size)
   ```

---

## 10. Conclusion & Success Definition

### 10.1 Project Success Criteria
This enhanced PRD defines success as:

1. **Technical Excellence**: Algorithm performs consistently across 5-year walk-forward validation
2. **Risk Compliance**: 100% adherence to prop firm rules across all test periods  
3. **Performance Consistency**: Positive risk-adjusted returns in majority of test periods
4. **Deployment Readiness**: Successful Python to MQL5 translation and validation

### 10.2 Value Proposition
The enhanced algorithm will provide:
- **Systematic Trading**: Removes emotional decision-making
- **Risk Control**: Strict adherence to prop firm requirements
- **Scalability**: Proven performance across multiple assets and market conditions
- **Consistency**: Statistical validation ensures robust performance

### 10.3 Competitive Advantages
- **Hybrid Approach**: Combines mean-reversion and momentum strategies
- **Regime Adaptation**: Adjusts parameters based on market conditions
- **Risk-First Design**: Built specifically for prop firm compliance
- **Statistical Validation**: Walk-forward testing ensures robustness

---

This updated PRD aligns with your existing codebase and provides a clear roadmap for enhancing your current implementation into a prop firm-ready trading algorithm. The development plan builds incrementally on your existing strong foundation while adding the specific features needed for prop firm success.