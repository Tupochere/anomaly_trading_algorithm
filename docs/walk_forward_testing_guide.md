# 🚀 Walk-Forward Testing Guide

## Overview
Walk-forward testing has been integrated into the `AdvancedTradingAlgorithm` class to provide more robust strategy evaluation. This approach tests the algorithm on multiple rolling time windows, simulating real-world deployment scenarios.

## Key Features Added

### 1. Helper Functions
- `calculate_max_drawdown(equity_curve)`: Calculates maximum drawdown from an equity curve
- `calculate_sharpe_ratio(pnl_series, risk_free_rate=0.0)`: Calculates annualized Sharpe ratio

### 2. New Methods in AdvancedTradingAlgorithm

#### `evaluate_performance(self, data)`
Evaluates performance metrics on a given data slice:
- Total return percentage
- Win rate
- Maximum drawdown
- Sharpe ratio
- Number of trades

#### `walk_forward_test(self, data, window_size_days=125, step_size_days=60)`
Performs walk-forward testing with rolling windows:
- **window_size_days**: Total length of each window (default: 125 trading days ≈ 6 months)
- **step_size_days**: How much to move forward each iteration (default: 60 trading days ≈ 3 months)
- Returns DataFrame with performance metrics for each window

## Usage Examples

### Basic Usage
```python
from strategies.current_algo import AdvancedTradingAlgorithm
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv", parse_dates=['date'], index_col='date')

# Create algorithm instance
algo = AdvancedTradingAlgorithm(debug=False)

# Run walk-forward test
results = algo.walk_forward_test(
    df, 
    window_size_days=125,  # 6-month windows
    step_size_days=60      # 3-month steps
)

# Analyze results
print(f"Total windows tested: {len(results)}")
print(f"Average return: {results['total_return_pct'].mean():.2f}%")
print(f"Win rate: {(results['total_return_pct'] > 0).mean()*100:.1f}%")
```

### Advanced Configuration
```python
# Test with different window sizes
short_term_results = algo.walk_forward_test(df, window_size_days=60, step_size_days=30)
long_term_results = algo.walk_forward_test(df, window_size_days=250, step_size_days=125)

# Compare stability across different time horizons
```

## Interpretation Guide

### Key Metrics to Monitor

1. **Consistency**: Look for stable performance across multiple windows
2. **Profitable Windows**: Aim for >60% of windows to be profitable
3. **Average Sharpe Ratio**: Values >1.0 indicate good risk-adjusted returns
4. **Maximum Drawdown**: Should be manageable (<20% ideally)

### Red Flags
- High variance in returns between windows
- Declining performance in recent windows
- Very few profitable windows
- Negative average Sharpe ratio

## Files Generated

The walk-forward testing creates several output files:

1. **Detailed Results**: `{symbol}_{period}_walk_forward.csv`
   - Performance metrics for each window
   
2. **Summary Report**: `{symbol}_{period}_walk_forward_summary.csv`
   - Aggregated statistics and key insights

## Integration with Existing Workflow

The walk-forward testing is now seamlessly integrated:

1. **Notebook Integration**: Added cells in `backtest_baseline.ipynb` for easy testing
2. **Visualization**: Built-in plotting for performance analysis
3. **Export Functionality**: Automatic saving of results and summaries

## Best Practices

1. **Window Size**: Choose based on your trading frequency
   - Day trading: 30-60 day windows
   - Swing trading: 60-125 day windows
   - Position trading: 125-250 day windows

2. **Step Size**: Typically 25-50% of window size for good overlap

3. **Data Requirements**: Ensure you have enough data for multiple windows (minimum 2-3 years for meaningful results)

4. **Validation**: Compare walk-forward results with single-period backtests to assess stability

## Example Output Interpretation

```
=== Walk-Forward Test Summary ===
Total Windows: 8
Average Return: 3.45%
Average Win Rate: 62.5%
Average Sharpe: 1.23
Positive Windows: 6/8
```

This example shows:
- ✅ Strong consistency (6/8 profitable windows)
- ✅ Good risk-adjusted returns (Sharpe > 1.0)
- ✅ Decent win rate (>60%)
- ✅ Positive average return

This indicates a robust strategy suitable for live trading consideration.
