import numpy as np

def calculate_comprehensive_metrics(trades):
    """Calculate metrics from trade list"""
    if not trades:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    returns = [trade['pnl_pct'] for trade in trades]
    positive_returns = [r for r in returns if r > 0]
    negative_returns = [r for r in returns if r < 0]
    
    total_return = sum(returns) * 100
    total_trades = len(returns)
    win_rate = len(positive_returns) / total_trades if total_trades > 0 else 0
    avg_win = sum(positive_returns)/len(positive_returns) if positive_returns else 0
    avg_loss = abs(sum(negative_returns)/len(negative_returns)) if negative_returns else 0
    profit_factor = (avg_win * win_rate) / (avg_loss * (1 - win_rate)) if win_rate < 1 else float('inf')
    
    # Calculate equity curve
    equity = [1]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    
    # Calculate max drawdown
    peak = equity[0]
    max_dd = 0
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Calculate Sharpe ratio (annualized)
    daily_returns = [equity[i]/equity[i-1] - 1 for i in range(1, len(equity))]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd * 100,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100
    }