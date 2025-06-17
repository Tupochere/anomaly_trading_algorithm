# backtest/metrics.py
def calculate_comprehensive_metrics(results):
    trades = results[results['action'].str.startswith('EXIT')]

    if trades.empty:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0
        }

    pnl = trades['entry_price'].combine(trades['close'], lambda e, x: (x - e) / e if e else 0)
    returns = pnl.tolist()

    total_return = sum(returns) * 100
    sharpe_ratio = (sum(returns) / len(returns)) / (pd.Series(returns).std() + 1e-9) * (252 ** 0.5)
    max_drawdown = -min(returns) * 100
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    avg_win = sum(r for r in returns if r > 0) / max(1, sum(r > 0 for r in returns))
    avg_loss = abs(sum(r for r in returns if r < 0) / max(1, sum(r < 0 for r in returns)))
    profit_factor = avg_win / avg_loss if avg_loss else float('inf')

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(returns)
    }
