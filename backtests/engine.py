# backtest/engine.py
def run_backtest(algo, df):
    """
    Run the strategy and return result DataFrame and algo object
    """
    data = algo.calculate_indicators(df)
    results = algo.execute_strategy(data)
    return results, algo
