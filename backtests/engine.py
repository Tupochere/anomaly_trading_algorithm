# backtest/engine.py
def run_backtest(algo, df):
    """
    Run the strategy and return the result DataFrame.
    Assumes algo has methods: calculate_indicators(), execute_strategy()
    """
    data = algo.calculate_indicators(df)
    results = algo.execute_strategy(data)
    return results
