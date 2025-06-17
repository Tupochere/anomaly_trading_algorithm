import os
import pandas as pd

def load_data(symbol, period):
    # Build path relative to this file's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(base_dir, 'data', 'processed', f"{symbol}_{period}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
    df = pd.read_csv(filename, index_col='date', parse_dates=True)
    return df
