# utils/data_loader.py
import pandas as pd
import os

def load_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    Example file path: data/processed/AAPL_1y.csv
    """
    filename = f"data/processed/{symbol}_{period}.csv"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    df = pd.read_csv(filename, index_col='date', parse_dates=True)
    return df
