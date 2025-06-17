# data/processed/download_data.py

import yfinance as yf
import os

def download_and_save(symbol, period="1y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    })

    data.index.name = 'date'
    output_path = f"data/processed/{symbol}_{period}.csv"
    os.makedirs("data/processed", exist_ok=True)
    data.to_csv(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    periods = ['1y', '2y']

    for symbol in symbols:
        for period in periods:
            download_and_save(symbol, period)
