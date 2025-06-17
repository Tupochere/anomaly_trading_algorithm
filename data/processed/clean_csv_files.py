import pandas as pd
import os

DATA_DIR = "data/processed"

def clean_csv_file(filepath):
    try:
        print(f"\nCleaning: {filepath}")
        
        # Read the file, skipping metadata rows
        df = pd.read_csv(filepath, skiprows=3, header=None)
        
        # Manually assign correct column names
        df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']

        # Ensure proper types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Drop rows without valid dates
        df.set_index('date', inplace=True)

        # Reorder columns to match expected format
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.sort_index(inplace=True)

        # Overwrite file with cleaned data
        df.to_csv(filepath)
        print(f"✅ Cleaned and saved: {filepath}")

    except Exception as e:
        print(f"❌ Failed to clean {filepath}: {e}")

def clean_all_csvs():
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".csv"):
            clean_csv_file(os.path.join(DATA_DIR, fname))

if __name__ == "__main__":
    clean_all_csvs()
