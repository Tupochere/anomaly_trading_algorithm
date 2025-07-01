import pandas as pd
from prop_firm_simulation import simulate_prop_firm_challenge
from strategies.current_algo_prop_v2 import AdvancedTradingAlgorithmPropV2

def main():
    """
    Main runner for prop firm challenge simulation
    """
    # Example load - adjust paths and column names as needed
    try:
        df = pd.read_csv("data/processed/AAPL_5y.csv")
        # Adjust to your actual date column name
        df['date'] = pd.to_datetime(df['timestamp']).dt.date  # Change 'timestamp' to your date column
    except FileNotFoundError:
        print("Data file not found. Please check the path and filename.")
        return
    except KeyError as e:
        print(f"Column not found: {e}. Please adjust the date column name.")
        return

    # Initialize the prop firm optimized algorithm
    algo = AdvancedTradingAlgorithmPropV2(debug=True)

    print("=== Prop Firm Challenge Simulation ===")
    print(f"Starting balance: $100,000")
    print(f"Daily loss limit: -$5,000")
    print(f"Total drawdown limit: -$10,000")
    print("=" * 40)

    # Run the simulation
    final_equity = simulate_prop_firm_challenge(df, algo)

    print("\n=== Simulation Results ===")
    print(f"Final equity: ${final_equity:.2f}")
    
    if final_equity > 100_000:
        profit = final_equity - 100_000
        print(f"Profit: ${profit:.2f} ({profit/100_000:.2%})")
        print("✅ Challenge PASSED!")
    else:
        loss = 100_000 - final_equity
        print(f"Loss: ${loss:.2f} ({loss/100_000:.2%})")
        print("❌ Challenge FAILED!")

    print("Simulation complete!")

if __name__ == "__main__":
    main()
