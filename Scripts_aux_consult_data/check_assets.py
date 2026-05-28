import pandas as pd
import os

csv_file = "/Users/davidcamachomorales/Desktop/TFG/results/analysis/all_results_consolidated.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print("Columns:", df.columns.tolist())
    # Group by asset and compute mean of sharpe_ratio_mean
    asset_means = df.groupby("asset")["sharpe_ratio_mean"].mean().sort_values(ascending=False)
    print("\nMean Sharpe by Asset:")
    for asset, val in asset_means.items():
        print(f"{asset}: {val:.4f}")
else:
    print(f"File {csv_file} does not exist")

