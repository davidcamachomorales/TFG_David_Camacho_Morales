import pandas as pd
import os

project_root = "/Users/davidcamachomorales/Desktop/TFG"
print("Checking batch summaries:")

for algo in ["PPO", "A2C", "DQN"]:
    csv_file = os.path.join(project_root, f"results/{algo}_batch_summary.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Find numeric columns with sharpe in them
        for col in df.columns:
            if 'sharpe' in col.lower() and not col.endswith('_values') and not col.endswith('_list'):
                # Check if it is numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                    print(f"Average of {algo} {col}: {df[col].mean():.6f}")
                except Exception as e:
                    print(f"Could not convert {col} in {algo}: {e}")
    else:
        print(f"File {csv_file} does not exist")

