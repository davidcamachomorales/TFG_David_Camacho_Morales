import os
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd


# file name -> ticker in yahoo
assets = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Nvidia": "NVDA",
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Inditex": "ITX.MC",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "TetherUSDT": "USDT-USD",
    "S&P_500_Vanguard": "VOO"
}

# dir to save
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

# Timelapse 3 years from the day it was created 16-03-2026
end_date = datetime(2026, 3, 16)

# Timelapse 3 years from the day it is executed
# in case you want to test with current data
#end_date = datetime.today()
start_date = end_date - timedelta(days=3 * 365)

print(f"Downloading data...\n")

for asset_name, ticker in assets.items():
    try:
        print(f"Downloading {asset_name} ({ticker})...")

        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        # if it fails
        if df.empty:
            print(f"  -> No data found for {asset_name}")
            continue

        # yfinance can return MultiIndex columns 
        # flatten it to keep only the column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index so that Date remains as a column
        df.reset_index(inplace=True)

        # Rename columns to the project format
        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        # Drop adj_close if it exists we do not use it
        if "Adj Close" in df.columns:
            df.drop(columns=["Adj Close"], inplace=True)

        # Sort columns as the original project-> date,open,high,low,close,volume
        df = df[["date", "open", "high", "low", "close", "volume"]]

        # Save CSV
        file_path = os.path.join(output_dir, f"{asset_name}.csv")
        df.to_csv(file_path, index=False)

        print(f"  -> Done: {len(df)} rows saved")

    except Exception as e:
        print(f"  -> Error with {asset_name}: {e}")

print("\nDownload finished.")