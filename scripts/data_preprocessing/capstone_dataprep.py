
# Import Libraries
import requests
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""**1. Data Collection**"""

# Collect BTCUSDT and ETHUSDT Daily Data
BASE_URL = "https://data-api.binance.vision/api/v3/klines"

def get_binance_daily_data(symbol, start_date, end_date):
    start_time = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        response = requests.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # move to next candle
        start_time = data[-1][0] + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    # convert date
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")

    # convert numeric columns
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # keep useful columns
    df = df[[
        "date", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]]

    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


# date range
start_date = "2021-01-01"
end_date = "2025-12-31"

# download datasets
btc_df = get_binance_daily_data("BTCUSDT", start_date, end_date)
eth_df = get_binance_daily_data("ETHUSDT", start_date, end_date)

# save to one Excel file with multiple sheets
with pd.ExcelWriter("crypto_daily_2021_2025.xlsx", engine="openpyxl") as writer:
    btc_df.to_excel(writer, sheet_name="BTC", index=False)
    eth_df.to_excel(writer, sheet_name="ETH", index=False)

# Inspect downloaded datasets
print("BTC rows:", len(btc_df))
print(btc_df.head())
print()
print("ETH rows:", len(eth_df))
print(eth_df.head())
print("\nSaved files:")
print("- btc_daily_2021_2025.csv")
print("- eth_daily_2021_2025.csv")
print("- crypto_daily_2021_2025.xlsx")

# Mount Google drive
from google.colab import drive
drive.mount('/content/drive')

# Upload Dataset
file_path = "/content/drive/MyDrive/Capstone Project 27 Group A/crypto_daily_2021_2025.xlsx"

btc = pd.read_excel(file_path, sheet_name="BTC")
eth = pd.read_excel(file_path, sheet_name="ETH")

print("\n=== BTC ===")
print(btc.head())

print("\n=== ETH ===")
print(eth.head())

""" **2. Data Preprocessing And Analysis**"""

# Examine BTC Data
print("\n=== BTC INFO ===")
btc.info()

print("\n=== BTC DESCRIBE ===")
print(btc.describe())

print("\n=== BTC MISSING VALUES ===")
print(btc.isnull().sum())

print("\n=== BTC DUPLICATES ===")
print("Duplicate rows:", btc.duplicated().sum())
print("Duplicate dates:", btc["date"].duplicated().sum())

# Examine ETH Data
print("\n=== ETH INFO ===")
eth.info()

print("\n=== ETH DESCRIBE ===")
print(eth.describe())

print("\n=== ETH MISSING VALUES ===")
print(eth.isnull().sum())

print("\n=== ETH DUPLICATES ===")
print("Duplicate rows:", eth.duplicated().sum())
print("Duplicate dates:", eth["date"].duplicated().sum())

print("BTC shape:", btc_df.shape)
print("ETH shape:", eth.shape)

print("BTC:", btc["date"].is_monotonic_increasing)
print("ETH:", eth["date"].is_monotonic_increasing)

# Daily Price Trend of BTC and ETH
fig, ax = plt.subplots(2,1, figsize=(12,8))

# BTC Daily Price Trend
ax[0].plot(btc['date'], btc['close'], color='#F7931A')
ax[0].set_title("BTC Daily Price (2021–2025)")
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Price (USDT)")
ax[0].grid(True, linestyle='--', alpha=0.5)

# ETH Daily Price Trend
ax[1].plot(eth['date'], eth['close'], color='#627EEA')
ax[1].set_title("ETH Daily Price (2021–2025)")
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Price (USDT)")
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Daily Price Range of BTC and ETH
btc_range = btc['high'] - btc['low']
eth_range = eth['high'] - eth['low']

fig, ax = plt.subplots(2,1, figsize=(12,8))

# BTC
ax[0].plot(btc['date'], btc_range, color="#F7931A")
ax[0].set_title("Bitcoin Daily Price Range")
ax[0].set_ylabel("Price Range (USDT)")
ax[0].set_xlabel("Date")
ax[0].grid(True, linestyle="--", alpha=0.5)

# ETH
ax[1].plot(eth['date'], eth_range, color="#627EEA")
ax[1].set_title("Ethereum Daily Price Range")
ax[1].set_ylabel("Price Range (USDT)")
ax[1].set_xlabel("Date")
ax[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Weekly Price and Volatility Trend of BTC and ETH
# Set index
btc_week = btc.set_index('date').resample('W').agg({
    'high': 'max',
    'low': 'min',
    'close': 'last'
})

eth_week = eth.set_index('date').resample('W').agg({
    'high': 'max',
    'low': 'min',
    'close': 'last'
})

# Plot
fig, ax = plt.subplots(2,1, figsize=(14,8))

# BTC
ax[0].fill_between(
    btc_week.index,
    btc_week['low'],
    btc_week['high'],
    color='#F7931A',
    alpha=0.25,
    label='Weekly High-Low Range'
)
ax[0].plot(
    btc_week.index,
    btc_week['close'],
    color='black',
    linewidth=1.5,
    label='Weekly Close'
)
ax[0].set_title("Bitcoin Weekly Price Range with Closing Trend (2021–2025)")
ax[0].set_ylabel("Price (USDT)")
ax[0].set_xlabel("Date")
ax[0].grid(True, linestyle='--', alpha=0.4)
ax[0].legend()

# ETH
ax[1].fill_between(
    eth_week.index,
    eth_week['low'],
    eth_week['high'],
    color='#627EEA',
    alpha=0.25,
    label='Weekly High-Low Range'
)
ax[1].plot(
    eth_week.index,
    eth_week['close'],
    color='black',
    linewidth=1.5,
    label='Weekly Close'
)
ax[1].set_title("Ethereum Weekly Price Range with Closing Trend (2021–2025)")
ax[1].set_ylabel("Price (USDT)")
ax[1].set_xlabel("Date")
ax[1].grid(True, linestyle='--', alpha=0.4)
ax[1].legend()

plt.tight_layout()
plt.show()

# Convert Date format
btc["date"] = pd.to_datetime(btc["date"])
eth["date"] = pd.to_datetime(eth["date"])

# Sort Date
btc = btc.sort_values("date").reset_index(drop=True)
eth = eth.sort_values("date").reset_index(drop=True)

# Double-check "date" data type
print("BTC date dtype:", btc["date"].dtype)
print("ETH date dtype:", eth["date"].dtype)

# Feature engineering

output_file = "crypto_daily_2021_2025_features.xlsx"

def add_features(df):

    df = df.copy()

    # Returns
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (7,14,30)
    df["volatility_7"] = df["daily_return"].rolling(7).std()
    df["volatility_14"] = df["daily_return"].rolling(14).std()
    df["volatility_30"] = df["daily_return"].rolling(30).std()

    # Moving averages
    df["MA_7"] = df["close"].rolling(7).mean()
    df["MA_14"] = df["close"].rolling(14).mean()
    df["MA_30"] = df["close"].rolling(30).mean()

    # EMA
    df["EMA_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["EMA_14"] = df["close"].ewm(span=14, adjust=False).mean()
    df["EMA_30"] = df["close"].ewm(span=30, adjust=False).mean()

    # Momentum
    df["momentum_7"] = df["close"] - df["close"].shift(7)
    df["momentum_14"] = df["close"] - df["close"].shift(14)
    df["momentum_30"] = df["close"] - df["close"].shift(30)

    # Volume trends
    df["volume_MA_7"] = df["volume"].rolling(7).mean()
    df["volume_MA_14"] = df["volume"].rolling(14).mean()
    df["volume_MA_30"] = df["volume"].rolling(30).mean()

    # RSI function
    def compute_rsi(series, window):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["RSI_7"] = compute_rsi(df["close"], 7)
    df["RSI_14"] = compute_rsi(df["close"], 14)
    df["RSI_30"] = compute_rsi(df["close"], 30)

    return df


# Use the existing data
btc_features = add_features(btc)
eth_features = add_features(eth)

# Save to new Excel file
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    btc_features.to_excel(writer, sheet_name="BTC", index=False)
    eth_features.to_excel(writer, sheet_name="ETH", index=False)

print("Feature engineering complete.")
print(f"Saved as: {output_file}")

# Examine Both Datasets After Featured
print("\n=== BTC NEW INFO ===")
btc_features.info()
print("\n=== ETH NEW INFO ===")
eth_features.info()

# Examine Feature-Engineered Dataset
# Preview rows
print("BTC first 10 rows:")
print(btc_features.head(10))

print("\nETH first 10 rows:")
print(eth_features.head(10))

# Shape
print("BTC shape:", btc_features.shape)
print("ETH shape:", eth_features.shape)

# Columns
print("BTC columns:")
print(btc_features.columns.tolist())

print("\nETH columns:")
print(eth_features.columns.tolist())

# Check variables data types
print("BTC data types:")
print(btc_features.dtypes)

print("\nETH data types:")
print(eth_features.dtypes)

# Check missing values
print("BTC missing values:")
print(btc_features.isnull().sum())

print("\nETH missing values:")
print(eth_features.isnull().sum())

# Summary statistics
print("BTC summary statistics:")
print(btc_features.describe())

print("\nETH summary statistics:")
print(eth_features.describe())

# Remove missing values
btc_clean = btc_features.dropna()
eth_clean = eth_features.dropna()

print("BTC shape after dropna:", btc_clean.shape)
print("ETH shape after dropna:", eth_clean.shape)

# Check data after dropping missing values
print("BTC (after dropna) - first 5 rows:")
print(btc_clean.head(10))

print("\nETH (after dropna) - first 5 rows:")
print(eth_clean.head(10))

# Data shape comparison before and after dropping missing values
print("BTC shape before:", btc_features.shape)
print("BTC shape after:", btc_clean.shape)

print("\nETH shape before:", eth_features.shape)
print("ETH shape after:", eth_clean.shape)

# Data summary after drop
print("BTC summary (after dropna):")
print(btc_clean.describe())

print("\nETH summary (after dropna):")
print(eth_clean.describe())

# Save cleaned dataset
output_clean = "crypto_daily_2021_2025_clean.xlsx"

with pd.ExcelWriter(output_clean, engine="openpyxl") as writer:
    btc_clean.to_excel(writer, sheet_name="BTC", index=False)
    eth_clean.to_excel(writer, sheet_name="ETH", index=False)

print("Cleaned dataset saved.")

# Normalised Price Comparison of BTC and ETH
btc_norm = btc_clean['close'] / btc_clean['close'].iloc[0]
eth_norm = eth_clean['close'] / eth_clean['close'].iloc[0]

plt.figure(figsize=(12,5))

plt.plot(btc_clean['date'], btc_norm, label="BTC", color='#F7931A')
plt.plot(eth_clean['date'], eth_norm, label="ETH", color='#627EEA')

plt.title("Normalised Price Comparison")
plt.xlabel("Date")
plt.ylabel("Relative Price")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# 30-Day Rolling Volatility Trend of BTC and ETH
plt.figure(figsize=(12,5))

plt.plot(btc_clean['date'], btc_clean['volatility_30'],
         label="BTC", color='#F7931A')

plt.plot(eth_clean['date'], eth_clean['volatility_30'],
         label="ETH", color='#627EEA')

plt.title("30-Day Rolling Volatility Comparison (BTC vs ETH)")
plt.xlabel("Date")
plt.ylabel("Volatility")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# Log Return Distribution of BTC and ETH
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# BTC
ax[0].hist(btc_clean['log_return'], bins='auto', color='#F7931A')
ax[0].set_title("BTC Log Return Distribution")
ax[0].set_xlabel("Log Return")
ax[0].set_ylabel("Frequency")

# ETH
ax[1].hist(eth_clean['log_return'], bins='auto', color='#627EEA')
ax[1].set_title("ETH Log Return Distribution")
ax[1].set_xlabel("Log Return")

plt.tight_layout()
plt.show()

# Correlation Heatmap of BTC and ETH Features
# BTC
plt.figure(figsize=(10,8))
sns.heatmap(btc_clean.corr(), cmap="coolwarm")
plt.title("BTC Correlation Heatmap")
plt.show()

# ETH
plt.figure(figsize=(10,8))
sns.heatmap(eth_clean.corr(), cmap="coolwarm")
plt.title("ETH Correlation Heatmap")
plt.show()

# BTC EDA
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Price
ax[0, 0].plot(btc_clean["date"], btc_clean["close"], color="#F7931A")
ax[0, 0].set_title("BTC Price")
ax[0, 0].set_ylabel("Price")
ax[0, 0].set_xlabel("Date")
ax[0, 0].grid(True)

# Log Return
ax[0, 1].plot(btc_clean["date"], btc_clean["log_return"], color="#F7931A")
ax[0, 1].set_title("BTC Log Return")
ax[0, 1].set_ylabel("Log Return")
ax[0, 1].set_xlabel("Date")
ax[0, 1].grid(True)

# Volatility
ax[1, 0].plot(btc_clean["date"], btc_clean["volatility_30"], color="#F7931A")
ax[1, 0].set_title("BTC 30-Day Rolling Volatility")
ax[1, 0].set_ylabel("Volatility")
ax[1, 0].set_xlabel("Date")
ax[1, 0].grid(True)

# Volume
ax[1, 1].plot(btc_clean["date"], btc_clean["volume"], color="#F7931A")
ax[1, 1].set_title("BTC Volume")
ax[1, 1].set_ylabel("Volume")
ax[1, 1].set_xlabel("Date")
ax[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ETH EDA
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Ethereum Market EDA (2021–2025)", fontsize=14)

# Price
ax[0, 0].plot(eth_clean["date"], eth_clean["close"], color="#627EEA")
ax[0, 0].set_title("ETH Price")
ax[0, 0].set_ylabel("Price")
ax[0, 0].set_xlabel("Date")
ax[0, 0].grid(True)

# Log Return
ax[0, 1].plot(eth_clean["date"], eth_clean["log_return"], color="#627EEA")
ax[0, 1].set_title("ETH Log Return")
ax[0, 1].set_ylabel("Log Return")
ax[0, 1].set_xlabel("Date")
ax[0, 1].grid(True)

# Volatility
ax[1, 0].plot(eth_clean["date"], eth_clean["volatility_30"], color="#627EEA")
ax[1, 0].set_title("ETH 30-Day Rolling Volatility")
ax[1, 0].set_ylabel("Volatility")
ax[1, 0].set_xlabel("Date")
ax[1, 0].grid(True)

# Volume
ax[1, 1].plot(eth_clean["date"], eth_clean["volume"], color="#627EEA")
ax[1, 1].set_title("ETH Volume")
ax[1, 1].set_ylabel("Volume")
ax[1, 1].set_xlabel("Date")
ax[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()