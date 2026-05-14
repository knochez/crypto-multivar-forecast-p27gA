import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# FINAL BASELINE LINEAR REGRESSION CODE FOR CLOSE PRICE PREDICTION
# GROUPED FIGURE + SINGLE LEGEND
# Target: close
# Coins: BTC, ETH
# Lookbacks: 3, 5, 8, 10
# Train: 2021-2024
# Test: 2025
# ============================================================

# -----------------------------
# 1. Config
# -----------------------------
file_path = "crypto_daily_2021_2025_clean.xlsx"
coins = ["BTC", "ETH"]
lookbacks = [3, 5, 8, 10]
target_col = "close"
feature_cols = ["close", "log_return", "volume", "volatility_30"]

train_start = "2021-01-01"
train_end   = "2024-12-31"
test_start  = "2025-01-01"
test_end    = "2025-12-31"

output_dir = Path("lr_close_outputs")
graphs_dir = output_dir / "graphs"
preds_dir = output_dir / "predictions"

graphs_dir.mkdir(parents=True, exist_ok=True)
preds_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Helper metrics
# -----------------------------
def mean_directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    if len(actual_diff) == 0:
        return np.nan

    return np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# -----------------------------
# 3. Regime windows
# -----------------------------
bull_start = "2025-01-01"
bull_end   = "2025-07-31"
bear_start = "2025-08-01"
bear_end   = "2025-12-31"

# -----------------------------
# 4. Storage
# -----------------------------
summary_results = {}

# -----------------------------
# 5. Main loop
# -----------------------------
for coin in coins:
    print(f"\n{'='*80}")
    print(f"RUNNING FINAL BASELINE LINEAR REGRESSION FOR {coin}")
    print(f"{'='*80}")

    df = pd.read_excel(file_path, sheet_name=coin)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    subplot_data = []

    for lookback in lookbacks:
        print(f"\n--- {coin} | lookback = {lookback} ---")

        data = df[["date"] + feature_cols].copy()

        for col in feature_cols:
            for lag in range(1, lookback + 1):
                data[f"{col}_lag{lag}"] = data[col].shift(lag)

        data["target"] = data[target_col]
        lagged_df = data.dropna().reset_index(drop=True)

        train_df = lagged_df[
            (lagged_df["date"] >= train_start) & (lagged_df["date"] <= train_end)
        ].copy()

        test_df = lagged_df[
            (lagged_df["date"] >= test_start) & (lagged_df["date"] <= test_end)
        ].copy()

        X_cols = []
        for col in feature_cols:
            for lag in range(1, lookback + 1):
                X_cols.append(f"{col}_lag{lag}")

        X_train = train_df[X_cols]
        y_train = train_df["target"]

        X_test = test_df[X_cols]
        y_test = test_df["target"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = safe_mape(y_test, y_pred)
        mda = mean_directional_accuracy(y_test.values, y_pred)

        results_df = test_df[["date"]].copy()
        results_df["actual_close"] = y_test.values
        results_df["predicted_close"] = y_pred
        results_df["error"] = results_df["actual_close"] - results_df["predicted_close"]

        pred_file = preds_dir / f"{coin.lower()}_lr_close_lb{lookback}_predictions.csv"
        results_df.to_csv(pred_file, index=False)

        summary_results[(coin, lookback)] = {
            "coin": coin,
            "target": target_col,
            "lookback": lookback,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "mda": mda
        }

        subplot_data.append({
            "lookback": lookback,
            "dates": results_df["date"].values,
            "y_true": results_df["actual_close"].values,
            "y_pred": results_df["predicted_close"].values,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "mda": mda
        })

        print(f"Train rows: {len(train_df)}")
        print(f"Test rows:  {len(test_df)}")
        print(f"RMSE:       {rmse:.6f}")
        print(f"MAE:        {mae:.6f}")
        print(f"MAPE:       {mape:.2f}%")
        print(f"MDA:        {mda:.2f}%")
        print(f"Saved predictions to: {pred_file}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    for ax, item in zip(axes, subplot_data):
        dates = item["dates"]
        y_true = item["y_true"]
        y_pred = item["y_pred"]
        lookback = item["lookback"]

        ax.axvspan(np.datetime64(bull_start), np.datetime64(bull_end), alpha=0.06, color="green")
        ax.axvspan(np.datetime64(bear_start), np.datetime64(bear_end), alpha=0.06, color="red")

        ax.fill_between(dates, y_true, y_pred, alpha=0.15, color="#D85A30")
        ax.plot(dates, y_true, color="#185FA5", linewidth=1.4)
        ax.plot(dates, y_pred, color="#D85A30", linewidth=1.1, linestyle="--", alpha=0.85)

        ax.set_title(
            f"Lookback = {lookback} days\n"
            f"RMSE = {item['rmse']:.2f}   "
            f"MAE = {item['mae']:.2f}   "
            f"MAPE = {item['mape']:.2f}%   "
            f"MDA = {item['mda']:.1f}%",
            fontsize=10,
            pad=8
        )
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Close Price", fontsize=10)
        ax.grid(True, alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], color="#185FA5", lw=1.5),
        plt.Line2D([0], [0], color="#D85A30", lw=1.5, linestyle="--"),
        plt.Rectangle((0, 0), 1, 1, color="#D85A30", alpha=0.15),
        plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.06),
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.06),
    ]

    legend_labels = ["Actual", "Predicted", "Error region", "Bull regime", "Bear regime"]

    fig.legend(legend_handles, legend_labels, loc="center right", fontsize=10)

    fig.suptitle(f"Linear Regression Forecast — {coin} Close Price", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    combined_file = graphs_dir / f"{coin.lower()}_lr_close_all_lookbacks.png"
    fig.savefig(combined_file, dpi=150)
    plt.close(fig)

    print(f"Saved combined graph to: {combined_file}")

summary_df = pd.DataFrame(summary_results.values())

print(f"\n{'='*80}")
print("FINAL SUMMARY TABLE")
print(f"{'='*80}")
print(summary_df.sort_values(["coin", "lookback"]))

summary_file = output_dir / "lr_close_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"\nSaved summary to: {summary_file}")