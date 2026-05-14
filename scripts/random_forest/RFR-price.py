import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# RANDOM FOREST — PRICE PREDICTION (GROUPED + SINGLE LEGEND)
# ============================================================

file_path = "crypto_daily_2021_2025_clean.xlsx"
coins = ["BTC", "ETH"]
lookbacks = [3, 5, 8, 10]

target_col = "close"
feature_cols = ["close", "log_return", "volume", "volatility_30"]

train_start = "2021-01-01"
train_end   = "2024-12-31"
test_start  = "2025-01-01"
test_end    = "2025-12-31"

rf_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "random_state": 42,
    "n_jobs": -1
}

output_dir = Path("rf_price_outputs")
graphs_dir = output_dir / "graphs"
preds_dir = output_dir / "predictions"

graphs_dir.mkdir(parents=True, exist_ok=True)
preds_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Metrics
# -----------------------------
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mean_directional_accuracy(y_true, y_pred):
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    return np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100

# -----------------------------
# Regime windows
# -----------------------------
bull_start = "2025-01-01"
bull_end   = "2025-07-31"
bear_start = "2025-08-01"
bear_end   = "2025-12-31"

# -----------------------------
# Main loop
# -----------------------------
for coin in coins:
    print(f"\n{'='*80}")
    print(f"RUNNING RANDOM FOREST PRICE PREDICTION FOR {coin}")
    print(f"{'='*80}")

    df = pd.read_excel(file_path, sheet_name=coin)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    subplot_data = []

    for lookback in lookbacks:

        data = df[["date"] + feature_cols].copy()

        for col in feature_cols:
            for lag in range(1, lookback + 1):
                data[f"{col}_lag{lag}"] = data[col].shift(lag)

        data["target"] = data[target_col]
        lagged_df = data.dropna().reset_index(drop=True)

        train_df = lagged_df[(lagged_df["date"] >= train_start) & (lagged_df["date"] <= train_end)]
        test_df  = lagged_df[(lagged_df["date"] >= test_start) & (lagged_df["date"] <= test_end)]

        X_cols = [f"{col}_lag{lag}" for col in feature_cols for lag in range(1, lookback+1)]

        X_train, y_train = train_df[X_cols], train_df["target"]
        X_test, y_test   = test_df[X_cols], test_df["target"]

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = safe_mape(y_test, y_pred)
        mda = mean_directional_accuracy(y_test.values, y_pred)

        results_df = test_df[["date"]].copy()
        results_df["actual"] = y_test.values
        results_df["predicted"] = y_pred

        subplot_data.append({
            "lookback": lookback,
            "dates": results_df["date"].values,
            "y_true": results_df["actual"].values,
            "y_pred": results_df["predicted"].values,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "mda": mda
        })

    # -----------------------------
    # Plot combined figure
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    for ax, item in zip(axes, subplot_data):
        ax.axvspan(np.datetime64(bull_start), np.datetime64(bull_end),
                   alpha=0.06, color="green")
        ax.axvspan(np.datetime64(bear_start), np.datetime64(bear_end),
                   alpha=0.06, color="red")

        ax.fill_between(item["dates"], item["y_true"], item["y_pred"],
                        alpha=0.15, color="#D85A30")

        ax.plot(item["dates"], item["y_true"], color="#185FA5", linewidth=1.4)
        ax.plot(item["dates"], item["y_pred"], color="#D85A30",
                linestyle="--", linewidth=1.1)

        ax.set_title(
            f"Lookback = {item['lookback']} days\n"
            f"RMSE={item['rmse']:.2f}  MAE={item['mae']:.2f}  "
            f"MAPE={item['mape']:.2f}%  MDA={item['mda']:.1f}%",
            fontsize=10
        )

        ax.grid(True, alpha=0.3)

    # -----------------------------
    # GLOBAL LEGEND (RIGHT SIDE)
    # -----------------------------
    lines = [
        plt.Line2D([0], [0], color="#185FA5", lw=1.5),
        plt.Line2D([0], [0], color="#D85A30", lw=1.5, linestyle="--"),
        plt.Rectangle((0,0),1,1, color="#D85A30", alpha=0.15),
        plt.Rectangle((0,0),1,1, color="green", alpha=0.06),
        plt.Rectangle((0,0),1,1, color="red", alpha=0.06),
    ]

    labels = [
        "Actual",
        "Predicted",
        "Error region",
        "Bull regime",
        "Bear regime"
    ]

    fig.legend(lines, labels, loc="center right", fontsize=10)

    fig.suptitle(f"Random Forest Forecast — {coin} Close Price", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])  # space for legend

    save_path = graphs_dir / f"{coin.lower()}_rf_price_all_lookbacks.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved combined graph to: {save_path}")