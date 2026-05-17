import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone

# ============================================================
# GPT-DESIGNED HYBRID FORECASTING MODEL
# Targets: close, log_return, volume, volatility_30
# Coins: BTC, ETH
# Lookbacks: 3, 5, 8, 10
# Train: 2021-2024
# Test: 2025
#
# Model idea:
#   - Naive persistence forecast
#   - Lag-mean forecast
#   - Regularised linear model: Ridge
#   - Nonlinear model: HistGradientBoostingRegressor
#   - Validation-weighted ensemble
#
# This is NOT an OpenAI API model.
# This is a ChatGPT-designed forecasting methodology.
# ============================================================

# -----------------------------
# 1. Config
# -----------------------------
file_path = "crypto_daily_2021_2025_clean.xlsx"

coins = ["BTC", "ETH"]
lookbacks = [3, 5, 8, 10]

targets = ["close", "log_return", "volume", "volatility_30"]

feature_cols = ["close", "log_return", "volume", "volatility_30"]

train_start = "2021-01-01"
train_end   = "2024-12-31"

validation_start = "2024-01-01"
validation_end   = "2024-12-31"

inner_train_start = "2021-01-01"
inner_train_end   = "2023-12-31"

test_start = "2025-01-01"
test_end   = "2025-12-31"

# Regime windows
bull_start = "2025-01-01"
bull_end   = "2025-07-31"
bear_start = "2025-08-01"
bear_end   = "2025-12-31"

# Output folders
output_dir = Path("gpt_hybrid_outputs")
graphs_dir = output_dir / "graphs"
preds_dir = output_dir / "predictions"

graphs_dir.mkdir(parents=True, exist_ok=True)
preds_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Helper metrics
# -----------------------------
def safe_mape(y_true, y_pred):
    """
    MAPE can explode when actual values are close to zero.
    This safely ignores exact zero actual values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    non_zero_mask = y_true != 0

    if np.sum(non_zero_mask) == 0:
        return np.nan

    return np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    ) * 100


def mean_directional_accuracy(y_true, y_pred, y_prev, target_col):
    """
    Directional accuracy is handled differently depending on the target.

    For log_return:
        Direction = sign of the return itself.

    For close, volume, volatility:
        Direction = whether the value increased or decreased compared to the previous day.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prev = np.asarray(y_prev)

    if target_col == "log_return":
        actual_direction = np.sign(y_true)
        predicted_direction = np.sign(y_pred)
    else:
        actual_direction = np.sign(y_true - y_prev)
        predicted_direction = np.sign(y_pred - y_prev)

    return np.mean(actual_direction == predicted_direction) * 100


def enforce_valid_range(preds, target_col):
    """
    Close price, volume, and volatility cannot be negative.
    Log returns can be negative.
    """
    preds = np.asarray(preds)

    if target_col in ["close", "volume", "volatility_30"]:
        preds = np.maximum(preds, 0)

    return preds


# -----------------------------
# 3. Lagged feature creation
# -----------------------------
def create_lagged_dataset(df, feature_cols, target_col, lookback):
    """
    Creates lagged features for the selected lookback.

    Example for lookback = 3:
        close_lag1, close_lag2, close_lag3
        log_return_lag1, log_return_lag2, log_return_lag3
        volume_lag1, volume_lag2, volume_lag3
        volatility_30_lag1, volatility_30_lag2, volatility_30_lag3

    Also creates summary lag features:
        close_lag_mean, close_lag_std, close_lag_min, close_lag_max, close_lag_change
    """
    data = df[["date"] + feature_cols].copy()

    new_cols = {}

    for col in feature_cols:
        lag_names = []

        for lag in range(1, lookback + 1):
            lag_col = f"{col}_lag{lag}"
            new_cols[lag_col] = data[col].shift(lag)
            lag_names.append(lag_col)

        lag_block = pd.concat([new_cols[name] for name in lag_names], axis=1)
        lag_block.columns = lag_names

        new_cols[f"{col}_lag_mean"] = lag_block.mean(axis=1)
        new_cols[f"{col}_lag_std"] = lag_block.std(axis=1)
        new_cols[f"{col}_lag_min"] = lag_block.min(axis=1)
        new_cols[f"{col}_lag_max"] = lag_block.max(axis=1)
        new_cols[f"{col}_lag_change"] = lag_block[f"{col}_lag1"] - lag_block[f"{col}_lag{lookback}"]

    lagged_features = pd.DataFrame(new_cols)

    data = pd.concat([data[["date"]], lagged_features], axis=1)

    data["target"] = df[target_col].values
    data["previous_target"] = df[target_col].shift(1).values

    data = data.dropna().reset_index(drop=True)

    return data


# -----------------------------
# 4. Model creation
# -----------------------------
def make_ridge_model(target_col):
    """
    Ridge is a regularised linear model.
    For positive targets, we use log1p target transformation to stabilise scale.
    """
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    if target_col in ["close", "volume", "volatility_30"]:
        return TransformedTargetRegressor(
            regressor=base_model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False
        )

    return base_model


def make_hgb_model(target_col):
    """
    HistGradientBoostingRegressor captures nonlinear relationships.
    For positive targets, we use log1p target transformation.
    """
    base_model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.05,
        random_state=42
    )

    if target_col in ["close", "volume", "volatility_30"]:
        return TransformedTargetRegressor(
            regressor=base_model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False
        )

    return base_model


# -----------------------------
# 5. Ensemble helper
# -----------------------------
def calculate_validation_weights(y_val, component_predictions):
    """
    Each component receives a weight based on validation RMSE.

    Lower RMSE = higher weight.
    """
    rmses = {}

    for name, preds in component_predictions.items():
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses[name] = rmse

    inverse_errors = {
        name: 1 / (rmse + 1e-9)
        for name, rmse in rmses.items()
    }

    total_inverse_error = sum(inverse_errors.values())

    weights = {
        name: inverse_errors[name] / total_inverse_error
        for name in inverse_errors
    }

    return weights, rmses


def weighted_ensemble_prediction(component_predictions, weights):
    """
    Combines component predictions using validation-derived weights.
    """
    final_pred = np.zeros(len(next(iter(component_predictions.values()))))

    for name, preds in component_predictions.items():
        final_pred += weights[name] * preds

    return final_pred


# -----------------------------
# 6. Main experiment loop
# -----------------------------
summary_results = {}

for target_col in targets:

    print(f"\n{'#' * 90}")
    print(f"RUNNING GPT-HYBRID FORECASTING MODEL FOR TARGET: {target_col}")
    print(f"{'#' * 90}")

    for coin in coins:

        print(f"\n{'=' * 80}")
        print(f"COIN: {coin} | TARGET: {target_col}")
        print(f"{'=' * 80}")

        df = pd.read_excel(file_path, sheet_name=coin)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        subplot_data = []

        for lookback in lookbacks:

            print(f"\n--- {coin} | target = {target_col} | lookback = {lookback} ---")

            # Create lagged dataset
            lagged_df = create_lagged_dataset(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                lookback=lookback
            )

            # Split data
            inner_train_df = lagged_df[
                (lagged_df["date"] >= inner_train_start) &
                (lagged_df["date"] <= inner_train_end)
            ].copy()

            validation_df = lagged_df[
                (lagged_df["date"] >= validation_start) &
                (lagged_df["date"] <= validation_end)
            ].copy()

            full_train_df = lagged_df[
                (lagged_df["date"] >= train_start) &
                (lagged_df["date"] <= train_end)
            ].copy()

            test_df = lagged_df[
                (lagged_df["date"] >= test_start) &
                (lagged_df["date"] <= test_end)
            ].copy()

            X_cols = [
                col for col in lagged_df.columns
                if col not in ["date", "target", "previous_target"]
            ]

            X_inner_train = inner_train_df[X_cols]
            y_inner_train = inner_train_df["target"]

            X_val = validation_df[X_cols]
            y_val = validation_df["target"]

            X_full_train = full_train_df[X_cols]
            y_full_train = full_train_df["target"]

            X_test = test_df[X_cols]
            y_test = test_df["target"]
            y_prev_test = test_df["previous_target"]

            # -----------------------------
            # 6.1 Validation phase
            # -----------------------------
            ridge_model_val = make_ridge_model(target_col)
            hgb_model_val = make_hgb_model(target_col)

            ridge_model_val.fit(X_inner_train, y_inner_train)
            hgb_model_val.fit(X_inner_train, y_inner_train)

            val_preds = {}

            # Component 1: naive persistence
            val_preds["naive_last"] = validation_df[f"{target_col}_lag1"].values

            # Component 2: mean of lag window
            val_preds["lag_mean"] = validation_df[f"{target_col}_lag_mean"].values

            # Component 3: Ridge
            val_preds["ridge"] = ridge_model_val.predict(X_val)

            # Component 4: HistGradientBoosting
            val_preds["hgb"] = hgb_model_val.predict(X_val)

            for name in val_preds:
                val_preds[name] = enforce_valid_range(val_preds[name], target_col)

            weights, validation_rmses = calculate_validation_weights(y_val, val_preds)

            # -----------------------------
            # 6.2 Final training phase
            # -----------------------------
            ridge_model_final = make_ridge_model(target_col)
            hgb_model_final = make_hgb_model(target_col)

            ridge_model_final.fit(X_full_train, y_full_train)
            hgb_model_final.fit(X_full_train, y_full_train)

            test_preds = {}

            test_preds["naive_last"] = test_df[f"{target_col}_lag1"].values
            test_preds["lag_mean"] = test_df[f"{target_col}_lag_mean"].values
            test_preds["ridge"] = ridge_model_final.predict(X_test)
            test_preds["hgb"] = hgb_model_final.predict(X_test)

            for name in test_preds:
                test_preds[name] = enforce_valid_range(test_preds[name], target_col)

            y_pred = weighted_ensemble_prediction(test_preds, weights)
            y_pred = enforce_valid_range(y_pred, target_col)

            # -----------------------------
            # 6.3 Metrics
            # -----------------------------
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = safe_mape(y_test, y_pred)
            mda = mean_directional_accuracy(
                y_true=y_test.values,
                y_pred=y_pred,
                y_prev=y_prev_test.values,
                target_col=target_col
            )

            # -----------------------------
            # 6.4 Save predictions
            # -----------------------------
            results_df = test_df[["date"]].copy()
            results_df[f"actual_{target_col}"] = y_test.values
            results_df[f"predicted_{target_col}"] = y_pred
            results_df["previous_target"] = y_prev_test.values
            results_df["error"] = results_df[f"actual_{target_col}"] - results_df[f"predicted_{target_col}"]

            # Add component predictions
            for name, preds in test_preds.items():
                results_df[f"{name}_prediction"] = preds

            # Add ensemble weights
            for name, weight in weights.items():
                results_df[f"{name}_weight"] = weight

            pred_file = preds_dir / f"{coin.lower()}_gpt_hybrid_{target_col}_lb{lookback}_predictions.csv"
            results_df.to_csv(pred_file, index=False)

            # Store summary
            summary_results[(coin, target_col, lookback)] = {
                "coin": coin,
                "model": "GPT-Hybrid",
                "target": target_col,
                "lookback": lookback,
                "inner_train_rows": len(inner_train_df),
                "validation_rows": len(validation_df),
                "full_train_rows": len(full_train_df),
                "test_rows": len(test_df),
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "mda": mda,
                "weight_naive_last": weights["naive_last"],
                "weight_lag_mean": weights["lag_mean"],
                "weight_ridge": weights["ridge"],
                "weight_hgb": weights["hgb"],
                "val_rmse_naive_last": validation_rmses["naive_last"],
                "val_rmse_lag_mean": validation_rmses["lag_mean"],
                "val_rmse_ridge": validation_rmses["ridge"],
                "val_rmse_hgb": validation_rmses["hgb"]
            }

            subplot_data.append({
                "lookback": lookback,
                "dates": results_df["date"].values,
                "y_true": results_df[f"actual_{target_col}"].values,
                "y_pred": results_df[f"predicted_{target_col}"].values,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "mda": mda,
                "weights": weights
            })

            print(f"Inner train rows: {len(inner_train_df)}")
            print(f"Validation rows:  {len(validation_df)}")
            print(f"Full train rows:  {len(full_train_df)}")
            print(f"Test rows:        {len(test_df)}")

            print(f"RMSE:             {rmse:.6f}")
            print(f"MAE:              {mae:.6f}")
            print(f"MAPE:             {mape:.2f}%")
            print(f"MDA:              {mda:.2f}%")

            print("Validation-based ensemble weights:")
            for name, weight in weights.items():
                print(f"  {name}: {weight:.3f}")

            print(f"Saved predictions to: {pred_file}")

        # -----------------------------
        # 7. Combined figure per coin + target
        # -----------------------------
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        axes = axes.flatten()

        for ax, item in zip(axes, subplot_data):

            dates = item["dates"]
            y_true = item["y_true"]
            y_pred = item["y_pred"]
            lookback = item["lookback"]

            ax.axvspan(
                np.datetime64(bull_start),
                np.datetime64(bull_end),
                alpha=0.06,
                color="green"
            )

            ax.axvspan(
                np.datetime64(bear_start),
                np.datetime64(bear_end),
                alpha=0.06,
                color="red"
            )

            ax.fill_between(
                dates,
                y_true,
                y_pred,
                alpha=0.15,
                color="#D85A30"
            )

            ax.plot(
                dates,
                y_true,
                color="#185FA5",
                linewidth=1.4
            )

            ax.plot(
                dates,
                y_pred,
                color="#D85A30",
                linewidth=1.1,
                linestyle="--",
                alpha=0.85
            )

            ax.set_title(
                f"Lookback = {lookback} days\n"
                f"RMSE = {item['rmse']:.6f}   "
                f"MAE = {item['mae']:.6f}   "
                f"MAPE = {item['mape']:.2f}%   "
                f"MDA = {item['mda']:.1f}%",
                fontsize=10,
                pad=8
            )

            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel(target_col, fontsize=10)
            ax.grid(True, alpha=0.3)

        legend_handles = [
            plt.Line2D([0], [0], color="#185FA5", lw=1.5),
            plt.Line2D([0], [0], color="#D85A30", lw=1.5, linestyle="--"),
            plt.Rectangle((0, 0), 1, 1, color="#D85A30", alpha=0.15),
            plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.06),
            plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.06),
        ]

        legend_labels = [
            "Actual",
            "Predicted",
            "Error region",
            "Bull regime",
            "Bear regime"
        ]

        fig.legend(
            legend_handles,
            legend_labels,
            loc="center right",
            fontsize=10
        )

        fig.suptitle(
            f"GPT-Hybrid Forecast — {coin} {target_col}",
            fontsize=14,
            y=0.98
        )

        fig.tight_layout(rect=[0, 0, 0.88, 0.96])

        combined_file = graphs_dir / f"{coin.lower()}_gpt_hybrid_{target_col}_all_lookbacks.png"
        fig.savefig(combined_file, dpi=150)
        plt.close(fig)

        print(f"\nSaved combined graph to: {combined_file}")

# -----------------------------
# 8. Final summary
# -----------------------------
summary_df = pd.DataFrame(summary_results.values())

print(f"\n{'=' * 90}")
print("FINAL GPT-HYBRID SUMMARY TABLE")
print(f"{'=' * 90}")

summary_df = summary_df.sort_values(["target", "coin", "lookback"])
print(summary_df)

summary_file = output_dir / "gpt_hybrid_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"\nSaved summary to: {summary_file}")