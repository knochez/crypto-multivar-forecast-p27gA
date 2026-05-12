"""
lstm1_volume.py — BTC Trading Volume forecasting using LSTM
=========================================================
Target   : volume (daily BTC trading volume)
Features : close, log_return, volatility_30, volume,
           RSI_14, EMA_14, momentum_14, number_of_trades
Lookbacks: 3, 5, 8, 10 days
Metrics  : RMSE, MAE, MAPE, MDA

Volume is the least predictable target — dominated by unpredictable spikes
driven by external news and market events not present in the feature set.
Expected findings: short lookbacks perform best, longer lookbacks overfit.


"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# config
# =============================================================================

DATA_PATH    = "crypto_daily_2021_2025_clean.xlsx"
TARGET       = "volume"
TARGET_LABEL = "BTC Trading Volume"
TARGET_UNIT  = "Volume (BTC)"
RESULTS_DIR  = "results_lstm_volume"

FEATURES = [
    "close", "log_return", "volatility_30", "volume",
    "RSI_14", "EMA_14", "momentum_14", "number_of_trades",
]

LOOKBACKS    = [3, 5, 8, 10]
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
EPOCHS       = 200
BATCH_SIZE   = 32
PATIENCE     = 25
MAX_LR       = 0.005   # reduced vs price to limit divergence at longer lookbacks
BULL_START   = "2025-04-01"
BULL_END     = "2025-10-01"
BEAR_START   = "2025-10-01"
BEAR_END     = "2026-01-01"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRS = {k: os.path.join(RESULTS_DIR, v) for k, v in {
    "predictions":   "1_prediction_plots",
    "loss_curves":   "2_loss_curves",
    "model_weights": "3_model_weights",
    "summary":       "4_summary"}.items()}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# =============================================================================
# data
# =============================================================================

# Loads excel file, sorts chronologically, selects required columns
def load_data():
    df = pd.read_excel(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date"] + FEATURES].copy()
    print(f"  Rows       : {len(df)}")
    print(f"  Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Missing    : {df[FEATURES].isnull().sum().sum()}")
    return df

# Splits chronologically, fits MinMaxScaler on train only, builds sliding windows
def split_and_scale(df, lookback):
    v  = df[FEATURES].values
    n  = len(v)
    te = int(n * TRAIN_RATIO)
    ve = int(n * (TRAIN_RATIO + VAL_RATIO))
    ti = FEATURES.index(TARGET)

    scaler = MinMaxScaler().fit(v[:te])

    def seqs(arr):
        X, y = [], []
        for i in range(lookback, len(arr)):
            X.append(arr[i - lookback:i])
            y.append(arr[i, ti])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    Xtr, ytr = seqs(scaler.transform(v[:te]))
    Xv,  yv  = seqs(scaler.transform(v[te:ve]))
    Xt,  yt  = seqs(scaler.transform(v[ve:]))
    return Xtr, ytr, Xv, yv, Xt, yt, scaler, ti, df["date"].values[ve + lookback:]

# =============================================================================
# lstm
# =============================================================================

class LSTMForecaster(nn.Module):
    """
    GRU -> BatchNorm -> Dropout -> Linear(128->32) -> ReLU -> Linear(32->1)
    Output unbounded — volume is strictly positive but model learns this from data.
    """
    def __init__(self, input_size):
        super().__init__()
        self.lstm        = nn.LSTM(
            input_size  = input_size,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            dropout     = DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first = True,
        )
        self.batch_norm = nn.BatchNorm1d(HIDDEN_SIZE)
        self.dropout    = nn.Dropout(DROPOUT)
        self.fc1        = nn.Linear(HIDDEN_SIZE, 32)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.batch_norm(out)
        out    = self.dropout(out)
        out    = self.relu(self.fc1(out))
        out    = self.dropout(out)
        return self.fc2(out).squeeze(-1)

# =============================================================================
# training
# =============================================================================

# Trains lstm with OneCycleLR warmup and early stopping, returns best model
def train_model(Xtr, ytr, Xv, yv):
    tl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    BATCH_SIZE, shuffle=False, drop_last=True)
    vl = DataLoader(TensorDataset(torch.tensor(Xv),  torch.tensor(yv)),
                    BATCH_SIZE, shuffle=False)

    model = LSTMForecaster(Xtr.shape[2]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=MAX_LR / 25)
    sch   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=MAX_LR, epochs=EPOCHS,
        steps_per_epoch=len(tl), pct_start=0.3,
        div_factor=25, final_div_factor=1000
    )
    crit  = nn.MSELoss()
    best, best_state, wait, tls, vls = float("inf"), None, 0, [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        el = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            el += loss.item() * len(xb)
        el /= len(tl.dataset)

        model.eval()
        with torch.no_grad():
            ev = sum(crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
                     for xb, yb in vl) / len(vl.dataset)

        tls.append(el); vls.append(ev)
        if ev < best:
            best = ev
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, tls, vls

# =============================================================================
# evaluation
# =============================================================================

# Inverse-transforms predictions and computes RMSE, MAE, MAPE, MDA
def evaluate(model, Xt, yt, scaler, ti):
    model.eval()
    with torch.no_grad():
        ps = model(torch.tensor(Xt).to(DEVICE)).cpu().numpy()

    def inv(v):
        d = np.zeros((len(v), len(FEATURES)))
        d[:, ti] = v
        return scaler.inverse_transform(d)[:, ti]

    yt, yp = inv(yt), inv(ps)
    mda  = np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))) * 100
    mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "MAE":  round(float(mean_absolute_error(yt, yp)),         4),
        "MAPE": round(float(mape),                                4),
        "MDA":  round(float(mda),                                 2),
    }, yt, yp

# =============================================================================
# Plots
# =============================================================================

# Plots actual vs predicted volume with regime shading, spike markers, metrics in title
def plot_predictions(dates, yt, yp, lb, m):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axvspan(np.datetime64(BULL_START), np.datetime64(BULL_END),
               alpha=0.06, color="green",
               label="Bull regime — period of rising BTC price (Apr–Oct 2025)")
    ax.axvspan(np.datetime64(BEAR_START), np.datetime64(BEAR_END),
               alpha=0.06, color="red",
               label="Bear regime — period of declining BTC price (Oct 2025–Jan 2026)")
    ax.fill_between(dates, yt, yp, alpha=0.15, color="#D85A30",
                    label="Error region — shaded area between actual and predicted values")
    ax.plot(dates, yt, color="#185FA5", linewidth=1.3,
            label="Actual btc trading volume — true daily volume from market data")
    ax.plot(dates, yp, color="#D85A30", linewidth=1.1,
            linestyle="--", alpha=0.85,
            label="Predicted btc trading volume — LSTM model one-day-ahead forecast")

    # mark volume spikes (>1.5x 7-day rolling mean)
    roll_mean = pd.Series(yt).rolling(7, min_periods=1).mean().values
    spikes    = yt > (roll_mean * 1.5)
    ax.scatter(np.array(dates)[spikes], yt[spikes],
               color="#7F77DD", s=25, zorder=5,
               label="Volume spike — day where volume exceeds 1.5x rolling 7-day mean")

    ax.set_title(
        f"LSTM Forecast — {TARGET_LABEL}  |  Lookback = {lb} days\n"
        f"RMSE = {m['RMSE']}   MAE = {m['MAE']}   "
        f"MAPE = {m['MAPE']:.2f}%   MDA = {m['MDA']:.1f}%",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(TARGET_UNIT, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRS["predictions"],
                f"Volume_Forecast_lookback_{lb}d.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved prediction plot -> {DIRS['predictions']}/Volume_Forecast_lookback_{lb}d.png")

# Plots training and validation MSE loss curves to check for overfitting
def plot_loss(tls, vls, lb):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(tls, label="Training loss",   color="#185FA5", linewidth=1.3)
    ax.plot(vls, label="Validation loss", color="#D85A30", linewidth=1.3)
    ax.set_title(f"Training & Validation Loss — {TARGET_LABEL}  |  Lookback = {lb} days",
                 fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(DIRS["loss_curves"],
                f"Loss_Curve_lookback_{lb}d.png"), dpi=150)
    plt.close(fig)

# Plots 2x2 grid of all four lookback forecasts with metrics in each subtitle
def plot_combined(all_preds):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, r in zip(axes.flatten(), all_preds):
        ax.fill_between(r["dates"], r["yt"], r["yp"],
                        alpha=0.12, color="#D85A30")
        ax.plot(r["dates"], r["yt"], color="#185FA5",
                linewidth=1.2, label="Actual")
        ax.plot(r["dates"], r["yp"], color="#D85A30",
                linewidth=1.0, linestyle="--", alpha=0.85, label="Predicted")
        m = r["m"]
        ax.set_title(
            f"Lookback = {r['lb']} days  |  RMSE = {m['RMSE']}   "
            f"MAE = {m['MAE']}   MAPE = {m['MAPE']:.2f}%   MDA = {m['MDA']:.1f}%",
            fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Volume (BTC)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"LSTM Forecasts — All Lookback Windows  |  {TARGET_LABEL}",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRS["summary"], "All_Lookbacks_Combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined plot -> {DIRS['summary']}/All_Lookbacks_Combined.png")

# Plots four bar charts showing each metric across all lookback windows
def plot_summary(results_df):
    metrics = ["RMSE", "MAE", "MAPE", "MDA"]
    colours = ["#185FA5", "#D85A30", "#1D9E75", "#7F77DD"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric, colour in zip(axes, metrics, colours):
        bars = ax.bar([f"{lb}d" for lb in results_df["lookback"]],
                      results_df[metric], color=colour, alpha=0.85, width=0.5)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.2f}" if metric != "MDA" else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("Lookback window", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if metric == "MDA":
            ax.axhline(50, color="black", linewidth=0.8,
                       linestyle="--", label="Random baseline (50%)")
            ax.legend(fontsize=8)
    fig.suptitle(
        f"LSTM Performance Summary — {TARGET_LABEL}\n"
        f"(RMSE/MAE/MAPE: lower is better  |  MDA: higher is better, 50% = coin flip)",
        fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRS["summary"], "Performance_Summary_Volume.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary chart -> {DIRS['summary']}/Performance_Summary_Volume.png")

# =============================================================================
# main
# =============================================================================

def main():
    print("=" * 55)
    print(f"  LSTM Forecasting — {TARGET_LABEL}")
    print(f"  Device: {DEVICE}  |  Lookbacks: {LOOKBACKS}")
    print(f"  Hidden: {HIDDEN_SIZE}  |  Scheduler: OneCycleLR max_lr={MAX_LR}")
    print(f"  Note: Volume is the noisiest target — expect highest errors")
    print("=" * 55)

    df = load_data()
    results, all_preds = [], []

    for i, lb in enumerate(LOOKBACKS, 1):
        print(f"\n[{i}/{len(LOOKBACKS)}] Lookback = {lb} days")
        Xtr, ytr, Xv, yv, Xt, yt, scaler, ti, dates = split_and_scale(df, lb)
        print(f"  Train={len(Xtr)}  Val={len(Xv)}  Test={len(Xt)}")

        model, tls, vls = train_model(Xtr, ytr, Xv, yv)
        m, yt_inv, yp_inv = evaluate(model, Xt, yt, scaler, ti)
        print(f"  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
              f"MAPE={m['MAPE']:.2f}%  MDA={m['MDA']:.1f}%")

        plot_predictions(dates, yt_inv, yp_inv, lb, m)
        plot_loss(tls, vls, lb)
        torch.save(model.state_dict(), os.path.join(
            DIRS["model_weights"],
            f"Model_Weights_LSTM_Volume_lookback_{lb}d.pt"))

        results.append({"lookback": lb, **m})
        all_preds.append({"dates": dates, "yt": yt_inv,
                          "yp": yp_inv, "lb": lb, "m": m})

    plot_combined(all_preds)
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 55)
    print(df_res.to_string(index=False))

    best = df_res.loc[df_res["RMSE"].idxmin()]
    print(f"\n  Best by RMSE: lookback {int(best['lookback'])}d  "
          f"RMSE={best['RMSE']:.2f}  MDA={best['MDA']:.1f}%")

    csv = os.path.join(DIRS["summary"], "Results_LSTM_Volume.csv")
    df_res.to_csv(csv, index=False)
    print(f"\n  Saved -> {csv}")
    plot_summary(df_res)


if __name__ == "__main__":
    main()
