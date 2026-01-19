import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Reproducibility & performance
# ----------------------------
np.random.seed(42)
tf.random.set_seed(42)

tf.get_logger().setLevel("ERROR")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = "/home/hamsa/Documents/BSc-thesis/Transfer Learning/Transfer/TL2"

ROT_PATH = os.path.join(BASE_DIR, "rotterdam_TL.csv")
UTR_PATH = os.path.join(BASE_DIR, "utrecht_TL.csv")

METRICS_OUT_PATH = os.path.join(BASE_DIR, "TL2_ROT_UTR_performance_metrics.csv")
GRID_OUT_PATH = os.path.join(BASE_DIR, "TL2_ROT_UTR_grid_search_results.csv")

JUST_DIR = os.path.join(BASE_DIR, "TL_justification")
os.makedirs(JUST_DIR, exist_ok=True)

TEMPORAL_SIM_PATH = os.path.join(JUST_DIR, "temporal_similarity.csv")
WEIGHT_CHANGE_PATH = os.path.join(JUST_DIR, "weight_change_per_layer.csv")

# ----------------------------
# Load data
# ----------------------------
rot = pd.read_csv(ROT_PATH)
utr = pd.read_csv(UTR_PATH)

rot["datetime"] = pd.to_datetime(rot["Datetime"])
utr["datetime"] = pd.to_datetime(utr["Datetime"])

rot = rot.rename(columns={"PM25": "pm25"})
utr = utr.rename(columns={"pm2.5 hour": "pm25"})

rot = rot.sort_values("datetime")
utr = utr.sort_values("datetime")

# ----------------------------
# Time split
# ----------------------------
rot_tr = rot[rot.datetime < "2024-01-01"].copy()
utr_tr = utr[utr.datetime < "2024-01-01"].copy()
utr_te = utr[utr.datetime >= "2024-01-01"].copy()

# ============================================================
# METHOD 1 — Temporal Structure Similarity (SAME AS TL1)
# ============================================================
rot_tr.loc[:, "hour"] = rot_tr.datetime.dt.hour
utr_tr.loc[:, "hour"] = utr_tr.datetime.dt.hour

rot_profile = rot_tr.groupby("hour")["pm25"].mean()
utr_profile = utr_tr.groupby("hour")["pm25"].mean()

df_temporal_similarity = pd.DataFrame({
    "hour": rot_profile.index,
    "rotterdam_mean": rot_profile.values,
    "utrecht_mean": utr_profile.values,
    "difference": rot_profile.values - utr_profile.values
})

df_temporal_similarity.to_csv(TEMPORAL_SIM_PATH, index=False)

print("\n=== Temporal Structure Similarity ===")
print("Mean absolute hourly difference:",
      np.abs(df_temporal_similarity["difference"]).mean())

# ----------------------------
# Sliding windows
# ----------------------------
WINDOW = 24

def make_windows(series, window):
    X = np.lib.stride_tricks.sliding_window_view(series, window)[:-1]
    y = series[window:]
    return X, y

X_rot, y_rot = make_windows(rot_tr.pm25.values, WINDOW)
X_utr_tr, y_utr_tr = make_windows(utr_tr.pm25.values, WINDOW)
X_utr_te, y_utr_te = make_windows(utr_te.pm25.values, WINDOW)

# ----------------------------
# Normalization (IDENTICAL to TL1)
# ----------------------------
x_scaler_rot = MinMaxScaler()
y_scaler_rot = MinMaxScaler()

X_rot = x_scaler_rot.fit_transform(X_rot)
y_rot = y_scaler_rot.fit_transform(y_rot.reshape(-1, 1)).ravel()

x_scaler_utr = MinMaxScaler()
y_scaler_utr = MinMaxScaler()

X_utr_tr = x_scaler_utr.fit_transform(X_utr_tr)
X_utr_te = x_scaler_utr.transform(X_utr_te)
y_utr_tr = y_scaler_utr.fit_transform(y_utr_tr.reshape(-1, 1)).ravel()

X_rot = X_rot.reshape(-1, WINDOW, 1)
X_utr_tr = X_utr_tr.reshape(-1, WINDOW, 1)
X_utr_te = X_utr_te.reshape(-1, WINDOW, 1)

# ----------------------------
# Models
# ----------------------------
def compile_model(model, lr):
    model.compile(
        optimizer=Adam(lr),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

def build_lstm(lr):
    return compile_model(Sequential([
        LSTM(32, return_sequences=True, name="LSTM_1"),
        LSTM(32, name="LSTM_2"),
        Dense(1, name="Dense")
    ]), lr)

def build_rnn(lr):
    return compile_model(Sequential([
        SimpleRNN(32, return_sequences=True, name="RNN_1"),
        SimpleRNN(32, name="RNN_2"),
        Dense(1, name="Dense")
    ]), lr)

early_stop = EarlyStopping(monitor="val_rmse", patience=5, restore_best_weights=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================================
# METHOD 2 — Layer-wise Weight Adaptation (SAME AS TL1)
# ============================================================
def compute_weight_changes(model_before, model_after, model_name):
    rows = []
    for lb, la in zip(model_before.layers, model_after.layers):
        if not lb.get_weights():
            continue
        wb = np.concatenate([w.flatten() for w in lb.get_weights()])
        wa = np.concatenate([w.flatten() for w in la.get_weights()])
        delta = np.linalg.norm(wa - wb) / np.linalg.norm(wb)
        rows.append({
            "Model": model_name,
            "Layer": lb.name,
            "RelativeWeightChange": delta
        })
    return rows

# ----------------------------
# Grid search
# ----------------------------
LEARNING_RATES = [5e-4, 1e-3]
BATCH_SIZES = [32, 64]

def grid_search(model_name, model_builder, X, y):
    records, best_cfg, best_rmse = [], None, np.inf
    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            tf.keras.backend.clear_session()
            log(f"{model_name} grid: lr={lr}, bs={bs}")
            model = model_builder(lr)
            h = model.fit(
                X, y,
                epochs=25,
                batch_size=bs,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )
            val_rmse = min(h.history["val_rmse"])
            records.append({
                "Model": model_name,
                "LearningRate": lr,
                "BatchSize": bs,
                "ValRMSE": val_rmse
            })
            if val_rmse < best_rmse:
                best_rmse, best_cfg = val_rmse, {"lr": lr, "batch_size": bs}
    return best_cfg, records

# ----------------------------
# Experiment
# ----------------------------
def run_experiment():
    metrics, grid_records, weight_records = [], [], []

    for model_name, builder in [("LSTM", build_lstm), ("RNN", build_rnn)]:

        log(f"Grid search {model_name}")
        cfg, grid = grid_search(model_name, builder, X_utr_tr, y_utr_tr)
        grid_records.extend(grid)

        # ----- Baseline -----
        log(f"Training {model_name} baseline")
        base = builder(cfg["lr"])
        base.fit(
            X_utr_tr, y_utr_tr,
            epochs=40,
            batch_size=cfg["batch_size"],
            validation_split=0.1,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        y_base = y_scaler_utr.inverse_transform(
            base.predict(X_utr_te, verbose=0)
        ).ravel()

        # ----- Pretraining -----
        log(f"Pretraining {model_name} source")
        src = builder(cfg["lr"])
        src.fit(
            X_rot, y_rot,
            epochs=40,
            batch_size=cfg["batch_size"],
            validation_split=0.1,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        tl = clone_model(src)
        tl.set_weights(src.get_weights())
        compile_model(tl, 1e-4)

        log(f"Fine-tuning {model_name}")
        tl.fit(
            X_utr_tr, y_utr_tr,
            epochs=25,
            batch_size=cfg["batch_size"],
            validation_split=0.1,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        weight_records.extend(
            compute_weight_changes(src, tl, model_name)
        )

        y_tl = y_scaler_utr.inverse_transform(
            tl.predict(X_utr_te, verbose=0)
        ).ravel()

        for setting, y_pred in [("Baseline", y_base), ("Transfer", y_tl)]:
            metrics.append({
                "Model": model_name,
                "Setting": setting,
                "RMSE": root_mean_squared_error(y_utr_te, y_pred),
                "R2": r2_score(y_utr_te, y_pred)
            })

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(grid_records),
        pd.DataFrame(weight_records)
    )

# ----------------------------
# Run
# ----------------------------
df_metrics, df_grid, df_weights = run_experiment()

df_metrics.to_csv(METRICS_OUT_PATH, index=False)
df_grid.to_csv(GRID_OUT_PATH, index=False)
df_weights.to_csv(WEIGHT_CHANGE_PATH, index=False)

print("\n=== DONE ===")
print(df_metrics)
