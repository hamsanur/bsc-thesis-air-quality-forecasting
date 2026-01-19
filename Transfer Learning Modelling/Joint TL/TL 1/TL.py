# Imports
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

# Configurations
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = "/home/hamsa/Documents/BSc-thesis/Transfer Learning/Transfer/TL1"
AMS_PATH = os.path.join(BASE_DIR, "amsterdam_TL.csv")
GRO_PATH = os.path.join(BASE_DIR, "groningen_TL.csv")
METRICS_OUT_PATH = os.path.join(BASE_DIR, "TL1_AMS_GRO_performance_metrics.csv")
GRID_OUT_PATH = os.path.join(BASE_DIR, "TL1_AMS_GRO_grid_search_results.csv")

JUST_DIR = os.path.join(BASE_DIR, "TL_justification")
os.makedirs(JUST_DIR, exist_ok=True)
TEMPORAL_SIM_PATH = os.path.join(JUST_DIR, "temporal_similarity.csv")
WEIGHT_CHANGE_PATH = os.path.join(JUST_DIR, "weight_change_per_layer.csv")

# Load data
ams = pd.read_csv(AMS_PATH)
gro = pd.read_csv(GRO_PATH)

ams["datetime"] = pd.to_datetime(ams["Datetime"])
gro["datetime"] = pd.to_datetime(gro["Start"])

ams = ams.rename(columns={"pm25_concentration": "pm25"})
gro = gro.rename(columns={"pm25_value": "pm25"})

ams = ams.sort_values("datetime")
gro = gro.sort_values("datetime")

# Time split
ams_tr = ams[ams.datetime < "2024-01-01"].copy()
gro_tr = gro[gro.datetime < "2024-01-01"].copy()
gro_te = gro[gro.datetime >= "2024-01-01"].copy()

# Temporal similarity
ams_tr["hour"] = ams_tr.datetime.dt.hour
gro_tr["hour"] = gro_tr.datetime.dt.hour

ams_profile = ams_tr.groupby("hour")["pm25"].mean()
gro_profile = gro_tr.groupby("hour")["pm25"].mean()

df_temporal_similarity = pd.DataFrame({
    "hour": ams_profile.index,
    "amsterdam_mean": ams_profile.values,
    "groningen_mean": gro_profile.values,
    "difference": ams_profile.values - gro_profile.values
})
df_temporal_similarity.to_csv(TEMPORAL_SIM_PATH, index=False)

print("Mean absolute hourly difference:",
      np.abs(df_temporal_similarity["difference"]).mean())

# Sliding windows
WINDOW = 24

def make_windows(series, window):
    X = np.lib.stride_tricks.sliding_window_view(series, window)[:-1]
    y = series[window:]
    return X, y

X_ams, y_ams = make_windows(ams_tr.pm25.values, WINDOW)
X_gro_tr, y_gro_tr = make_windows(gro_tr.pm25.values, WINDOW)
X_gro_te, y_gro_te = make_windows(gro_te.pm25.values, WINDOW)

# Scaling
x_scaler_ams = MinMaxScaler()
y_scaler_ams = MinMaxScaler()
X_ams = x_scaler_ams.fit_transform(X_ams)
y_ams = y_scaler_ams.fit_transform(y_ams.reshape(-1, 1)).ravel()

x_scaler_gro = MinMaxScaler()
y_scaler_gro = MinMaxScaler()
X_gro_tr = x_scaler_gro.fit_transform(X_gro_tr)
X_gro_te = x_scaler_gro.transform(X_gro_te)
y_gro_tr = y_scaler_gro.fit_transform(y_gro_tr.reshape(-1, 1)).ravel()

X_ams = X_ams.reshape(-1, WINDOW, 1)
X_gro_tr = X_gro_tr.reshape(-1, WINDOW, 1)
X_gro_te = X_gro_te.reshape(-1, WINDOW, 1)

# Models
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

early_stop = EarlyStopping(
    monitor="val_rmse", patience=5, restore_best_weights=True
)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Weight change analysis
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

# Grid search
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

# Experiment
def run_experiment():
    metrics, grid_records, weight_records = [], [], []

    for model_name, builder in [("LSTM", build_lstm), ("RNN", build_rnn)]:
        log(f"Grid search {model_name}")
        cfg, grid = grid_search(model_name, builder, X_gro_tr, y_gro_tr)
        grid_records.extend(grid)

        log(f"Training {model_name} baseline")
        base = builder(cfg["lr"])
        base.fit(
            X_gro_tr, y_gro_tr,
            epochs=40,
            batch_size=cfg["batch_size"],
            validation_split=0.1,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        y_base = y_scaler_gro.inverse_transform(
            base.predict(X_gro_te, verbose=0)
        ).ravel()

        log(f"Pretraining {model_name} source")
        src = builder(cfg["lr"])
        src.fit(
            X_ams, y_ams,
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
            X_gro_tr, y_gro_tr,
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

        y_tl = y_scaler_gro.inverse_transform(
            tl.predict(X_gro_te, verbose=0)
        ).ravel()

        for setting, y_pred in [("Baseline", y_base), ("Transfer", y_tl)]:
            metrics.append({
                "Model": model_name,
                "Setting": setting,
                "RMSE": root_mean_squared_error(y_gro_te, y_pred),
                "R2": r2_score(y_gro_te, y_pred)
            })

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(grid_records),
        pd.DataFrame(weight_records)
    )

# Run
df_metrics, df_grid, df_weights = run_experiment()
df_metrics.to_csv(METRICS_OUT_PATH, index=False)
df_grid.to_csv(GRID_OUT_PATH, index=False)
df_weights.to_csv(WEIGHT_CHANGE_PATH, index=False)

print(df_metrics)
