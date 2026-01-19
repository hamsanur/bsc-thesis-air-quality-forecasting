# Imports
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configurations
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = "/home/hamsa/Documents/BSc-thesis/Transfer Learning/Transfer/Transfer 4"

GRO_PATH = os.path.join(BASE_DIR, "groningen_TL.csv")
EIN_PATH = os.path.join(BASE_DIR, "eindhoven_TL.csv")
UTR_PATH = os.path.join(BASE_DIR, "utrecht_TL.csv")

METRICS_OUT_PATH = os.path.join(BASE_DIR, "TL4_metrics.csv")
GRID_OUT_PATH = os.path.join(BASE_DIR, "TL4_grid_search.csv")

JUST_DIR = os.path.join(BASE_DIR, "TL_justification")
os.makedirs(JUST_DIR, exist_ok=True)
TEMPORAL_SIM_PATH = os.path.join(JUST_DIR, "temporal_similarity.csv")
WEIGHT_CHANGE_PATH = os.path.join(JUST_DIR, "weight_change_per_layer.csv")

# Load data
gro = pd.read_csv(GRO_PATH)
ein = pd.read_csv(EIN_PATH)
utr = pd.read_csv(UTR_PATH)

gro["datetime"] = pd.to_datetime(gro["Start"])
ein["datetime"] = pd.to_datetime(ein["Start"])
utr["datetime"] = pd.to_datetime(utr["Start"])

gro = gro.rename(columns={"pm25_value": "pm25"})
ein = ein.rename(columns={"Value": "pm25"})
utr = utr.rename(columns={"pm2.5 hour": "pm25"})

gro = gro.sort_values("datetime")
ein = ein.sort_values("datetime")
utr = utr.sort_values("datetime")

# Time split
gro_tr = gro[gro.datetime < "2024-01-01"].copy()
ein_tr = ein[ein.datetime < "2024-01-01"].copy()
utr_tr = utr[utr.datetime < "2024-01-01"].copy()
utr_te = utr[utr.datetime >= "2024-01-01"].copy()

# Temporal profiles
for df in [gro_tr, ein_tr, utr_tr]:
    df["hour"] = df.datetime.dt.hour

hours = np.arange(24)

gro_profile = gro_tr.groupby("hour")["pm25"].mean().reindex(hours)
ein_profile = ein_tr.groupby("hour")["pm25"].mean().reindex(hours)
utr_profile = utr_tr.groupby("hour")["pm25"].mean().reindex(hours)

df_temporal = pd.DataFrame({
    "hour": hours,
    "groningen_mean": gro_profile.values,
    "eindhoven_mean": ein_profile.values,
    "utrecht_mean": utr_profile.values
})
df_temporal.to_csv(TEMPORAL_SIM_PATH, index=False)

# Sliding windows
WINDOW = 24

def make_windows(series, window):
    X = np.lib.stride_tricks.sliding_window_view(series, window)[:-1]
    y = series[window:]
    return X, y

X_gro, y_gro = make_windows(gro_tr.pm25.values, WINDOW)
X_ein, y_ein = make_windows(ein_tr.pm25.values, WINDOW)
X_utr_tr_raw, y_utr_tr_raw = make_windows(utr_tr.pm25.values, WINDOW)
X_utr_te_raw, y_utr_te_raw = make_windows(utr_te.pm25.values, WINDOW)

# Scaling
x_scaler_joint = MinMaxScaler()
y_scaler_joint = MinMaxScaler()
x_scaler_utr = MinMaxScaler()
y_scaler_utr = MinMaxScaler()

X_joint_raw = np.concatenate([X_gro, X_ein])
y_joint_raw = np.concatenate([y_gro, y_ein])

X_joint = x_scaler_joint.fit_transform(X_joint_raw)
y_joint = y_scaler_joint.fit_transform(y_joint_raw.reshape(-1, 1)).ravel()

X_utr_tr = x_scaler_utr.fit_transform(X_utr_tr_raw)
X_utr_te = x_scaler_utr.transform(X_utr_te_raw)
y_utr_tr = y_scaler_utr.fit_transform(y_utr_tr_raw.reshape(-1, 1)).ravel()

X_joint = X_joint.reshape(-1, WINDOW, 1)
X_utr_tr = X_utr_tr.reshape(-1, WINDOW, 1)
X_utr_te = X_utr_te.reshape(-1, WINDOW, 1)

# Models
def compile_model(model, lr):
    model.compile(
        optimizer=Adam(lr),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

def build_lstm(lr):
    return compile_model(
        Sequential([
            LSTM(32, return_sequences=True),
            LSTM(32),
            Dense(1)
        ]),
        lr
    )

def build_rnn(lr):
    return compile_model(
        Sequential([
            SimpleRNN(32, return_sequences=True),
            SimpleRNN(32),
            Dense(1)
        ]),
        lr
    )

early_stop = EarlyStopping(
    monitor="val_rmse", patience=5, restore_best_weights=True
)

# Weight change
def compute_weight_changes(model_before, model_after, model_name):
    rows = []
    for lb, la in zip(model_before.layers, model_after.layers):
        if not lb.get_weights():
            continue
        wb = np.concatenate([w.flatten() for w in lb.get_weights()])
        wa = np.concatenate([w.flatten() for w in la.get_weights()])
        rows.append({
            "Model": model_name,
            "Layer": lb.name,
            "RelativeWeightChange": np.linalg.norm(wa - wb) / np.linalg.norm(wb)
        })
    return rows

# Grid search
LEARNING_RATES = [5e-4, 1e-3]
BATCH_SIZES = [32, 64]

def grid_search(builder, X, y):
    best_cfg, best_rmse = None, np.inf
    records = []
    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            tf.keras.backend.clear_session()
            model = builder(lr)
            h = model.fit(
                X, y,
                epochs=25,
                batch_size=bs,
                shuffle=False,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0
            )
            rmse = min(h.history["val_rmse"])
            records.append({"LR": lr, "BatchSize": bs, "ValRMSE": rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = {"lr": lr, "batch_size": bs}
    return best_cfg, records

# Experiment
metrics, grids, weights = [], [], []

for name, builder in [("LSTM", build_lstm), ("RNN", build_rnn)]:
    cfg, grid = grid_search(builder, X_utr_tr, y_utr_tr)
    grids.extend([{**g, "Model": name} for g in grid])

    base = builder(cfg["lr"])
    base.fit(
        X_utr_tr,
        y_utr_tr,
        epochs=40,
        batch_size=cfg["batch_size"],
        shuffle=False,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    y_base = y_scaler_utr.inverse_transform(
        base.predict(X_utr_te, verbose=0)
    ).ravel()

    src = builder(cfg["lr"])
    src.fit(
        X_joint,
        y_joint,
        epochs=40,
        batch_size=cfg["batch_size"],
        shuffle=False,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    tl = clone_model(src)
    tl.set_weights(src.get_weights())

    for layer in tl.layers[:-1]:
        layer.trainable = False
    compile_model(tl, 1e-3)
    tl.fit(X_utr_tr, y_utr_tr, epochs=5,
           batch_size=cfg["batch_size"], verbose=0)

    for layer in tl.layers:
        layer.trainable = True
    compile_model(tl, 1e-4)
    tl.fit(
        X_utr_tr,
        y_utr_tr,
        epochs=10,
        batch_size=cfg["batch_size"],
        shuffle=False,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    weights.extend(compute_weight_changes(src, tl, name))

    y_tl = y_scaler_utr.inverse_transform(
        tl.predict(X_utr_te, verbose=0)
    ).ravel()

    for label, y_pred in [("Baseline", y_base), ("Joint TL", y_tl)]:
        metrics.append({
            "Model": name,
            "Setting": label,
            "RMSE": root_mean_squared_error(y_utr_te_raw, y_pred),
            "R2": r2_score(y_utr_te_raw, y_pred)
        })

# Save results
pd.DataFrame(metrics).to_csv(METRICS_OUT_PATH, index=False)
pd.DataFrame(grids).to_csv(GRID_OUT_PATH, index=False)
pd.DataFrame(weights).to_csv(WEIGHT_CHANGE_PATH, index=False)

print(pd.DataFrame(metrics))
