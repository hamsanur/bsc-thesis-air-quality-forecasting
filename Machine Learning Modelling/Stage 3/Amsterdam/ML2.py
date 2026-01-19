# Imports
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN


# Paths
BASE = "/home/hamsa/Documents/BSc-thesis/ML_Models/Stage 2/Ams"
PREP = f"{BASE}/Prepared"
PRED = f"{BASE}/Predictions"
os.makedirs(PRED, exist_ok=True)

# Data
train_df = pd.read_csv(f"{PREP}/train_dataset.csv", index_col=0)
test_df  = pd.read_csv(f"{PREP}/test_dataset.csv", index_col=0)
train_df.index = pd.to_datetime(train_df.index)
test_df.index  = pd.to_datetime(test_df.index)

# Configuration
TARGET = "pm25_concentration"
STAGE1_LAGS = {
    TARGET: 1,
    "temperature_celsius": 24,
    "humidity_percent": 2,
    "wind_direction_deg": 3,
    "pressure_hpa": 24,
    "visibility_km": 1
}
STATIC = [
    "Households_total",
    "Avg_Natural_Gas_Use_m3",
    "Land_area_ha",
    "Cars_total",
    "Population_total"
]

# Features
final_features = [f"{v}_lag{l}" for v, l in STAGE1_LAGS.items()] + STATIC
X_train = train_df[final_features].values
X_test  = test_df[final_features].values
y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Metrics
def compute_metrics(y, p):
    return (
        root_mean_squared_error(y, p),
        mean_absolute_error(y, p),
        r2_score(y, p)
    )

# Classical models
models = {
    "Linear": LinearRegression(),
    "SVR": SVR(C=100, gamma=0.01),
    "RF": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGB": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    ),
}

results = []
pfi_dict = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(
        f"{PRED}/{name}_predictions.csv", index=False
    )

    rmse, mae, r2 = compute_metrics(y_test, preds)
    results.append([name, rmse, mae, r2])

    imp = permutation_importance(
        model, X_test_s, y_test, n_repeats=5, random_state=42
    )
    pfi_dict[name] = imp.importances_mean

# Sequence data
Xtr_seq = X_train_s.reshape(len(X_train_s), 1, X_train_s.shape[1])
Xte_seq = X_test_s.reshape(len(X_test_s), 1, X_test_s.shape[1])

# RNN
rnn = Sequential([
    SimpleRNN(32, input_shape=(1, X_train_s.shape[1])),
    Dense(1)
])
rnn.compile(optimizer="adam", loss="mse")
rnn.fit(Xtr_seq, y_train, epochs=15, batch_size=32, verbose=0)

pred_rnn = rnn.predict(Xte_seq).flatten()
results.append(["RNN", *compute_metrics(y_test, pred_rnn)])

baseline = root_mean_squared_error(y_test, pred_rnn)
pfi_dict["RNN"] = [
    root_mean_squared_error(
        y_test,
        rnn.predict(
            np.random.permutation(X_test_s)
            .reshape(len(X_test_s), 1, X_test_s.shape[1])
        ).flatten()
    ) - baseline
    for _ in range(len(final_features))
]

# LSTM
lstm = Sequential([
    LSTM(32, input_shape=(1, X_train_s.shape[1])),
    Dense(1)
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(Xtr_seq, y_train, epochs=15, batch_size=32, verbose=0)

pred_lstm = lstm.predict(Xte_seq).flatten()
results.append(["LSTM", *compute_metrics(y_test, pred_lstm)])

baseline = root_mean_squared_error(y_test, pred_lstm)
pfi_dict["LSTM"] = [
    root_mean_squared_error(
        y_test,
        lstm.predict(
            np.random.permutation(X_test_s)
            .reshape(len(X_test_s), 1, X_test_s.shape[1])
        ).flatten()
    ) - baseline
    for _ in range(len(final_features))
]

# Save
pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"]).to_csv(
    f"{PRED}/model_performance.csv", index=False
)
pd.DataFrame(pfi_dict, index=final_features).to_csv(
    f"{PRED}/pfi_results.csv"
)

