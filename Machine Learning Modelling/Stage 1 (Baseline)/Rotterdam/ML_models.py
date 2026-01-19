# imports
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam


# paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_PATH, "merged_rotterdam_cleaned_iqr.csv")
TRAIN_FILE = os.path.join(BASE_PATH, "airq_rotterdam_train.csv")
TEST_FILE = os.path.join(BASE_PATH, "airq_rotterdam_test.csv")
LAG_RESULTS_FILE = os.path.join(BASE_PATH, "lag_results_rotterdam.csv")
METRICS_FILE = os.path.join(BASE_PATH, "metrics_pm25_rotterdam.csv")

PRED_DIR = os.path.join(BASE_PATH, "Model_Predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# metrics
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true, y_pred):
    return (
        mean_squared_error(y_true, y_pred, squared=False),
        r2_score(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        mape(y_true, y_pred),
    )

# load and prepare data
df = pd.read_csv(INPUT_FILE)

df["Datetime"] = pd.to_datetime(df["Start"])
df = df.sort_values("Datetime").dropna(subset=["Value"])
df = df.groupby("Datetime", as_index=False)["Value"].mean()
df["Year"] = df["Datetime"].dt.year

train_df = df[df["Year"] < 2024].copy()
test_df = df[df["Year"] == 2024].copy()

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# lag feature
def add_lag(data, lag):
    data = data.copy()
    data[f"Value_lag{lag}"] = data["Value"].shift(lag)
    return data

# lag selection
lag_results = []

for lag in range(1, 25):
    tr = add_lag(train_df, lag).dropna()
    te = add_lag(test_df, lag).dropna()

    X_tr = tr[[f"Value_lag{lag}"]].values
    y_tr = tr["Value"].values
    X_te = te[[f"Value_lag{lag}"]].values
    y_te = te["Value"].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    rmse, r2, _, _ = evaluate(y_te, preds)
    lag_results.append({"Lag": lag, "RMSE": rmse, "R²": r2})

lag_df = pd.DataFrame(lag_results)
lag_df.to_csv(LAG_RESULTS_FILE, index=False)

best_lag = lag_df.loc[lag_df["RMSE"].idxmin(), "Lag"]

# final dataset
train_df = add_lag(train_df, best_lag).dropna()
test_df = add_lag(test_df, best_lag).dropna()

X_train = train_df[[f"Value_lag{best_lag}"]].values
y_train = train_df["Value"].values
X_test = test_df[[f"Value_lag{best_lag}"]].values
y_test = test_df["Value"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# baseline models
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(C=10, gamma="scale"),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    results.append(
        dict(zip(
            ["Model", "RMSE", "R²", "MAE", "MAPE (%)"],
            [name, *evaluate(y_test, preds)],
        ))
    )

    pd.DataFrame({
        "Datetime": test_df["Datetime"],
        "Actual_PM2.5": y_test,
        "Predicted_PM2.5": preds,
    }).to_csv(
        os.path.join(PRED_DIR, f"{name.replace(' ', '_')}_preds.csv"),
        index=False
    )

# rnn
X_train_rnn = X_train_s.reshape(-1, 1, 1)
X_test_rnn = X_test_s.reshape(-1, 1, 1)

rnn = Sequential([
    SimpleRNN(64, activation="tanh", input_shape=(1, 1)),
    Dense(1),
])
rnn.compile(optimizer=Adam(0.001), loss="mse")
rnn.fit(X_train_rnn, y_train, epochs=25, batch_size=16, verbose=0)

preds_rnn = rnn.predict(X_test_rnn).ravel()
results.append(
    dict(zip(
        ["Model", "RMSE", "R²", "MAE", "MAPE (%)"],
        ["RNN", *evaluate(y_test, preds_rnn)],
    ))
)

# lstm
lstm = Sequential([
    LSTM(64, activation="tanh", input_shape=(1, 1)),
    Dense(1),
])
lstm.compile(optimizer=Adam(0.001), loss="mse")
lstm.fit(X_train_rnn, y_train, epochs=25, batch_size=16, verbose=0)

preds_lstm = lstm.predict(X_test_rnn).ravel()
results.append(
    dict(zip(
        ["Model", "RMSE", "R²", "MAE", "MAPE (%)"],
        ["LSTM", *evaluate(y_test, preds_lstm)],
    ))
)

# save results
results_df = pd.DataFrame(results)
results_df.to_csv(METRICS_FILE, index=False)

print(results_df)
