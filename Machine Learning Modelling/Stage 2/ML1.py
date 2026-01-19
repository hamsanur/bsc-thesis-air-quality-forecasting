# imports
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# metrics
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def evaluate(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred, squared=False),
        r2_score(y_true, y_pred),
        mape(y_true, y_pred),
        smape(y_true, y_pred),
    )

# paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(BASE_PATH, "train_rotterdam.csv")
TEST_FILE  = os.path.join(BASE_PATH, "test_rotterdam.csv")

LAG_RESULTS_FILE = os.path.join(BASE_PATH, "lag_results_weather_rotterdam.csv")
METRICS_FILE = os.path.join(BASE_PATH, "metrics_weather_rotterdam.csv")
PFI_FILE = os.path.join(BASE_PATH, "permutation_importance_weather_rotterdam.csv")

PRED_DIR = os.path.join(BASE_PATH, "Predictions_Weather_Rotterdam")
os.makedirs(PRED_DIR, exist_ok=True)

# load data
train_df = pd.read_csv(TRAIN_FILE, parse_dates=["Datetime"])
test_df = pd.read_csv(TEST_FILE, parse_dates=["Datetime"])

weather_cols = ["Temp", "Humidity", "WindDir", "Pressure", "Visibility"]
lags = range(1, 25)

# lag helper
def add_lag(df, col, lag):
    return df[col].shift(lag)

# pm25 lag search
pm_results = []

for lag in lags:
    tr, te = train_df.copy(), test_df.copy()

    tr[f"PM25_lag{lag}"] = add_lag(tr, "PM25", lag)
    te[f"PM25_lag{lag}"] = add_lag(te, "PM25", lag)

    tr, te = tr.dropna(), te.dropna()

    Xtr, Xte = tr[[f"PM25_lag{lag}"]], te[[f"PM25_lag{lag}"]]
    ytr, yte = tr["PM25"], te["PM25"]

    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

    model = LinearRegression()
    model.fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)

    pm_results.append({
        "Variable": "PM25",
        "Lag": lag,
        "RMSE": mean_squared_error(yte, pred, squared=False),
        "R2": r2_score(yte, pred),
    })

pm_df = pd.DataFrame(pm_results)
best_pm_lag = int(pm_df.loc[pm_df["RMSE"].idxmin(), "Lag"])

# weather lag search
weather_best_lags = {}
weather_results = []

for col in weather_cols:
    best_rmse, best_lag = np.inf, 1

    for lag in lags:
        tr, te = train_df.copy(), test_df.copy()

        tr[f"{col}_lag{lag}"] = add_lag(tr, col, lag)
        te[f"{col}_lag{lag}"] = add_lag(te, col, lag)

        tr, te = tr.dropna(), te.dropna()

        Xtr, Xte = tr[[f"{col}_lag{lag}"]], te[[f"{col}_lag{lag}"]]
        ytr, yte = tr["PM25"], te["PM25"]

        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

        model = LinearRegression()
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)

        rmse = mean_squared_error(yte, pred, squared=False)

        weather_results.append({
            "Variable": col,
            "Lag": lag,
            "RMSE": rmse,
            "R2": r2_score(yte, pred),
        })

        if rmse < best_rmse:
            best_rmse, best_lag = rmse, lag

    weather_best_lags[col] = best_lag

lag_df = pd.concat([pm_df, pd.DataFrame(weather_results)], ignore_index=True)
lag_df.to_csv(LAG_RESULTS_FILE, index=False)

# final dataset
train_f, test_f = train_df.copy(), test_df.copy()

train_f[f"PM25_lag{best_pm_lag}"] = add_lag(train_f, "PM25", best_pm_lag)
test_f[f"PM25_lag{best_pm_lag}"] = add_lag(test_f, "PM25", best_pm_lag)

for col, lag in weather_best_lags.items():
    train_f[f"{col}_lag{lag}"] = add_lag(train_f, col, lag)
    test_f[f"{col}_lag{lag}"] = add_lag(test_f, col, lag)

train_f, test_f = train_f.dropna(), test_f.dropna()

features = [f"PM25_lag{best_pm_lag}"] + [
    f"{c}_lag{l}" for c, l in weather_best_lags.items()
]

X_train, X_test = train_f[features], test_f[features]
y_train, y_test = train_f["PM25"], test_f["PM25"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# classical models
models = {
    "Linear": LinearRegression(),
    "SVR": SVR(C=10, gamma="scale"),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    ),
}

results = {}
pfi = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    results[name] = evaluate(y_test, pred)

    pd.DataFrame({
        "Datetime": test_f["Datetime"],
        "Actual_PM25": y_test.values,
        "Predicted_PM25": pred,
    }).to_csv(os.path.join(PRED_DIR, f"{name}.csv"), index=False)

    pfi[name] = permutation_importance(
        model, X_test_s, y_test, n_repeats=3
    ).importances_mean

# rnn and lstm
Xtr_nn = X_train_s.reshape(-1, 1, X_train_s.shape[1])
Xte_nn = X_test_s.reshape(-1, 1, X_test_s.shape[1])

for name, layer in [("RNN", SimpleRNN), ("LSTM", LSTM)]:
    nn = Sequential([
        layer(64, activation="tanh", input_shape=(1, X_train_s.shape[1])),
        Dense(1),
    ])
    nn.compile(optimizer=Adam(0.001), loss="mse")
    nn.fit(Xtr_nn, y_train, epochs=25, batch_size=16, verbose=0)

    pred = nn.predict(Xte_nn).ravel()
    results[name] = evaluate(y_test, pred)

    baseline = mean_squared_error(y_test, pred)
    pfi[name] = []

    for i in range(len(features)):
        Xp = X_test_s.copy()
        np.random.shuffle(Xp[:, i])
        pred_p = nn.predict(Xp.reshape(-1, 1, Xp.shape[1])).ravel()
        pfi[name].append(mean_squared_error(y_test, pred_p) - baseline)

# save results
results_df = pd.DataFrame(
    results,
    index=["MAE", "RMSE", "R²", "MAPE", "SMAPE"]
).T
results_df.to_csv(METRICS_FILE)

pd.DataFrame(pfi, index=features).to_csv(PFI_FILE)

print(results_df)
print("Permutation Feature Importance saved.")
