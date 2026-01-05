# Run in your environment (Python >=3.8). Requires: pandas, numpy, scikit-learn, xgboost, joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import joblib
import os
import time

# ------------- config -------------
DATA_PATH = "PRSA_Aotizhongxin_cleaned.csv"   # replace with your file path
TARGET = "PM2.5"
LAGS = 3
HORIZONS = [1, 3, 6, 12, 24]
TRAIN_FRAC = 0.8
RESULT_DIR = "results_multi_horizon"
os.makedirs(RESULT_DIR, exist_ok=True)
# -----------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)  # index should be datetime

def make_supervised(df, target_col, lags=3, horizon=1):
    df_sup = df.copy()
    # create lag features for all columns (or select features)
    for lag in range(1, lags+1):
        df_sup = df_sup.assign(**{f"{c}_lag{lag}": df[c].shift(lag) for c in df.columns})
    # target shifted up by horizon (predict future)
    df_sup[f"y_h{horizon}"] = df_sup[target_col].shift(-horizon)
    df_sup = df_sup.dropna()
    X = df_sup.drop(columns=[f"y_h{horizon}"])
    y = df_sup[f"y_h{horizon}"]
    return X, y

def train_test_split_time(X, y, train_frac=0.8):
    n = len(X)
    split = int(n * train_frac)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test

models = {
    "Naive": None,  # handled separately
    "Linear": make_pipeline(StandardScaler(), LinearRegression()),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=4, verbosity=0)
}

results = []
per_hour_errors = {}  # store arrays for paired tests

for h in HORIZONS:
    print(f"\n=== Horizon: {h} hours ===")
    X, y = make_supervised(df, TARGET, lags=LAGS, horizon=h)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, TRAIN_FRAC)

    # Naive baseline: predict y_hat_t = last observed value at t-1 (which in supervised framing corresponds to PM2.5_lag1)
    naive_pred = X_test[f"{TARGET}_lag1"].values  # since lag1 is available as feature

    # older sklearn: no 'squared' param, so do RMSE manually
    naive_mse = mean_squared_error(y_test, naive_pred)
    naive_rmse = np.sqrt(naive_mse)

    print("Naive RMSE:", naive_rmse)
    results.append({"horizon": h, "model": "Naive", "rmse": naive_rmse})
    per_hour_errors[(h, "Naive")] = (y_test.values - naive_pred)  # actual - pred (error)

    for mname, m in models.items():
        if mname == "Naive":
            continue

        print(" Training:", mname)

        # measure training time for RandomForest only
        if mname == "RandomForest":
            start_time = time.time()
            m.fit(X_train, y_train)
            elapsed = time.time() - start_time
            print(f"  RandomForest training time: {elapsed:.3f} seconds")
        else:
            m.fit(X_train, y_train)

        yhat = m.predict(X_test)

        # RMSE without 'squared' arg
        mse = mean_squared_error(y_test, yhat)
        rmse = np.sqrt(mse)

        print(f"  {mname} RMSE: {rmse:.4f}")
        results.append({"horizon": h, "model": mname, "rmse": rmse})
        per_hour_errors[(h, mname)] = (y_test.values - yhat)

        # optional: save model
        joblib.dump(m, os.path.join(RESULT_DIR, f"{mname}_h{h}.joblib"))

# save summary RMSE table
pd.DataFrame(results).pivot(index="model", columns="horizon", values="rmse").to_csv(
    os.path.join(RESULT_DIR, "rmse_table.csv")
)

# save per-hour errors (for statistical tests later)
joblib.dump(per_hour_errors, os.path.join(RESULT_DIR, "per_hour_errors.joblib"))

print("Done. Results saved to:", RESULT_DIR)
