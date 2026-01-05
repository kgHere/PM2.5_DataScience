# rf_with_time_metrics.py
# Train RandomForest per horizon, log total RF training time, show progress while building trees.
# Compatible with older scikit-learn (avoids `squared=` kwarg).
# Computes RMSE, MAE, R2, MAPE and saves results.

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------- CONFIG -------------
DATA_PATH = "PRSA_Aotizhongxin_cleaned.csv"   # update path if needed
TARGET = "PM2.5"
LAGS = 3
HORIZONS = [1, 3, 6, 12, 24]
TRAIN_FRAC = 0.8
RESULT_DIR = "results_multi_horizon_rf_metrics"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
RF_N_ESTIMATORS = 200       # final number of trees
RF_CHUNK = 10               # add trees in chunks of this size to show progress
RF_N_JOBS = -1              # use all cores for tree-building
# -----------------------------------

def make_supervised(df, target_col, lags=3, horizon=1):
    df_sup = df.copy()
    for lag in range(1, lags+1):
        df_sup = df_sup.assign(**{f"{c}_lag{lag}": df[c].shift(lag) for c in df.columns})
    df_sup[f"y_h{horizon}"] = df_sup[target_col].shift(-horizon)
    df_sup = df_sup.dropna()
    X = df_sup.drop(columns=[f"y_h{horizon}"])
    y = df_sup[f"y_h{horizon}"]
    return X, y

def train_test_split_time(X, y, train_frac=0.8):
    n = len(X)
    split = int(n * train_frac)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def ensure_numeric(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    # coerce object columns to numeric, map boolean-like strings, fillna
    obj_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
        X_test[c] = pd.to_numeric(X_test[c], errors='coerce')
        if X_train[c].dtype == 'object' or X_test[c].dtype == 'object':
            X_train[c] = X_train[c].map({'True':1, 'False':0, True:1, False:0}).astype('float')
            X_test[c]  = X_test[c].map({'True':1, 'False':0, True:1, False:0}).astype('float')
    X_train = X_train.fillna(0).astype(float)
    X_test  = X_test.fillna(0).astype(float)
    return X_train, X_test

def safe_mape(y_true, y_pred):
    # Avoid divide-by-zero: ignore zero true values in denominator
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

# load df
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)
print("Data loaded. Shape:", df.shape)

metrics_rows = []   # to save metrics per (horizon, model)
per_hour_errors = {}

for h in HORIZONS:
    print("\n" + "="*60)
    print(f"HORIZON = {h} hour(s)")
    X, y = make_supervised(df, TARGET, lags=LAGS, horizon=h)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, TRAIN_FRAC)
    X_train_num, X_test_num = ensure_numeric(X_train, X_test)

    # Naive baseline (use last observed PM2.5: lag1)
    naive_pred = X_test_num[f"{TARGET}_lag1"].values
    naive_mse = mean_squared_error(y_test.values, naive_pred)              # MSE
    naive_rmse = np.sqrt(naive_mse)
    naive_mae = mean_absolute_error(y_test.values, naive_pred)
    naive_r2 = r2_score(y_test.values, naive_pred)
    naive_mape = safe_mape(y_test.values, naive_pred)
    print(f"Naive - RMSE: {naive_rmse:.4f}, MAE: {naive_mae:.4f}, R2: {naive_r2:.4f}, MAPE: {np.round(naive_mape,3)}%")

    metrics_rows.append({
        "horizon": h, "model": "Naive",
        "rmse": float(naive_rmse), "mae": float(naive_mae),
        "r2": float(naive_r2), "mape_pct": float(naive_mape)
    })
    per_hour_errors[(h, "Naive")] = (y_test.values - naive_pred)

    # ------------------ Random Forest with progress ------------------
    print("Training RandomForest with progress reporting...")
    rf = RandomForestRegressor(
        n_estimators=RF_CHUNK,
        warm_start=True,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS
    )

    start_time = time.time()
    current_trees = 0
    while current_trees < RF_N_ESTIMATORS:
        next_trees = min(current_trees + RF_CHUNK, RF_N_ESTIMATORS)
        rf.n_estimators = next_trees
        # fit will add new trees because warm_start=True
        rf.fit(X_train_num, y_train)
        current_trees = next_trees
        pct = current_trees / RF_N_ESTIMATORS * 100.0
        print(f"  Progress: built {current_trees}/{RF_N_ESTIMATORS} trees ({pct:.0f}%)")

    end_time = time.time()
    total_time_sec = end_time - start_time
    print(f"Total RandomForest training time (H{h}): {total_time_sec:.2f} seconds")

    # evaluate on test set
    rf_pred = rf.predict(X_test_num)
    rf_mse = mean_squared_error(y_test.values, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test.values, rf_pred)
    rf_r2  = r2_score(y_test.values, rf_pred)
    rf_mape = safe_mape(y_test.values, rf_pred)

    print(f"RandomForest - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R2: {rf_r2:.4f}, MAPE: {np.round(rf_mape,3)}%")

    # save model and results
    model_path = os.path.join(RESULT_DIR, f"RandomForest_h{h}.joblib")
    joblib.dump(rf, model_path)
    print("Saved RandomForest model to:", model_path)

    metrics_rows.append({
        "horizon": h, "model": "RandomForest",
        "rmse": float(rf_rmse), "mae": float(rf_mae),
        "r2": float(rf_r2), "mape_pct": float(rf_mape),
        "train_time_sec": float(total_time_sec)
    })
    per_hour_errors[(h, "RandomForest")] = (y_test.values - rf_pred)

# Save metrics table
metrics_df = pd.DataFrame(metrics_rows)
# pivot for RMSE table (models as rows, horizons as columns)
rmse_pivot = metrics_df.pivot_table(index="model", columns="horizon", values="rmse", aggfunc='first')
rmse_csv_path = os.path.join(RESULT_DIR, "rmse_table_rf.csv")
rmse_pivot.to_csv(rmse_csv_path)

# save full metrics (flat)
metrics_full_csv = os.path.join(RESULT_DIR, "metrics_full_rf.csv")
metrics_df.to_csv(metrics_full_csv, index=False)

# Save per-hour errors
joblib.dump(per_hour_errors, os.path.join(RESULT_DIR, "per_hour_errors_rf.joblib"))

print("\nAll done. Results saved to:", RESULT_DIR)
print("RMSE pivot:", rmse_csv_path)
print("Full metrics CSV:", metrics_full_csv)
