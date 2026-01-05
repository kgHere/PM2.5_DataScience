# rf_scaling_experiments_with_times.py
# Run randomized-scaling experiments (random projection + subsampling) and compare RF runtime & RMSE.
# Records prep_time (projection/resampling), train_time, total_time, and accuracy metrics (rmse, mae, r2).
import time
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from sklearn.random_projection import GaussianRandomProjection
import joblib

RESULT_DIR = "results_scaling"
os.makedirs(RESULT_DIR, exist_ok=True)
DATA_PATH = "PRSA_Data_Aotizhongxin_20130301-20170228.csv"  # change if needed

# ----- helper: build same preprocessed X,y as your earlier script -----
def build_dataset(csv_path, n_lags=3):
    df = pd.read_csv(csv_path)
    # create datetime index and cleanup columns (same as your script)
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df.set_index('datetime', inplace=True)
    drop_cols = ['No','year','month','day','hour','station']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # fill / interpolate
    if 'wd' in df.columns:
        df['wd'].fillna(method='ffill', inplace=True)
    cont_cols = [c for c in df.columns if c != 'wd']
    if len(cont_cols)>0:
        df[cont_cols] = df[cont_cols].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    # one-hot wind direction
    if 'wd' in df.columns:
        df = pd.get_dummies(df, columns=['wd'], drop_first=True)
    # cyclical features
    df['hour_sin'] = np.sin(2*np.pi*df.index.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*df.index.hour/24)
    df['month_sin'] = np.sin(2*np.pi*df.index.month/12)
    df['month_cos'] = np.cos(2*np.pi*df.index.month/12)
    # lag features
    df_model = df.copy()
    for col in df.columns:
        for lag in range(1, n_lags+1):
            df_model[f"{col}_lag{lag}"] = df[col].shift(lag)
    df_model.dropna(inplace=True)
    target_col = 'PM2.5'
    if target_col not in df_model.columns:
        raise ValueError("PM2.5 not found in data after preprocessing")
    y = df_model[target_col].copy()
    X = df_model.drop(columns=[target_col]).copy()
    return X, y

print("Loading and preparing data...")
X, y = build_dataset(DATA_PATH, n_lags=3)
n, d = X.shape
print(f"Prepared dataset: n={n}, d={d}")

# time-series split (chronological)
train_frac = 0.8
split = int(n * train_frac)
X_train_df, X_test_df = X.iloc[:split].copy(), X.iloc[split:].copy()
y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()

# scale features (fit on train)
scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X_train_df)
X_test_scaled_full  = scaler.transform(X_test_df)

# Experiment RF params
rf_params = dict(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, min_samples_leaf=5)

results = []  # list of dicts

def eval_and_record(Xtr, Xte, ytr, yte, tag, prep_time_s=0.0):
    """Train RF, measure train time and accuracy, return record dict including prep_time."""
    model = RandomForestRegressor(**rf_params)
    t0 = time.time()
    model.fit(Xtr, ytr)
    t1 = time.time()
    train_time = t1 - t0
    total_time = prep_time_s + train_time
    yhat = model.predict(Xte)
    rmse = math.sqrt(mean_squared_error(yte, yhat))
    mae = mean_absolute_error(yte, yhat)
    r2 = r2_score(yte, yhat)
    # save model for inspection
    joblib.dump(model, os.path.join(RESULT_DIR, f"rf_{tag}.joblib"))
    rec = {
        "tag": tag,
        "prep_time_s": float(prep_time_s),
        "train_time_s": float(train_time),
        "total_time_s": float(total_time),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "n_train": int(Xtr.shape[0]),
        "d": int(Xtr.shape[1])
    }
    return rec

# --- Baseline: full features, full train set ---
print("Running baseline (full features, full train set)...")
rec = eval_and_record(X_train_scaled_full, X_test_scaled_full, y_train, y_test, tag="full", prep_time_s=0.0)
results.append(rec)
print(rec)

# --- Random Projection experiments (reduce dimensionality) ---
proj_dims = [5, 15, 20, 25]  # target dims to try
for k in proj_dims:
    print(f"\nRandom Projection -> d' = {k}")
    t_p0 = time.time()
    rp = GaussianRandomProjection(n_components=k, random_state=42)
    # We project scaled features (scale then project)
    Xtr_rp = rp.fit_transform(X_train_scaled_full)
    Xte_rp = rp.transform(X_test_scaled_full)
    t_p1 = time.time()
    prep_time = t_p1 - t_p0
    rec = eval_and_record(Xtr_rp, Xte_rp, y_train, y_test, tag=f"rp_{k}", prep_time_s=prep_time)
    results.append(rec)
    print(rec)

# --- Random subsampling (reduce n) experiments ---
sub_fracs = [0.1, 0.25, 0.5]
for frac in sub_fracs:
    print(f"\nRandom Subsampling train fraction = {frac}")
    t_p0 = time.time()
    n_sub = max(10, int(X_train_scaled_full.shape[0] * frac))
    Xtr_sub, ytr_sub = resample(X_train_scaled_full, y_train.values, n_samples=n_sub, replace=False, random_state=42)
    t_p1 = time.time()
    prep_time = t_p1 - t_p0
    rec = eval_and_record(Xtr_sub, X_test_scaled_full, ytr_sub, y_test, tag=f"sub_{int(frac*100)}", prep_time_s=prep_time)
    results.append(rec)
    print(rec)

# --- Save results and make summary table/plots ---
df_res = pd.DataFrame(results).sort_values(by="tag").reset_index(drop=True)
csvpath = os.path.join(RESULT_DIR, "scaling_experiment_results_with_times.csv")
df_res.to_csv(csvpath, index=False)
print("\nSaved experiment results to:", csvpath)
print(df_res)

# Pretty print table
print("\nSummary table (ordered by total_time):")
print(df_res.sort_values("total_time_s")[[
    "tag","n_train","d","prep_time_s","train_time_s","total_time_s","rmse","mae","r2"
]].to_string(index=False))

# Plot RMSE vs method
plt.figure(figsize=(9,5))
plt.bar(df_res['tag'], df_res['rmse'], color='C0')
plt.xticks(rotation=45)
plt.ylabel('Test RMSE (µg/m³)')
plt.title('RF test RMSE: baseline vs random-projection vs subsampling')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "rmse_comparison.png"))
plt.close()

# Plot training time vs method (train time only)
plt.figure(figsize=(9,5))
plt.bar(df_res['tag'], df_res['train_time_s'], color='C1')
plt.xticks(rotation=45)
plt.ylabel('Train time (seconds)')
plt.title('RF training time (train only)')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "train_time_comparison.png"))
plt.close()

# Plot total time vs method (prep + train)
plt.figure(figsize=(9,5))
plt.bar(df_res['tag'], df_res['total_time_s'], color='C2')
plt.xticks(rotation=45)
plt.ylabel('Total time (seconds)')
plt.title('Total experiment time (prep + train)')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "total_time_comparison.png"))
plt.close()

print("Plots saved to:", RESULT_DIR)
print("All done.")
