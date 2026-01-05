# rf_scaling_custom_jl_full.py
# Scaling experiments using custom JL matrices (Gaussian & Achlioptas) + subsampling
# Includes: prep time, train time, total time, RMSE, MAE, R2, JL distance distortion plots.

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
from sklearn.metrics import pairwise_distances
import joblib

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
RESULT_DIR = "results_scaling_custom_jl"
os.makedirs(RESULT_DIR, exist_ok=True)
DATA_PATH = "PRSA_Data_Aotizhongxin_20130301-20170228.csv"

proj_dims = [5, 15, 20, 25]
jl_kind = "gauss"          # or "achlioptas"
n_repeats = 5
sub_fracs = [0.1, 0.25, 0.5]

rf_params = dict(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=5
)

# ------------------------------------------------------------
# JL MATRIX DEFINITIONS
# ------------------------------------------------------------

def make_gaussian_jl(d, k, random_state=None):
    rng = np.random.default_rng(random_state)
    return rng.normal(0.0, 1.0/np.sqrt(k), size=(k, d))

def make_achlioptas_jl(d, k, random_state=None):
    rng = np.random.default_rng(random_state)
    values = [np.sqrt(3/k), 0.0, -np.sqrt(3/k)]
    probs  = [1/6, 2/3, 1/6]
    return rng.choice(values, size=(k, d), p=probs)

# ------------------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------------------

def build_dataset(csv_path, n_lags=3):
    df = pd.read_csv(csv_path)

    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df.set_index('datetime', inplace=True)
    df.drop(['No','year','month','day','hour','station'], axis=1, inplace=True)

    if 'wd' in df.columns:
        df['wd'].fillna(method='ffill', inplace=True)

    cont_cols = [c for c in df.columns if c != 'wd']
    df[cont_cols] = df[cont_cols].interpolate(method='time').fillna(method='bfill')

    if 'wd' in df.columns:
        df = pd.get_dummies(df, columns=['wd'], drop_first=True)

    df['hour_sin']  = np.sin(2*np.pi*df.index.hour/24)
    df['hour_cos']  = np.cos(2*np.pi*df.index.hour/24)
    df['month_sin'] = np.sin(2*np.pi*df.index.month/12)
    df['month_cos'] = np.cos(2*np.pi*df.index.month/12)

    df_model = df.copy()
    for col in df.columns:
        for lag in range(1, n_lags+1):
            df_model[f"{col}_lag{lag}"] = df[col].shift(lag)

    df_model.dropna(inplace=True)

    y = df_model['PM2.5']
    X = df_model.drop('PM2.5', axis=1)
    return X, y

# ------------------------------------------------------------
# TRAIN + METRICS
# ------------------------------------------------------------

def eval_and_record(Xtr, Xte, ytr, yte, tag, prep_time_s=0):
    model = RandomForestRegressor(**rf_params)

    t0 = time.time()
    model.fit(Xtr, ytr)
    t1 = time.time()
    train_time = t1 - t0
    total_time = prep_time_s + train_time

    yhat = model.predict(Xte)
    rmse = math.sqrt(mean_squared_error(yte, yhat))
    mae  = mean_absolute_error(yte, yhat)
    r2   = r2_score(yte, yhat)

    joblib.dump(model, os.path.join(RESULT_DIR, f"rf_{tag}.joblib"))

    return {
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

# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

print("Loading dataset…")
X, y = build_dataset(DATA_PATH, n_lags=3)
n, d = X.shape
print(f"Dataset prepared: n={n}, d={d}")

split = int(0.8*n)
X_train_df, X_test_df = X.iloc[:split], X.iloc[split:]
y_train, y_test       = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled  = scaler.transform(X_test_df)

results = []

# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------
print("\nRunning baseline…")
baseline = eval_and_record(X_train_scaled, X_test_scaled, y_train, y_test, "full")
results.append(baseline)
print(baseline)

# ------------------------------------------------------------
# JL PROJECTION EXPERIMENTS
# ------------------------------------------------------------
for k in proj_dims:
    print(f"\nJL projection: target dimension = {k} ({jl_kind}), repeats = {n_repeats}")

    rmse_vals, train_vals, prep_vals, total_vals = [], [], [], []

    for run in range(n_repeats):
        seed = 123 + run

        # build JL
        t0 = time.time()
        if jl_kind == "gauss":
            M = make_gaussian_jl(d, k, random_state=seed)
        else:
            M = make_achlioptas_jl(d, k, random_state=seed)

        # project
        Xtr_jl = X_train_scaled @ M.T
        Xte_jl = X_test_scaled  @ M.T
        t1 = time.time()

        prep_time = t1 - t0
        tag = f"jl_{jl_kind}_{k}_run{run}"

        rec = eval_and_record(Xtr_jl, Xte_jl, y_train, y_test, tag, prep_time)
        results.append(rec)

        rmse_vals.append(rec["rmse"])
        train_vals.append(rec["train_time_s"])
        prep_vals.append(rec["prep_time_s"])
        total_vals.append(rec["total_time_s"])

        print(f"  {tag}: RMSE={rec['rmse']:.3f}, train={rec['train_time_s']:.3f}s")

    # Summary row
    summary = {
        "tag": f"jl_{jl_kind}_{k}_mean",
        "prep_time_s": float(np.mean(prep_vals)),
        "train_time_s": float(np.mean(train_vals)),
        "total_time_s": float(np.mean(total_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "mae": np.nan,
        "r2": np.nan,
        "n_train": X_train_scaled.shape[0],
        "d": k,
        "rmse_std": float(np.std(rmse_vals, ddof=1)),
        "train_time_std": float(np.std(train_vals, ddof=1)),
        "total_time_std": float(np.std(total_vals, ddof=1))
    }

    results.append(summary)
    print(f"  Summary (k={k}): RMSE={summary['rmse']:.3f} ± {summary['rmse_std']:.3f}")

# ------------------------------------------------------------
# SUBSAMPLING EXPERIMENTS
# ------------------------------------------------------------
print("\nSubsampling experiments…")
for frac in sub_fracs:
    n_sub = max(20, int(len(X_train_scaled) * frac))
    print(f"  Using {frac*100:.0f}% ({n_sub} samples)")

    t0 = time.time()
    Xtr_sub, ytr_sub = resample(X_train_scaled, y_train, n_samples=n_sub, replace=False, random_state=42)
    t1 = time.time()
    prep_time = t1 - t0

    tag = f"sub_{int(frac*100)}"
    rec = eval_and_record(Xtr_sub, X_test_scaled, ytr_sub, y_test, tag, prep_time)
    results.append(rec)
    print(rec)

# ------------------------------------------------------------
# JL DISTORTION CHECK
# ------------------------------------------------------------
print("\nComputing JL distance distortion…")

subset_n = min(600, X_train_scaled.shape[0])
X_sub = X_train_scaled[:subset_n]

dist_records = []

for k in proj_dims:
    distortions = []
    for run in range(n_repeats):
        seed = 500 + run

        # Build JL
        if jl_kind == "gauss":
            M = make_gaussian_jl(d, k, random_state=seed)
        else:
            M = make_achlioptas_jl(d, k, random_state=seed)

        Xp = X_sub @ M.T

        D_orig = pairwise_distances(X_sub)
        D_proj = pairwise_distances(Xp)

        delta = np.linalg.norm(D_orig - D_proj, "fro") / np.linalg.norm(D_orig, "fro")
        distortions.append(delta)

    dist_records.append({
        "k": k,
        "mean_distortion": float(np.mean(distortions)),
        "std_distortion": float(np.std(distortions, ddof=1))
    })

dist_df = pd.DataFrame(dist_records)
dist_df.to_csv(os.path.join(RESULT_DIR, "jl_distance_distortion.csv"), index=False)
print(dist_df)

# Distortion plot
plt.figure(figsize=(8,5))
plt.errorbar(dist_df["k"], dist_df["mean_distortion"], yerr=dist_df["std_distortion"], fmt="o-", capsize=4)
plt.xlabel("Projected dimension k")
plt.ylabel("Relative distortion")
plt.title(f"JL Distance Distortion ({jl_kind})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"jl_distortion_plot_{jl_kind}.png"))
plt.close()

# ------------------------------------------------------------
# SAVE RESULTS & PLOTS
# ------------------------------------------------------------
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(RESULT_DIR,"scaling_results_custom_jl.csv"), index=False)
print("\nSaved results to scaling_results_custom_jl.csv")

# Make comparison plots (use JL means only)
plot_df = df_res[df_res['tag'].str.contains("_mean") | df_res['tag'].isin(['full'] + [f"sub_{int(f*100)}" for f in sub_fracs])]

plt.figure(figsize=(10,5))
plt.bar(plot_df['tag'], plot_df['rmse'], color="C0")
plt.xticks(rotation=45, ha="right")
plt.ylabel("RMSE")
plt.title("RMSE comparison")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"rmse_comparison.png"))
plt.close()

plt.figure(figsize=(10,5))
plt.bar(plot_df['tag'], plot_df['total_time_s'], color="C1")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total time (s)")
plt.title("Total time comparison")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"total_time_comparison.png"))
plt.close()

print("\nAll done! Results in:", RESULT_DIR)
