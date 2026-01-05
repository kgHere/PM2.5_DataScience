# rf_scaling_custom_jl.py
# Scaling experiments using custom JL matrices (Gaussian & Achlioptas) + subsampling
# Records prep_time, train_time, total_time and accuracy metrics; repeats JL experiments.
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
import joblib

# ---------- CONFIG ----------
RESULT_DIR = "results_scaling_custom_jl"
os.makedirs(RESULT_DIR, exist_ok=True)
DATA_PATH = "PRSA_Data_Aotizhongxin_20130301-20170228.csv"  # change if needed

# RF hyperparams
rf_params = dict(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, min_samples_leaf=5)

# experiments
proj_dims = [5, 15, 20, 25]       # target dims for JL
jl_kind = "gauss"                 # "gauss" or "achlioptas" - controls which JL matrix to use
n_repeats = 5                     # repeats per JL setting (different seeds)
sub_fracs = [0.1, 0.25, 0.5]      # subsampling fractions
# --------------------------------

def make_gaussian_jl(d, k, random_state=None):
    """Return M with shape (k, d) where entries ~ N(0, 1/k)."""
    rng = np.random.default_rng(random_state)
    M = rng.normal(loc=0.0, scale=1.0/np.sqrt(k), size=(k, d))
    return M

def make_achlioptas_jl(d, k, random_state=None):
    """
    Achlioptas sparse JL matrix:
    entries: +sqrt(3/k) w.p. 1/6, 0 w.p. 2/3, -sqrt(3/k) w.p. 1/6
    shape: (k, d)
    """
    rng = np.random.default_rng(random_state)
    probs = [1/6, 2/3, 1/6]
    choices = [np.sqrt(3.0/k), 0.0, -np.sqrt(3.0/k)]
    M = rng.choice(choices, size=(k, d), p=probs)
    return M

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

# ---------------- main ----------------
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

results = []

# --- Baseline: full features, full train set ---
print("Running baseline (full features, full train set)...")
rec = eval_and_record(X_train_scaled_full, X_test_scaled_full, y_train, y_test, tag="full", prep_time_s=0.0)
results.append(rec)
print(rec)

# --- Custom JL experiments (repeated) ---
for k in proj_dims:
    print(f"\nCustom JL -> d' = {k}, repeats = {n_repeats}, kind = {jl_kind}")
    rmse_list = []
    train_time_list = []
    prep_time_list = []
    total_time_list = []
    for run in range(n_repeats):
        seed = 42 + run
        t_p0 = time.time()
        if jl_kind == "gauss":
            M = make_gaussian_jl(d, k, random_state=seed)
        elif jl_kind == "achlioptas":
            M = make_achlioptas_jl(d, k, random_state=seed)
        else:
            raise ValueError("jl_kind must be 'gauss' or 'achlioptas'")

        # project scaled features: X' = X @ M.T
        Xtr_rp = X_train_scaled_full.dot(M.T)
        Xte_rp = X_test_scaled_full.dot(M.T)
        t_p1 = time.time()
        prep_time = t_p1 - t_p0

        tag_run = f"jl_{jl_kind}_{k}_run{run}"
        rec_run = eval_and_record(Xtr_rp, Xte_rp, y_train, y_test, tag=tag_run, prep_time_s=prep_time)
        results.append(rec_run)

        rmse_list.append(rec_run['rmse'])
        train_time_list.append(rec_run['train_time_s'])
        prep_time_list.append(rec_run['prep_time_s'])
        total_time_list.append(rec_run['total_time_s'])

        print(f"  {tag_run}: rmse={rec_run['rmse']:.3f}, train_time={rec_run['train_time_s']:.3f}, prep_time={rec_run['prep_time_s']:.3f}")

    # aggregated summary
    rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list, ddof=1)
    train_mean, train_std = np.mean(train_time_list), np.std(train_time_list, ddof=1)
    prep_mean, prep_std = np.mean(prep_time_list), np.std(prep_time_list, ddof=1)
    total_mean, total_std = np.mean(total_time_list), np.std(total_time_list, ddof=1)

    summary_tag = f"jl_{jl_kind}_{k}_mean"
    summary_rec = {
        "tag": summary_tag,
        "prep_time_s": float(prep_mean),
        "train_time_s": float(train_mean),
        "total_time_s": float(total_mean),
        "rmse": float(rmse_mean),
        "mae": float(np.nan),
        "r2": float(np.nan),
        "n_train": int(X_train_scaled_full.shape[0]),
        "d": int(k),
        "rmse_std": float(rmse_std),
        "train_time_std": float(train_std),
        "prep_time_std": float(prep_std),
        "total_time_std": float(total_std)
    }
    results.append(summary_rec)
    print(f"  {summary_tag}: rmse_mean={rmse_mean:.3f} ± {rmse_std:.3f}, train_mean={train_mean:.3f} ± {train_std:.3f}")

# --- Random subsampling (reduce n) experiments (single-run) ---
for frac in sub_fracs:
    print(f"\nRandom Subsampling train fraction = {frac}")
    t_p0 = time.time()
    n_sub = max(10, int(X_train_scaled_full.shape[0] * frac))
    Xtr_sub, ytr_sub = resample(X_train_scaled_full, y_train.values, n_samples=n_sub, replace=False, random_state=42)
    t_p1 = time.time()
    prep_time = t_p1 - t_p0
    tag_sub = f"sub_{int(frac*100)}"
    rec_sub = eval_and_record(Xtr_sub, X_test_scaled_full, ytr_sub, y_test, tag=tag_sub, prep_time_s=prep_time)
    results.append(rec_sub)
    print(rec_sub)

# --- Save results and make summary table/plots ---
df_res = pd.DataFrame(results)
csvpath = os.path.join(RESULT_DIR, "scaling_experiment_results_custom_jl.csv")
df_res.to_csv(csvpath, index=False)
print("\nSaved experiment results to:", csvpath)

# Pretty print aggregated summary for JL means + baseline + subs
print("\nSummary (JL means and baseline/subs):")
summary_rows = df_res[df_res['tag'].str.contains("_mean") | df_res['tag'].isin(['full'] + [f"sub_{int(f*100)}" for f in sub_fracs])]
if not summary_rows.empty:
    display_df = summary_rows[[
        "tag","n_train","d","prep_time_s","train_time_s","total_time_s","rmse"
    ]].sort_values("total_time_s")
    print(display_df.to_string(index=False))
else:
    print("No summary rows found (unexpected).")

# Plot RMSE vs method (use means for JL if present)
# Build a plotting dataframe: pick summary rows for JL, and full/sub rows
plot_rows = []

# baseline:
plot_rows.append({"tag":"full", "rmse": df_res.loc[df_res['tag']=='full','rmse'].values[0], "total_time_s": df_res.loc[df_res['tag']=='full','total_time_s'].values[0]})

# JL means:
jl_means = df_res[df_res['tag'].str.contains("jl_") & df_res['tag'].str.contains("_mean")]
for _, r in jl_means.iterrows():
    plot_rows.append({"tag": r['tag'], "rmse": r['rmse'], "total_time_s": r['total_time_s']})

# subs:
for f in sub_fracs:
    tag_sub = f"sub_{int(f*100)}"
    row = df_res[df_res['tag']==tag_sub].iloc[0]
    plot_rows.append({"tag": row['tag'], "rmse": row['rmse'], "total_time_s": row['total_time_s']})

plot_df = pd.DataFrame(plot_rows)

# RMSE bar chart
plt.figure(figsize=(10,5))
plt.bar(plot_df['tag'], plot_df['rmse'], color='C0')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Test RMSE (µg/m³)')
plt.title(f'RF test RMSE: baseline vs custom-JL ({jl_kind}) vs subsampling')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"rmse_comparison_custom_jl_{jl_kind}.png"))
plt.close()

# Total time bar chart
plt.figure(figsize=(10,5))
plt.bar(plot_df['tag'], plot_df['total_time_s'], color='C2')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total time (seconds)')
plt.title(f'Total time (prep + train): custom-JL ({jl_kind}) vs subsampling')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"total_time_comparison_custom_jl_{jl_kind}.png"))
plt.close()

print("Plots saved to:", RESULT_DIR)
print("Done.")
