# PM2.5 Forecasting with Random Forest
# Streamlined version focusing only on Random Forest model

import time
start_time = time.time()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestRegressor

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 5)

print("--- 1. Data Loading ---")
try:
    df = pd.read_csv('PRSA_Data_Aotizhongxin_20130301-20170228.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# Create datetime index
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.set_index('datetime', inplace=True)
df.drop(['No', 'year', 'month', 'day', 'hour', 'station'], axis=1, inplace=True)

print("\n--- 2. Preprocessing ---")
# Handle missing data
df['wd'].fillna(method='ffill', inplace=True)
continuous_cols = df.columns.drop('wd')
df[continuous_cols] = df[continuous_cols].interpolate(method='time')
df.fillna(method='bfill', inplace=True)

# Encode categorical data
df = pd.get_dummies(df, columns=['wd'], drop_first=True)

# Engineer cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

# Create sliding window
n_lags = 3
target_col = 'PM2.5'
df_model = df.copy()

print(f"Creating sliding window with {n_lags} lags...")
for col in df.columns:
    for lag in range(1, n_lags + 1):
        df_model[f'{col}_lag{lag}'] = df[col].shift(lag)

df_model.dropna(inplace=True)

# Data splitting
y = df_model[target_col]
X = df_model.drop(target_col, axis=1)

split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- 3. Train Random Forest Model ---")
model_start = time.time()

model_rf = RandomForestRegressor(n_estimators=100,
                                 max_depth=15,
                                 random_state=42,
                                 n_jobs=-1,
                                 min_samples_leaf=5)

model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)

rf_runtime = time.time() - model_start
print(f"Random Forest training time: {rf_runtime:.2f} seconds")

print("\n--- 4. Evaluate Model ---")
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae = mean_absolute_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

non_zero_mask = y_test != 0
mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred_rf[non_zero_mask]) / y_test[non_zero_mask])) * 100

print(f"\nRandom Forest Performance:")
print(f"RMSE: {rmse:.4f} µg/m³")
print(f"MAE: {mae:.4f} µg/m³")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Visualize predictions
print("\nPlotting forecast...")
plt.figure(figsize=(15, 6))
plt.plot(y_test.values[:168], label='Actual PM2.5', alpha=0.8)
plt.plot(y_pred_rf[:168], label='RF Prediction', linestyle='--')
plt.title('Random Forest: PM2.5 Predictions vs Actual (First Week)')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.savefig('rf_forecast.png')
plt.close()
print("Saved plot to 'rf_forecast.png'")

print("\n--- 5. Classification Metrics (Optional) ---")

def categorize_pm25(value):
    """Categorize PM2.5 based on US EPA AQI standards"""
    if value <= 12:
        return 'Good'
    elif value <= 35.4:
        return 'Moderate'
    elif value <= 55.4:
        return 'Unhealthy_SG'
    elif value <= 150.4:
        return 'Unhealthy'
    elif value <= 250.4:
        return 'Very_Unhealthy'
    else:
        return 'Hazardous'

y_test_cat = y_test.apply(categorize_pm25)
y_pred_rf_cat = pd.Series(y_pred_rf, index=y_test.index).apply(categorize_pm25)

category_order = ['Good', 'Moderate', 'Unhealthy_SG', 'Unhealthy', 'Very_Unhealthy', 'Hazardous']
y_test_cat = pd.Categorical(y_test_cat, categories=category_order, ordered=True)
y_pred_rf_cat = pd.Categorical(y_pred_rf_cat, categories=category_order, ordered=True)

cm = confusion_matrix(y_test_cat, y_pred_rf_cat, labels=category_order)

accuracy = accuracy_score(y_test_cat, y_pred_rf_cat)
precision = precision_score(y_test_cat, y_pred_rf_cat, average='weighted', zero_division=0)
recall = recall_score(y_test_cat, y_pred_rf_cat, average='weighted', zero_division=0)
f1 = f1_score(y_test_cat, y_pred_rf_cat, average='weighted', zero_division=0)

print(f"\nClassification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=category_order, yticklabels=category_order)
plt.title('Random Forest Confusion Matrix (PM2.5 Categories)', fontsize=14, fontweight='bold')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved confusion matrix to 'rf_confusion_matrix.png'")

print("\nDetailed Classification Report:")
print(classification_report(y_test_cat, y_pred_rf_cat, zero_division=0))

print("\n--- 6. Train Final Model ---")
X_full = df_model.drop(target_col, axis=1)
y_full = df_model[target_col]

scaler_final = StandardScaler()
X_full_scaled = scaler_final.fit_transform(X_full)

final_model = RandomForestRegressor(n_estimators=100,
                                    max_depth=15,
                                    random_state=42,
                                    n_jobs=-1,
                                    min_samples_leaf=5)
final_model.fit(X_full_scaled, y_full)
print("Final model trained on all data.")

total_runtime = time.time() - start_time
print(f"\nTotal runtime: {rf_runtime:.2f} seconds ({rf_runtime/60:.2f} minutes)")
print("\n--- Script Finished ---")