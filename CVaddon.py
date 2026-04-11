import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 2. FILES TO LOAD
# =========================================================
city_files = {
    "Beijing": "cleaned_BeijingPM20100101_20151231.csv",
    "Chengdu": "cleaned_ChengduPM20100101_20151231.csv",
    "Guangzhou": "cleaned_GuangzhouPM20100101_20151231.csv",
    "Shanghai": "cleaned_ShanghaiPM20100101_20151231.csv",
    "Shenyang": "cleaned_ShenyangPM20100101_20151231.csv",
}

# =========================================================
# 3. LOAD AND MERGE DATA
# =========================================================
all_dfs = []

for city, filename in city_files.items():
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    df["city"] = city
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

print("Merged dataset shape:", df.shape)
print(df.head())

# =========================================================
# 4. DATETIME FEATURE ENGINEERING
# =========================================================
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

df = df.dropna(subset=["datetime", "PM_Average"]).copy()

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

# Build lag features per city so each row only uses historical PM values.
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)
city_pm = df.groupby("city")["PM_Average"]
df["PM_lag_1"] = city_pm.shift(1)
df["PM_lag_3"] = city_pm.shift(3)
df["PM_lag_24"] = city_pm.shift(24)
df["PM_roll_24_mean"] = df.groupby("city")["PM_Average"].transform(
    lambda s: s.shift(1).rolling(24, min_periods=24).mean()
)

# =========================================================
# 5. SELECT FEATURES
# IMPORTANT:
# We EXCLUDE station-level PM columns so the model predicts
# PM_Average from weather/time/city factors only.
# =========================================================
target_col = "PM_Average"

feature_cols = [
    "season",
    "DEWP",
    "HUMI",
    "PRES",
    "TEMP",
    "cbwd",
    "Iws",
    "precipitation",
    "Iprec",
    "city",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "PM_lag_1",
    "PM_lag_3",
    "PM_lag_24",
    "PM_roll_24_mean",
]

df_model = df[feature_cols + [target_col, "datetime"]].copy()
df_model = df_model.dropna().reset_index(drop=True)

# =========================================================
# 6. TIME-BASED TRAIN/TEST SPLIT
# Use earlier data for training, later data for testing
# =========================================================
df_model = df_model.sort_values("datetime").reset_index(drop=True)

split_index = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_index].copy()
test_df = df_model.iloc[split_index:].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# =========================================================
# 7. PREPROCESSING
# =========================================================
categorical_features = ["season", "cbwd", "city"]
numeric_features = [
    "DEWP",
    "HUMI",
    "PRES",
    "TEMP",
    "Iws",
    "precipitation",
    "Iprec",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "PM_lag_1",
    "PM_lag_3",
    "PM_lag_24",
    "PM_roll_24_mean",
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =========================================================
# 8. DEFINE MODELS
# =========================================================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
cv_results = []
trained_pipelines = {}


def evaluate_time_series_cv(pipeline, X, y, splitter):
    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for train_idx, val_idx in splitter.split(X):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_val_pred = pipeline.predict(X_val)

        fold_mae.append(mean_absolute_error(y_val, y_val_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        fold_r2.append(r2_score(y_val, y_val_pred))

    return {
        "CV_MAE_Mean": float(np.mean(fold_mae)),
        "CV_MAE_Std": float(np.std(fold_mae)),
        "CV_RMSE_Mean": float(np.mean(fold_rmse)),
        "CV_RMSE_Std": float(np.std(fold_rmse)),
        "CV_R2_Mean": float(np.mean(fold_r2)),
        "CV_R2_Std": float(np.std(fold_r2)),
    }

# =========================================================
# 9. TRAIN AND EVALUATE
# =========================================================
tscv = TimeSeriesSplit(n_splits=5)

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv_metrics = evaluate_time_series_cv(pipeline, X_train, y_train, tscv)
    cv_results.append({
        "Model": model_name,
        **cv_metrics
    })

    print(f"\n{model_name} (CV on train, 5 folds)")
    print(f"CV MAE : {cv_metrics['CV_MAE_Mean']:.4f} +/- {cv_metrics['CV_MAE_Std']:.4f}")
    print(f"CV RMSE: {cv_metrics['CV_RMSE_Mean']:.4f} +/- {cv_metrics['CV_RMSE_Std']:.4f}")
    print(f"CV R2  : {cv_metrics['CV_R2_Mean']:.4f} +/- {cv_metrics['CV_R2_Std']:.4f}")

    # Refit on full training set, then evaluate once on held-out test set.
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_pipelines[model_name] = pipeline

    print(f"\n{model_name}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

# =========================================================
# 10. SAVE MODEL COMPARISON
# =========================================================
results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"), index=False)

cv_results_df = pd.DataFrame(cv_results).sort_values(by="CV_RMSE_Mean", ascending=True)
cv_results_df.to_csv(os.path.join(OUTPUT_DIR, "model_cv_metrics.csv"), index=False)

print("\nModel comparison:")
print(results_df)

print("\nCross-validation summary (train folds):")
print(cv_results_df)

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

print(f"\nBest model selected: {best_model_name}")

# =========================================================
# 11. SAVE BEST MODEL
# =========================================================
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.joblib"))

# =========================================================
# 12. SAVE TEST PREDICTIONS
# =========================================================
best_preds = best_pipeline.predict(X_test)

predictions_df = test_df.copy()
predictions_df["actual_PM_Average"] = y_test.values
predictions_df["predicted_PM_Average"] = best_preds
predictions_df["residual"] = predictions_df["actual_PM_Average"] - predictions_df["predicted_PM_Average"]

predictions_df.to_csv(os.path.join(OUTPUT_DIR, "best_model_test_predictions.csv"), index=False)

# =========================================================
# 13. FEATURE IMPORTANCE
# Only available for tree-based models here
# =========================================================
feature_importance_df = None

if best_model_name in ["RandomForest", "GradientBoosting"]:
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_model = best_pipeline.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    feature_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False
    )

    # Plot top 15 features
    top_n = 15
    top_features = feature_importance_df.head(top_n).sort_values(by="importance")

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_top15.png"), dpi=200)
    plt.close()

# =========================================================
# 14. ACTUAL VS PREDICTED PLOT
# =========================================================
plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_preds, alpha=0.3)
plt.xlabel("Actual PM_Average")
plt.ylabel("Predicted PM_Average")
plt.title(f"Actual vs Predicted - {best_model_name}")

min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=200)
plt.close()

# =========================================================
# 15. RESIDUAL PLOT
# =========================================================
residuals = y_test - best_preds

plt.figure(figsize=(8, 5))
plt.scatter(best_preds, residuals, alpha=0.3)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted PM_Average")
plt.ylabel("Residual")
plt.title(f"Residual Plot - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"), dpi=200)
plt.close()

# =========================================================
# 16. SAVE SUMMARY TXT
# =========================================================
with open(os.path.join(OUTPUT_DIR, "training_summary.txt"), "w", encoding="utf-8") as f:
    f.write("PM2.5 Model Training Summary\n")
    f.write("===========================\n\n")
    f.write(f"Target variable: {target_col}\n")
    f.write(f"Features used: {', '.join(feature_cols)}\n\n")
    f.write("Models tested:\n")
    for _, row in results_df.iterrows():
        f.write(
            f"- {row['Model']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R2={row['R2']:.4f}\n"
        )

    f.write("\nCross-validation (TimeSeriesSplit, train only):\n")
    for _, row in cv_results_df.iterrows():
        f.write(
            f"- {row['Model']}: "
            f"CV_MAE={row['CV_MAE_Mean']:.4f}+/-{row['CV_MAE_Std']:.4f}, "
            f"CV_RMSE={row['CV_RMSE_Mean']:.4f}+/-{row['CV_RMSE_Std']:.4f}, "
            f"CV_R2={row['CV_R2_Mean']:.4f}+/-{row['CV_R2_Std']:.4f}\n"
        )
    f.write(f"\nBest model: {best_model_name}\n")

print("\nDone.")
print("Saved files in:")
print(f"- {OUTPUT_DIR}/")
print(f"- {MODEL_DIR}/")
