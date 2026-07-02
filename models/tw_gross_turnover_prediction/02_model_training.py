# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 – Model Training (TW Gross Turnover Prediction)
# MAGIC
# MAGIC Trains an XGBoost Tweedie regressor on **weekly aggregate** TRUEWIN gross turnover.
# MAGIC Each training row = one week; the target is total weekly gross turnover.
# MAGIC
# MAGIC - **Model:** XGBoost `reg:tweedie` — handles non-negative, right-skewed revenue natively
# MAGIC - **CV:** time-series walk-forward split (not random k-fold)
# MAGIC - **Auto-promote:** challenger → champion when CV MAPE improves by ≥ 1 pp

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_tw_gross_turnover_prediction

# COMMAND ----------

for _w in ["analysis_id", "seed", "candidate_alias", "champion_alias", "required_signature"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id",        "analysis_1")
dbutils.widgets.text("seed",               "42")
dbutils.widgets.text("candidate_alias",    "challenger")
dbutils.widgets.text("champion_alias",     "champion")
dbutils.widgets.text("required_signature", "true")

# COMMAND ----------

import json
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType


def _to_bool(v, default=False):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"} if v else default


analysis_id        = dbutils.widgets.get("analysis_id").strip()
seed               = int(dbutils.widgets.get("seed"))
candidate_alias    = dbutils.widgets.get("candidate_alias").strip()
champion_alias     = dbutils.widgets.get("champion_alias").strip()
required_signature = _to_bool(dbutils.widgets.get("required_signature"), default=True)

ANALYSIS_NAME = ANALYSIS_NAMES[analysis_id]

# COMMAND ----------

# Weekly aggregate feature columns — must match 01_feature_engineering output
FEATURE_COLS = [
    "revenue_lag_1w",
    "revenue_lag_4w",
    "revenue_lag_8w",
    "revenue_lag_12w",
    "revenue_rolling_4w_avg",
    "revenue_rolling_4w_std",
    "revenue_rolling_8w_avg",
    "revenue_rolling_12w_avg",
    "revenue_trend_4w",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_month_start",
    "is_month_end",
    "is_summer",
    "is_winter",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "is_ramadan",
    "is_eid",
    "is_uae_national_day",
    "has_payday",
]

LABEL_COL = "revenue"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load training features

# COMMAND ----------

train_spark = read_current_table(
    spark=spark,
    table_base=f"{T_FEATURES_PREFIX}_{ANALYSIS_NAME}",
)

for col in [f.name for f in train_spark.schema.fields if isinstance(f.dataType, DecimalType)]:
    train_spark = train_spark.withColumn(col, F.col(col).cast("double"))

# Sort by week to preserve time ordering for walk-forward CV
train_pd = train_spark.orderBy("week_start").toPandas().reset_index(drop=True)

# Hold back the last 4 weeks for evaluation — model must NOT see them during training
N_EVAL_HOLDOUT = 4
train_pd_model = train_pd.iloc[:-N_EVAL_HOLDOUT]

available_feat_cols = [c for c in FEATURE_COLS if c in train_pd_model.columns]
# Use numpy arrays to avoid pandas groupby SIGSEGV inside sklearn's TimeSeriesSplit
X = train_pd_model[available_feat_cols].astype("float64").values
y = train_pd_model[LABEL_COL].astype("float64").values

print(f"Total weeks in feature table : {len(train_pd):,}")
print(f"Held back for evaluation     : {N_EVAL_HOLDOUT} weeks  ({train_pd.iloc[-N_EVAL_HOLDOUT]['week_start']} → {train_pd.iloc[-1]['week_start']})")
print(f"Training weeks               : {len(X):,}  |  Features: {len(available_feat_cols)}")
print(f"Revenue — mean: {y.mean():,.2f}  median: {float(np.median(y)):,.2f}  max: {y.max():,.2f}")
print(f"Training date range: {train_pd_model['week_start'].min()} → {train_pd_model['week_start'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time-series walk-forward cross-validation

# COMMAND ----------

def create_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("model",   XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


# Walk-forward CV: at least 26 weeks of initial training, 4-week test windows
n_splits    = min(8, max(3, (len(X) - 26) // 4))
tscv        = TimeSeriesSplit(n_splits=n_splits, test_size=4)
cv_mapes    = []

for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    fold_pipe = create_pipeline(seed)
    fold_pipe.fit(X_tr, y_tr)
    y_pred = np.clip(fold_pipe.predict(X_te), 0, None)

    fold_mape = mean_absolute_percentage_error(y_te, y_pred) * 100
    cv_mapes.append(fold_mape)
    print(f"  Fold {fold_i+1:02d} | train={len(X_tr)} test={len(X_te)} | MAPE={fold_mape:.2f}%")

cv_mape_scores = np.array(cv_mapes)
print(f"\nCV MAPE: {cv_mape_scores.mean():.2f}% ± {cv_mape_scores.std():.2f}%  (n_splits={n_splits})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train final model on all history except the held-back evaluation weeks

# COMMAND ----------

pipeline = create_pipeline(seed)
pipeline.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register with MLflow

# COMMAND ----------

model_name = build_model_name(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=WRITE_CATALOG,
)

X_example = train_pd_model[available_feat_cols].head(5).copy()
y_example = pipeline.predict(X_example)
signature = infer_signature(X_example, y_example)

with mlflow.start_run(run_name=f"train_{ANALYSIS_NAME}"):
    mlflow.log_params({
        "subject":         SUBJECT_NAME,
        "analysis_id":     analysis_id,
        "analysis_name":   ANALYSIS_NAME,
        "seed":            seed,
        "n_weeks":         int(X.shape[0]),
        "n_weeks_total":   int(len(train_pd)),
        "n_eval_holdout":  N_EVAL_HOLDOUT,
        "n_features":      int(X.shape[1]),
        "n_cv_splits":     n_splits,
        "source_table":    FACT_SETTLED_WAGER,
        "source_system":   SOURCE_SYSTEM,
        "label_col":       LABEL_COL,
        "candidate_alias": candidate_alias,
    })
    mlflow.log_metric("cv_mape_mean", float(cv_mape_scores.mean()))
    mlflow.log_metric("cv_mape_std",  float(cv_mape_scores.std()))
    mlflow.log_text("\n".join(available_feat_cols), "features.txt")

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        registered_model_name=model_name,
        input_example=X_example,
        signature=signature,
    )

    run_id    = mlflow.active_run().info.run_id
    candidate = register_candidate(
        model_name=model_name,
        run_id=run_id,
        alias=candidate_alias,
        tags={
            "analysis":          ANALYSIS_NAME,
            "governance_status": "candidate",
            "subject":           SUBJECT_NAME,
            "source_table":      FACT_SETTLED_WAGER,
            "source_system":     SOURCE_SYSTEM,
        },
    )
    validation = validate_candidate(
        model_name=model_name,
        alias=candidate_alias,
        required_signature=required_signature,
        sample_input=X_example,
    )
    if not validation["passed"]:
        raise ValueError(f"Validation failed: {validation}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto-promote: challenger → champion if MAPE improves ≥ 1 pp

# COMMAND ----------

client  = MlflowClient()
promote = False
MIN_IMPROVEMENT_PP = 1.0

try:
    champ_version   = client.get_model_version_by_alias(model_name, champion_alias)
    champ_mape      = client.get_run(champ_version.run_id).data.metrics.get("cv_mape_mean", float("inf"))
    challenger_mape = float(cv_mape_scores.mean())
    delta           = champ_mape - challenger_mape

    print(f"Champion   CV MAPE: {champ_mape:.2f}%")
    print(f"Challenger CV MAPE: {challenger_mape:.2f}%  (Δ={delta:+.2f} pp)")
    promote = delta >= MIN_IMPROVEMENT_PP

except Exception as e:
    print(f"No existing champion ({e}) — auto-promoting on first run.")
    promote = True

if promote:
    promote_alias(model_name=model_name, source_alias=candidate_alias, target_alias=champion_alias)
    print(f"Promoted {candidate_alias} → {champion_alias}")
else:
    print("Challenger did not improve enough — champion retained.")

# COMMAND ----------

print(json.dumps({
    "analysis_name":     ANALYSIS_NAME,
    "model_name":        model_name,
    "candidate_version": candidate["version"],
    "validated":         bool(validation["passed"]),
    "auto_promoted":     bool(promote),
    "cv_mape_mean":      float(cv_mape_scores.mean()),
    "cv_mape_std":       float(cv_mape_scores.std()),
    "n_weeks_trained":   int(len(X)),
    "n_cv_splits":       n_splits,
}, indent=2, default=str))
