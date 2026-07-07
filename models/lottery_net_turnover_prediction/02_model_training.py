# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 – Model Training (Lottery Net Turnover Prediction)
# MAGIC
# MAGIC Trains an XGBoost regressor on weekly lottery net turnover history.
# MAGIC Registers the model to MLflow as `challenger`.

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_lottery_net_turnover_prediction

# COMMAND ----------

for _w in ["analysis_id", "seed", "candidate_alias", "model_catalog", "champion_alias", "required_signature"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id",        "analysis_1")
dbutils.widgets.text("seed",               "42")
dbutils.widgets.text("candidate_alias",    "challenger")
dbutils.widgets.text("model_catalog",      "ml_train")
dbutils.widgets.text("champion_alias",     "champion")
dbutils.widgets.text("required_signature", "true")

# COMMAND ----------

import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
seed               = int(dbutils.widgets.get("seed").strip())
candidate_alias    = dbutils.widgets.get("candidate_alias").strip()
model_catalog      = dbutils.widgets.get("model_catalog").strip()
champion_alias     = dbutils.widgets.get("champion_alias").strip()
required_signature = _to_bool(dbutils.widgets.get("required_signature"), default=True)
ANALYSIS_NAME      = ANALYSIS_NAMES[analysis_id]

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

SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
]

# COMMAND ----------

train_table_base = f"{T_LOTTERY_FEATURES_PREFIX}_{ANALYSIS_NAME}_train"
train_spark = read_current_table(spark=spark, table_base=train_table_base)

decimal_cols = [f.name for f in train_spark.schema.fields if isinstance(f.dataType, DecimalType)]
for c in decimal_cols:
    train_spark = train_spark.withColumn(c, F.col(c).cast("double"))

train_pd = train_spark.toPandas().reset_index(drop=True)

available_cols = [c for c in FEATURE_COLS if c in train_pd.columns]
# Use numpy arrays to avoid pandas groupby SIGSEGV inside sklearn's TimeSeriesSplit
X = train_pd[available_cols].astype("float64").values
y = train_pd[LABEL_COL].astype("float64").values

print(f"Training rows: {len(train_pd):,}  |  Features: {len(available_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-validation (time series split)

# COMMAND ----------

tscv  = TimeSeriesSplit(n_splits=4)
mapes = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    m = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                     subsample=0.8, colsample_bytree=0.8, random_state=seed)
    m.fit(X_tr, y_tr)
    preds = np.clip(m.predict(X_val), 0, None)
    mape  = mean_absolute_percentage_error(y_val, preds) * 100
    mapes.append(mape)
    print(f"  Fold {fold+1}: MAPE = {mape:.2f}%")

cv_mape = float(np.mean(mapes))
print(f"CV MAPE: {cv_mape:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train final model on full dataset

# COMMAND ----------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
    )),
])
pipeline.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register with MLflow

# COMMAND ----------

model_name = build_model_name(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=model_catalog,
)

X_example = train_pd[available_cols].head(3).astype("float64")
y_example = pipeline.predict(X_example)
signature = infer_signature(X_example, y_example)

with mlflow.start_run(run_name=f"train_{ANALYSIS_NAME}") as run:
    mlflow.log_params({
        "analysis_id":     analysis_id,
        "analysis_name":   ANALYSIS_NAME,
        "n_estimators":    200,
        "learning_rate":   0.05,
        "max_depth":       4,
        "seed":            seed,
        "feature_cols":    json.dumps(available_cols),
        "candidate_alias": candidate_alias,
        "model_catalog":   model_catalog,
    })
    mlflow.log_metrics({
        "cv_mape_mean": cv_mape,
        "train_rows":   len(train_pd),
        "n_features":   len(available_cols),
    })

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        registered_model_name=model_name,
        input_example=X_example,
        signature=signature,
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

    run_id = run.info.run_id
    print(f"MLflow run_id: {run_id}")
    print(f"Model:         {model_name}")

# COMMAND ----------

candidate = register_candidate(
    model_name=model_name,
    run_id=run_id,
    alias=candidate_alias,
    tags={
        "analysis":          ANALYSIS_NAME,
        "governance_status": "candidate",
        "subject":           SUBJECT_NAME,
        "cv_mape":           str(round(cv_mape, 4)),
    },
)
validation = validate_candidate(
    model_name=model_name,
    alias=candidate_alias,
    required_signature=required_signature,
    sample_input=X_example,
)
if not validation["passed"]:
    raise ValueError(f"Candidate validation failed: {validation}")

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
    challenger_mape = cv_mape
    delta           = champ_mape - challenger_mape

    print(f"Champion   CV MAPE: {champ_mape:.2f}%")
    print(f"Challenger CV MAPE: {challenger_mape:.2f}%  (Δ={delta:+.2f} pp)")
    promote = delta >= MIN_IMPROVEMENT_PP

except Exception as e:
    print(f"No existing champion ({e}) — auto-promoting on first run.")
    promote = True

if promote:
    promote_alias(
        model_name=model_name,
        source_alias=candidate_alias,
        target_alias=champion_alias,
        remove_source_alias=False,
    )
    print(f"Promoted {candidate_alias} → {champion_alias}")
else:
    print("Challenger did not improve enough — champion retained.")

# COMMAND ----------

print(json.dumps({
    "model_name":      model_name,
    "version":         candidate["version"],
    "candidate_alias": candidate_alias,
    "cv_mape":         f"{cv_mape:.2f}%",
    "validated":       bool(validation["passed"]),
    "auto_promoted":   bool(promote),
    "run_id":          run_id,
}, indent=2))
