# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 – Model Training (Active User Prediction)
# MAGIC
# MAGIC Trains a binary churn classifier on labeled historical data:
# MAGIC - `label = 1` → user was active last week AND placed ≥ 1 bet this week (retained)
# MAGIC - `label = 0` → user was active last week but placed 0 bets this week (churned)
# MAGIC
# MAGIC Model: Gradient Boosted Trees via scikit-learn, registered to MLflow.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_active_user_prediction

# COMMAND ----------

dbutils.widgets.text("analysis_id",           "analysis_4")
dbutils.widgets.text("seed",                  "42")
dbutils.widgets.text("candidate_alias",       "challenger")
dbutils.widgets.text("required_signature",    "true")
dbutils.widgets.text("auto_promote_champion", "false")
dbutils.widgets.text("champion_alias",        "champion")

# COMMAND ----------

import json
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from mlflow.models.signature import infer_signature
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


analysis_id           = dbutils.widgets.get("analysis_id").strip()
seed                  = int(dbutils.widgets.get("seed"))
candidate_alias       = dbutils.widgets.get("candidate_alias").strip()
required_signature    = _to_bool(dbutils.widgets.get("required_signature"), default=True)
auto_promote_champion = _to_bool(dbutils.widgets.get("auto_promote_champion"), default=False)
champion_alias        = dbutils.widgets.get("champion_alias").strip()

ANALYSIS_NAME = ANALYSIS_NAMES[analysis_id]

# COMMAND ----------

FEATURE_COLS = [
    "num_bets_total",
    "num_active_days",
    "num_active_weeks",
    "days_since_last_bet",
    "avg_bets_per_active_day",
    "num_bets_90d",
    "active_days_90d",
    "active_weeks_90d",
    "stake_sum_90d",
    "stake_avg_90d",
    "bets_last_4w",
    "bets_prior_4w",
    "bet_count_trend",
    "stake_sum_total",
    "stake_cv",
    "win_rate",
]

LABEL_COL = "label"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load training features

# COMMAND ----------

train_table_base = f"{T_ACTIVE_USER_FEATURES_PREFIX}_{ANALYSIS_NAME}_train"
train_spark = read_current_table(spark=spark, table_base=train_table_base)

# Cast Decimal columns to double before toPandas() to avoid JSON serialisation errors
decimal_cols = [f.name for f in train_spark.schema.fields if isinstance(f.dataType, DecimalType)]
for col in decimal_cols:
    train_spark = train_spark.withColumn(col, F.col(col).cast("double"))

train_pd = train_spark.toPandas()

available_feat_cols = [c for c in FEATURE_COLS if c in train_pd.columns]
X = train_pd[available_feat_cols].copy()
y = train_pd[LABEL_COL].astype(int)

print(f"Training samples: {len(X):,}  |  churn rate: {1 - y.mean():.2%}  |  features: {len(available_feat_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build & train pipeline

# COMMAND ----------

def create_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler",     StandardScaler(with_mean=True, with_std=True)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed,
        )),
    ])


# Compute sample weights to compensate for class imbalance:
# churn (~25%) gets upweighted so the model treats it equally with retained (~75%)
churn_rate   = 1 - y.mean()
retain_rate  = y.mean()
sample_weight = np.where(y == 0, retain_rate / churn_rate, 1.0)

pipeline  = create_pipeline(seed)
# ROC-AUC is rank-based and invariant to class weights, so CV scoring
# without sample_weight is still valid for model comparison.
# (fit_params was removed from cross_val_score in scikit-learn >= 1.4)
cv_scores = cross_val_score(
    pipeline, X, y,
    cv=5, scoring="roc_auc", n_jobs=-1,
)
pipeline.fit(X, y, classifier__sample_weight=sample_weight)

print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model with MLflow

# COMMAND ----------

model_name = build_model_name(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=WRITE_CATALOG,
)

# Cast all feature columns to float64 so MLflow signature has no integer columns,
# which cannot represent missing values and cause schema enforcement errors at inference.
X = X.astype("float64")

X_example = X.head(5).copy()
y_example = pipeline.predict_proba(X_example)
signature = infer_signature(X_example, y_example)

with mlflow.start_run(run_name=f"train_{ANALYSIS_NAME}"):
    mlflow.log_params({
        "subject":         SUBJECT_NAME,
        "analysis_id":     analysis_id,
        "analysis_name":   ANALYSIS_NAME,
        "seed":            seed,
        "n_rows":          int(X.shape[0]),
        "n_features":      int(X.shape[1]),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std":  float(cv_scores.std()),
        "write_catalog":   WRITE_CATALOG,
        "candidate_alias": candidate_alias,
    })
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
        },
    )
    validation = validate_candidate(
        model_name=model_name,
        alias=candidate_alias,
        required_signature=required_signature,
        sample_input=X_example,
    )
    if not validation["passed"]:
        raise ValueError(
            f"Candidate validation failed for {model_name} "
            f"(version={candidate['version']}): {validation}"
        )

    if auto_promote_champion:
        promote_alias(
            model_name=model_name,
            source_alias=candidate_alias,
            target_alias=champion_alias,
        )

result = {
    "analysis_id":       analysis_id,
    "analysis_name":     ANALYSIS_NAME,
    "model_name":        model_name,
    "candidate_alias":   candidate_alias,
    "candidate_version": candidate["version"],
    "validated":         bool(validation["passed"]),
    "auto_promoted":     bool(auto_promote_champion),
    "champion_alias":    champion_alias,
    "cv_roc_auc_mean":   float(cv_scores.mean()),
}
print(json.dumps(result, indent=2, default=str))
