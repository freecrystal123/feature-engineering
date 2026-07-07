# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 – Model Prediction (Active User Prediction)
# MAGIC
# MAGIC **Formula:**
# MAGIC ```
# MAGIC predicted_active_users_this_week
# MAGIC     = active_users_last_week - predicted_churn_users_this_week
# MAGIC ```
# MAGIC
# MAGIC **Churn definition:** a user who was active last week (placed ≥ 1 bet) but is
# MAGIC predicted to place **zero** bets this week.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Load inference features for users who were active last week
# MAGIC 2. Score each user with the champion churn classifier
# MAGIC 3. Apply churn threshold to produce a binary churn flag
# MAGIC 4. Compute and publish the weekly active-user forecast

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_active_user_prediction

# COMMAND ----------

dbutils.widgets.text("analysis_id",     "analysis_4")
dbutils.widgets.text("model_alias",     "champion")
dbutils.widgets.text("model_catalog",   "ml_train")
dbutils.widgets.text("churn_threshold", "0.7")

# COMMAND ----------

import json
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

analysis_id     = dbutils.widgets.get("analysis_id").strip()
model_alias     = dbutils.widgets.get("model_alias").strip()
model_catalog   = dbutils.widgets.get("model_catalog").strip()
churn_threshold = float(dbutils.widgets.get("churn_threshold").strip())

ANALYSIS_NAME = ANALYSIS_NAMES[analysis_id]

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load inference features (users active last week)

# COMMAND ----------

infer_table_base = f"{T_ACTIVE_USER_FEATURES_PREFIX}_{ANALYSIS_NAME}_infer"
infer_spark = read_current_table(spark=spark, table_base=infer_table_base)

# Cast Decimal columns to double before toPandas() to avoid JSON serialisation errors
decimal_cols = [f.name for f in infer_spark.schema.fields if isinstance(f.dataType, DecimalType)]
for col in decimal_cols:
    infer_spark = infer_spark.withColumn(col, F.col(col).cast("double"))

infer_pd = infer_spark.toPandas().reset_index(drop=True)
user_ids = infer_pd["user_id"].values

available_feat_cols = [c for c in FEATURE_COLS if c in infer_pd.columns]
# Use numpy array to avoid pandas groupby SIGSEGV inside sklearn
X_infer = infer_pd[available_feat_cols].astype("float64").values

active_users_last_week = len(user_ids)
print(f"Active users last week: {active_users_last_week:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score with champion churn classifier

# COMMAND ----------

model = load_model(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=model_catalog,
    alias=model_alias,
)

# predict_proba columns: [P(label=0/churn), P(label=1/retained)]
proba = model.predict_proba(X_infer)
prob_churn    = proba[:, 0]
prob_retained = proba[:, 1]

predicted_churn_flag = (prob_churn >= churn_threshold).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build prediction DataFrame

# COMMAND ----------

pred_pd = pd.DataFrame({
    "user_id":         user_ids,
    "prob_churn":      prob_churn,
    "prob_retained":   prob_retained,
    "predicted_churn": predicted_churn_flag,  # 1 = predicted to churn, 0 = predicted to stay
    "churn_threshold": churn_threshold,
})

pred_spark = (
    spark.createDataFrame(pred_pd)
    .withColumn("created_at", F.current_timestamp())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute weekly active-user forecast
# MAGIC
# MAGIC ```
# MAGIC predicted_active_users_this_week = active_users_last_week - predicted_churn_users
# MAGIC ```

# COMMAND ----------

predicted_churn_users      = int(predicted_churn_flag.sum())
predicted_active_this_week = active_users_last_week - predicted_churn_users
churn_rate_predicted       = predicted_churn_users / active_users_last_week if active_users_last_week > 0 else 0.0

summary_pd = pd.DataFrame([{
    "analysis_id":                analysis_id,
    "analysis_name":              ANALYSIS_NAME,
    "model_alias":                model_alias,
    "churn_threshold":            churn_threshold,
    "active_users_last_week":     active_users_last_week,
    "predicted_churn_users":      predicted_churn_users,
    "predicted_active_this_week": predicted_active_this_week,
    "predicted_churn_rate":       churn_rate_predicted,
}])

summary_spark = (
    spark.createDataFrame(summary_pd)
    .withColumn("created_at", F.current_timestamp())
)

print(json.dumps({
    "active_users_last_week":     active_users_last_week,
    "predicted_churn_users":      predicted_churn_users,
    "predicted_active_this_week": predicted_active_this_week,
    "predicted_churn_rate":       f"{churn_rate_predicted:.2%}",
}, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Publish results

# COMMAND ----------

pred_base    = f"{T_ACTIVE_USER_PREDICTED_PREFIX}_{ANALYSIS_NAME}"
summary_base = f"{T_ACTIVE_USER_SUMMARY_PREFIX}_{ANALYSIS_NAME}"

pred_publish = publish_table(
    spark=spark,
    df=pred_spark,
    table_base=pred_base,
    analysis_name=ANALYSIS_NAME,
    current_mode="view",
)

summary_publish = publish_table(
    spark=spark,
    df=summary_spark,
    table_base=summary_base,
    analysis_name=ANALYSIS_NAME,
    run_id=pred_publish["run_id"],
    current_mode="view",
)

print(json.dumps({
    "pred_current":    pred_publish["current_table"],
    "summary_current": summary_publish["current_table"],
    "run_id":          pred_publish["run_id"],
}, indent=2, default=str))
