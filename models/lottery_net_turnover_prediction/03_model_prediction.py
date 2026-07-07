# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 – Model Prediction (Lottery Net Turnover Prediction)
# MAGIC
# MAGIC Scores the next week inference row with the champion lottery net turnover model.
# MAGIC
# MAGIC **Output:**
# MAGIC - `predicted_revenue` — net turnover forecast for next week
# MAGIC - Summary row published to `{T_LOTTERY_SUMMARY_PREFIX}_{ANALYSIS_NAME}_current`

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_lottery_net_turnover_prediction

# COMMAND ----------

for _w in ["analysis_id", "model_alias"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id", "analysis_1")
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("model_catalog", "ml_train")

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

analysis_id   = dbutils.widgets.get("analysis_id").strip()
model_alias   = dbutils.widgets.get("model_alias").strip()
model_catalog = dbutils.widgets.get("model_catalog").strip()
ANALYSIS_NAME = ANALYSIS_NAMES[analysis_id]

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

# COMMAND ----------

infer_spark = read_current_table(
    spark=spark,
    table_base=f"{T_LOTTERY_FEATURES_PREFIX}_{ANALYSIS_NAME}_infer",
)

# Use collect() on a single-row inference table to avoid multi-job toPandas() kernel crash
infer_row  = infer_spark.limit(1).collect()[0]
infer_dict = infer_row.asDict()

print(f"Target week: {infer_dict['week_start']}")

# COMMAND ----------

model = load_model(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=model_catalog,
    alias=model_alias,
)

available_cols    = [c for c in FEATURE_COLS if c in infer_dict]
X_infer           = np.array([[float(infer_dict[c]) for c in available_cols]])
predicted_revenue = float(np.clip(model.predict(X_infer), 0, None)[0])

print(f"Predicted lottery net turnover next week: {predicted_revenue:,.2f}")

# COMMAND ----------

summary_pd = pd.DataFrame([{
    "analysis_id":            analysis_id,
    "analysis_name":          ANALYSIS_NAME,
    "model_alias":            model_alias,
    "week_start":             str(infer_dict["week_start"]),
    "predicted_revenue":      predicted_revenue,
    "revenue_lag_1w":         float(infer_dict["revenue_lag_1w"]),
    "revenue_rolling_4w_avg": float(infer_dict["revenue_rolling_4w_avg"]),
}])

summary_spark = (
    spark.createDataFrame(summary_pd)
    .withColumn("week_start",        F.to_date("week_start"))
    .withColumn("predicted_revenue", F.col("predicted_revenue").cast("double"))
    .withColumn("created_at",        F.current_timestamp())
)

summary_publish = publish_table(
    spark=spark,
    df=summary_spark,
    table_base=f"{T_LOTTERY_SUMMARY_PREFIX}_{ANALYSIS_NAME}",
    analysis_name=ANALYSIS_NAME,
    current_mode="view",
    primary_keys=["week_start", "run_ts"],
    timeseries_column="run_ts",
)

print(json.dumps({
    "summary_current":   summary_publish["current_table"],
    "week_start":        str(infer_dict["week_start"]),
    "predicted_revenue": f"{predicted_revenue:,.2f}",
    "run_id":            summary_publish["run_id"],
}, indent=2, default=str))
