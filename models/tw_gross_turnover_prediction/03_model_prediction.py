# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 – Model Prediction (TW Gross Turnover Prediction)
# MAGIC
# MAGIC Scores the **next week** inference row with the champion gross-turnover model.
# MAGIC
# MAGIC **Output:**
# MAGIC - `predicted_revenue` — model's point estimate for total TRUEWIN gross turnover next week

# COMMAND ----------

# MAGIC %pip install -q xgboost scikit-learn==1.9.0

# COMMAND ----------

# MAGIC %run ./00_config_tw_gross_turnover_prediction

# COMMAND ----------

for _w in ["analysis_id", "model_alias"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id", "analysis_1")
dbutils.widgets.text("model_alias", "champion")

# COMMAND ----------

import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

analysis_id   = dbutils.widgets.get("analysis_id").strip()
model_alias   = dbutils.widgets.get("model_alias").strip()
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

# MAGIC %md
# MAGIC ### Load inference features (one row = next week)

# COMMAND ----------

infer_spark = read_current_table(
    spark=spark,
    table_base=f"{T_INFER_PREFIX}_{ANALYSIS_NAME}",
)

for col in [f.name for f in infer_spark.schema.fields if isinstance(f.dataType, DecimalType)]:
    infer_spark = infer_spark.withColumn(col, F.col(col).cast("double"))

infer_pd = infer_spark.toPandas()
print(f"Inference rows: {len(infer_pd)}")
print(f"Target week:    {infer_pd['week_start'].iloc[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score with champion model

# COMMAND ----------

model = load_model(
    subject=SUBJECT_NAME,
    analysis_name=ANALYSIS_NAME,
    catalog=WRITE_CATALOG,
    alias=model_alias,
)

available_cols    = [c for c in FEATURE_COLS if c in infer_pd.columns]
X_infer           = infer_pd[available_cols].astype("float64")
predicted_revenue = float(np.clip(model.predict(X_infer), 0, None)[0])

print(f"Predicted gross turnover next week: {predicted_revenue:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build and publish prediction summary

# COMMAND ----------

summary_pd = pd.DataFrame([{
    "analysis_id":        analysis_id,
    "analysis_name":      ANALYSIS_NAME,
    "model_alias":        model_alias,
    "week_start":         str(infer_pd["week_start"].iloc[0]),
    "predicted_revenue":  predicted_revenue,
    "revenue_lag_1w":     float(infer_pd["revenue_lag_1w"].iloc[0]),
    "revenue_rolling_4w_avg": float(infer_pd["revenue_rolling_4w_avg"].iloc[0]),
}])

summary_spark = (
    spark.createDataFrame(summary_pd)
    .withColumn("week_start",  F.to_date("week_start"))
    .withColumn("created_at",  F.current_timestamp())
)

summary_publish = publish_table(
    spark=spark,
    df=summary_spark,
    table_base=f"{T_PREDICTIONS_PREFIX}_{ANALYSIS_NAME}",
    analysis_name=ANALYSIS_NAME,
    current_mode="view",
    primary_keys=["week_start", "run_ts"],
    timeseries_column="run_ts",
)

print(json.dumps({
    "pred_current":      summary_publish["current_table"],
    "week_start":        str(infer_pd["week_start"].iloc[0]),
    "predicted_revenue": f"{predicted_revenue:,.2f}",
    "run_id":            summary_publish["run_id"],
}, indent=2, default=str))
