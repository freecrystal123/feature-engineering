# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 – Feature Engineering (Lottery Net Turnover Prediction)
# MAGIC
# MAGIC **Revenue formula:** Net Turnover = Settled Wager Amount − VAT
# MAGIC
# MAGIC Steps:
# MAGIC 1. Aggregate daily settled wager and VAT for LOTTERY
# MAGIC 2. Compute daily net turnover and roll up to weekly totals
# MAGIC 3. Build lag + temporal features
# MAGIC 4. Publish `_train` (with label) and `_infer` (next week, no label)

# COMMAND ----------

# MAGIC %run ./00_config_lottery_net_turnover_prediction

# COMMAND ----------

for _w in ["analysis_id", "current_date", "n_eval_folds"]:
    try: dbutils.widgets.remove(_w)
    except: pass

dbutils.widgets.text("analysis_id",  "analysis_1")
dbutils.widgets.text("current_date", "")
dbutils.widgets.text("n_eval_folds", "4")

# COMMAND ----------

import json
import math
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DecimalType
from datetime import date as _date, timedelta as _timedelta

analysis_id   = dbutils.widgets.get("analysis_id").strip()
ANALYSIS_NAME = ANALYSIS_NAMES[analysis_id]
n_eval_folds  = int(dbutils.widgets.get("n_eval_folds").strip())

_current_date_str = dbutils.widgets.get("current_date").strip()
ref_date   = _date.fromisoformat(_current_date_str) if _current_date_str else _date.today()
week_start = ref_date - _timedelta(days=ref_date.weekday())   # Monday of current week
today_sk   = int(ref_date.strftime("%Y%m%d"))

print(f"Analysis:   {ANALYSIS_NAME}")
print(f"Week start: {week_start}  |  Excluding date_sk >= {today_sk}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Daily net turnover (settled wager − VAT)

# COMMAND ----------

daily_settled = spark.sql(f"""
    SELECT
        date_sk,
        to_date(cast(date_sk AS string), 'yyyyMMdd') AS bet_date,
        SUM(wager_amount_native)                     AS settled_wager_amount
    FROM {FACT_SETTLED_WAGER}
    WHERE source_system = '{SOURCE_SYSTEM}'
      AND date_sk < {today_sk}
    GROUP BY date_sk
""")

daily_vat = spark.sql(f"""
    SELECT
        date_sk,
        SUM(vat_on_wager_amount_native) AS vat_amount
    FROM {FACT_WAGER}
    WHERE source_system = '{SOURCE_SYSTEM}'
      AND date_sk < {today_sk}
    GROUP BY date_sk
""")

daily_revenue = (
    daily_settled
    .join(daily_vat, on="date_sk", how="left")
    .withColumn("vat_amount",           F.coalesce(F.col("vat_amount").cast("double"), F.lit(0.0)))
    .withColumn("settled_wager_amount", F.col("settled_wager_amount").cast("double"))
    .withColumn("daily_revenue",        F.col("settled_wager_amount") - F.col("vat_amount"))
    .select("date_sk", "bet_date", "settled_wager_amount", "vat_amount", "daily_revenue")
    .orderBy("date_sk")
)

daily_revenue.cache()

date_range = daily_revenue.agg(F.min("bet_date"), F.max("bet_date"), F.sum("daily_revenue")).collect()[0]
print(f"Daily rows:    {daily_revenue.count():,}")
print(f"Date range:    {date_range[0]} → {date_range[1]}")
print(f"Total revenue: {date_range[2]:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Weekly aggregation

# COMMAND ----------

weekly_revenue = (
    daily_revenue
    .withColumn("week_start", F.date_trunc("week", F.col("bet_date")).cast("date"))
    .groupBy("week_start")
    .agg(
        F.sum("daily_revenue").alias("revenue"),
        F.sum("settled_wager_amount").alias("settled_wager_amount"),
        F.sum("vat_amount").alias("vat_amount"),
    )
    .filter(F.col("week_start") < F.lit(str(week_start)))
    .orderBy("week_start")
)
weekly_revenue.cache()
print(f"Weekly rows: {weekly_revenue.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Lag + temporal features

# COMMAND ----------

w_spec = Window.orderBy("week_start")
w_4wk  = Window.orderBy("week_start").rowsBetween(-4, -1)
w_8wk  = Window.orderBy("week_start").rowsBetween(-8, -1)
w_12wk = Window.orderBy("week_start").rowsBetween(-12, -1)

PI2 = 2 * math.pi

features = (
    weekly_revenue
    # ── Revenue lag features ─────────────────────────────────────────
    .withColumn("revenue_lag_1w",  F.lag("revenue", 1).over(w_spec))
    .withColumn("revenue_lag_4w",  F.lag("revenue", 4).over(w_spec))
    .withColumn("revenue_lag_8w",  F.lag("revenue", 8).over(w_spec))
    .withColumn("revenue_lag_12w", F.lag("revenue", 12).over(w_spec))
    # ── Rolling aggregates ───────────────────────────────────────────
    .withColumn("revenue_rolling_4w_avg",  F.avg("revenue").over(w_4wk))
    .withColumn("revenue_rolling_4w_std",  F.stddev("revenue").over(w_4wk))
    .withColumn("revenue_rolling_8w_avg",  F.avg("revenue").over(w_8wk))
    .withColumn("revenue_rolling_12w_avg", F.avg("revenue").over(w_12wk))
    # ── Revenue trend: last 4w vs prior 4w ───────────────────────────
    .withColumn("revenue_lag_4w_sum",  F.sum("revenue").over(w_4wk))
    .withColumn("revenue_lag_8_5w_sum",
        F.sum("revenue").over(Window.orderBy("week_start").rowsBetween(-8, -5)))
    .withColumn("revenue_trend_4w",
        F.col("revenue_lag_4w_sum") - F.col("revenue_lag_8_5w_sum"))
    # ── Temporal features ────────────────────────────────────────────
    .withColumn("week_of_year",  F.weekofyear("week_start"))
    .withColumn("month",         F.month("week_start"))
    .withColumn("quarter",       F.quarter("week_start"))
    .withColumn("year",          F.year("week_start"))
    .withColumn("is_month_start",
        F.when(F.dayofmonth("week_start") <= 7, True).otherwise(False))
    .withColumn("is_month_end",
        F.when(F.dayofmonth("week_start") >= 22, True).otherwise(False))
    .withColumn("is_summer",
        F.when(F.month("week_start").isin(6, 7, 8), True).otherwise(False))
    .withColumn("is_winter",
        F.when(F.month("week_start").isin(12, 1, 2), True).otherwise(False))
    # ── Cyclical encodings ───────────────────────────────────────────
    .withColumn("week_sin",  F.sin(F.col("week_of_year") * F.lit(PI2 / 52)))
    .withColumn("week_cos",  F.cos(F.col("week_of_year") * F.lit(PI2 / 52)))
    .withColumn("month_sin", F.sin(F.col("month")        * F.lit(PI2 / 12)))
    .withColumn("month_cos", F.cos(F.col("month")        * F.lit(PI2 / 12)))
    # ── UAE holiday flags ────────────────────────────────────────────
    .withColumn("is_ramadan",
        F.when(
            (F.col("week_start").between("2024-03-11", "2024-04-09")) |
            (F.col("week_start").between("2025-03-01", "2025-03-30")) |
            (F.col("week_start").between("2026-02-18", "2026-03-19")),
        True).otherwise(False))
    .withColumn("is_eid",
        F.when(
            (F.col("week_start").between("2024-04-10", "2024-04-15")) |
            (F.col("week_start").between("2024-06-16", "2024-06-22")) |
            (F.col("week_start").between("2025-03-31", "2025-04-06")) |
            (F.col("week_start").between("2025-06-07", "2025-06-13")),
        True).otherwise(False))
    .withColumn("is_uae_national_day",
        F.when(F.month("week_start") == 12, True).otherwise(False))
    # ── Payday flag ──────────────────────────────────────────────────
    .withColumn("week_end", F.date_add("week_start", 6))
    .withColumn("has_payday",
        F.when(
            (F.dayofmonth("week_start") <= 1)  | (F.dayofmonth("week_end") >= 1)  &
            (F.dayofmonth("week_start") <= 15) | (F.dayofmonth("week_end") >= 15) &
            (F.dayofmonth("week_start") <= 25) | (F.dayofmonth("week_end") >= 25),
        True).otherwise(False))
    .drop("revenue_lag_4w_sum", "revenue_lag_8_5w_sum", "week_end",
          "settled_wager_amount", "vat_amount")
)

features = features.dropna(subset=["revenue_lag_1w", "revenue_lag_4w", "revenue_rolling_4w_avg"])
print(f"Feature rows (after dropping null-lag rows): {features.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Build train table and infer row (next week)

# COMMAND ----------

# Exclude the last n_eval_folds weeks — held back as true out-of-sample evaluation
w_rank = Window.orderBy(F.col("week_start").desc())
train_spark = (
    features
    .withColumn("_rank", F.row_number().over(w_rank))
    .filter(F.col("_rank") > n_eval_folds)
    .drop("_rank")
    .withColumn("created_at", F.current_timestamp())
)
print(f"Training rows (excluding last {n_eval_folds} weeks): {train_spark.count():,}")

latest_row = features.orderBy(F.col("week_start").desc()).limit(1).toPandas()

infer_data = {
    "week_start":              str(week_start),
    "revenue":                 None,
    "revenue_lag_1w":          float(latest_row["revenue"].iloc[0]),
    "revenue_lag_4w":          float(latest_row["revenue_lag_1w"].iloc[0]) if latest_row["revenue_lag_1w"].iloc[0] else 0.0,
    "revenue_lag_8w":          float(latest_row["revenue_lag_4w"].iloc[0]) if latest_row["revenue_lag_4w"].iloc[0] else 0.0,
    "revenue_lag_12w":         float(latest_row["revenue_lag_8w"].iloc[0]) if latest_row["revenue_lag_8w"].iloc[0] else 0.0,
    "revenue_rolling_4w_avg":  float(latest_row["revenue_rolling_4w_avg"].iloc[0]),
    "revenue_rolling_4w_std":  float(latest_row["revenue_rolling_4w_std"].iloc[0]) if latest_row["revenue_rolling_4w_std"].iloc[0] else 0.0,
    "revenue_rolling_8w_avg":  float(latest_row["revenue_rolling_8w_avg"].iloc[0]),
    "revenue_rolling_12w_avg": float(latest_row["revenue_rolling_12w_avg"].iloc[0]),
    "revenue_trend_4w":        float(latest_row["revenue_trend_4w"].iloc[0]) if latest_row["revenue_trend_4w"].iloc[0] else 0.0,
}

infer_data.update({
    "week_of_year":        int(week_start.strftime("%V")),
    "month":               week_start.month,
    "quarter":             (week_start.month - 1) // 3 + 1,
    "year":                week_start.year,
    "is_month_start":      week_start.day <= 7,
    "is_month_end":        week_start.day >= 22,
    "is_summer":           week_start.month in (6, 7, 8),
    "is_winter":           week_start.month in (12, 1, 2),
    "week_sin":            math.sin(int(week_start.strftime("%V")) * PI2 / 52),
    "week_cos":            math.cos(int(week_start.strftime("%V")) * PI2 / 52),
    "month_sin":           math.sin(week_start.month * PI2 / 12),
    "month_cos":           math.cos(week_start.month * PI2 / 12),
    "is_ramadan":          False,
    "is_eid":              False,
    "is_uae_national_day": week_start.month == 12,
    "has_payday":          any(d in range(week_start.day, week_start.day + 7) for d in [1, 15, 25]),
})

import pandas as pd
infer_pd    = pd.DataFrame([infer_data])
infer_spark = (
    spark.createDataFrame(infer_pd)
    .withColumn("week_start", F.to_date("week_start"))
    .withColumn("created_at", F.current_timestamp())
)

print(f"Next week to predict: {week_start}")
print(f"revenue_lag_1w (last week actual): {infer_data['revenue_lag_1w']:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Publish feature tables

# COMMAND ----------

train_publish = publish_table(
    spark=spark,
    df=train_spark,
    table_base=f"{T_LOTTERY_FEATURES_PREFIX}_{ANALYSIS_NAME}_train",
    analysis_name=f"{ANALYSIS_NAME}_train",
    current_mode="view",
    primary_keys=["week_start", "run_ts"],
    timeseries_column="run_ts",
)

infer_publish = publish_table(
    spark=spark,
    df=infer_spark,
    table_base=f"{T_LOTTERY_FEATURES_PREFIX}_{ANALYSIS_NAME}_infer",
    analysis_name=f"{ANALYSIS_NAME}_infer",
    current_mode="view",
    primary_keys=["week_start", "run_ts"],
    timeseries_column="run_ts",
)

print(json.dumps({
    "train_table": train_publish["current_table"],
    "infer_table": infer_publish["current_table"],
    "train_rows":  train_spark.count(),
    "run_id":      train_publish["run_id"],
}, indent=2))
