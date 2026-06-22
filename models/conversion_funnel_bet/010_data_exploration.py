# Databricks notebook source
# MAGIC %md
# MAGIC ## Deposit-to-first-wager data exploration

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_bet

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "analysis_2")
analysis_id = dbutils.widgets.get("analysis_id").strip()

if analysis_id not in ANALYSIS_NAMES:
    raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)}")

analysis_name = ANALYSIS_NAMES[analysis_id]
feat_df = read_current_table(
    spark=spark,
    table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}",
)

display(
    feat_df
    .groupBy("deposit_month", "source_system")
    .agg(
        F.count("*").alias("depositors"),
        F.sum(F.col("label_wager_success_30d")).alias("wagered_30d"),
        F.avg(F.col("label_wager_success_30d").cast("double")).alias("wager_conversion_rate_30d"),
        F.avg(F.col("first_deposit_amount")).alias("avg_first_deposit_amount"),
        F.expr("percentile_approx(first_wager_delay_hours_30d, 0.5)").alias("median_hours_to_first_wager"),
        F.avg(F.col("num_wagers_30d")).alias("avg_wagers_30d"),
        F.avg(F.col("sum_wager_amount_30d")).alias("avg_wager_amount_30d"),
    )
    .orderBy(F.col("deposit_month").desc(), F.col("source_system"))
)

display(
    feat_df
    .groupBy(
        F.when(F.col("first_wager_delay_hours_30d").isNull(), F.lit("No wager"))
        .when(F.col("first_wager_delay_hours_30d") < F.lit(1.0), F.lit("<1h"))
        .when(F.col("first_wager_delay_hours_30d") < F.lit(24.0), F.lit("1-24h"))
        .when(F.col("first_wager_delay_hours_30d") < F.lit(168.0), F.lit("1-7d"))
        .otherwise(F.lit("7-30d"))
        .alias("time_to_wager")
    )
    .agg(F.count("*").alias("users"))
    .orderBy(F.col("users").desc())
)
