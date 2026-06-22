# Databricks notebook source
# MAGIC %md
# MAGIC ## 7 - Deposit-to-first-wager customer journey dataset

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_bet

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
analysis_id = dbutils.widgets.get("analysis_id").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    feat_df = read_current_table(
        spark=spark,
        table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}",
    )

    journey = (
        feat_df
        .select(
            "unique_user_id",
            "user_id",
            "source_system",
            "registration_ts",
            "first_deposit_ts",
            "deposit_month",
            "first_deposit_amount",
            "first_deposit_channel_type",
            "first_deposit_mode_type",
            "registration_to_deposit_hours",
            "observation_complete",
            "label_wager_success_30d",
            "first_wager_ts",
            "first_wager_delay_hours_30d",
            "first_wager_amount_30d",
            "num_wagers_30d",
            "sum_wager_amount_30d",
            "avg_wager_amount_30d",
            "num_wager_platforms_30d",
            "num_wager_coupons_30d",
        )
        .withColumn(
            "time_to_first_wager_bucket",
            F.when(F.col("label_wager_success_30d") == F.lit(0), F.lit("No wager within 30d"))
            .when(F.col("first_wager_delay_hours_30d") < F.lit(1.0), F.lit("< 1 hour"))
            .when(F.col("first_wager_delay_hours_30d") < F.lit(6.0), F.lit("1-6 hours"))
            .when(F.col("first_wager_delay_hours_30d") < F.lit(24.0), F.lit("6-24 hours"))
            .when(F.col("first_wager_delay_hours_30d") < F.lit(72.0), F.lit("1-3 days"))
            .when(F.col("first_wager_delay_hours_30d") < F.lit(168.0), F.lit("3-7 days"))
            .otherwise(F.lit("7-30 days")),
        )
        .withColumn(
            "deposit_amount_bucket",
            F.when(F.col("first_deposit_amount").isNull(), F.lit("Missing"))
            .when(F.col("first_deposit_amount") <= F.lit(10.0), F.lit("<= 10"))
            .when(F.col("first_deposit_amount") <= F.lit(25.0), F.lit("10-25"))
            .when(F.col("first_deposit_amount") <= F.lit(50.0), F.lit("25-50"))
            .when(F.col("first_deposit_amount") <= F.lit(100.0), F.lit("50-100"))
            .otherwise(F.lit("> 100")),
        )
        .withColumn(
            "journey_outcome",
            F.when(F.col("label_wager_success_30d") == F.lit(1), F.lit("Deposit -> Wager"))
            .otherwise(F.lit("Deposit -> No wager")),
        )
        .withColumn("journey_window_days", F.lit(int(DEPOSIT_JOURNEY_WINDOW_DAYS)))
        .withColumn("created_at", F.current_timestamp())
    )

    journey_publish = publish_table(
        spark=spark,
        df=journey,
        table_base=f"{T_FUNNEL_REPORT_PREFIX}_deposit_journey_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["unique_user_id", "run_ts"],
        timeseries_column="run_ts",
        append_merge_schema=True,
    )

    print(
        {
            "analysis": analysis_name,
            "journey_window_days": int(DEPOSIT_JOURNEY_WINDOW_DAYS),
            "journey_table": journey_publish["current_table"],
        }
    )
