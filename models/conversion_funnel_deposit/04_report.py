# Databricks notebook source
# MAGIC %md
# MAGIC ## 4 - Conversion funnel report datasets (for Streamlit)

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_deposit

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


def _read_if_exists(table_base: str):
    current_name = current_table_name(table_base)
    snapshot_name = snapshot_table_name(table_base)
    if spark.catalog.tableExists(current_name):
        return spark.table(current_name)
    if spark.catalog.tableExists(snapshot_name):
        return spark.table(snapshot_name)
    return None


spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]

    feat_df = read_current_table(
        spark=spark,
        table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}",
    )
    pred_df = _read_if_exists(f"{T_FUNNEL_PREDICTIONS_PREFIX}_{analysis_name}")
    eval_df = _read_if_exists(f"{T_FUNNEL_EVAL_PREFIX}_{analysis_name}")
    shap_df = _read_if_exists(f"{T_FUNNEL_SHAP_PREFIX}_{analysis_name}")

    monthly = (
        feat_df
        .groupBy("registration_month")
        .agg(
            F.count("*").alias("registrations"),
            F.sum(F.col("label_deposit_attempt_30d").cast("double")).alias("attempted_30d"),
            F.sum(F.col("label_deposit_success_30d").cast("double")).alias("success_30d"),
            F.sum(F.col("label_failed_after_attempt_30d").cast("double")).alias("failed_after_attempt_30d"),
            F.sum(F.col("observation_complete").cast("double")).alias("matured_registrations"),
        )
        .withColumn("never_attempted_30d", F.col("registrations") - F.col("attempted_30d"))
        .withColumn("attempt_rate_30d", F.when(F.col("registrations") > 0, F.col("attempted_30d") / F.col("registrations")).otherwise(F.lit(0.0)))
        .withColumn("success_rate_30d", F.when(F.col("registrations") > 0, F.col("success_30d") / F.col("registrations")).otherwise(F.lit(0.0)))
        .withColumn("success_given_attempt_rate_30d", F.when(F.col("attempted_30d") > 0, F.col("success_30d") / F.col("attempted_30d")).otherwise(F.lit(0.0)))
        .withColumn("failed_given_attempt_rate_30d", F.when(F.col("attempted_30d") > 0, F.col("failed_after_attempt_30d") / F.col("attempted_30d")).otherwise(F.lit(0.0)))
    )

    if pred_df is not None:
        pred_enriched = (
            feat_df.alias("f")
            .join(
                pred_df.alias("p").select("unique_user_id", "predicted_label"),
                on="unique_user_id",
                how="left",
            )
            .groupBy("registration_month")
            .agg(
                F.avg(F.col("predicted_label").cast("double")).alias("avg_predicted_positive_rate"),
                F.sum(F.when(F.col("predicted_label") >= 0.5, F.lit(1)).otherwise(F.lit(0))).alias("predicted_positive_users"),
            )
        )
        monthly = monthly.join(pred_enriched, on="registration_month", how="left")
    else:
        monthly = monthly.withColumn("avg_predicted_positive_rate", F.lit(None).cast("double")).withColumn("predicted_positive_users", F.lit(None).cast("double"))

    monthly_publish = publish_table(
        spark=spark,
        df=monthly.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_FUNNEL_REPORT_PREFIX}_funnel_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["registration_month", "run_ts"],
        timeseries_column="run_ts",
    )

    if shap_df is not None:
        shap_top = (
            shap_df
            .select("feature_name", "feature_rank", "mean_abs_shap", "mean_shap", "sample_size")
            .orderBy(F.col("feature_rank").asc())
            .limit(30)
        )
        shap_publish = publish_table(
            spark=spark,
            df=shap_top.withColumn("created_at", F.current_timestamp()),
            table_base=f"{T_FUNNEL_REPORT_PREFIX}_shap_{analysis_name}",
            analysis_name=analysis_name,
            current_mode="view",
            primary_keys=["feature_name", "run_ts"],
            timeseries_column="run_ts",
        )
    else:
        shap_publish = {"current_table": None}

    if eval_df is not None:
        eval_publish = publish_table(
            spark=spark,
            df=eval_df.withColumn("created_at", F.current_timestamp()),
            table_base=f"{T_FUNNEL_REPORT_PREFIX}_eval_{analysis_name}",
            analysis_name=analysis_name,
            current_mode="view",
            primary_keys=["split_name", "run_ts"],
            timeseries_column="run_ts",
        )
    else:
        eval_publish = {"current_table": None}

    print(
        {
            "analysis": analysis_name,
            "funnel_table": monthly_publish["current_table"],
            "shap_table": shap_publish["current_table"],
            "eval_table": eval_publish["current_table"],
        }
    )
