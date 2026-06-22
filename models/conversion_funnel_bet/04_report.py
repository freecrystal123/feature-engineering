# Databricks notebook source
# MAGIC %md
# MAGIC ## 4 - Deposit-to-wager report datasets (for Streamlit)

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
        .groupBy("deposit_month")
        .agg(
            F.count("*").alias("depositors"),
            F.sum(F.col("label_wager_success_30d").cast("double")).alias("wagered_30d"),
            F.sum(F.col("observation_complete").cast("double")).alias("matured_depositors"),
            F.sum(
                F.when(
                    F.col("observation_complete") & (F.col("label_wager_success_30d") == F.lit(1)),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("matured_wagered_30d"),
            F.avg(
                F.when(
                    F.col("label_wager_success_30d") == F.lit(1),
                    F.col("first_wager_delay_hours_30d").cast("double"),
                )
            ).alias("avg_first_wager_delay_hours_30d"),
            F.expr(
                "percentile_approx("
                "CASE WHEN label_wager_success_30d = 1 "
                "THEN CAST(first_wager_delay_hours_30d AS DOUBLE) END, 0.5)"
            ).alias("median_first_wager_delay_hours_30d"),
        )
        .withColumn("not_wagered_30d", F.col("depositors") - F.col("wagered_30d"))
        .withColumn(
            "wager_conversion_rate_30d",
            F.when(F.col("depositors") > 0, F.col("wagered_30d") / F.col("depositors")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "matured_wager_conversion_rate_30d",
            F.when(
                F.col("matured_depositors") > 0,
                F.col("matured_wagered_30d") / F.col("matured_depositors"),
            ).otherwise(F.lit(0.0)),
        )
    )

    if pred_df is not None:
        pred_enriched = (
            feat_df.alias("f")
            .join(
                pred_df.alias("p").select("unique_user_id", "predicted_label"),
                on="unique_user_id",
                how="left",
            )
            .groupBy("deposit_month")
            .agg(
                F.avg(F.col("predicted_label").cast("double")).alias("avg_predicted_positive_rate"),
                F.sum(
                    F.when(F.col("predicted_label") >= F.lit(0.5), F.lit(1)).otherwise(F.lit(0))
                ).alias("predicted_positive_users"),
            )
        )
        monthly = monthly.join(pred_enriched, on="deposit_month", how="left")
    else:
        monthly = (
            monthly
            .withColumn("avg_predicted_positive_rate", F.lit(None).cast("double"))
            .withColumn("predicted_positive_users", F.lit(None).cast("double"))
        )

    monthly_publish = publish_table(
        spark=spark,
        df=monthly.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_FUNNEL_REPORT_PREFIX}_funnel_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["deposit_month", "run_ts"],
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
        eval_report_cols = [
            "split_name",
            "split_strategy",
            "holdout_months",
            "n_rows",
            "positive_rate",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
        eval_payload = eval_df.select(*[column for column in eval_report_cols if column in eval_df.columns])
        eval_publish = publish_table(
            spark=spark,
            df=eval_payload.withColumn("created_at", F.current_timestamp()),
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
