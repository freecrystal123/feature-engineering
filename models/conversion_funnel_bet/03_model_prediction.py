# Databricks notebook source
# MAGIC %md
# MAGIC ## 3 - Conversion funnel model prediction

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_bet

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------
# MAGIC %run ../../core/uc_feature_store

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("model_alias", DEFAULT_CHAMPION_ALIAS)
dbutils.widgets.text("model_catalog", "ml_train")
dbutils.widgets.text("use_score_batch", "true")

# COMMAND ----------

import json
import mlflow
from pyspark.sql import functions as F

analysis_id = dbutils.widgets.get("analysis_id").strip()
model_alias = dbutils.widgets.get("model_alias").strip()
model_catalog = dbutils.widgets.get("model_catalog").strip()
use_score_batch = dbutils.widgets.get("use_score_batch").strip().lower() in {"1", "true", "yes", "y", "on"}


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def _resolve_prediction_col(columns: list[str]) -> str | None:
    candidates = ["prediction", "predictions", "score", "predicted_label"]
    for c in candidates:
        if c in columns:
            return c
    return None


def _required_model_input_cols(model_uri: str) -> list[str]:
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    signature = getattr(getattr(pyfunc_model, "metadata", None), "signature", None)
    if signature is None or signature.inputs is None:
        return []

    inputs = signature.inputs
    try:
        cols = [c.name for c in inputs.inputs if getattr(c, "name", None)]
    except Exception:
        cols = [c.name for c in inputs if getattr(c, "name", None)]
    return cols


def _score_with_feature_engineering(model_uri: str, entity_df):
    fe = get_feature_engineering_client()
    return fe.score_batch(model_uri=model_uri, df=entity_df)


def predict_analysis(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    table_base = f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}"
    entity_source_table = current_table_name(table_base)

    model_name = build_model_name(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=model_catalog,
    )
    uri = f"models:/{model_name}@{model_alias}"

    if not use_score_batch:
        raise ValueError("This solution requires Feature Engineering score_batch. Set use_score_batch=true.")

    source_df = spark.table(entity_source_table)
    source_cols = set(source_df.columns)

    mandatory_lookup_cols = ["unique_user_id"]
    if "run_ts" in source_cols:
        mandatory_lookup_cols.append("run_ts")
    missing_lookup_cols = [c for c in mandatory_lookup_cols if c not in source_cols]
    if missing_lookup_cols:
        raise ValueError(
            f"Entity source table '{entity_source_table}' is missing required lookup columns: {missing_lookup_cols}"
        )

    entity_df = source_df.select(*mandatory_lookup_cols).dropDuplicates(mandatory_lookup_cols)
    try:
        pred_df = _score_with_feature_engineering(uri, entity_df)
    except Exception as exc:
        msg = str(exc)
        if "Feature names unseen at fit time" in msg or "feature names should match those that were passed during fit" in msg.lower():
            raise RuntimeError(
                "Prediction failed due to training-serving schema mismatch in this model version. "
                "This usually happens with old versions trained before the latest fix in 02_model_training. "
                "Retrain and score with the new alias/version."
            ) from exc
        raise
    pred_col = _resolve_prediction_col(pred_df.columns)
    if pred_col is None:
        raise ValueError(f"No prediction column returned by score_batch. Columns={pred_df.columns}")

    select_cols = [F.col("unique_user_id")]
    if "run_ts" in pred_df.columns:
        select_cols.append(F.col("run_ts"))
    select_cols.append(F.col(pred_col).cast("double").alias("predicted_label"))

    output_df = (
        pred_df
        .select(*select_cols)
        .withColumn("model_alias", F.lit(model_alias))
        .withColumn("created_at", F.current_timestamp())
    )
    publish_result = publish_table(
        spark=spark,
        df=output_df,
        table_base=f"{T_FUNNEL_PREDICTIONS_PREFIX}_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["unique_user_id", "run_ts"],
        timeseries_column="run_ts",
    )

    return {
        "analysis": analysis,
        "analysis_name": analysis_name,
        "model_uri": uri,
        "used_score_batch": True,
        "score_batch_error": None,
        "pred_current_table": publish_result["current_table"],
        "run_id": publish_result["run_id"],
    }


results = [predict_analysis(a) for a in get_analyses_to_run()]
print(json.dumps({"results": results}, indent=2, default=str))
