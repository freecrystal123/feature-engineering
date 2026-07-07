# Databricks notebook source
# MAGIC %run ../00_config

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

SUBJECT_NAME = "active_user_prediction"
WRITE_SCHEMA = get_write_schema(SUBJECT_NAME)

REPORT_DIR         = get_report_dir(SUBJECT_NAME)
REPORT_SPACE       = get_report_space(SUBJECT_NAME)
MODEL_SPACE_PREFIX = get_model_space_prefix(SUBJECT_NAME)

# Source tables not in the global 00_config
T_WAGER_HISTORY = "prod_silver.wager.wager_history"   # contains wager_winning_status
WAGER_WIN_STATUS = 3   # wager_winning_status = 3 → Won

# Table prefixes — all in WRITE_SCHEMA (ml_train for train/dev, ml_prod for prod)
T_ACTIVE_USER_FEATURES_PREFIX  = f"{WRITE_SCHEMA}.feat"
T_ACTIVE_USER_PREDICTED_PREFIX = f"{WRITE_SCHEMA}.pred"
T_ACTIVE_USER_SUMMARY_PREFIX   = f"{WRITE_SCHEMA}.summary"

# COMMAND ----------

ANALYSIS_NAMES = {
    "analysis_1": "online_casino",
    "analysis_2": "live_casino",
    "analysis_3": "sports",
    "analysis_4": "all_verticals",
}

ANALYSIS_FILTERS = {
    "analysis_1": ((F.col("source_system") == "TRUEWIN") & (F.col("game_category_id").isin(1, 99))),
    "analysis_2": ((F.col("source_system") == "TRUEWIN") & (F.col("game_category_id").isin(3, 99))),
    "analysis_3": ((F.col("source_system") == "TRUEWIN") & (F.col("game_category_id").isin(6, 99))),
    "analysis_4": (F.col("source_system") == "TRUEWIN"),
}

TRAINING_WEEKS_LOOKBACK = 12

# COMMAND ----------

def get_analyses_to_run():
    aid = dbutils.widgets.get("analysis_id").strip()
    if aid.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if aid not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{aid}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [aid]
