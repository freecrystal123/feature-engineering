# Databricks notebook source
# MAGIC %md
# MAGIC ## Config – Lottery Net Turnover Prediction
# MAGIC
# MAGIC **Revenue formula:** Net Turnover = Settled Wager Amount − VAT
# MAGIC
# MAGIC | Source | Table | Key columns |
# MAGIC |---|---|---|
# MAGIC | Settled wager | `FACT_SETTLED_WAGER` | `date_sk`, `source_system`, `wager_amount_native` |
# MAGIC | VAT | `FACT_WAGER` | `date_sk`, `source_system`, `vat_on_wager_amount_native` |

# COMMAND ----------

# MAGIC %run ../00_config

# COMMAND ----------

# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

SUBJECT_NAME = "lottery_net_turnover_prediction"
WRITE_SCHEMA = get_write_schema(SUBJECT_NAME)

REPORT_DIR         = get_report_dir(SUBJECT_NAME)
REPORT_SPACE       = get_report_space(SUBJECT_NAME)
MODEL_SPACE_PREFIX = get_model_space_prefix(SUBJECT_NAME)

# Table prefixes — all in WRITE_SCHEMA
T_LOTTERY_FEATURES_PREFIX  = f"{WRITE_SCHEMA}.feat"
T_LOTTERY_PREDICTED_PREFIX = f"{WRITE_SCHEMA}.pred"
T_LOTTERY_SUMMARY_PREFIX   = f"{WRITE_SCHEMA}.summary"

# Source system identifier for LOTTERY data
SOURCE_SYSTEM = "LOTTERY"
REVENUE_COL   = "revenue"

TRAINING_WEEKS_LOOKBACK = 12

# COMMAND ----------

ANALYSIS_NAMES = {
    "analysis_1": "lottery_all",
}

ANALYSIS_FILTERS = {
    "analysis_1": (F.col("source_system") == SOURCE_SYSTEM),
}

# COMMAND ----------

def get_analyses_to_run():
    aid = dbutils.widgets.get("analysis_id").strip()
    if aid.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if aid not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{aid}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [aid]
