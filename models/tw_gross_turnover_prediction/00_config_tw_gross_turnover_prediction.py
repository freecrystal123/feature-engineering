# Databricks notebook source
# MAGIC %run ../00_config

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

SUBJECT_NAME = "tw_gross_turnover_prediction"
WRITE_SCHEMA = get_write_schema(SUBJECT_NAME)
REPORT_DIR   = get_report_dir(SUBJECT_NAME)
REPORT_SPACE = get_report_space(SUBJECT_NAME)
MODEL_SPACE_PREFIX = get_model_space_prefix(SUBJECT_NAME)

ANALYSIS_NAMES = {
    "analysis_1": "truewin_weekly",
}

SOURCE_SYSTEM = "TRUEWIN"

T_FEATURES_PREFIX    = f"{WRITE_SCHEMA}.feat"
T_INFER_PREFIX       = f"{WRITE_SCHEMA}.infer"
T_PREDICTIONS_PREFIX = f"{WRITE_SCHEMA}.pred"

LABEL_COL = "revenue"   # total weekly gross turnover for TRUEWIN

OUTLIER_DATE_SKS = [20240521]   # known bad data dates to exclude

DEFAULT_CANDIDATE_ALIAS = "challenger"
DEFAULT_CHAMPION_ALIAS  = "champion"
DEFAULT_RANDOM_SEED     = 42
