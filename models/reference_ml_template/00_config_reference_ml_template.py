# Databricks notebook source
# MAGIC %run ../00_config

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

SUBJECT_NAME = "reference_ml_template"
WRITE_SCHEMA_NAME = SUBJECT_NAME
WRITE_SCHEMA = get_write_schema(SUBJECT_NAME)

REPORT_DIR = get_report_dir(SUBJECT_NAME)
REPORT_SPACE = get_report_space(SUBJECT_NAME)
MODEL_SPACE_PREFIX = get_model_space_prefix(SUBJECT_NAME)

ANALYSIS_NAMES = {
    "analysis_1": "baseline",
}

# Feature + output tables
T_REFERENCE_FEATURES_PREFIX = f"{WRITE_SCHEMA}.feat"
T_REFERENCE_PREDICTIONS_PREFIX = f"{WRITE_SCHEMA}.pred"

# Modeling windows
FEATURE_LOOKBACK_DAYS = 90
LABEL_WINDOW_DAYS = 7

# Model defaults
DEFAULT_CANDIDATE_ALIAS = "challenger"
DEFAULT_CHAMPION_ALIAS = "champion"
DEFAULT_RANDOM_SEED = 42


def get_reference_feature_columns(table_df):
    metadata_cols = {"run_id", "run_ts", "ref_date", "analysis_name", "created_at"}
    excluded = metadata_cols | {"unique_user_id", "label_active_7d", "user_sk"}
    cols = []
    for f in table_df.schema.fields:
        if f.name in excluded:
            continue
        if f.dataType.typeName() in {"byte", "short", "integer", "long", "float", "double", "decimal"}:
            cols.append(f.name)
    return cols


def prepare_reference_model_input(pdf, feature_cols):
    import pandas as pd

    X = pdf[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X
