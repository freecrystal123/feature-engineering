# Databricks notebook source
# MAGIC %run ../00_config

# COMMAND ----------

# Domain-specific settings
SUBJECT_NAME = "user_segmentation"
WRITE_SCHEMA_NAME = SUBJECT_NAME
WRITE_SCHEMA = get_write_schema(SUBJECT_NAME)

# Output tables
T_GOLD_CUSTOMER_FEATURES_PREFIX = f"{WRITE_SCHEMA}.feat"
T_GOLD_CUSTOMER_PCA_COMP_PREFIX = f"{WRITE_SCHEMA}.pca"
T_GOLD_CUSTOMER_PREDICTED_PREFIX = f"{WRITE_SCHEMA}.pred"
T_GOLD_CUSTOMER_EVAL_SCORE = f"{WRITE_SCHEMA}.eval_score"
T_GOLD_CUSTOMER_CLUSTERS_PREFIX = f"{WRITE_SCHEMA}.cluster"
T_GOLD_CUSTOMER_SUGGESTIONS_PREFIX = f"{WRITE_SCHEMA}.suggestion"

REPORT_DIR = get_report_dir(SUBJECT_NAME)
REPORT_SPACE = get_report_space(SUBJECT_NAME)
MODEL_SPACE_PREFIX = get_model_space_prefix(SUBJECT_NAME)

# Keep a reference to the shared helper before defining analysis-level wrappers
_shared_get_model_uri = get_model_uri

FIXED_KMEANS_K = 5  # None for testing all K's

# COMMAND ----------

ANALYSIS_NAMES = {
    "analysis_1": "online_casino",
    "analysis_2": "live_casino",
    "analysis_3": "sports",
    "analysis_4": "ols",
    "analysis_6": "online_live_casino_ap2",
}

# Parameters
GENERAL_DAYS_BEFORE = 365
DAYS_RECENT = 90

ONLINE_CASINO_CATEGORY_IDS = [1, 99]
LIVE_CASINO_CATEGORY_IDS = [3, 99]
SPORTS_CATEGORY_IDS = [6, 99]
GAME_CATS_ALLOCATION = [1, 3, 6]

ONLINE_FILTER = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*ONLINE_CASINO_CATEGORY_IDS)))
LIVE_FILTER = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*LIVE_CASINO_CATEGORY_IDS)))
SPORTS_FILTER = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*SPORTS_CATEGORY_IDS)))

ANALYSIS_FILTERS = {
    "analysis_1": ONLINE_FILTER,
    "analysis_2": LIVE_FILTER,
    "analysis_3": SPORTS_FILTER,
    "analysis_4": (ONLINE_FILTER | LIVE_FILTER | SPORTS_FILTER),
    "analysis_5": (ONLINE_FILTER | LIVE_FILTER),
    "analysis_6": (ONLINE_FILTER | LIVE_FILTER),
}

# COMMAND ----------

def get_features_table(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    return spark.table(f"{T_GOLD_CUSTOMER_FEATURES_PREFIX}_{analysis_name}")


def get_pca_table(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    return spark.table(f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}")


def get_predicted_table(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    return spark.table(f"{T_GOLD_CUSTOMER_PREDICTED_PREFIX}_{analysis_name}")


def get_eval_score_table():
    return spark.table(f"{T_GOLD_CUSTOMER_EVAL_SCORE}")


def get_clusters_table(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    return spark.table(f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}")


def get_suggestions_table(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    return spark.table(f"{T_GOLD_CUSTOMER_SUGGESTIONS_PREFIX}_{analysis_name}")


# COMMAND ----------

def get_model_uri(analysis: str, version: int | None = None, alias: str | None = None):
    analysis_name = ANALYSIS_NAMES[analysis]
    return _shared_get_model_uri(
        SUBJECT_NAME,
        analysis_name,
        version=version,
        alias=alias,
    )


def get_model_uri_from_name(full_model_name: str, version: int | None = None, alias: str | None = None):
    if version is not None:
        return f"{MODEL_URI_PREFIX}{full_model_name}/{int(version)}"
    if alias:
        return f"{MODEL_URI_PREFIX}{full_model_name}@{alias}"
    return f"{MODEL_URI_PREFIX}{full_model_name}"
