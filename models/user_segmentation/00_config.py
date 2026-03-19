# Databricks notebook source
ENVIRONMENT = "uat"

# COMMAND ----------


if ENVIRONMENT == "dev":
    READ_ENVIRONMENT = "dev"
else:
    READ_ENVIRONMENT = "prod"

WRITE_ENVIRONMENT = ENVIRONMENT

READ_SILVER_DB = f"{READ_ENVIRONMENT}_silver"
READ_GOLD_DB   = f"{READ_ENVIRONMENT}_gold"

WRITE_CATALOG   = f"{ENVIRONMENT}_analytics"
WRITE_SCHEMA_NAME = "user_segmentation"
# WRITE_CATALOG   = f"{ENVIRONMENT}_gold"
# WRITE_SCHEMA_NAME = "machine_learning"


WRITE_SCHEMA = f"{WRITE_CATALOG}.{WRITE_SCHEMA_NAME}"


# Gold tables
T_WAGER             = f"{READ_GOLD_DB}.finance.fact_wager"
T_SETTLED_WAGER       = f"{READ_GOLD_DB}.finance.fact_settled_wager"
T_RTP       =           f"{READ_GOLD_DB}.finance.fact_rtp"
DIM_WAGER             = f"{READ_GOLD_DB}.shared.dim_wager"
T_DEPOSIT           = f"{READ_GOLD_DB}.finance.fact_deposit"
T_WITHDRAW          = f"{READ_GOLD_DB}.finance.fact_withdrawal"
DIM_COUPON            = f"{READ_GOLD_DB}.shared.dim_coupon"
DIM_GAME            = f"{READ_GOLD_DB}.shared.dim_game"
T_USER              = f"{READ_GOLD_DB}.shared.dim_user"

REFERENCE_DATA_STATUS = f"{READ_SILVER_DB}.reference_data.status_master"

# Output (Gold)
T_GOLD_CUSTOMER_FEATURES_PREFIX = f"{WRITE_SCHEMA}.feat"
T_GOLD_CUSTOMER_PCA_COMP_PREFIX = f"{WRITE_SCHEMA}.pca"
T_GOLD_CUSTOMER_PREDICTED_PREFIX = f"{WRITE_SCHEMA}.pred"
T_GOLD_CUSTOMER_EVAL_SCORE = f"{WRITE_SCHEMA}.eval_score"
T_GOLD_CUSTOMER_CLUSTERS_PREFIX = f"{WRITE_SCHEMA}.cluster"
T_GOLD_CUSTOMER_SUGGESTIONS_PREFIX = f"{WRITE_SCHEMA}.suggestion"



REPORTS_FOLDER_NAME = 'segmentation_reports'
EXPERIMENT_FOLDER_NAME = 'segmentation_experiments'

REPORT_DIR = f"/Volumes/{WRITE_CATALOG}/{WRITE_SCHEMA_NAME}/{REPORTS_FOLDER_NAME}"
REPORT_SPACE = f"{WRITE_CATALOG}.{WRITE_SCHEMA_NAME}.{REPORTS_FOLDER_NAME}"

EXPERIMENT_DIR = f"/Volumes/{WRITE_CATALOG}/{WRITE_SCHEMA_NAME}/{EXPERIMENT_FOLDER_NAME}"
EXPERIMENT_SPACE = f"{WRITE_CATALOG}.{WRITE_SCHEMA_NAME}.{EXPERIMENT_FOLDER_NAME}"

MODEL_SPACE_PREFIX = f"{WRITE_CATALOG}.{WRITE_SCHEMA_NAME}.mdl"
MODEL_URI_PREFIX = f"models:/"

MODEL_FULL_NAME = "uat_analytics.user_segmentation.mdl_online_live_casino_ap2"
model_uri = f"models:/{MODEL_FULL_NAME}/9"

FIXED_KMEANS_K = 5  # None for testing all K's


from pyspark.sql import functions as F
# CURRENT_DATE = F.lit("2026-01-19").cast("date")
CURRENT_DATE = F.current_date()


# COMMAND ----------

# ANALYSIS_NAMES = { "analysis_1": "online_casino",
#                    "analysis_2": "live_casino",
#                    "analysis_3": "sports",
#                    "analysis_4": "lottery" }
# ANALYSIS_NAMES = {"analysis_5":"online_live_casino"}
ANALYSIS_NAMES = {
    "analysis_1": "online_casino",
    "analysis_2": "live_casino",
    "analysis_3": "sports",
    "analysis_4": "ols"
}
# ANALYSIS_NAMES = {
#     "analysis_6":"online_live_casino_ap2"
# }
    



# COMMAND ----------

# Parameters
GENERAL_DAYS_BEFORE = 365 # days
DAYS_RECENT = 90

ONLINE_CASINO_CATEGORY_IDS = [1, 99]
LIVE_CASINO_CATEGORY_IDS = [3, 99]
SPORTS_CATEGORY_IDS = [6, 99]
GAME_CATS_ALLOCATION = [1, 3, 6]

# Combined scope = Online Casino + Live Casino (excluding Lottery)
# ONLINE_FILTER = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*ONLINE_CASINO_CATEGORY_IDS)))
# LIVE_FILTER   = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*LIVE_CASINO_CATEGORY_IDS)))

ONLINE_FILTER = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*ONLINE_CASINO_CATEGORY_IDS)))
LIVE_FILTER   = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*LIVE_CASINO_CATEGORY_IDS)))
SPORTS_FILTER   = ((F.col("source_system") != "LOTTERY") & (F.col("game_category_id").isin(*SPORTS_CATEGORY_IDS)))



ANALYSIS_FILTERS = {
    "analysis_1": ONLINE_FILTER,
    "analysis_2": LIVE_FILTER,
    "analysis_3": SPORTS_FILTER,
    "analysis_4": (ONLINE_FILTER | LIVE_FILTER | SPORTS_FILTER),
    # "analysis_5": (ONLINE_FILTER | LIVE_FILTER),  # APPROACH 1
    # "analysis_6": (ONLINE_FILTER | LIVE_FILTER)     # APPROACH 2
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

def get_model_uri(analysis: str, version: int):
    analysis_name = ANALYSIS_NAMES[analysis]

    return f"{MODEL_URI_PREFIX}{MODEL_SPACE_PREFIX}_{analysis_name}/{version}"

