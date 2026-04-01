# Databricks notebook source
# Shared config across ML analyses

ENVIRONMENT = "uat"

if ENVIRONMENT == "dev":
    READ_ENVIRONMENT = "dev"
else:
    READ_ENVIRONMENT = "prod"

WRITE_ENVIRONMENT = ENVIRONMENT

READ_SILVER_DB = f"{READ_ENVIRONMENT}_silver"
READ_GOLD_DB = f"{READ_ENVIRONMENT}_gold"

# Keep writes environment-scoped by default
WRITE_CATALOG = f"{WRITE_ENVIRONMENT}_analytics"

# Shared folder conventions
REPORTS_FOLDER_NAME = "reports"

# Shared source tables
T_WAGER = f"{READ_GOLD_DB}.finance.fact_wager"
T_SETTLED_WAGER = f"{READ_GOLD_DB}.finance.fact_settled_wager"
T_RTP = f"{READ_GOLD_DB}.finance.fact_rtp"
DIM_WAGER = f"{READ_GOLD_DB}.shared.dim_wager"
T_DEPOSIT = f"{READ_GOLD_DB}.finance.fact_deposit"
T_WITHDRAW = f"{READ_GOLD_DB}.finance.fact_withdrawal"
DIM_COUPON = f"{READ_GOLD_DB}.shared.dim_coupon"
DIM_GAME = f"{READ_GOLD_DB}.shared.dim_game"
T_USER = f"{READ_GOLD_DB}.shared.dim_user"

REFERENCE_DATA_STATUS = f"{READ_SILVER_DB}.reference_data.status_master"

MODEL_URI_PREFIX = "models:/"

from pyspark.sql import functions as F
CURRENT_DATE = F.current_date()


def get_write_schema(subject_name: str) -> str:
    return f"{WRITE_CATALOG}.{subject_name}"


def get_report_space(subject_name: str) -> str:
    return f"{WRITE_CATALOG}.{subject_name}.{REPORTS_FOLDER_NAME}"


def get_report_dir(subject_name: str) -> str:
    return f"/Volumes/{WRITE_CATALOG}/{subject_name}/{REPORTS_FOLDER_NAME}"


def get_model_space_prefix(subject_name: str) -> str:
    return f"{WRITE_CATALOG}.{subject_name}.mdl"


def get_model_uri(subject_name: str, model_name: str, *, version: int | None = None, alias: str | None = None) -> str:
    full_name = f"{get_model_space_prefix(subject_name)}_{model_name}"
    if version is not None:
        return f"{MODEL_URI_PREFIX}{full_name}/{int(version)}"
    if alias:
        return f"{MODEL_URI_PREFIX}{full_name}@{alias}"
    return f"{MODEL_URI_PREFIX}{full_name}"
