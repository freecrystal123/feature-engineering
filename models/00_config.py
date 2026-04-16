# Databricks notebook source
# Shared config across ML analyses

import os
import re


def normalize_environment(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if v in {"dev", "train", "prod"}:
        return v
    m = re.search(r"/(dev|train|prod)(?:/|$)", v)
    if m:
        return m.group(1)
    return None


def detect_environment(default: str = "train") -> str:
    env = normalize_environment(os.getenv("ENVIRONMENT"))
    if env:
        return env

    try:
        dbutils_obj = globals().get("dbutils")
        if dbutils_obj is not None:
            notebook_path = (
                dbutils_obj.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .notebookPath()
                .get()
            )
            env = normalize_environment(notebook_path)
            if env:
                return env
    except Exception:
        pass

    env = normalize_environment(os.getcwd())
    if env:
        return env

    return default


ENVIRONMENT = detect_environment(default="train")

if ENVIRONMENT == "dev":
    READ_ENVIRONMENT = "dev"
else:
    READ_ENVIRONMENT = "prod"

WRITE_ENVIRONMENT = ENVIRONMENT

READ_SILVER_DB = f"{READ_ENVIRONMENT}_silver"
READ_GOLD_DB = f"{READ_ENVIRONMENT}_gold"

# Keep writes environment-scoped by default
if WRITE_ENVIRONMENT == "prod":
    WRITE_CATALOG = "ml_prod"
elif WRITE_ENVIRONMENT in {"train", "dev"}:
    WRITE_CATALOG = "ml_train"
else:
    raise ValueError(f"Unsupported WRITE_ENVIRONMENT='{WRITE_ENVIRONMENT}'.")

# Shared folder conventions
REPORTS_FOLDER_NAME = "reports"

# Shared source tables
FACT_WAGER = f"{READ_GOLD_DB}.finance.fact_wager"
FACT_SETTLED_WAGER = f"{READ_GOLD_DB}.finance.fact_settled_wager"
FACT_RTP = f"{READ_GOLD_DB}.finance.fact_rtp"
DIM_WAGER = f"{READ_GOLD_DB}.shared.dim_wager"
FACT_DEPOSIT = f"{READ_GOLD_DB}.finance.fact_deposit"
FACT_WITHDRAWAL = f"{READ_GOLD_DB}.finance.fact_withdrawal"
DIM_COUPON = f"{READ_GOLD_DB}.shared.dim_coupon"
DIM_GAME = f"{READ_GOLD_DB}.shared.dim_game"
DIM_USER = f"{READ_GOLD_DB}.shared.dim_user"

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
