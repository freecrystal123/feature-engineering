# Databricks notebook source
# MAGIC %run ../00_config

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

SUBJECT_NAME = "conversion_funnel_deposit"
WRITE_SCHEMA_NAME = SUBJECT_NAME
WRITE_SCHEMA = get_write_schema(SUBJECT_NAME)

REPORT_DIR = get_report_dir(SUBJECT_NAME)
REPORT_SPACE = get_report_space(SUBJECT_NAME)
MODEL_SPACE_PREFIX = get_model_space_prefix(SUBJECT_NAME)

ANALYSIS_NAMES = {
    "analysis_1": "register_to_deposit_30d",
}

# Feature + output tables
T_FUNNEL_FEATURES_PREFIX = f"{WRITE_SCHEMA}.feat"
T_FUNNEL_PREDICTIONS_PREFIX = f"{WRITE_SCHEMA}.pred"
T_FUNNEL_EVAL_PREFIX = f"{WRITE_SCHEMA}.eval"
T_FUNNEL_SHAP_PREFIX = f"{WRITE_SCHEMA}.shap"
T_FUNNEL_REPORT_PREFIX = f"{WRITE_SCHEMA}.rpt"

# Additional SILVER sources for this analysis
SILVER_USER = f"{READ_SILVER_DB}.user.user"
SILVER_USER_PROFILE = f"{READ_SILVER_DB}.user.user_profile"
SILVER_USER_GROUP = f"{READ_SILVER_DB}.user.user_group"
SILVER_USER_LOGIN = f"{READ_SILVER_DB}.user.user_login"
SILVER_USER_LOGIN_LOG = f"{READ_SILVER_DB}.user.user_login_log"
SILVER_USER_VERIFICATION = f"{READ_SILVER_DB}.user.user_verification"
SILVER_WALLET_DEPOSIT = f"{READ_SILVER_DB}.wallet.deposit"

# Cohort + modeling windows
COHORT_LOOKBACK_MONTHS = 12
LABEL_WINDOW_DAYS = 30
EARLY_ACTIVITY_HOURS = 24
VERIFICATION_WINDOW_DAYS = 7
SOURCE_SYSTEM_TOP_K = 10

# Model defaults
DEFAULT_CANDIDATE_ALIAS = "challenger"
DEFAULT_CHAMPION_ALIAS = "champion"
DEFAULT_RANDOM_SEED = 42


# Features kept in the feature table for diagnostics, but excluded from model fitting.
EXCLUDED_MODEL_FEATURES = {
    # Superseded by source_system one-hot features.
    "source_system_hash",
    # Constant / near-constant / no-signal columns found in feature distribution review.
    "username_length",
    "sale_status_id",
    "user_type_id",
    "profile_platform_id",
    "profile_os_type_id",
    "resident_country_id",
    "distributor_status_id",
    "enable_status_id",
    "has_sign_login_name",
    "verified_in_7d",
    "has_verification_event_7d",
    "has_marketing_channel",
    "has_marketing_source",
    "has_referrer",
    "has_referrer_code",
    "has_device_code",
    "has_user_agent",
    "has_imei",
    "has_idfa",
    "has_android_id",
    "has_cookie_data",
    "num_test_groups",
    "has_any_group",
    # Redundant with kept counterparts.
    "has_mail",
    "has_phone",
    "num_non_test_groups",
    "num_active_login_methods",
    "num_third_party_login_methods",
}


def get_funnel_feature_columns(table_df):
    metadata_cols = {"run_id", "run_ts", "ref_date", "analysis_name", "created_at"}
    key_cols = {"unique_user_id", "user_id", "source_system"}
    leak_cols = {
        "label_deposit_success_30d",
        "label_deposit_attempt_30d",
        "label_failed_after_attempt_30d",
        "num_deposit_attempts_30d",
        "num_deposit_success_30d",
        "num_deposit_failure_30d",
        "num_deposit_pending_30d",
        "num_deposit_channels_30d",
        "num_deposit_modes_30d",
        "sum_deposit_amount_attempted_30d",
        "avg_deposit_amount_attempted_30d",
        "first_attempt_delay_hours_30d",
        "first_success_delay_hours_30d",
        "observation_complete",
    }
    leak_prefixes = (
        "label_",
        "num_deposit_",
        "sum_deposit_",
        "avg_deposit_",
        "first_attempt_delay_",
        "first_success_delay_",
    )
    excluded = metadata_cols | key_cols | leak_cols | EXCLUDED_MODEL_FEATURES

    cols = []
    for f in table_df.schema.fields:
        if f.name in excluded:
            continue
        if any(f.name.startswith(prefix) for prefix in leak_prefixes):
            continue
        if f.dataType.typeName() in {"byte", "short", "integer", "long", "float", "double", "decimal", "boolean"}:
            cols.append(f.name)
    return cols


def prepare_funnel_model_input(pdf, feature_cols):
    import pandas as pd

    X = pdf[feature_cols].copy()
    for c in feature_cols:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("int64")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X
