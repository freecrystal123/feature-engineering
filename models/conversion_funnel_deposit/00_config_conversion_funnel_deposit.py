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
    "analysis_2": "register_to_deposit_30d_truewin",
    "analysis_3": "register_to_deposit_30d_igaming",
    "analysis_4": "register_to_deposit_30d_truewin_no_ekyc",
    "analysis_5": "register_to_deposit_30d_truewin_no_ekyc_no_remarks_coupon",
}

# Per-analysis source_system filters applied in feature engineering publish step.
# Rules:
# - include: when provided, keep only these systems
# - exclude: remove these systems
ANALYSIS_SOURCE_SYSTEM_FILTERS = {
    "analysis_1": {"include": None, "exclude": []},
    "analysis_2": {"include": ["TRUEWIN"], "exclude": []},
    "analysis_3": {"include": None, "exclude": ["LOTTERY"]},
    "analysis_4": {"include": ["TRUEWIN"], "exclude": []},
    "analysis_5": {"include": ["TRUEWIN"], "exclude": []},
}

# Per-analysis feature exclusions on top of EXCLUDED_MODEL_FEATURES.
# Used to create controlled training variants without changing feature engineering.
NO_EKYC_EXCLUDED_FEATURES = {
    # Demographic fields sourced from verification profile.
    "customer_age_years",
    "customer_nationality",
    "customer_nationality_is_unknown",
    "customer_nationality_is_other",
}

NO_EKYC_EXCLUDED_PREFIXES = (
    # Remove all one-hot columns derived from verification nationality.
    "customer_nationality_is_",
)

NO_EKYC_EXCLUDED_SUBSTRINGS = (
    # Remove verification/eKYC/KYC related features.
    "ekyc",
    "kyc",
    "verification",
    "verify",
    "id_card",
    "birth_date",
)

COUPON_FEATURE_PREFIXES = (
    # Direct coupon features are reserved for analysis_5.
    "coupon_",
    "num_coupon_",
    "num_distinct_coupon",
    "has_coupon_",
    "sum_coupon_",
    "avg_coupon_",
    "max_coupon_",
    "first_coupon_",
)

ANALYSIS_EXTRA_EXCLUDED_FEATURES = {
    # Truewin variant without eKYC/KYC/verification-derived features.
    "analysis_4": set(NO_EKYC_EXCLUDED_FEATURES),
    # Same as analysis_4, but with no user_remarks-derived inputs and direct coupon features.
    "analysis_5": set(NO_EKYC_EXCLUDED_FEATURES),
}

ANALYSIS_EXTRA_EXCLUDED_PREFIXES = {
    "analysis_1": COUPON_FEATURE_PREFIXES,
    "analysis_2": COUPON_FEATURE_PREFIXES,
    "analysis_3": COUPON_FEATURE_PREFIXES,
    "analysis_4": NO_EKYC_EXCLUDED_PREFIXES + COUPON_FEATURE_PREFIXES,
    "analysis_5": NO_EKYC_EXCLUDED_PREFIXES,
}

ANALYSIS_EXTRA_EXCLUDED_SUBSTRINGS = {
    "analysis_4": NO_EKYC_EXCLUDED_SUBSTRINGS,
    "analysis_5": NO_EKYC_EXCLUDED_SUBSTRINGS + ("remark", "user_remarks"),
}

# Feature + output tables
T_FUNNEL_FEATURES_PREFIX = f"{WRITE_SCHEMA}.feat"
T_FUNNEL_PREDICTIONS_PREFIX = f"{WRITE_SCHEMA}.pred"
T_FUNNEL_EVAL_PREFIX = f"{WRITE_SCHEMA}.eval"
T_FUNNEL_SHAP_PREFIX = f"{WRITE_SCHEMA}.shap"
T_FUNNEL_REPORT_PREFIX = f"{WRITE_SCHEMA}.rpt"

# Additional SILVER sources for this analysis
SILVER_USER = f"{READ_SILVER_DB}.user.user"
SILVER_USER_REGISTRATION = f"{READ_SILVER_DB}.user.user_registration"
SILVER_USER_PROFILE = f"{READ_SILVER_DB}.user.user_profile"
SILVER_USER_GROUP = f"{READ_SILVER_DB}.user.user_group"
SILVER_USER_LOGIN = f"{READ_SILVER_DB}.user.user_login"
SILVER_USER_LOGIN_LOG = f"{READ_SILVER_DB}.user.user_login_log"
SILVER_USER_VERIFICATION = f"{READ_SILVER_DB}.user.user_verification"
SILVER_USER_VERIFICATION_LOG = f"{READ_SILVER_DB}.user.user_verification_log"
SILVER_USER_REMARK = f"{READ_SILVER_DB}.user.user_remark"
SILVER_WALLET_DEPOSIT = f"{READ_SILVER_DB}.wallet.deposit"
SILVER_WALLET_COUPON = f"{READ_SILVER_DB}.wallet.coupon"
SILVER_WALLET_WALLET = f"{READ_SILVER_DB}.wallet.wallet"
SILVER_WALLET_TRANSACTION_ORDER = f"{READ_SILVER_DB}.wallet.wallet_transaction_order"

# Cohort + modeling windows
COHORT_LOOKBACK_MONTHS = 12
# Legacy name kept for compatibility:
# used as minimum observation/follow-up days and as fallback for predeposit
# feature cutoffs. Deposit conversion labels are computed as-of run date
# (not capped to this 30-day window).
LABEL_WINDOW_DAYS = 30
EARLY_ACTIVITY_HOURS = 24
VERIFICATION_WINDOW_DAYS = 7
REMARK_WINDOW_DAYS = 7
WALLET_WINDOW_DAYS = 7
EKYC_JOURNEY_WINDOW_DAYS = 30
SOURCE_SYSTEM_TOP_K = 10
VERIFICATION_COUNTRY_TOP_K = 10
# Fixed global list for nationality one-hot encoding in model features.
# Values are normalized to uppercase in feature engineering.
CUSTOMER_NATIONALITY_TOP_COUNTRIES_GLOBAL = [
    "INDIA",
    "PHILIPPINES",
    "PAKISTAN",
    "NEPAL",
    "UNITED ARAB EMIRATES",
    "BANGLADESH",
    "SRI LANKA",
    "EGYPT",
    "SYRIA",
    "UNITED KINGDOM",
    "LEBANON",
    "IRAN",
    "UNITED STATES",
    "MYANMAR",
    "JORDAN",
    "MOROCCO",
    "ETHIOPIA",
    "SUDAN",
    "SOUTH AFRICA",
    "GHANA",
]

# Model defaults
DEFAULT_CANDIDATE_ALIAS = "challenger"
DEFAULT_CHAMPION_ALIAS = "champion"
DEFAULT_RANDOM_SEED = 42

# Demographic deep-dive defaults (used by notebook/report generation)
DEMOGRAPHIC_TOP_FEATURES_DEFAULT = 12
DEMOGRAPHIC_MAX_SEGMENTS = 12
DEMOGRAPHIC_NUMERIC_BINS = 5
DEMOGRAPHIC_CLASSIC_FEATURES = [
    "registration_hour",
    "registration_weekday",
    "registration_day_of_month",
    "registration_weekend_flag",
    "registered_by_mobile",
    "registered_by_desktop",
    "days_since_registration",
    "customer_age_years",
    "customer_nationality",
]

# Keep calendar and device flags as exact values in demographic segmentation
# (no numeric quantile bins), so report visuals stay hour/day level.
DEMOGRAPHIC_FORCE_DISCRETE_FEATURES = {
    "registration_hour",
    "registration_weekday",
    "registration_day_of_month",
    "registration_weekend_flag",
    "registered_by_mobile",
    "registered_by_desktop",
    "customer_nationality",
}


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
    "country_id",
    "resident_country_id",
    # Demographic-only (kept for report diagnostics; not for model fitting).
    "customer_nationality",
    # Potential leakage proxy: unknown nationality is strongly tied to failed/no eKYC.
    "customer_nationality_is_unknown",
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
    # Legacy wallet features removed from feature engineering (kept here to avoid
    # stale historical columns being picked during training from merged schemas).
    "num_wallet_events_7d",
    "has_wallet_event_7d",
    "wallet_frozen_max_7d",
    "wallet_frozen_avg_7d",
    "first_wallet_event_delay_hours_7d",
    "has_wallet_frozen_positive_7d",
    # Leakage guard: `betrecharge` is the deposit transaction itself.
    "num_wallet_txn_trans_type_betrecharge_predeposit_30d",
    "has_wallet_txn_trans_type_betrecharge_predeposit_30d",
    # Wallet transaction types too sparse to be useful (near-all zeros).
    "num_wallet_txn_trans_type_eventdecrease_predeposit_30d",
    "has_wallet_txn_trans_type_eventdecrease_predeposit_30d",
    "num_wallet_txn_trans_type_eventrecharge_predeposit_30d",
    "has_wallet_txn_trans_type_eventrecharge_predeposit_30d",
    "num_wallet_txn_trans_type_decreaserollback_predeposit_30d",
    "has_wallet_txn_trans_type_decreaserollback_predeposit_30d",
    "num_wallet_txn_trans_type_decreasemerchant_predeposit_30d",
    "has_wallet_txn_trans_type_decreasemerchant_predeposit_30d",
    "num_wallet_txn_trans_type_prizeevent_predeposit_30d",
    "has_wallet_txn_trans_type_prizeevent_predeposit_30d",
    "num_wallet_txn_trans_type_decrease_predeposit_30d",
    "has_wallet_txn_trans_type_decrease_predeposit_30d",
    "num_wallet_txn_trans_type_prize_predeposit_30d",
    "has_wallet_txn_trans_type_prize_predeposit_30d",
    # Remark features with constant value (no signal).
    "has_user_remark_7d",
    "has_remark_account_created_7d",
    "has_remark_flow_step_account_created_7d",
    "num_remark_account_created_7d",
    "num_remark_flow_step_account_created_7d",
    # High-missing remark delay features.
    "first_remark_customer_request_delay_hours_7d",
    "first_remark_ekyc_failed_delay_hours_7d",
    "first_remark_lock_delay_hours_7d",
    # Constant/no-signal features from current exploration.
    "has_kyc_attempt_7d",
    "has_sign_id_card_verification_7d",
    "has_verification_fail_reason_7d",
    "login_status_id",
    "num_distinct_id_card_types_7d",
    "num_distinct_verification_ip_7d",
}


def _resolve_analysis_key(*, analysis_id: str | None = None, analysis_name: str | None = None) -> str | None:
    if analysis_id in ANALYSIS_NAMES:
        return analysis_id
    if analysis_name is not None:
        for _k, _v in ANALYSIS_NAMES.items():
            if _v == analysis_name:
                return _k
    return None


def _analysis_specific_excluded_features(table_df, *, analysis_id: str | None = None, analysis_name: str | None = None) -> set[str]:
    key = _resolve_analysis_key(analysis_id=analysis_id, analysis_name=analysis_name)
    if key is None:
        return set()

    exact = set(ANALYSIS_EXTRA_EXCLUDED_FEATURES.get(key, set()))
    prefixes = tuple(str(p).lower() for p in ANALYSIS_EXTRA_EXCLUDED_PREFIXES.get(key, tuple()))
    substrings = tuple(str(s).lower() for s in ANALYSIS_EXTRA_EXCLUDED_SUBSTRINGS.get(key, tuple()))

    if not exact and not prefixes and not substrings:
        return set()

    out = set()
    for f in table_df.schema.fields:
        name = f.name
        lname = name.lower()
        if name in exact:
            out.add(name)
            continue
        if prefixes and any(lname.startswith(p) for p in prefixes):
            out.add(name)
            continue
        if substrings and any(s in lname for s in substrings):
            out.add(name)
            continue
    return out


def get_funnel_feature_columns(table_df, *, analysis_id: str | None = None, analysis_name: str | None = None):
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
    analysis_extra_excluded = _analysis_specific_excluded_features(
        table_df,
        analysis_id=analysis_id,
        analysis_name=analysis_name,
    )
    excluded = metadata_cols | key_cols | leak_cols | EXCLUDED_MODEL_FEATURES | analysis_extra_excluded

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
