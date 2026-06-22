# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 - Conversion funnel feature engineering (deposit -> first wager)

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_bet

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

import re

from pyspark.sql import Window
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
analysis_id = dbutils.widgets.get("analysis_id").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


ANALYSES_TO_RUN = get_analyses_to_run()
REQUIRES_USER_REMARKS = any(a != "analysis_5" for a in ANALYSES_TO_RUN)


def _pick_col(df, candidates: list[str], label: str, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Column not found for {label}. Tried {candidates}. Available: {df.columns}")
    return None


def _present_flag(column_expr):
    return F.when(F.trim(F.coalesce(column_expr.cast("string"), F.lit(""))) != F.lit(""), F.lit(1)).otherwise(F.lit(0))


def _sanitize_source_system_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _normalize_status_int(column_expr):
    cleaned = F.regexp_extract(F.trim(F.coalesce(column_expr.cast("string"), F.lit(""))), r"([0-9]+)", 1)
    return F.when(cleaned == F.lit(""), F.lit(None).cast("int")).otherwise(cleaned.cast("int"))


def _quote_identifier(column_name: str) -> str:
    raw = str(column_name).strip()
    parts = [p.strip() for p in raw.split(".") if p.strip()]

    if not parts:
        return "``"

    quoted_parts = []
    for part in parts:
        clean = part
        if clean.startswith("`") and clean.endswith("`") and len(clean) >= 2:
            clean = clean[1:-1]
        quoted_parts.append(f"`{clean.replace('`', '``')}`")

    return ".".join(quoted_parts)


def _safe_to_timestamp(column_name: str):
    q = _quote_identifier(column_name)
    return F.coalesce(
        F.expr(f"try_to_timestamp({q})"),
        F.expr(f"try_to_timestamp({q}, 'yyyy-MM-dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp({q}, 'yyyy-MM-dd HH:mm:ss.SSS')"),
        F.expr(f"try_to_timestamp({q}, 'yyyy/MM/dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp({q}, 'dd/MM/yyyy HH:mm:ss')"),
        F.expr(f"try_to_timestamp({q}, 'yyyy-MM-dd')"),
        F.expr(f"try_to_timestamp({q}, 'yyyy/MM/dd')"),
        F.expr(f"try_to_timestamp({q}, 'dd/MM/yyyy')"),
    )


def _safe_to_date(column_name: str):
    q = _quote_identifier(column_name)
    return F.coalesce(
        F.expr(f"try_to_date({q})"),
        F.expr(f"try_to_date({q}, 'yyyy-MM-dd')"),
        F.expr(f"try_to_date({q}, 'yyyy/MM/dd')"),
        F.expr(f"try_to_date({q}, 'dd/MM/yyyy')"),
    )


def _apply_analysis_source_filter(df, analysis_id_value: str):
    rules = ANALYSIS_SOURCE_SYSTEM_FILTERS.get(analysis_id_value, {"include": None, "exclude": []})
    include = rules.get("include")
    exclude = rules.get("exclude", [])

    out = df
    if include:
        include_norm = [str(v).strip().upper() for v in include if str(v).strip()]
        if include_norm:
            out = out.filter(F.upper(F.col("source_system")).isin(include_norm))

    if exclude:
        exclude_norm = [str(v).strip().upper() for v in exclude if str(v).strip()]
        if exclude_norm:
            out = out.filter(~F.upper(F.col("source_system")).isin(exclude_norm))

    return out


NO_EKYC_REMARK_OVERRIDE_COL_MAP = {
    "num_user_remarks_7d": "__num_user_remarks_7d_no_ekyc",
    "num_distinct_remark_operation_types_7d": "__num_distinct_remark_operation_types_7d_no_ekyc",
    "num_distinct_remark_flow_steps_7d": "__num_distinct_remark_flow_steps_7d_no_ekyc",
    "has_user_remark_7d": "__has_user_remark_7d_no_ekyc",
    "first_user_remark_delay_hours_7d": "__first_user_remark_delay_hours_7d_no_ekyc",
}
NO_EKYC_REMARK_HELPER_COLUMNS = tuple(NO_EKYC_REMARK_OVERRIDE_COL_MAP.values())
USER_REMARK_DERIVED_COLUMN_SUBSTRINGS = ("remark", "user_remarks")

COUPON_FEATURE_DEFAULTS = {
    "num_coupons_predeposit_30d": 0.0,
    "num_distinct_coupons_predeposit_30d": 0.0,
    "has_coupon_predeposit_30d": 0,
    "sum_coupon_amount_predeposit_30d": 0.0,
    "avg_coupon_amount_predeposit_30d": None,
    "max_coupon_amount_predeposit_30d": None,
    "num_coupon_types_predeposit_30d": 0.0,
    "num_coupon_statuses_predeposit_30d": 0.0,
    "num_coupon_channels_predeposit_30d": 0.0,
    "num_coupon_templates_predeposit_30d": 0.0,
    "num_coupon_business_ids_predeposit_30d": 0.0,
    "num_coupon_event_ids_predeposit_30d": 0.0,
    "first_coupon_delay_hours_predeposit_30d": None,
}


def _drop_no_ekyc_remark_helper_cols(df):
    helper_cols = [c for c in NO_EKYC_REMARK_HELPER_COLUMNS if c in df.columns]
    if helper_cols:
        return df.drop(*helper_cols)
    return df


def _drop_user_remark_derived_cols(df):
    cols_to_drop = [
        c
        for c in df.columns
        if any(token in c.lower() for token in USER_REMARK_DERIVED_COLUMN_SUBSTRINGS)
    ]
    if cols_to_drop:
        return df.drop(*cols_to_drop)
    return df


def _with_default_features(df, feature_defaults: dict[str, object]):
    out = df
    for feature_name, feature_default in feature_defaults.items():
        if feature_default is None:
            out = out.withColumn(feature_name, F.lit(None).cast("double"))
        elif isinstance(feature_default, float):
            out = out.withColumn(feature_name, F.lit(feature_default).cast("double"))
        elif isinstance(feature_default, int):
            out = out.withColumn(feature_name, F.lit(feature_default).cast("int"))
        else:
            out = out.withColumn(feature_name, F.lit(feature_default))
    return out


def _apply_no_ekyc_remark_overrides(df):
    # Replace aggregate remark features using values recalculated from
    # row-level remark records after removing eKYC/KYC/system_face steps.
    out = df
    for target_col, helper_col in NO_EKYC_REMARK_OVERRIDE_COL_MAP.items():
        if target_col in out.columns and helper_col in out.columns:
            out = out.withColumn(target_col, F.col(helper_col))
    return out


def _apply_analysis_feature_overrides(df, analysis_id_value: str):
    out = df
    if analysis_id_value == "analysis_4":
        out = _apply_no_ekyc_remark_overrides(out)
    if analysis_id_value == "analysis_5":
        out = _drop_user_remark_derived_cols(out)
    return _drop_no_ekyc_remark_helper_cols(out)


# COMMAND ----------

users_dim = spark.table(DIM_USER)
users = spark.table(SILVER_USER)
registrations = spark.table(SILVER_USER_REGISTRATION)
profiles = spark.table(SILVER_USER_PROFILE)
groups = spark.table(SILVER_USER_GROUP)
login_methods = spark.table(SILVER_USER_LOGIN)
login_logs = spark.table(SILVER_USER_LOGIN_LOG)
verification = spark.table(SILVER_USER_VERIFICATION)
remarks = spark.table(SILVER_USER_REMARK) if REQUIRES_USER_REMARKS else None
deposit = spark.table(SILVER_WALLET_DEPOSIT)
wager = spark.table(FACT_WAGER)
coupons = spark.table(SILVER_WALLET_COUPON) if spark.catalog.tableExists(SILVER_WALLET_COUPON) else None
wallet_transactions = (
    spark.table(SILVER_WALLET_TRANSACTION_ORDER)
    if spark.catalog.tableExists(SILVER_WALLET_TRANSACTION_ORDER)
    else None
)

# COMMAND ----------

# Current non-test users from gold dimension
user_dim_id_col = _pick_col(users_dim, ["user_id"], "dim user_id")
user_dim_source_col = _pick_col(users_dim, ["source_system"], "dim source_system")
user_dim_is_test_col = _pick_col(users_dim, ["is_test_user"], "dim is_test_user")

valid_users = (
    users_dim
    .filter(F.coalesce(F.col(user_dim_is_test_col), F.lit(False)) == F.lit(False))
    .select(
        F.col(user_dim_id_col).cast("long").alias("user_id"),
        F.col(user_dim_source_col).alias("source_system"),
    )
    .dropna(subset=["user_id", "source_system"])
    .dropDuplicates(["user_id", "source_system"])
)

# COMMAND ----------

# Registration cohort base (monthly cohorts)
user_id_col = _pick_col(users, ["user_id"], "user.user_id")
user_source_col = _pick_col(users, ["source_system"], "user.source_system")
user_ts_col = _pick_col(
    users,
    ["created_time_uae", "created_time", "register_time_uae", "register_time"],
    "user registration timestamp",
)

username_col = _pick_col(users, ["username"], "user.username", required=False)
sign_mail_col = _pick_col(users, ["sign_mail"], "user.sign_mail", required=False)
sign_phone_col = _pick_col(users, ["sign_phone"], "user.sign_phone", required=False)
mail_col = _pick_col(users, ["mail"], "user.mail", required=False)
phone_col = _pick_col(users, ["phone"], "user.phone", required=False)

status_cols = {
    "user_status_id": _pick_col(users, ["user_status"], "user_status", required=False),
    "login_status_id": _pick_col(users, ["login_status"], "login_status", required=False),
    "distributor_status_id": _pick_col(users, ["distributor_status"], "distributor_status", required=False),
    "sale_status_id": _pick_col(users, ["sale_status"], "sale_status", required=False),
    "enable_status_id": _pick_col(users, ["enable_status"], "enable_status", required=False),
    "user_type_id": _pick_col(users, ["user_type"], "user_type", required=False),
}

cohort_start = F.add_months(F.date_trunc("month", F.current_timestamp()), -int(COHORT_LOOKBACK_MONTHS))

registration_base = (
    users.alias("u")
    .join(
        valid_users.alias("v"),
        (F.col(f"u.{user_id_col}").cast("long") == F.col("v.user_id"))
        & (F.col(f"u.{user_source_col}") == F.col("v.source_system")),
        "inner",
    )
    .withColumn("registration_ts", _safe_to_timestamp(f"u.{user_ts_col}"))
    .filter(F.col("registration_ts").isNotNull())
    .filter(F.col("registration_ts") >= cohort_start)
    .filter(F.col("registration_ts") < F.current_timestamp())
    .select(
        F.col("v.user_id"),
        F.col("v.source_system"),
        F.col("registration_ts"),
        F.date_trunc("month", F.col("registration_ts")).cast("date").alias("registration_month"),
        F.hour(F.col("registration_ts")).cast("int").alias("registration_hour"),
        F.dayofweek(F.col("registration_ts")).cast("int").alias("registration_weekday"),
        F.dayofmonth(F.col("registration_ts")).cast("int").alias("registration_day_of_month"),
        F.when(F.dayofweek(F.col("registration_ts")).isin(1, 7), F.lit(1)).otherwise(F.lit(0)).alias("registration_weekend_flag"),
        F.length(F.coalesce(F.col(f"u.{username_col}").cast("string"), F.lit(""))).cast("int").alias("username_length") if username_col else F.lit(0).alias("username_length"),
        _present_flag(F.col(f"u.{sign_mail_col}") if sign_mail_col else F.lit(None)).alias("has_sign_mail"),
        _present_flag(F.col(f"u.{sign_phone_col}") if sign_phone_col else F.lit(None)).alias("has_sign_phone"),
        _present_flag(F.col(f"u.{mail_col}") if mail_col else F.lit(None)).alias("has_mail"),
        _present_flag(F.col(f"u.{phone_col}") if phone_col else F.lit(None)).alias("has_phone"),
        F.pmod(F.abs(F.hash(F.col("v.source_system"))), F.lit(100000)).cast("int").alias("source_system_hash"),  # legacy diagnostic
        *[
            F.coalesce(F.col(f"u.{col_name}").cast("double"), F.lit(0.0)).alias(alias)
            if col_name else F.lit(0.0).alias(alias)
            for alias, col_name in status_cols.items()
        ],
    )
    .dropDuplicates(["user_id", "source_system"])
)

# Device class at registration from user_registration.plat_form
reg_user_id_col = _pick_col(registrations, ["user_id"], "registration.user_id", required=False)
reg_source_col = _pick_col(registrations, ["source_system"], "registration.source_system", required=False)
reg_platform_col = _pick_col(registrations, ["plat_form"], "registration.plat_form", required=False)
reg_ts_col = _pick_col(
    registrations,
    ["register_time_uae", "register_time", "created_time_uae", "created_time", "updated_time_uae", "updated_time"],
    "registration event timestamp",
    required=False,
)

if reg_user_id_col and reg_source_col:
    reg_partition = Window.partitionBy(reg_user_id_col, reg_source_col).orderBy(
        _safe_to_timestamp(reg_ts_col).asc_nulls_last() if reg_ts_col else F.lit(1)
    )
    registration_platform_features = (
        registrations
        .withColumn("_rn", F.row_number().over(reg_partition))
        .filter(F.col("_rn") == 1)
        .select(
            F.col(reg_user_id_col).cast("long").alias("user_id"),
            F.col(reg_source_col).alias("source_system"),
            F.col(reg_platform_col).cast("double").alias("registration_platform_id") if reg_platform_col else F.lit(None).cast("double").alias("registration_platform_id"),
        )
        .withColumn(
            "registered_by_mobile",
            F.when(F.col("registration_platform_id").isin(1, 2, 4), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "registered_by_desktop",
            F.when(F.col("registration_platform_id").isin(0, 3), F.lit(1)).otherwise(F.lit(0)),
        )
    )
else:
    registration_platform_features = (
        registration_base
        .select("user_id", "source_system")
        .withColumn("registration_platform_id", F.lit(None).cast("double"))
        .withColumn("registered_by_mobile", F.lit(0))
        .withColumn("registered_by_desktop", F.lit(0))
    )

registration_base = (
    registration_base.alias("r")
    .join(registration_platform_features.alias("rp"), on=["user_id", "source_system"], how="left")
    .select("r.*", "rp.registration_platform_id", "rp.registered_by_mobile", "rp.registered_by_desktop")
)

source_system_rows = (
    registration_base
    .groupBy("source_system")
    .count()
    .orderBy(F.col("count").desc(), F.col("source_system").asc())
    .limit(int(SOURCE_SYSTEM_TOP_K))
    .collect()
)
source_system_values = [r["source_system"] for r in source_system_rows if r["source_system"] is not None]

source_system_ohe_cols: list[str] = []
used_source_system_tokens: dict[str, int] = {}
for source_value in source_system_values:
    base_token = _sanitize_source_system_token(source_value)
    used_source_system_tokens[base_token] = used_source_system_tokens.get(base_token, 0) + 1
    suffix = used_source_system_tokens[base_token]
    token = f"{base_token}_{suffix}" if suffix > 1 else base_token
    col_name = f"source_system_is_{token}"
    source_system_ohe_cols.append(col_name)
    registration_base = registration_base.withColumn(
        col_name,
        F.when(F.col("source_system") == F.lit(source_value), F.lit(1)).otherwise(F.lit(0)),
    )

if source_system_ohe_cols:
    ohe_sum_expr = None
    for col_name in source_system_ohe_cols:
        col_expr = F.col(col_name)
        ohe_sum_expr = col_expr if ohe_sum_expr is None else (ohe_sum_expr + col_expr)
    registration_base = registration_base.withColumn(
        "source_system_is_other",
        F.when(ohe_sum_expr > F.lit(0), F.lit(0)).otherwise(F.lit(1)),
    )
else:
    registration_base = registration_base.withColumn("source_system_is_other", F.lit(1))

# COMMAND ----------

# Pre-deposit cutoff per user:
# first successful deposit after registration; if none, registration + 30d.
dep_cutoff_user_col = _pick_col(deposit, ["user_id"], "deposit user_id for predeposit cutoff", required=False)
dep_cutoff_source_col = _pick_col(deposit, ["source_system"], "deposit source_system for predeposit cutoff", required=False)
dep_cutoff_ts_col = _pick_col(
    deposit,
    ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
    "deposit attempt timestamp for predeposit cutoff",
    required=False,
)
dep_cutoff_success_ts_col = _pick_col(
    deposit,
    ["success_time_uae", "success_time"],
    "deposit success timestamp for predeposit cutoff",
    required=False,
)
dep_cutoff_status_col = _pick_col(deposit, ["deposit_status"], "deposit status for predeposit cutoff", required=False)
dep_cutoff_channel_col = _pick_col(deposit, ["deposit_channel_type"], "deposit channel for predeposit cutoff", required=False)

registration_with_predeposit_cutoff = (
    registration_base
    .select("user_id", "source_system", "registration_ts")
    .withColumn("predeposit_cutoff_ts", F.col("registration_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"))
)

if dep_cutoff_user_col and dep_cutoff_source_col and dep_cutoff_ts_col:
    deposit_cutoff_base = (
        deposit
        .select(
            F.col(dep_cutoff_user_col).cast("long").alias("user_id"),
            F.col(dep_cutoff_source_col).alias("source_system"),
            _safe_to_timestamp(dep_cutoff_ts_col).alias("attempt_ts"),
            _safe_to_timestamp(dep_cutoff_success_ts_col).alias("success_ts") if dep_cutoff_success_ts_col else F.lit(None).cast("timestamp").alias("success_ts"),
            F.col(dep_cutoff_status_col).cast("int").alias("deposit_status") if dep_cutoff_status_col else F.lit(None).cast("int").alias("deposit_status"),
            F.col(dep_cutoff_channel_col).cast("int").alias("deposit_channel_type") if dep_cutoff_channel_col else F.lit(None).cast("int").alias("deposit_channel_type"),
        )
        .filter(F.col("attempt_ts").isNotNull())
    )

    if dep_cutoff_channel_col:
        deposit_cutoff_base = deposit_cutoff_base.filter(
            (F.col("deposit_channel_type") < 100)
            | (F.col("deposit_channel_type") >= 200)
            | F.col("deposit_channel_type").isNull()
        )

    deposit_success_event_expr = F.when(
        (F.col("d.deposit_status") == 3) | F.col("d.success_ts").isNotNull(),
        F.coalesce(F.col("d.success_ts"), F.col("d.attempt_ts")),
    )

    first_deposit_success_30d = (
        registration_base.alias("r")
        .join(
            deposit_cutoff_base.alias("d"),
            (F.col("r.user_id") == F.col("d.user_id"))
            & (F.col("r.source_system") == F.col("d.source_system"))
            & (deposit_success_event_expr >= F.col("r.registration_ts"))
            & (deposit_success_event_expr < F.current_timestamp()),
            "left",
        )
        .groupBy("r.user_id", "r.source_system", "r.registration_ts")
        .agg(F.min(deposit_success_event_expr).alias("first_deposit_success_ts"))
    )

    registration_with_predeposit_cutoff = (
        registration_base.alias("r")
        .join(first_deposit_success_30d.alias("d"), on=["user_id", "source_system", "registration_ts"], how="left")
        .withColumn(
            "predeposit_cutoff_ts",
            F.coalesce(
                F.col("d.first_deposit_success_ts"),
                F.col("r.registration_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"),
            ),
        )
        .select("r.user_id", "r.source_system", "r.registration_ts", "predeposit_cutoff_ts")
    )

# COMMAND ----------

# Latest user profile snapshot per user
profile_user_id_col = _pick_col(profiles, ["user_id"], "profile user_id")
profile_source_col = _pick_col(profiles, ["source_system"], "profile source_system")
profile_ts_col = _pick_col(
    profiles,
    ["updated_time_uae", "updated_time", "created_time_uae", "created_time"],
    "profile timestamp",
    required=False,
)

profile_partition = Window.partitionBy(profile_user_id_col, profile_source_col).orderBy(
    _safe_to_timestamp(profile_ts_col).desc_nulls_last() if profile_ts_col else F.lit(1)
)

profile_latest = (
    profiles
    .withColumn("_rn", F.row_number().over(profile_partition))
    .filter(F.col("_rn") == 1)
    .select(
        F.col(profile_user_id_col).cast("long").alias("user_id"),
        F.col(profile_source_col).alias("source_system"),
        F.coalesce(F.col("register_type").cast("double"), F.lit(0.0)).alias("register_type_id") if "register_type" in profiles.columns else F.lit(0.0).alias("register_type_id"),
        F.coalesce(F.col("product_type").cast("double"), F.lit(0.0)).alias("profile_product_type_id") if "product_type" in profiles.columns else F.lit(0.0).alias("profile_product_type_id"),
        F.coalesce(F.col("plat_form").cast("double"), F.lit(0.0)).alias("profile_platform_id") if "plat_form" in profiles.columns else F.lit(0.0).alias("profile_platform_id"),
        F.coalesce(F.col("os_type").cast("double"), F.lit(0.0)).alias("profile_os_type_id") if "os_type" in profiles.columns else F.lit(0.0).alias("profile_os_type_id"),
        F.coalesce(F.col("mail_valid_status").cast("double"), F.lit(0.0)).alias("mail_valid_status_id") if "mail_valid_status" in profiles.columns else F.lit(0.0).alias("mail_valid_status_id"),
        F.coalesce(F.col("phone_valid_status").cast("double"), F.lit(0.0)).alias("phone_valid_status_id") if "phone_valid_status" in profiles.columns else F.lit(0.0).alias("phone_valid_status_id"),
        F.coalesce(F.col("country_id").cast("double"), F.lit(0.0)).alias("country_id") if "country_id" in profiles.columns else F.lit(0.0).alias("country_id"),
        F.coalesce(F.col("nation_id").cast("double"), F.lit(0.0)).alias("nation_id") if "nation_id" in profiles.columns else F.lit(0.0).alias("nation_id"),
        F.coalesce(F.col("resident_country_id").cast("double"), F.lit(0.0)).alias("resident_country_id") if "resident_country_id" in profiles.columns else F.lit(0.0).alias("resident_country_id"),
        _present_flag(F.col("channel") if "channel" in profiles.columns else F.lit(None)).alias("has_marketing_channel"),
        _present_flag(F.col("source") if "source" in profiles.columns else F.lit(None)).alias("has_marketing_source"),
        _present_flag(F.col("referrer") if "referrer" in profiles.columns else F.lit(None)).alias("has_referrer"),
        _present_flag(F.col("referrer_code") if "referrer_code" in profiles.columns else F.lit(None)).alias("has_referrer_code"),
        _present_flag(F.col("device_code") if "device_code" in profiles.columns else F.lit(None)).alias("has_device_code"),
        _present_flag(F.col("user_agent") if "user_agent" in profiles.columns else F.lit(None)).alias("has_user_agent"),
        _present_flag(F.col("imei") if "imei" in profiles.columns else F.lit(None)).alias("has_imei"),
        _present_flag(F.col("idfa") if "idfa" in profiles.columns else F.lit(None)).alias("has_idfa"),
        _present_flag(F.col("android_id") if "android_id" in profiles.columns else F.lit(None)).alias("has_android_id"),
        _present_flag(F.col("cookie_data") if "cookie_data" in profiles.columns else F.lit(None)).alias("has_cookie_data"),
    )
    .dropDuplicates(["user_id", "source_system"])
)

# COMMAND ----------

# User group features
if "type" in groups.columns:
    group_features = (
        groups
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            F.col("type").cast("int").alias("group_type"),
            F.col("id").cast("string").alias("group_row_id") if "id" in groups.columns else F.lit(None).cast("string").alias("group_row_id"),
        )
        .groupBy("user_id", "source_system")
        .agg(
            F.countDistinct("group_row_id").cast("double").alias("num_user_groups"),
            F.countDistinct("group_type").cast("double").alias("num_distinct_group_types"),
            F.sum(F.when(F.col("group_type").isin(50, 121), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_test_groups"),
        )
        .withColumn("has_any_group", F.when(F.col("num_user_groups") > 0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("num_non_test_groups", F.greatest(F.col("num_user_groups") - F.col("num_test_groups"), F.lit(0.0)))
    )
else:
    group_features = (
        registration_base
        .select("user_id", "source_system")
        .withColumn("num_user_groups", F.lit(0.0))
        .withColumn("num_distinct_group_types", F.lit(0.0))
        .withColumn("num_test_groups", F.lit(0.0))
        .withColumn("has_any_group", F.lit(0))
        .withColumn("num_non_test_groups", F.lit(0.0))
    )

# COMMAND ----------

# Login methods (static)
login_method_features = (
    login_methods
    .select(
        F.col("user_id").cast("long").alias("user_id"),
        F.col("source_system").alias("source_system"),
        F.col("id").cast("string").alias("login_method_id") if "id" in login_methods.columns else F.lit(None).cast("string").alias("login_method_id"),
        F.col("status").cast("int").alias("login_method_status") if "status" in login_methods.columns else F.lit(None).cast("int").alias("login_method_status"),
        F.col("third_party_type").cast("int").alias("third_party_type") if "third_party_type" in login_methods.columns else F.lit(None).cast("int").alias("third_party_type"),
        _present_flag(F.col("sign_login_name") if "sign_login_name" in login_methods.columns else F.lit(None)).alias("has_sign_login_name"),
    )
    .groupBy("user_id", "source_system")
    .agg(
        F.countDistinct("login_method_id").cast("double").alias("num_login_methods"),
        F.sum(F.when(F.col("login_method_status") == 1, F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_active_login_methods"),
        F.sum(F.when(F.col("third_party_type").isNotNull(), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_third_party_login_methods"),
        F.max("has_sign_login_name").cast("int").alias("has_sign_login_name"),
    )
)

# COMMAND ----------

# Early login activity after registration
login_log_ts_col = _pick_col(
    login_logs,
    ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
    "login log timestamp",
    required=False,
)

if login_log_ts_col:
    login_logs_base = (
        login_logs
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            _safe_to_timestamp(login_log_ts_col).alias("login_event_ts"),
            F.col("login_status").cast("int").alias("login_status") if "login_status" in login_logs.columns else F.lit(None).cast("int").alias("login_status"),
            F.col("ip").cast("string").alias("ip") if "ip" in login_logs.columns else F.lit(None).cast("string").alias("ip"),
            F.col("channel").cast("string").alias("channel") if "channel" in login_logs.columns else F.lit(None).cast("string").alias("channel"),
        )
        .filter(F.col("login_event_ts").isNotNull())
    )

    early_login_features = (
        registration_with_predeposit_cutoff.alias("r")
        .join(
            login_logs_base.alias("l"),
            (F.col("r.user_id") == F.col("l.user_id"))
            & (F.col("r.source_system") == F.col("l.source_system"))
            & (F.col("l.login_event_ts") >= F.col("r.registration_ts"))
            & (
                F.col("l.login_event_ts")
                < F.least(
                    F.col("r.predeposit_cutoff_ts"),
                    (F.col("r.registration_ts") + F.expr(f"INTERVAL {int(EARLY_ACTIVITY_HOURS)} HOURS")),
                )
            ),
            "left",
        )
        .groupBy("r.user_id", "r.source_system")
        .agg(
            F.count(F.col("l.login_event_ts")).cast("double").alias("num_login_events_24h"),
            F.sum(F.when((F.col("l.login_status").isNotNull()) & (F.col("l.login_status") != 1), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_unsuccessful_login_events_24h"),
            F.countDistinct("l.ip").cast("double").alias("num_distinct_login_ip_24h"),
            F.countDistinct("l.channel").cast("double").alias("num_distinct_login_channel_24h"),
        )
    )
else:
    early_login_features = registration_base.select("user_id", "source_system").withColumn("num_login_events_24h", F.lit(0.0)).withColumn("num_unsuccessful_login_events_24h", F.lit(0.0)).withColumn("num_distinct_login_ip_24h", F.lit(0.0)).withColumn("num_distinct_login_channel_24h", F.lit(0.0))

# COMMAND ----------

# Verification behavior in first days after registration
verification_ts_col = _pick_col(
    verification,
    ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
    "verification timestamp",
    required=False,
)
verify_type_col = _pick_col(verification, ["verify_type"], "verify_type", required=False)
verified_status_col = _pick_col(verification, ["verified_status"], "verified_status", required=False)
fail_reason_col = _pick_col(verification, ["fail_reason"], "fail_reason", required=False)
country_region_col = _pick_col(verification, ["country_region"], "country_region", required=False)
gender_col = _pick_col(verification, ["gender"], "gender", required=False)
verification_ip_col = _pick_col(verification, ["ip"], "verification.ip", required=False)
id_card_type_col = _pick_col(verification, ["id_card_type"], "id_card_type", required=False)
name_similar_col = _pick_col(verification, ["name_similar"], "name_similar", required=False)
birth_date_v_col = _pick_col(verification, ["birth_date"], "verification.birth_date", required=False)
id_card_img_col = _pick_col(verification, ["id_card_img"], "id_card_img", required=False)
sign_phone_v_col = _pick_col(verification, ["sign_phone"], "sign_phone", required=False)
sign_id_card_v_col = _pick_col(verification, ["sign_id_card"], "sign_id_card", required=False)

verification_country_ohe_cols: list[str] = []
verification_country_agg_exprs: list = []

if verification_ts_col:
    verified_status_raw_col = (
        F.col(verified_status_col).cast("string") if verified_status_col else F.lit(None).cast("string")
    )
    verified_status_id_col = _normalize_status_int(verified_status_raw_col)
    verify_type_id_col = (
        _normalize_status_int(F.col(verify_type_col).cast("string"))
        if verify_type_col
        else F.lit(None).cast("int")
    )
    fail_reason_id_col = (
        _normalize_status_int(F.col(fail_reason_col).cast("string"))
        if fail_reason_col
        else F.lit(None).cast("int")
    )
    country_region_norm_col = (
        F.upper(F.trim(F.col(country_region_col).cast("string")))
        if country_region_col
        else F.lit(None).cast("string")
    )
    gender_norm_col = (
        F.lower(F.trim(F.col(gender_col).cast("string")))
        if gender_col
        else F.lit(None).cast("string")
    )

    verification_base = (
        verification
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            _safe_to_timestamp(verification_ts_col).alias("verification_ts"),
            F.col("verified").cast("int").alias("verified") if "verified" in verification.columns else F.lit(None).cast("int").alias("verified"),
            verify_type_id_col.alias("verify_type_id"),
            verified_status_raw_col.alias("verified_status_raw"),
            verified_status_id_col.alias("verified_status_id"),
            fail_reason_id_col.alias("fail_reason_id"),
            country_region_norm_col.alias("country_region"),
            gender_norm_col.alias("gender_norm"),
            F.col(verification_ip_col).cast("string").alias("verification_ip") if verification_ip_col else F.lit(None).cast("string").alias("verification_ip"),
            F.col(id_card_type_col).cast("int").alias("id_card_type") if id_card_type_col else F.lit(None).cast("int").alias("id_card_type"),
            F.col(name_similar_col).cast("double").alias("name_similar") if name_similar_col else F.lit(None).cast("double").alias("name_similar"),
            _safe_to_date(birth_date_v_col).alias("birth_date_verification") if birth_date_v_col else F.lit(None).cast("date").alias("birth_date_verification"),
            _present_flag(F.col(id_card_img_col) if id_card_img_col else F.lit(None)).alias("has_id_card_img"),
            _present_flag(F.col(sign_phone_v_col) if sign_phone_v_col else F.lit(None)).alias("has_sign_phone_verification"),
            _present_flag(F.col(sign_id_card_v_col) if sign_id_card_v_col else F.lit(None)).alias("has_sign_id_card_verification"),
        )
        .filter(F.col("verification_ts").isNotNull())
    )

    country_rows = (
        verification_base
        .filter(
            (F.col("verify_type_id") == 3)
            & (F.col("verified_status_id") == 3)
            & F.col("country_region").isNotNull()
            & (F.col("country_region") != "")
        )
        .groupBy("country_region")
        .count()
        .orderBy(F.col("count").desc(), F.col("country_region").asc())
        .limit(int(VERIFICATION_COUNTRY_TOP_K))
        .collect()
    )
    country_values = [r["country_region"] for r in country_rows if r["country_region"]]
    used_country_tokens: dict[str, int] = {}
    for country_value in country_values:
        base_token = _sanitize_source_system_token(country_value)
        used_country_tokens[base_token] = used_country_tokens.get(base_token, 0) + 1
        suffix = used_country_tokens[base_token]
        token = f"{base_token}_{suffix}" if suffix > 1 else base_token
        col_name = f"verification_country_region_is_{token}_7d"
        verification_country_ohe_cols.append(col_name)
        verification_country_agg_exprs.append(
            F.max(
                F.when(F.col("v.country_region") == F.lit(country_value), F.lit(1)).otherwise(F.lit(0))
            ).cast("int").alias(col_name)
        )

    kyc_event_condition = (
        F.col("v.verify_type_id").isin(2, 3)
        if verify_type_col
        else F.col("v.verification_ts").isNotNull()
    )
    ekyc_event_condition = (
        F.col("v.verify_type_id") == 3
        if verify_type_col
        else F.lit(False)
    )
    verification_success_condition = (F.col("v.verified") == 1) | (F.col("v.verified_status_id") == 3)
    verification_fail_condition = F.col("v.verification_ts").isNotNull() & (~verification_success_condition)
    ekyc_fail_condition = ekyc_event_condition & (~verification_success_condition)
    risk_fail_reason_condition = F.col("v.fail_reason_id").isNotNull()
    gender_male_condition = F.col("v.gender_norm").rlike("^(m|male|masculino|1)$")
    gender_female_condition = F.col("v.gender_norm").rlike("^(f|female|feminino|2)$")

    verification_features = (
        registration_with_predeposit_cutoff.alias("r")
        .join(
            verification_base.alias("v"),
            (F.col("r.user_id") == F.col("v.user_id"))
            & (F.col("r.source_system") == F.col("v.source_system"))
            & (F.col("v.verification_ts") >= F.col("r.registration_ts"))
            & (
                F.col("v.verification_ts")
                < F.least(
                    F.col("r.predeposit_cutoff_ts"),
                    (F.col("r.registration_ts") + F.expr(f"INTERVAL {int(VERIFICATION_WINDOW_DAYS)} DAYS")),
                )
            ),
            "left",
        )
        .withColumn(
            "verified_success_flag",
            F.when(verification_success_condition, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "verification_fail_flag",
            F.when(verification_fail_condition, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("kyc_event_flag", F.when(kyc_event_condition, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("ekyc_event_flag", F.when(ekyc_event_condition, F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "kyc_success_flag",
            F.when(kyc_event_condition & (F.col("v.verified_status_id") == 3), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "ekyc_success_flag",
            F.when(ekyc_event_condition & (F.col("v.verified_status_id") == 3), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "ekyc_fail_flag",
            F.when(ekyc_fail_condition, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "gender_male_flag",
            F.when(gender_male_condition, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "gender_female_flag",
            F.when(gender_female_condition, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "verification_age_years",
            F.when(
                F.col("v.birth_date_verification").isNotNull(),
                F.months_between(F.to_date(F.col("r.registration_ts")), F.col("v.birth_date_verification")) / F.lit(12.0),
            ).otherwise(F.lit(None).cast("double")),
        )
        .groupBy("r.user_id", "r.source_system")
        .agg(
            F.count(F.col("v.verification_ts")).cast("double").alias("num_verification_events_7d"),
            F.sum(F.col("verified_success_flag")).cast("double").alias("num_verification_success_events_7d"),
            F.sum(F.col("verification_fail_flag")).cast("double").alias("num_verification_fail_events_7d"),
            F.max(F.col("verified_success_flag")).cast("int").alias("verified_in_7d"),
            F.min(
                F.when(
                    F.col("v.verification_ts").isNotNull(),
                    (F.col("v.verification_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                )
            ).cast("double").alias("first_verification_delay_hours_7d"),
            F.sum(F.col("kyc_event_flag")).cast("double").alias("num_kyc_events_7d"),
            F.max(F.col("kyc_event_flag")).cast("int").alias("has_kyc_attempt_7d"),
            F.max(F.col("kyc_success_flag")).cast("int").alias("kyc_verified_success_7d"),
            F.min(
                F.when(
                    F.col("kyc_event_flag") == 1,
                    (F.col("v.verification_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                )
            ).cast("double").alias("first_kyc_attempt_delay_hours_7d"),
            F.min(
                F.when(
                    F.col("kyc_success_flag") == 1,
                    (F.col("v.verification_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                )
            ).cast("double").alias("first_kyc_success_delay_hours_7d"),
            F.sum(F.col("ekyc_event_flag")).cast("double").alias("num_ekyc_events_7d"),
            F.max(F.col("ekyc_event_flag")).cast("int").alias("has_ekyc_attempt_7d"),
            F.max(F.col("ekyc_success_flag")).cast("int").alias("ekyc_verified_success_7d"),
            F.sum(F.col("ekyc_success_flag")).cast("double").alias("num_ekyc_success_events_7d"),
            F.sum(F.col("ekyc_fail_flag")).cast("double").alias("num_ekyc_fail_events_7d"),
            F.max(F.col("ekyc_fail_flag")).cast("int").alias("has_ekyc_fail_7d"),
            F.min(
                F.when(
                    F.col("ekyc_event_flag") == 1,
                    (F.col("v.verification_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                )
            ).cast("double").alias("first_ekyc_attempt_delay_hours_7d"),
            F.min(
                F.when(
                    F.col("ekyc_success_flag") == 1,
                    (F.col("v.verification_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                )
            ).cast("double").alias("first_ekyc_success_delay_hours_7d"),
            F.countDistinct("v.verify_type_id").cast("double").alias("num_distinct_verify_types_7d"),
            F.countDistinct("v.id_card_type").cast("double").alias("num_distinct_id_card_types_7d"),
            F.avg("v.name_similar").cast("double").alias("avg_name_similar_7d"),
            F.countDistinct("v.verification_ip").cast("double").alias("num_distinct_verification_ip_7d"),
            F.countDistinct("v.country_region").cast("double").alias("num_distinct_verification_country_region_7d"),
            F.max(F.when(F.col("v.country_region").isNotNull() & (F.col("v.country_region") != ""), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_verification_country_region_7d"),
            F.max(F.col("gender_male_flag")).cast("int").alias("verification_gender_male_7d"),
            F.max(F.col("gender_female_flag")).cast("int").alias("verification_gender_female_7d"),
            F.max(
                F.when(
                    F.col("v.gender_norm").isNotNull()
                    & (F.col("v.gender_norm") != "")
                    & (~gender_male_condition)
                    & (~gender_female_condition),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).cast("int").alias("verification_gender_other_7d"),
            F.max(F.col("v.has_id_card_img")).cast("int").alias("has_id_card_img_7d"),
            F.max(F.col("v.has_sign_phone_verification")).cast("int").alias("has_sign_phone_verification_7d"),
            F.max(F.col("v.has_sign_id_card_verification")).cast("int").alias("has_sign_id_card_verification_7d"),
            F.max(F.when(F.col("v.birth_date_verification").isNotNull(), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_birth_date_verification_7d"),
            F.avg(F.col("verification_age_years")).cast("double").alias("verification_age_years_7d"),
            F.countDistinct(F.when(risk_fail_reason_condition, F.col("v.fail_reason_id"))).cast("double").alias("num_verification_fail_reason_distinct_7d"),
            F.max(F.when(risk_fail_reason_condition, F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_verification_fail_reason_7d"),
            *verification_country_agg_exprs,
        )
        .withColumn("has_verification_event_7d", F.when(F.col("num_verification_events_7d") > 0, F.lit(1)).otherwise(F.lit(0)))
    )
else:
    verification_features_base = (
        registration_base
        .select("user_id", "source_system")
        .withColumn("num_verification_events_7d", F.lit(0.0))
        .withColumn("num_verification_success_events_7d", F.lit(0.0))
        .withColumn("num_verification_fail_events_7d", F.lit(0.0))
        .withColumn("verified_in_7d", F.lit(0))
        .withColumn("first_verification_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("has_verification_event_7d", F.lit(0))
        .withColumn("num_kyc_events_7d", F.lit(0.0))
        .withColumn("has_kyc_attempt_7d", F.lit(0))
        .withColumn("kyc_verified_success_7d", F.lit(0))
        .withColumn("first_kyc_attempt_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("first_kyc_success_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("num_ekyc_events_7d", F.lit(0.0))
        .withColumn("has_ekyc_attempt_7d", F.lit(0))
        .withColumn("ekyc_verified_success_7d", F.lit(0))
        .withColumn("num_ekyc_success_events_7d", F.lit(0.0))
        .withColumn("num_ekyc_fail_events_7d", F.lit(0.0))
        .withColumn("has_ekyc_fail_7d", F.lit(0))
        .withColumn("first_ekyc_attempt_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("first_ekyc_success_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("num_distinct_verify_types_7d", F.lit(0.0))
        .withColumn("num_distinct_id_card_types_7d", F.lit(0.0))
        .withColumn("avg_name_similar_7d", F.lit(None).cast("double"))
        .withColumn("num_distinct_verification_ip_7d", F.lit(0.0))
        .withColumn("num_distinct_verification_country_region_7d", F.lit(0.0))
        .withColumn("has_verification_country_region_7d", F.lit(0))
        .withColumn("verification_gender_male_7d", F.lit(0))
        .withColumn("verification_gender_female_7d", F.lit(0))
        .withColumn("verification_gender_other_7d", F.lit(0))
        .withColumn("has_id_card_img_7d", F.lit(0))
        .withColumn("has_sign_phone_verification_7d", F.lit(0))
        .withColumn("has_sign_id_card_verification_7d", F.lit(0))
        .withColumn("has_birth_date_verification_7d", F.lit(0))
        .withColumn("verification_age_years_7d", F.lit(None).cast("double"))
        .withColumn("num_verification_fail_reason_distinct_7d", F.lit(0.0))
        .withColumn("has_verification_fail_reason_7d", F.lit(0))
    )
    verification_features = verification_features_base
    for col_name in verification_country_ohe_cols:
        verification_features = verification_features.withColumn(col_name, F.lit(0))

# COMMAND ----------

# User age and nationality from verified eKYC profile (not limited to 7d window)
verification_nationality_candidates = ["country_region", "nationality", "country", "country_name"]
verification_nationality_cols = [c for c in verification_nationality_candidates if c in verification.columns]

customer_nationality_top_values = []
for _value in CUSTOMER_NATIONALITY_TOP_COUNTRIES_GLOBAL:
    _norm = str(_value).strip().upper()
    if _norm and _norm != "UNKNOWN" and _norm not in customer_nationality_top_values:
        customer_nationality_top_values.append(_norm)

customer_nationality_ohe_cols = [
    f"customer_nationality_is_{_sanitize_source_system_token(_value)}"
    for _value in customer_nationality_top_values
]
customer_nationality_unknown_col = "customer_nationality_is_unknown"
customer_nationality_other_col = "customer_nationality_is_other"


def _with_customer_nationality_ohe(df):
    out = df
    for _value, _col_name in zip(customer_nationality_top_values, customer_nationality_ohe_cols):
        out = out.withColumn(
            _col_name,
            F.when(F.col("customer_nationality") == F.lit(_value), F.lit(1)).otherwise(F.lit(0)).cast("int"),
        )

    out = out.withColumn(
        customer_nationality_unknown_col,
        F.when(F.col("customer_nationality") == F.lit("UNKNOWN"), F.lit(1)).otherwise(F.lit(0)).cast("int"),
    )

    if customer_nationality_top_values:
        out = out.withColumn(
            customer_nationality_other_col,
            F.when(
                (~F.col("customer_nationality").isin(*customer_nationality_top_values))
                & (F.col("customer_nationality") != F.lit("UNKNOWN")),
                F.lit(1),
            ).otherwise(F.lit(0)).cast("int"),
        )
    else:
        out = out.withColumn(
            customer_nationality_other_col,
            F.when(F.col("customer_nationality") != F.lit("UNKNOWN"), F.lit(1)).otherwise(F.lit(0)).cast("int"),
        )

    return out

demographic_features = (
    registration_base
    .select("user_id", "source_system")
    .withColumn("customer_age_years", F.lit(None).cast("double"))
    .withColumn("customer_nationality", F.lit("UNKNOWN"))
)
demographic_features = _with_customer_nationality_ohe(demographic_features)

if verified_status_col and verify_type_col and (birth_date_v_col or verification_nationality_cols):
    nationality_exprs = []
    for _col_name in verification_nationality_cols:
        _raw = F.trim(F.col(_col_name).cast("string"))
        nationality_exprs.append(
            F.when(_raw.isNotNull() & (F.length(_raw) > 0), F.upper(_raw))
        )
    nationality_raw_expr = (
        F.coalesce(*nationality_exprs)
        if nationality_exprs
        else F.lit(None).cast("string")
    )

    verification_demographic_base = (
        verification
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            _safe_to_timestamp(verification_ts_col).alias("verification_ts")
            if verification_ts_col
            else F.lit(None).cast("timestamp").alias("verification_ts"),
            _normalize_status_int(F.col(verified_status_col).cast("string")).alias("verified_status_id"),
            _normalize_status_int(F.col(verify_type_col).cast("string")).alias("verify_type_id"),
            _safe_to_date(birth_date_v_col).alias("birth_date_verification")
            if birth_date_v_col
            else F.lit(None).cast("date").alias("birth_date_verification"),
            nationality_raw_expr.alias("nationality_raw"),
        )
        .filter((F.col("verified_status_id") == F.lit(3)) & (F.col("verify_type_id") == F.lit(3)))
    )

    verification_demographic_partition = Window.partitionBy("user_id", "source_system").orderBy(
        F.col("verification_ts").desc_nulls_last()
    )

    verification_demographic_latest = (
        verification_demographic_base
        .withColumn("_rn", F.row_number().over(verification_demographic_partition))
        .filter(F.col("_rn") == 1)
        .select(
            "user_id",
            "source_system",
            F.when(
                F.col("birth_date_verification").isNotNull(),
                F.floor(
                    F.months_between(F.current_date(), F.col("birth_date_verification")) / F.lit(12.0)
                ).cast("double"),
            ).otherwise(F.lit(None).cast("double")).alias("customer_age_years"),
            F.when(
                F.col("nationality_raw").isNull() | (F.length(F.col("nationality_raw")) == 0),
                F.lit("UNKNOWN"),
            ).otherwise(F.col("nationality_raw")).alias("customer_nationality"),
        )
        .withColumn(
            "customer_age_years",
            F.when(
                (F.col("customer_age_years") >= F.lit(0.0))
                & (F.col("customer_age_years") <= F.lit(110.0)),
                F.col("customer_age_years"),
            ).otherwise(F.lit(None).cast("double")),
        )
    )

    demographic_features = (
        registration_base
        .select("user_id", "source_system")
        .join(verification_demographic_latest, on=["user_id", "source_system"], how="left")
        .withColumn(
            "customer_nationality",
            F.when(
                F.col("customer_nationality").isNull() | (F.length(F.trim(F.col("customer_nationality"))) == 0),
                F.lit("UNKNOWN"),
            ).otherwise(F.upper(F.trim(F.col("customer_nationality")))),
        )
    )
    demographic_features = _with_customer_nationality_ohe(demographic_features)

# COMMAND ----------

# User remark features in first days after registration
remark_ts_col = (
    _pick_col(
        remarks,
        ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
        "remark timestamp",
        required=False,
    )
    if remarks is not None
    else None
)
remark_operation_col = (
    _pick_col(remarks, ["operation_type"], "remark operation_type", required=False)
    if remarks is not None
    else None
)
remark_text_col = (
    _pick_col(remarks, ["remark"], "remark text", required=False)
    if remarks is not None
    else None
)
remark_flow_step_ohe_values = [
    "account_created",
    "ekyc_failed",
    "ekyc_verified",
    "kyc_failed",
    "kyc_verified",
    "system_face_failed",
    "system_face_verified",
    "address_added",
    "address_changed",
    "email_added",
    "email_changed",
    "phone_changed",
    "password_changed",
    "self_exclusion",
    "deposit_limit",
    "bet_limit",
    "customer_request",
    "lock_suspend_ban",
    "risk_investigation",
    "other",
]

if remark_ts_col:
    post_deposit_marker_pattern = r"(first deposit|first withdrawal)"
    risk_remark_pattern = r"(fraud|chargeback|suspicious|suspit|suspec|suspect|scam|under investigation|suspend|blocked|block|lock|locked|frozen|exclude|exclusion|abuse|risk|aml)"
    fraud_remark_pattern = r"(fraud|chargeback|abuse|aml|ato attack)"
    exclusion_remark_pattern = r"(exclude|exclusion|self-exclusion)"
    lock_remark_pattern = r"(blocked|block|lock|locked|frozen|suspend|suspended|black list|blacklist|ban)"
    account_created_pattern = r"^creation account$"
    address_added_pattern = r"add address"
    address_changed_pattern = r"address .* (is changed to|changed to)"
    email_added_pattern = r"(^mail: null -|initial email address: null new email address|mail account: null is changed to)"
    email_changed_pattern = r"(^mail: .* - .*|mail account: .* is changed to|initial email address: .* new email address)"
    phone_changed_pattern = r"(phone number: .* is changed to|phone number changed|mobile number update|phone number: .* - .*)"
    password_changed_pattern = r"(change password)"
    ekyc_verified_pattern = r"(ekyc information modification: verified|ekyc.*verified)"
    ekyc_failed_pattern = r"(ekyc information modification: verification failed|ekyc information modification: not verified|ekyc information modification: black list|ekyc.*verification failed|ekyc.*not verified|ekyc.*black list|ekyc.*under review)"
    kyc_verified_pattern = r"((^|[^a-z])kyc information modification: verified)"
    kyc_failed_pattern = r"((^|[^a-z])kyc information modification: verification failed|(^|[^a-z])kyc information modification: not verified|(^|[^a-z])kyc information modification: black list|(^|[^a-z])kyc.*verification failed|(^|[^a-z])kyc.*not verified|(^|[^a-z])kyc.*black list|(^|[^a-z])kyc.*under review)"
    system_face_verified_pattern = r"(system_face: verified)"
    system_face_failed_pattern = r"(system_face: verification failed|system_face: not verified|system_face: black list|system_face: under review)"
    self_exclusion_pattern = r"(user exclusion type self-exclusion)"
    deposit_limit_pattern = r"(user exclusion type deposit limit)"
    bet_limit_pattern = r"(user exclusion type bet limit)"
    customer_request_pattern = r"(customer request|customer requested|user requested|requested to|requesting|account deletion|close the account|account suspension|unsuspend)"

    remarks_base = (
        remarks
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            _safe_to_timestamp(remark_ts_col).alias("remark_ts"),
            F.col(remark_operation_col).cast("int").alias("remark_operation_type") if remark_operation_col else F.lit(None).cast("int").alias("remark_operation_type"),
            F.trim(
                F.regexp_replace(
                    F.regexp_replace(
                        F.lower(F.coalesce(F.col(remark_text_col).cast("string"), F.lit(""))),
                        r"<[^>]+>",
                        " ",
                    ),
                    r"\s+",
                    " ",
                )
            ).alias("remark_text") if remark_text_col else F.lit(None).cast("string").alias("remark_text"),
        )
        .filter(F.col("remark_ts").isNotNull())
        # Avoid leakage: these remarks explicitly indicate successful deposit/withdrawal.
        .filter(~F.coalesce(F.col("remark_text").rlike(post_deposit_marker_pattern), F.lit(False)))
    )

    remark_enriched = (
        registration_with_predeposit_cutoff.alias("r")
        .join(
            remarks_base.alias("rm"),
            (F.col("r.user_id") == F.col("rm.user_id"))
            & (F.col("r.source_system") == F.col("rm.source_system"))
            & (F.col("rm.remark_ts") >= F.col("r.registration_ts"))
            & (
                F.col("rm.remark_ts")
                < F.least(
                    F.col("r.predeposit_cutoff_ts"),
                    (F.col("r.registration_ts") + F.expr(f"INTERVAL {int(REMARK_WINDOW_DAYS)} DAYS")),
                )
            ),
            "left",
        )
        .withColumn(
            "remark_delay_hours",
            F.when(
                F.col("rm.remark_ts").isNotNull(),
                (F.col("rm.remark_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
            ).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "remark_flow_step",
            F.when(F.col("rm.remark_ts").isNull(), F.lit(None).cast("string"))
            .when(F.col("rm.remark_text").rlike(account_created_pattern), F.lit("account_created"))
            .when(F.col("rm.remark_text").rlike(ekyc_failed_pattern), F.lit("ekyc_failed"))
            .when(F.col("rm.remark_text").rlike(ekyc_verified_pattern), F.lit("ekyc_verified"))
            .when(F.col("rm.remark_text").rlike(kyc_failed_pattern), F.lit("kyc_failed"))
            .when(F.col("rm.remark_text").rlike(kyc_verified_pattern), F.lit("kyc_verified"))
            .when(F.col("rm.remark_text").rlike(system_face_failed_pattern), F.lit("system_face_failed"))
            .when(F.col("rm.remark_text").rlike(system_face_verified_pattern), F.lit("system_face_verified"))
            .when(F.col("rm.remark_text").rlike(address_added_pattern), F.lit("address_added"))
            .when(F.col("rm.remark_text").rlike(address_changed_pattern), F.lit("address_changed"))
            .when(F.col("rm.remark_text").rlike(email_added_pattern), F.lit("email_added"))
            .when(F.col("rm.remark_text").rlike(email_changed_pattern), F.lit("email_changed"))
            .when(F.col("rm.remark_text").rlike(phone_changed_pattern), F.lit("phone_changed"))
            .when(F.col("rm.remark_text").rlike(password_changed_pattern), F.lit("password_changed"))
            .when(F.col("rm.remark_text").rlike(self_exclusion_pattern), F.lit("self_exclusion"))
            .when(F.col("rm.remark_text").rlike(deposit_limit_pattern), F.lit("deposit_limit"))
            .when(F.col("rm.remark_text").rlike(bet_limit_pattern), F.lit("bet_limit"))
            .when(F.col("rm.remark_text").rlike(customer_request_pattern), F.lit("customer_request"))
            .when(F.col("rm.remark_text").rlike(lock_remark_pattern), F.lit("lock_suspend_ban"))
            .when(F.col("rm.remark_text").rlike(risk_remark_pattern), F.lit("risk_investigation"))
            .otherwise(F.lit("other"))
        )
    )
    remark_flow_step_ohe_agg_exprs = []
    for flow_step in remark_flow_step_ohe_values:
        remark_flow_step_ohe_agg_exprs.append(
            F.sum(
                F.when(F.col("remark_flow_step") == F.lit(flow_step), F.lit(1)).otherwise(F.lit(0))
            ).cast("double").alias(f"num_remark_flow_step_{flow_step}_7d")
        )
        remark_flow_step_ohe_agg_exprs.append(
            F.max(
                F.when(F.col("remark_flow_step") == F.lit(flow_step), F.lit(1)).otherwise(F.lit(0))
            ).cast("int").alias(f"has_remark_flow_step_{flow_step}_7d")
        )

    # Match the business rule used in ad-hoc SQL checks:
    # keep only remark rows where UPPER(remark) NOT LIKE '%EKYC%'.
    no_ekyc_row_condition = (
        F.col("rm.remark_ts").isNotNull()
        & (~F.coalesce(F.col("rm.remark_text").rlike("ekyc"), F.lit(False)))
    )

    remark_features = (
        remark_enriched
        .groupBy("r.user_id", "r.source_system")
        .agg(
            F.count("rm.remark_ts").cast("double").alias("num_user_remarks_7d"),
            F.countDistinct("rm.remark_operation_type").cast("double").alias("num_distinct_remark_operation_types_7d"),
            F.countDistinct(F.when(F.col("rm.remark_ts").isNotNull(), F.col("remark_flow_step"))).cast("double").alias("num_distinct_remark_flow_steps_7d"),
            F.max(F.when(F.col("rm.remark_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_user_remark_7d"),
            F.min(F.col("remark_delay_hours")).cast("double").alias("first_user_remark_delay_hours_7d"),
            F.count(F.when(no_ekyc_row_condition, F.col("rm.remark_ts"))).cast("double").alias("__num_user_remarks_7d_no_ekyc"),
            F.countDistinct(F.when(no_ekyc_row_condition, F.col("rm.remark_operation_type"))).cast("double").alias("__num_distinct_remark_operation_types_7d_no_ekyc"),
            F.countDistinct(F.when(no_ekyc_row_condition, F.col("remark_flow_step"))).cast("double").alias("__num_distinct_remark_flow_steps_7d_no_ekyc"),
            F.max(F.when(no_ekyc_row_condition, F.lit(1)).otherwise(F.lit(0))).cast("int").alias("__has_user_remark_7d_no_ekyc"),
            F.min(F.when(no_ekyc_row_condition, F.col("remark_delay_hours"))).cast("double").alias("__first_user_remark_delay_hours_7d_no_ekyc"),
            F.sum(F.when(F.col("rm.remark_text").rlike(risk_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_risk_remarks_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(risk_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_risk_remark_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(fraud_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_fraud_remarks_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(fraud_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_fraud_remark_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(exclusion_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_exclusion_remarks_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(exclusion_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_exclusion_remark_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(lock_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_lock_remarks_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(lock_remark_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_lock_remark_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(account_created_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_account_created_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(account_created_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_account_created_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(address_added_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_address_added_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(address_added_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_address_added_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(address_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_address_changed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(address_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_address_changed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(email_added_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_email_added_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(email_added_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_email_added_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(email_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_email_changed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(email_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_email_changed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(phone_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_phone_changed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(phone_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_phone_changed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(password_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_password_changed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(password_changed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_password_changed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(ekyc_verified_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_ekyc_verified_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(ekyc_verified_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_ekyc_verified_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(ekyc_failed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_ekyc_failed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(ekyc_failed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_ekyc_failed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(kyc_verified_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_kyc_verified_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(kyc_verified_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_kyc_verified_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(kyc_failed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_kyc_failed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(kyc_failed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_kyc_failed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(system_face_verified_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_system_face_verified_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(system_face_verified_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_system_face_verified_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(system_face_failed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_system_face_failed_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(system_face_failed_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_system_face_failed_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(self_exclusion_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_self_exclusion_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(self_exclusion_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_self_exclusion_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(deposit_limit_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_deposit_limit_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(deposit_limit_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_deposit_limit_7d"),
            F.sum(F.when(F.col("rm.remark_text").rlike(customer_request_pattern), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_remark_customer_request_7d"),
            F.max(F.when(F.col("rm.remark_text").rlike(customer_request_pattern), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_remark_customer_request_7d"),
            F.min(F.when(F.col("rm.remark_text").rlike(address_added_pattern), F.col("remark_delay_hours"))).cast("double").alias("first_remark_address_added_delay_hours_7d"),
            F.min(F.when(F.col("rm.remark_text").rlike(ekyc_verified_pattern), F.col("remark_delay_hours"))).cast("double").alias("first_remark_ekyc_verified_delay_hours_7d"),
            F.min(F.when(F.col("rm.remark_text").rlike(ekyc_failed_pattern), F.col("remark_delay_hours"))).cast("double").alias("first_remark_ekyc_failed_delay_hours_7d"),
            F.min(F.when(F.col("rm.remark_text").rlike(customer_request_pattern), F.col("remark_delay_hours"))).cast("double").alias("first_remark_customer_request_delay_hours_7d"),
            F.min(F.when(F.col("rm.remark_text").rlike(lock_remark_pattern), F.col("remark_delay_hours"))).cast("double").alias("first_remark_lock_delay_hours_7d"),
            *remark_flow_step_ohe_agg_exprs,
        )
    )
else:
    remark_features = (
        registration_base
        .select("user_id", "source_system")
        .withColumn("num_user_remarks_7d", F.lit(0.0))
        .withColumn("num_distinct_remark_operation_types_7d", F.lit(0.0))
        .withColumn("num_distinct_remark_flow_steps_7d", F.lit(0.0))
        .withColumn("has_user_remark_7d", F.lit(0))
        .withColumn("first_user_remark_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("num_risk_remarks_7d", F.lit(0.0))
        .withColumn("has_risk_remark_7d", F.lit(0))
        .withColumn("num_fraud_remarks_7d", F.lit(0.0))
        .withColumn("has_fraud_remark_7d", F.lit(0))
        .withColumn("num_exclusion_remarks_7d", F.lit(0.0))
        .withColumn("has_exclusion_remark_7d", F.lit(0))
        .withColumn("num_lock_remarks_7d", F.lit(0.0))
        .withColumn("has_lock_remark_7d", F.lit(0))
        .withColumn("num_remark_account_created_7d", F.lit(0.0))
        .withColumn("has_remark_account_created_7d", F.lit(0))
        .withColumn("num_remark_address_added_7d", F.lit(0.0))
        .withColumn("has_remark_address_added_7d", F.lit(0))
        .withColumn("num_remark_address_changed_7d", F.lit(0.0))
        .withColumn("has_remark_address_changed_7d", F.lit(0))
        .withColumn("num_remark_email_added_7d", F.lit(0.0))
        .withColumn("has_remark_email_added_7d", F.lit(0))
        .withColumn("num_remark_email_changed_7d", F.lit(0.0))
        .withColumn("has_remark_email_changed_7d", F.lit(0))
        .withColumn("num_remark_phone_changed_7d", F.lit(0.0))
        .withColumn("has_remark_phone_changed_7d", F.lit(0))
        .withColumn("num_remark_password_changed_7d", F.lit(0.0))
        .withColumn("has_remark_password_changed_7d", F.lit(0))
        .withColumn("num_remark_ekyc_verified_7d", F.lit(0.0))
        .withColumn("has_remark_ekyc_verified_7d", F.lit(0))
        .withColumn("num_remark_ekyc_failed_7d", F.lit(0.0))
        .withColumn("has_remark_ekyc_failed_7d", F.lit(0))
        .withColumn("num_remark_kyc_verified_7d", F.lit(0.0))
        .withColumn("has_remark_kyc_verified_7d", F.lit(0))
        .withColumn("num_remark_kyc_failed_7d", F.lit(0.0))
        .withColumn("has_remark_kyc_failed_7d", F.lit(0))
        .withColumn("num_remark_system_face_verified_7d", F.lit(0.0))
        .withColumn("has_remark_system_face_verified_7d", F.lit(0))
        .withColumn("num_remark_system_face_failed_7d", F.lit(0.0))
        .withColumn("has_remark_system_face_failed_7d", F.lit(0))
        .withColumn("num_remark_self_exclusion_7d", F.lit(0.0))
        .withColumn("has_remark_self_exclusion_7d", F.lit(0))
        .withColumn("num_remark_deposit_limit_7d", F.lit(0.0))
        .withColumn("has_remark_deposit_limit_7d", F.lit(0))
        .withColumn("num_remark_customer_request_7d", F.lit(0.0))
        .withColumn("has_remark_customer_request_7d", F.lit(0))
        .withColumn("first_remark_address_added_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("first_remark_ekyc_verified_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("first_remark_ekyc_failed_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("first_remark_customer_request_delay_hours_7d", F.lit(None).cast("double"))
        .withColumn("first_remark_lock_delay_hours_7d", F.lit(None).cast("double"))
    )
    for flow_step in remark_flow_step_ohe_values:
        remark_features = (
            remark_features
            .withColumn(f"num_remark_flow_step_{flow_step}_7d", F.lit(0.0))
            .withColumn(f"has_remark_flow_step_{flow_step}_7d", F.lit(0))
        )

# COMMAND ----------

# Wallet transaction behavior before first successful deposit
# (or until registration + 30d for users without successful deposit)
# IMPORTANT: `betrecharge` maps to the deposit transaction, so it is excluded
# from wallet features to prevent target leakage.
wallet_trans_types = [
    "systembetincrease",
    "bonusrecharge",
    "decrease",
    "prize",
    "prizecard",
    "prizeevent",
    "transferprizeincrease",
    "eventrecharge",
    "systemprizeincrease",
    "transferbetincrease",
    "eventdecrease",
    "decreaserollback",
    "decreasemerchant",
    "withdrawreverted",
]

wallet_tx_defaults = {
    "num_wallet_txn_rows_predeposit_30d": 0.0,
}
for _trans_type in wallet_trans_types:
    _token = _sanitize_source_system_token(_trans_type)
    wallet_tx_defaults[f"num_wallet_txn_trans_type_{_token}_predeposit_30d"] = 0.0
    wallet_tx_defaults[f"has_wallet_txn_trans_type_{_token}_predeposit_30d"] = 0

if wallet_transactions is not None:
    wtx_ts_col = _pick_col(
        wallet_transactions,
        ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
        "wallet transaction timestamp",
        required=False,
    )
    wtx_user_col = _pick_col(wallet_transactions, ["user_id"], "wallet transaction user_id", required=False)
    wtx_source_col = _pick_col(wallet_transactions, ["source_system"], "wallet transaction source_system", required=False)
    wtx_trans_type_col = _pick_col(wallet_transactions, ["trans_type"], "wallet transaction trans_type", required=False)

    if wtx_ts_col and wtx_user_col and wtx_source_col and wtx_trans_type_col:
        wallet_tx_base = (
            wallet_transactions
            .select(
                F.col(wtx_user_col).cast("long").alias("user_id"),
                F.col(wtx_source_col).alias("source_system"),
                _safe_to_timestamp(wtx_ts_col).alias("wallet_tx_ts"),
                F.lower(F.trim(F.coalesce(F.col(wtx_trans_type_col).cast("string"), F.lit("")))).alias("wallet_tx_trans_type"),
            )
            .filter(F.col("wallet_tx_ts").isNotNull())
            .filter(F.col("wallet_tx_trans_type").isin(wallet_trans_types))
        )

        wallet_tx_joined = (
            registration_with_predeposit_cutoff.alias("r")
            .join(
                wallet_tx_base.alias("w"),
                (F.col("r.user_id") == F.col("w.user_id"))
                & (F.col("r.source_system") == F.col("w.source_system"))
                & (F.col("w.wallet_tx_ts") >= F.col("r.registration_ts"))
                & (F.col("w.wallet_tx_ts") < F.col("r.predeposit_cutoff_ts")),
                "left",
            )
        )

        wallet_feature_aggs = [
            F.count("w.wallet_tx_ts").cast("double").alias("num_wallet_txn_rows_predeposit_30d"),
        ]

        for _trans_type in wallet_trans_types:
            _token = _sanitize_source_system_token(_trans_type)
            wallet_feature_aggs.append(
                F.sum(
                    F.when(F.col("w.wallet_tx_trans_type") == F.lit(_trans_type), F.lit(1)).otherwise(F.lit(0))
                ).cast("double").alias(f"num_wallet_txn_trans_type_{_token}_predeposit_30d")
            )
            wallet_feature_aggs.append(
                F.max(
                    F.when(F.col("w.wallet_tx_trans_type") == F.lit(_trans_type), F.lit(1)).otherwise(F.lit(0))
                ).cast("int").alias(f"has_wallet_txn_trans_type_{_token}_predeposit_30d")
            )

        wallet_features = (
            wallet_tx_joined
            .groupBy("r.user_id", "r.source_system")
            .agg(*wallet_feature_aggs)
        )
    else:
        wallet_features = registration_base.select("user_id", "source_system")
        for _feature_name, _feature_default in wallet_tx_defaults.items():
            if _feature_default is None:
                wallet_features = wallet_features.withColumn(_feature_name, F.lit(None).cast("double"))
            elif isinstance(_feature_default, float):
                wallet_features = wallet_features.withColumn(_feature_name, F.lit(_feature_default).cast("double"))
            elif isinstance(_feature_default, int):
                wallet_features = wallet_features.withColumn(_feature_name, F.lit(_feature_default).cast("int"))
            else:
                wallet_features = wallet_features.withColumn(_feature_name, F.lit(_feature_default))
else:
    wallet_features = registration_base.select("user_id", "source_system")
    for _feature_name, _feature_default in wallet_tx_defaults.items():
        if _feature_default is None:
            wallet_features = wallet_features.withColumn(_feature_name, F.lit(None).cast("double"))
        elif isinstance(_feature_default, float):
            wallet_features = wallet_features.withColumn(_feature_name, F.lit(_feature_default).cast("double"))
        elif isinstance(_feature_default, int):
            wallet_features = wallet_features.withColumn(_feature_name, F.lit(_feature_default).cast("int"))
        else:
            wallet_features = wallet_features.withColumn(_feature_name, F.lit(_feature_default))

# COMMAND ----------

# Direct coupon exposure before first successful deposit
# (or until registration + 30d for users without successful deposit).
if coupons is not None:
    coupon_user_col = _pick_col(coupons, ["uid", "user_id"], "coupon user_id", required=False)
    coupon_source_col = _pick_col(coupons, ["source_system"], "coupon source_system", required=False)
    coupon_id_col = _pick_col(coupons, ["id"], "coupon id", required=False)
    coupon_ts_col = _pick_col(
        coupons,
        ["created_time_uae", "created_time", "valid_start", "updated_time_uae", "updated_time"],
        "coupon timestamp",
        required=False,
    )
    coupon_amount_col = _pick_col(coupons, ["amount"], "coupon amount", required=False)
    coupon_type_col = _pick_col(coupons, ["type"], "coupon type", required=False)
    coupon_status_col = _pick_col(coupons, ["status"], "coupon status", required=False)
    coupon_channel_col = _pick_col(coupons, ["channel"], "coupon channel", required=False)
    coupon_template_col = _pick_col(coupons, ["coupon_tpl_id"], "coupon template", required=False)
    coupon_business_col = _pick_col(coupons, ["business_id"], "coupon business", required=False)
    coupon_event_col = _pick_col(coupons, ["event_id"], "coupon event", required=False)

    if coupon_user_col and coupon_source_col and coupon_ts_col:
        coupon_base = (
            coupons
            .select(
                F.col(coupon_user_col).cast("long").alias("user_id"),
                F.col(coupon_source_col).alias("source_system"),
                F.col(coupon_id_col).cast("string").alias("coupon_id") if coupon_id_col else F.lit(None).cast("string").alias("coupon_id"),
                _safe_to_timestamp(coupon_ts_col).alias("coupon_ts"),
                F.col(coupon_amount_col).cast("double").alias("coupon_amount") if coupon_amount_col else F.lit(None).cast("double").alias("coupon_amount"),
                F.col(coupon_type_col).cast("string").alias("coupon_type_id") if coupon_type_col else F.lit(None).cast("string").alias("coupon_type_id"),
                F.col(coupon_status_col).cast("string").alias("coupon_status_id") if coupon_status_col else F.lit(None).cast("string").alias("coupon_status_id"),
                F.col(coupon_channel_col).cast("string").alias("coupon_channel_id") if coupon_channel_col else F.lit(None).cast("string").alias("coupon_channel_id"),
                F.col(coupon_template_col).cast("string").alias("coupon_template_id") if coupon_template_col else F.lit(None).cast("string").alias("coupon_template_id"),
                F.col(coupon_business_col).cast("string").alias("coupon_business_id") if coupon_business_col else F.lit(None).cast("string").alias("coupon_business_id"),
                F.col(coupon_event_col).cast("string").alias("coupon_event_id") if coupon_event_col else F.lit(None).cast("string").alias("coupon_event_id"),
            )
            .filter(F.col("coupon_ts").isNotNull())
        )

        coupon_features = (
            registration_with_predeposit_cutoff.alias("r")
            .join(
                coupon_base.alias("c"),
                (F.col("r.user_id") == F.col("c.user_id"))
                & (F.col("r.source_system") == F.col("c.source_system"))
                & (F.col("c.coupon_ts") >= F.col("r.registration_ts"))
                & (F.col("c.coupon_ts") < F.col("r.predeposit_cutoff_ts")),
                "left",
            )
            .groupBy("r.user_id", "r.source_system")
            .agg(
                F.count("c.coupon_ts").cast("double").alias("num_coupons_predeposit_30d"),
                F.countDistinct("c.coupon_id").cast("double").alias("num_distinct_coupons_predeposit_30d"),
                F.max(F.when(F.col("c.coupon_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0))).cast("int").alias("has_coupon_predeposit_30d"),
                F.sum(F.coalesce(F.col("c.coupon_amount"), F.lit(0.0))).cast("double").alias("sum_coupon_amount_predeposit_30d"),
                F.avg(F.col("c.coupon_amount")).cast("double").alias("avg_coupon_amount_predeposit_30d"),
                F.max(F.col("c.coupon_amount")).cast("double").alias("max_coupon_amount_predeposit_30d"),
                F.countDistinct("c.coupon_type_id").cast("double").alias("num_coupon_types_predeposit_30d"),
                F.countDistinct("c.coupon_status_id").cast("double").alias("num_coupon_statuses_predeposit_30d"),
                F.countDistinct("c.coupon_channel_id").cast("double").alias("num_coupon_channels_predeposit_30d"),
                F.countDistinct("c.coupon_template_id").cast("double").alias("num_coupon_templates_predeposit_30d"),
                F.countDistinct("c.coupon_business_id").cast("double").alias("num_coupon_business_ids_predeposit_30d"),
                F.countDistinct("c.coupon_event_id").cast("double").alias("num_coupon_event_ids_predeposit_30d"),
                F.min(
                    F.when(
                        F.col("c.coupon_ts").isNotNull(),
                        (F.col("c.coupon_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                    )
                ).cast("double").alias("first_coupon_delay_hours_predeposit_30d"),
            )
        )
    else:
        coupon_features = _with_default_features(
            registration_base.select("user_id", "source_system"),
            COUPON_FEATURE_DEFAULTS,
        )
else:
    coupon_features = _with_default_features(
        registration_base.select("user_id", "source_system"),
        COUPON_FEATURE_DEFAULTS,
    )

# COMMAND ----------

# First successful deposit cohort and first valid wager within 30 days.
dep_user_id_col = _pick_col(deposit, ["user_id"], "deposit user_id")
dep_source_col = _pick_col(deposit, ["source_system"], "deposit source_system")
dep_id_col = _pick_col(deposit, ["id"], "deposit id")
dep_status_col = _pick_col(deposit, ["deposit_status"], "deposit status", required=False)
dep_channel_col = _pick_col(deposit, ["deposit_channel_type"], "deposit channel type", required=False)
dep_mode_col = _pick_col(deposit, ["deposit_mode_type"], "deposit mode type", required=False)
dep_created_ts_col = _pick_col(
    deposit,
    ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
    "deposit attempt timestamp",
)
dep_success_ts_col = _pick_col(
    deposit,
    ["success_time_uae", "success_time"],
    "deposit success timestamp",
    required=False,
)
dep_amount_col = _pick_col(deposit, ["actual_amount", "original_amount"], "deposit amount", required=False)

deposit_base = (
    deposit
    .select(
        F.col(dep_user_id_col).cast("long").alias("user_id"),
        F.col(dep_source_col).cast("string").alias("source_system"),
        F.col(dep_id_col).cast("string").alias("deposit_id"),
        F.col(dep_status_col).cast("int").alias("deposit_status") if dep_status_col else F.lit(None).cast("int").alias("deposit_status"),
        F.col(dep_channel_col).cast("int").alias("deposit_channel_type") if dep_channel_col else F.lit(None).cast("int").alias("deposit_channel_type"),
        F.col(dep_mode_col).cast("int").alias("deposit_mode_type") if dep_mode_col else F.lit(None).cast("int").alias("deposit_mode_type"),
        _safe_to_timestamp(dep_created_ts_col).alias("attempt_ts"),
        _safe_to_timestamp(dep_success_ts_col).alias("success_ts") if dep_success_ts_col else F.lit(None).cast("timestamp").alias("success_ts"),
        F.coalesce(F.col(dep_amount_col).cast("double"), F.lit(0.0)).alias("deposit_amount") if dep_amount_col else F.lit(0.0).alias("deposit_amount"),
    )
    .filter(F.col("attempt_ts").isNotNull())
)

if dep_channel_col:
    deposit_base = deposit_base.filter(
        (F.col("deposit_channel_type") < F.lit(100))
        | (F.col("deposit_channel_type") >= F.lit(200))
        | F.col("deposit_channel_type").isNull()
    )

deposit_success_event_ts = F.when(
    (F.col("d.deposit_status") == F.lit(3)) | F.col("d.success_ts").isNotNull(),
    F.coalesce(F.col("d.success_ts"), F.col("d.attempt_ts")),
)

first_deposit_window = Window.partitionBy(
    "user_id",
    "source_system",
).orderBy(
    F.col("first_deposit_ts").asc(),
    F.col("deposit_id").asc(),
)

first_deposit_features = (
    registration_base.alias("r")
    .join(
        deposit_base.alias("d"),
        (F.col("r.user_id") == F.col("d.user_id"))
        & (F.col("r.source_system") == F.col("d.source_system"))
        & (deposit_success_event_ts >= F.col("r.registration_ts"))
        & (deposit_success_event_ts < F.current_timestamp()),
        "inner",
    )
    .select(
        F.col("r.user_id").alias("user_id"),
        F.col("r.source_system").alias("source_system"),
        F.col("r.registration_ts").alias("registration_ts"),
        F.col("d.deposit_id").alias("deposit_id"),
        deposit_success_event_ts.alias("first_deposit_ts"),
        F.col("d.deposit_amount").cast("double").alias("first_deposit_amount"),
        F.col("d.deposit_channel_type").cast("double").alias("first_deposit_channel_type"),
        F.col("d.deposit_mode_type").cast("double").alias("first_deposit_mode_type"),
    )
    .withColumn("_rn", F.row_number().over(first_deposit_window))
    .filter(F.col("_rn") == F.lit(1))
    .drop("_rn")
    .withColumn("deposit_month", F.date_trunc("month", F.col("first_deposit_ts")).cast("date"))
    .withColumn("deposit_hour", F.hour(F.col("first_deposit_ts")).cast("int"))
    .withColumn("deposit_weekday", F.dayofweek(F.col("first_deposit_ts")).cast("int"))
    .withColumn("deposit_day_of_month", F.dayofmonth(F.col("first_deposit_ts")).cast("int"))
    .withColumn(
        "deposit_weekend_flag",
        F.when(F.dayofweek(F.col("first_deposit_ts")).isin(1, 7), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "registration_to_deposit_hours",
        ((F.col("first_deposit_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0)).cast("double"),
    )
)

deposit_attempt_features = (
    first_deposit_features.alias("fd")
    .join(
        deposit_base.alias("d"),
        (F.col("fd.user_id") == F.col("d.user_id"))
        & (F.col("fd.source_system") == F.col("d.source_system"))
        & (F.col("d.attempt_ts") >= F.col("fd.registration_ts"))
        & (F.col("d.attempt_ts") <= F.col("fd.first_deposit_ts")),
        "left",
    )
    .groupBy("fd.user_id", "fd.source_system")
    .agg(
        F.count(F.col("d.deposit_id")).cast("double").alias("num_deposit_attempts_before_first_success"),
        F.sum(F.when(F.col("d.deposit_status").isin(4, 5), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_failed_deposit_attempts_before_first_success"),
        F.sum(F.coalesce(F.col("d.deposit_amount"), F.lit(0.0))).cast("double").alias("sum_deposit_amount_attempted_before_first_success"),
        F.min(
            ((F.col("d.attempt_ts").cast("long") - F.col("fd.registration_ts").cast("long")) / F.lit(3600.0))
        ).cast("double").alias("first_deposit_attempt_delay_hours"),
    )
)

current_users_dim = users_dim
if "scd_is_current" in current_users_dim.columns:
    current_users_dim = current_users_dim.filter(F.col("scd_is_current") == F.lit(True))

user_sk_col = _pick_col(current_users_dim, ["user_sk"], "dim user_sk")
user_sk_lookup = (
    current_users_dim
    .filter(F.coalesce(F.col(user_dim_is_test_col), F.lit(False)) == F.lit(False))
    .select(
        F.col(user_sk_col).cast("long").alias("user_sk"),
        F.col(user_dim_id_col).cast("long").alias("user_id"),
        F.col(user_dim_source_col).cast("string").alias("source_system"),
    )
    .dropna(subset=["user_sk", "user_id", "source_system"])
    .dropDuplicates(["user_id", "source_system"])
)

deposit_cohort = (
    first_deposit_features.alias("fd")
    .join(user_sk_lookup.alias("u"), on=["user_id", "source_system"], how="inner")
    .select("fd.*", F.col("u.user_sk").alias("user_sk"))
)

wager_user_sk_col = _pick_col(wager, ["user_sk"], "wager user_sk")
wager_source_col = _pick_col(wager, ["source_system", "main_source_system"], "wager source system")
wager_id_col = _pick_col(wager, ["wager_id"], "wager id")
wager_status_col = _pick_col(wager, ["wager_pay_status"], "wager pay status")
wager_ts_col = _pick_col(
    wager,
    ["wager_placement_timestamp", "wager_placement_timestamp_uae", "created_at", "updated_at"],
    "wager placement timestamp",
)
wager_amount_col = _pick_col(wager, ["wager_amount", "wager_amount_native", "amount"], "wager amount", required=False)
wager_platform_col = _pick_col(wager, ["platform_sk"], "wager platform", required=False)
wager_coupon_col = _pick_col(wager, ["coupon_id", "coupon_sk"], "wager coupon", required=False)

wager_base = (
    wager
    .select(
        F.col(wager_user_sk_col).cast("long").alias("user_sk"),
        F.col(wager_source_col).cast("string").alias("source_system"),
        F.col(wager_id_col).cast("string").alias("wager_id"),
        F.col(wager_status_col).cast("string").alias("wager_pay_status"),
        F.to_timestamp(F.col(wager_ts_col)).alias("wager_ts"),
        F.abs(F.col(wager_amount_col).cast("double")).alias("wager_amount") if wager_amount_col else F.lit(0.0).cast("double").alias("wager_amount"),
        F.col(wager_platform_col).cast("string").alias("wager_platform") if wager_platform_col else F.lit(None).cast("string").alias("wager_platform"),
        F.col(wager_coupon_col).cast("string").alias("wager_coupon") if wager_coupon_col else F.lit(None).cast("string").alias("wager_coupon"),
    )
    .filter(F.col("user_sk").isNotNull())
    .filter(F.col("wager_ts").isNotNull())
    .filter(F.col("wager_pay_status") == F.lit("2"))
    .dropDuplicates(["user_sk", "source_system", "wager_id", "wager_ts"])
)

wager_after_deposit = (
    deposit_cohort.alias("d")
    .join(
        wager_base.alias("w"),
        (F.col("d.user_sk") == F.col("w.user_sk"))
        & (F.col("d.source_system") == F.col("w.source_system"))
        & (F.col("w.wager_ts") >= F.col("d.first_deposit_ts"))
        & (F.col("w.wager_ts") < (F.col("d.first_deposit_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"))),
        "left",
    )
)

wager_funnel_features = (
    wager_after_deposit
    .groupBy("d.user_id", "d.source_system")
    .agg(
        F.countDistinct(F.col("w.wager_id")).cast("double").alias("num_wagers_30d"),
        F.sum(F.coalesce(F.col("w.wager_amount"), F.lit(0.0))).cast("double").alias("sum_wager_amount_30d"),
        F.avg(F.col("w.wager_amount")).cast("double").alias("avg_wager_amount_30d"),
        F.countDistinct(F.col("w.wager_platform")).cast("double").alias("num_wager_platforms_30d"),
        F.countDistinct(F.col("w.wager_coupon")).cast("double").alias("num_wager_coupons_30d"),
        F.min(F.col("w.wager_ts")).alias("first_wager_ts"),
    )
)

first_wager_window = Window.partitionBy(
    F.col("d.user_id"),
    F.col("d.source_system"),
).orderBy(
    F.col("w.wager_ts").asc_nulls_last(),
    F.col("w.wager_id").asc_nulls_last(),
)

first_wager_features = (
    wager_after_deposit
    .withColumn("_rn", F.row_number().over(first_wager_window))
    .filter((F.col("_rn") == F.lit(1)) & F.col("w.wager_ts").isNotNull())
    .select(
        F.col("d.user_id").alias("user_id"),
        F.col("d.source_system").alias("source_system"),
        F.col("w.wager_amount").cast("double").alias("first_wager_amount_30d"),
    )
)

wager_funnel_features = (
    wager_funnel_features
    .join(first_wager_features, on=["user_id", "source_system"], how="left")
)

# COMMAND ----------

features = (
    registration_base.alias("r")
    .join(profile_latest.alias("p"), on=["user_id", "source_system"], how="left")
    .join(demographic_features.alias("dv"), on=["user_id", "source_system"], how="left")
    .join(group_features.alias("g"), on=["user_id", "source_system"], how="left")
    .join(login_method_features.alias("m"), on=["user_id", "source_system"], how="left")
    .join(early_login_features.alias("l"), on=["user_id", "source_system"], how="left")
    .join(verification_features.alias("v"), on=["user_id", "source_system"], how="left")
    .join(remark_features.alias("rm"), on=["user_id", "source_system"], how="left")
    .join(wallet_features.alias("w"), on=["user_id", "source_system"], how="left")
    .join(coupon_features.alias("c"), on=["user_id", "source_system"], how="left")
    .join(first_deposit_features.alias("fd"), on=["user_id", "source_system", "registration_ts"], how="inner")
    .join(deposit_attempt_features.alias("da"), on=["user_id", "source_system"], how="left")
    .join(wager_funnel_features.alias("wf"), on=["user_id", "source_system"], how="left")
    .withColumn(
        "observation_complete",
        F.current_timestamp() >= (F.col("first_deposit_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS")),
    )
    .withColumn(
        "label_wager_success_30d",
        F.when(F.coalesce(F.col("num_wagers_30d"), F.lit(0.0)) > F.lit(0.0), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "label_no_wager_30d",
        F.when(F.col("label_wager_success_30d") == F.lit(0), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "first_wager_delay_hours_30d",
        F.when(
            F.col("first_wager_ts").isNotNull(),
            ((F.col("first_wager_ts").cast("long") - F.col("first_deposit_ts").cast("long")) / F.lit(3600.0)).cast("double"),
        ),
    )
    .withColumn(
        "days_since_registration",
        F.datediff(F.current_date(), F.to_date(F.col("registration_ts"))).cast("int"),
    )
    .withColumn(
        "days_since_first_deposit",
        F.datediff(F.current_date(), F.to_date(F.col("first_deposit_ts"))).cast("int"),
    )
    .withColumn("unique_user_id", F.concat_ws("-", F.col("source_system"), F.col("user_id")))
    .filter(F.col("unique_user_id").isNotNull())
    .fillna(0)
    .fillna({"customer_nationality": "UNKNOWN"})
)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in ANALYSES_TO_RUN:
    analysis_name = ANALYSIS_NAMES[analysis]
    filtered_features = _apply_analysis_source_filter(features, analysis)
    filtered_features = _apply_analysis_feature_overrides(filtered_features, analysis)
    published = publish_table(
        spark=spark,
        df=filtered_features.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["unique_user_id", "run_ts"],
        timeseries_column="run_ts",
        append_merge_schema=True,
    )
    print(
        f"Published features: {published['snapshot_table']} and {published['current_table']} "
        f"(run_id={published['run_id']})"
    )
