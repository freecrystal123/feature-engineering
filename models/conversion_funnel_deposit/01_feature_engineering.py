# Databricks notebook source
# MAGIC %md
# MAGIC ## 1 - Conversion funnel feature engineering (register -> deposit)

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_deposit

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

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


def _pick_col(df, candidates: list[str], label: str, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Column not found for {label}. Tried {candidates}. Available: {df.columns}")
    return None


def _present_flag(column_expr):
    return F.when(F.trim(F.coalesce(column_expr.cast("string"), F.lit(""))) != F.lit(""), F.lit(1)).otherwise(F.lit(0))


# COMMAND ----------

users_dim = spark.table(DIM_USER)
users = spark.table(SILVER_USER)
profiles = spark.table(SILVER_USER_PROFILE)
groups = spark.table(SILVER_USER_GROUP)
login_methods = spark.table(SILVER_USER_LOGIN)
login_logs = spark.table(SILVER_USER_LOGIN_LOG)
verification = spark.table(SILVER_USER_VERIFICATION)
deposit = spark.table(SILVER_WALLET_DEPOSIT)

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
    .withColumn("registration_ts", F.to_timestamp(F.col(f"u.{user_ts_col}")))
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
        F.pmod(F.abs(F.hash(F.col("v.source_system"))), F.lit(100000)).cast("int").alias("source_system_hash"),
        *[
            F.coalesce(F.col(f"u.{col_name}").cast("double"), F.lit(0.0)).alias(alias)
            if col_name else F.lit(0.0).alias(alias)
            for alias, col_name in status_cols.items()
        ],
    )
    .dropDuplicates(["user_id", "source_system"])
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
    F.to_timestamp(F.col(profile_ts_col)).desc_nulls_last() if profile_ts_col else F.lit(1)
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
            F.sum(F.when(F.col("group_type").isin(50, 121), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_test_groups"),
        )
        .withColumn("has_any_group", F.when(F.col("num_user_groups") > 0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("num_non_test_groups", F.greatest(F.col("num_user_groups") - F.col("num_test_groups"), F.lit(0.0)))
    )
else:
    group_features = registration_base.select("user_id", "source_system").withColumn("num_user_groups", F.lit(0.0)).withColumn("num_test_groups", F.lit(0.0)).withColumn("has_any_group", F.lit(0)).withColumn("num_non_test_groups", F.lit(0.0))

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
            F.to_timestamp(F.col(login_log_ts_col)).alias("login_event_ts"),
            F.col("login_status").cast("int").alias("login_status") if "login_status" in login_logs.columns else F.lit(None).cast("int").alias("login_status"),
            F.col("ip").cast("string").alias("ip") if "ip" in login_logs.columns else F.lit(None).cast("string").alias("ip"),
            F.col("channel").cast("string").alias("channel") if "channel" in login_logs.columns else F.lit(None).cast("string").alias("channel"),
        )
        .filter(F.col("login_event_ts").isNotNull())
    )

    early_login_features = (
        registration_base.alias("r")
        .join(
            login_logs_base.alias("l"),
            (F.col("r.user_id") == F.col("l.user_id"))
            & (F.col("r.source_system") == F.col("l.source_system"))
            & (F.col("l.login_event_ts") >= F.col("r.registration_ts"))
            & (F.col("l.login_event_ts") < (F.col("r.registration_ts") + F.expr(f"INTERVAL {int(EARLY_ACTIVITY_HOURS)} HOURS"))),
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

if verification_ts_col:
    verification_base = (
        verification
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            F.to_timestamp(F.col(verification_ts_col)).alias("verification_ts"),
            F.col("verified").cast("int").alias("verified") if "verified" in verification.columns else F.lit(None).cast("int").alias("verified"),
            F.col("verified_status").cast("int").alias("verified_status") if "verified_status" in verification.columns else F.lit(None).cast("int").alias("verified_status"),
        )
        .filter(F.col("verification_ts").isNotNull())
    )

    verification_features = (
        registration_base.alias("r")
        .join(
            verification_base.alias("v"),
            (F.col("r.user_id") == F.col("v.user_id"))
            & (F.col("r.source_system") == F.col("v.source_system"))
            & (F.col("v.verification_ts") >= F.col("r.registration_ts"))
            & (F.col("v.verification_ts") < (F.col("r.registration_ts") + F.expr(f"INTERVAL {int(VERIFICATION_WINDOW_DAYS)} DAYS"))),
            "left",
        )
        .withColumn(
            "verified_success_flag",
            F.when((F.col("v.verified") == 1) | (F.col("v.verified_status") == 1), F.lit(1)).otherwise(F.lit(0)),
        )
        .groupBy("r.user_id", "r.source_system")
        .agg(
            F.count(F.col("v.verification_ts")).cast("double").alias("num_verification_events_7d"),
            F.max(F.col("verified_success_flag")).cast("int").alias("verified_in_7d"),
            F.min(
                F.when(
                    F.col("v.verification_ts").isNotNull(),
                    (F.col("v.verification_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
                )
            ).cast("double").alias("first_verification_delay_hours_7d"),
        )
        .withColumn("has_verification_event_7d", F.when(F.col("num_verification_events_7d") > 0, F.lit(1)).otherwise(F.lit(0)))
    )
else:
    verification_features = registration_base.select("user_id", "source_system").withColumn("num_verification_events_7d", F.lit(0.0)).withColumn("verified_in_7d", F.lit(0)).withColumn("first_verification_delay_hours_7d", F.lit(None).cast("double")).withColumn("has_verification_event_7d", F.lit(0))

# COMMAND ----------

# Deposit attempts/success in first 30d (target and funnel diagnostics)
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
        F.col(dep_source_col).alias("source_system"),
        F.col(dep_id_col).cast("string").alias("deposit_id"),
        F.col(dep_status_col).cast("int").alias("deposit_status") if dep_status_col else F.lit(None).cast("int").alias("deposit_status"),
        F.col(dep_channel_col).cast("int").alias("deposit_channel_type") if dep_channel_col else F.lit(None).cast("int").alias("deposit_channel_type"),
        F.col(dep_mode_col).cast("int").alias("deposit_mode_type") if dep_mode_col else F.lit(None).cast("int").alias("deposit_mode_type"),
        F.to_timestamp(F.col(dep_created_ts_col)).alias("attempt_ts"),
        F.to_timestamp(F.col(dep_success_ts_col)).alias("success_ts") if dep_success_ts_col else F.lit(None).cast("timestamp").alias("success_ts"),
        F.coalesce(F.col(dep_amount_col).cast("double"), F.lit(0.0)).alias("attempt_amount") if dep_amount_col else F.lit(0.0).alias("attempt_amount"),
    )
    .filter(F.col("attempt_ts").isNotNull())
)

if dep_channel_col:
    deposit_base = deposit_base.filter(
        (F.col("deposit_channel_type") < 100) | (F.col("deposit_channel_type") >= 200) | F.col("deposit_channel_type").isNull()
    )

deposit_funnel_features = (
    registration_base.alias("r")
    .join(
        deposit_base.alias("d"),
        (F.col("r.user_id") == F.col("d.user_id"))
        & (F.col("r.source_system") == F.col("d.source_system"))
        & (F.col("d.attempt_ts") >= F.col("r.registration_ts"))
        & (F.col("d.attempt_ts") < (F.col("r.registration_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"))),
        "left",
    )
    .groupBy("r.user_id", "r.source_system", "r.registration_ts")
    .agg(
        F.count(F.col("d.deposit_id")).cast("double").alias("num_deposit_attempts_30d"),
        F.sum(F.when(F.col("d.deposit_status") == 3, F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_deposit_success_30d"),
        F.sum(F.when(F.col("d.deposit_status").isin(4, 5), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_deposit_failure_30d"),
        F.sum(F.when(F.col("d.deposit_status").isin(1, 2), F.lit(1)).otherwise(F.lit(0))).cast("double").alias("num_deposit_pending_30d"),
        F.sum(F.col("d.attempt_amount")).cast("double").alias("sum_deposit_amount_attempted_30d"),
        F.avg(F.col("d.attempt_amount")).cast("double").alias("avg_deposit_amount_attempted_30d"),
        F.countDistinct(F.col("d.deposit_channel_type")).cast("double").alias("num_deposit_channels_30d"),
        F.countDistinct(F.col("d.deposit_mode_type")).cast("double").alias("num_deposit_modes_30d"),
        F.min(
            F.when(
                F.col("d.attempt_ts").isNotNull(),
                (F.col("d.attempt_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
            )
        ).cast("double").alias("first_attempt_delay_hours_30d"),
        F.min(
            F.when(
                (F.col("d.deposit_status") == 3) & F.col("d.success_ts").isNotNull(),
                (F.col("d.success_ts").cast("long") - F.col("r.registration_ts").cast("long")) / 3600.0,
            )
        ).cast("double").alias("first_success_delay_hours_30d"),
    )
)

# COMMAND ----------

features = (
    registration_base.alias("r")
    .join(profile_latest.alias("p"), on=["user_id", "source_system"], how="left")
    .join(group_features.alias("g"), on=["user_id", "source_system"], how="left")
    .join(login_method_features.alias("m"), on=["user_id", "source_system"], how="left")
    .join(early_login_features.alias("l"), on=["user_id", "source_system"], how="left")
    .join(verification_features.alias("v"), on=["user_id", "source_system"], how="left")
    .join(deposit_funnel_features.alias("d"), on=["user_id", "source_system", "registration_ts"], how="left")
    .withColumn("observation_complete", (F.current_timestamp() >= (F.col("registration_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"))))
    .withColumn("label_deposit_attempt_30d", F.when(F.coalesce(F.col("num_deposit_attempts_30d"), F.lit(0.0)) > 0, F.lit(1)).otherwise(F.lit(0)))
    .withColumn("label_deposit_success_30d", F.when(F.coalesce(F.col("num_deposit_success_30d"), F.lit(0.0)) > 0, F.lit(1)).otherwise(F.lit(0)))
    .withColumn(
        "label_failed_after_attempt_30d",
        F.when(
            (F.coalesce(F.col("num_deposit_attempts_30d"), F.lit(0.0)) > 0)
            & (F.coalesce(F.col("num_deposit_success_30d"), F.lit(0.0)) == 0),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    .withColumn("days_since_registration", F.datediff(F.current_date(), F.to_date(F.col("registration_ts"))).cast("int"))
    .withColumn("unique_user_id", F.concat_ws("-", F.col("source_system"), F.col("user_id")))
    .filter(F.col("unique_user_id").isNotNull())
    .fillna(
        {
            "register_type_id": 0.0,
            "profile_product_type_id": 0.0,
            "profile_platform_id": 0.0,
            "profile_os_type_id": 0.0,
            "mail_valid_status_id": 0.0,
            "phone_valid_status_id": 0.0,
            "country_id": 0.0,
            "nation_id": 0.0,
            "resident_country_id": 0.0,
            "has_marketing_channel": 0,
            "has_marketing_source": 0,
            "has_referrer": 0,
            "has_referrer_code": 0,
            "has_device_code": 0,
            "has_user_agent": 0,
            "has_imei": 0,
            "has_idfa": 0,
            "has_android_id": 0,
            "has_cookie_data": 0,
            "num_user_groups": 0.0,
            "num_test_groups": 0.0,
            "has_any_group": 0,
            "num_non_test_groups": 0.0,
            "num_login_methods": 0.0,
            "num_active_login_methods": 0.0,
            "num_third_party_login_methods": 0.0,
            "has_sign_login_name": 0,
            "num_login_events_24h": 0.0,
            "num_unsuccessful_login_events_24h": 0.0,
            "num_distinct_login_ip_24h": 0.0,
            "num_distinct_login_channel_24h": 0.0,
            "num_verification_events_7d": 0.0,
            "verified_in_7d": 0,
            "has_verification_event_7d": 0,
            "num_deposit_attempts_30d": 0.0,
            "num_deposit_success_30d": 0.0,
            "num_deposit_failure_30d": 0.0,
            "num_deposit_pending_30d": 0.0,
            "sum_deposit_amount_attempted_30d": 0.0,
            "avg_deposit_amount_attempted_30d": 0.0,
            "num_deposit_channels_30d": 0.0,
            "num_deposit_modes_30d": 0.0,
        }
    )
)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    published = publish_table(
        spark=spark,
        df=features.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["unique_user_id", "run_ts"],
        timeseries_column="run_ts",
    )
    print(
        f"Published features: {published['snapshot_table']} and {published['current_table']} "
        f"(run_id={published['run_id']})"
    )
