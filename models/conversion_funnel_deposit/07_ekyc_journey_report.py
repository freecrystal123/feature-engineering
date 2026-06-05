# Databricks notebook source
# MAGIC %md
# MAGIC ## 7 - eKYC-to-FTD customer journey report dataset

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_deposit

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("journey_window_days", str(EKYC_JOURNEY_WINDOW_DAYS))

analysis_id = dbutils.widgets.get("analysis_id").strip()


def _as_positive_int(raw_value: str, default_value: int, minimum: int = 1) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except Exception:
        return int(default_value)
    return max(int(minimum), parsed)


journey_window_days = _as_positive_int(
    dbutils.widgets.get("journey_window_days"),
    int(EKYC_JOURNEY_WINDOW_DAYS),
    minimum=1,
)


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def _pick_col(df, candidates: list[str], label: str, required: bool = True) -> str | None:
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    if required:
        raise ValueError(f"Column not found for {label}. Tried {candidates}. Available: {df.columns}")
    return None


def _quote_identifier(column_name: str) -> str:
    raw = str(column_name).strip()
    parts = [part.strip() for part in raw.split(".") if part.strip()]
    quoted_parts = []
    for part in parts:
        clean = part[1:-1] if part.startswith("`") and part.endswith("`") else part
        quoted_parts.append(f"`{clean.replace('`', '``')}`")
    return ".".join(quoted_parts)


def _safe_to_timestamp(column_name: str):
    quoted = _quote_identifier(column_name)
    return F.coalesce(
        F.expr(f"try_to_timestamp({quoted})"),
        F.expr(f"try_to_timestamp({quoted}, 'yyyy-MM-dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp({quoted}, 'yyyy-MM-dd HH:mm:ss.SSS')"),
        F.expr(f"try_to_timestamp({quoted}, 'yyyy/MM/dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp({quoted}, 'dd/MM/yyyy HH:mm:ss')"),
        F.expr(f"try_to_timestamp({quoted}, 'yyyy-MM-dd')"),
    )


def _available_timestamp_expr(df, candidates: list[str], label: str, required: bool = True):
    available = [column_name for column_name in candidates if column_name in df.columns]
    if not available:
        if required:
            raise ValueError(f"Timestamp columns not found for {label}. Tried {candidates}. Available: {df.columns}")
        return None
    return F.coalesce(*[_safe_to_timestamp(column_name) for column_name in available])


def _optional_table(table_name: str):
    try:
        if not spark.catalog.tableExists(table_name):
            return None
        return spark.table(table_name)
    except Exception as exc:
        print({"optional_table_unavailable": table_name, "reason": str(exc)})
        return None


def _normalize_status_int(column_expr):
    cleaned = F.regexp_extract(F.trim(F.coalesce(column_expr.cast("string"), F.lit(""))), r"(-?[0-9]+)", 1)
    return F.when(cleaned == F.lit(""), F.lit(None).cast("int")).otherwise(cleaned.cast("int"))


def _label_from_mapping(column_expr, mapping: dict[int, str], unknown_prefix: str):
    result = F.when(column_expr.isNull(), F.lit("Missing"))
    for code, label in sorted(mapping.items()):
        result = result.when(column_expr == F.lit(int(code)), F.lit(str(label)))
    return result.otherwise(F.concat(F.lit(f"{unknown_prefix} "), column_expr.cast("string")))


VERIFICATION_STATUS_LABELS = {
    1: "Not verified",
    2: "Under review",
    3: "Verified",
    4: "Verification failed",
    5: "Black list",
    6: "Register init",
    7: "Verification expired",
}

FAIL_REASON_LABELS = {
    1: "Picture fuzzy",
    2: "Inconsistent info",
    3: "Other",
    4: "DOB not match",
    5: "EID expired",
    6: "Invalid EID",
    7: "Data missing",
    8: "Over max time",
    9: "Name not match",
    10: "Face not match",
    11: "Live not match",
    12: "Expire date not match",
    13: "KYC service error",
    14: "Invalid passport",
    15: "Name matching failed",
    16: "Birth date matching failed",
    17: "Birth date not match register",
    18: "EID verification service error",
    19: "Expire date existence review",
    20: "EID nationality code check fail",
    21: "Birthday existence review",
    22: "Full name existence review",
    23: "Unknown error",
}

DEPOSIT_STATUS_LABELS = {
    1: "Pending",
    2: "Handling",
    3: "Success",
    4: "Failure",
    5: "Timeout",
}


def _verification_events_from_df(verification_df, event_source: str, status_history_source: bool):
    user_id_col = _pick_col(verification_df, ["user_id", "uid"], f"{event_source} user_id", required=False)
    source_col = _pick_col(verification_df, ["source_system"], f"{event_source} source_system", required=False)
    event_id_col = _pick_col(verification_df, ["id", "ticket_id"], f"{event_source} event id", required=False)
    attempt_ts_expr = _available_timestamp_expr(
        verification_df,
        [
            "created_time_uae",
            "created_time",
            "create_time_uae",
            "create_time",
            "row_last_updated_uae",
            "row_last_updated",
            "updated_time_uae",
            "updated_time",
        ],
        f"{event_source} attempt timestamp",
        required=False,
    )
    status_ts_candidates = (
        [
            "verified_time_uae",
            "verified_time",
            "updated_time_uae",
            "updated_time",
            "row_last_updated_uae",
            "row_last_updated",
            "created_time_uae",
            "created_time",
            "create_time_uae",
            "create_time",
        ]
        if status_history_source
        else [
            "created_time_uae",
            "created_time",
            "create_time_uae",
            "create_time",
            "updated_time_uae",
            "updated_time",
            "row_last_updated_uae",
            "row_last_updated",
        ]
    )
    status_ts_expr = _available_timestamp_expr(
        verification_df,
        status_ts_candidates,
        f"{event_source} status timestamp",
        required=False,
    )
    verify_type_col = _pick_col(verification_df, ["verify_type"], f"{event_source} verify_type", required=False)
    verified_status_col = _pick_col(
        verification_df,
        ["verified_status"],
        f"{event_source} verified_status",
        required=False,
    )
    fail_reason_col = _pick_col(verification_df, ["fail_reason"], f"{event_source} fail_reason", required=False)
    succeed_col = _pick_col(verification_df, ["succeed", "verified"], f"{event_source} success flag", required=False)

    if (
        not user_id_col
        or not source_col
        or attempt_ts_expr is None
        or status_ts_expr is None
        or not verify_type_col
    ):
        return None

    event_id_expr = F.col(event_id_col).cast("string") if event_id_col else F.lit("")
    verified_status_expr = (
        _normalize_status_int(F.col(verified_status_col))
        if verified_status_col
        else F.lit(None).cast("int")
    )
    fail_reason_expr = (
        _normalize_status_int(F.col(fail_reason_col))
        if fail_reason_col
        else F.lit(None).cast("int")
    )
    succeed_expr = (
        _normalize_status_int(F.col(succeed_col))
        if succeed_col
        else F.lit(None).cast("int")
    )

    events = (
        verification_df
        .select(
            F.col(user_id_col).cast("long").alias("user_id"),
            F.col(source_col).cast("string").alias("source_system"),
            event_id_expr.alias("verification_event_id"),
            F.coalesce(attempt_ts_expr, status_ts_expr).alias("verification_attempt_ts"),
            F.coalesce(status_ts_expr, attempt_ts_expr).alias("verification_event_ts"),
            _normalize_status_int(F.col(verify_type_col)).alias("verify_type_id"),
            verified_status_expr.alias("verified_status_id"),
            fail_reason_expr.alias("fail_reason_id"),
            succeed_expr.alias("succeed_flag"),
            F.lit(event_source).alias("verification_event_source"),
        )
        .withColumn(
            "valid_fail_reason_id",
            F.when(F.col("fail_reason_id") > F.lit(0), F.col("fail_reason_id")),
        )
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("source_system").isNotNull())
        .filter(F.col("verification_event_ts").isNotNull())
        .filter(F.col("verify_type_id") == F.lit(3))
        .withColumn(
            "ekyc_status",
            F.when(F.col("verified_status_id").isNotNull(), _label_from_mapping(F.col("verified_status_id"), VERIFICATION_STATUS_LABELS, "Status"))
            .when(F.col("succeed_flag") == F.lit(1), F.lit("Verified"))
            .when(F.col("valid_fail_reason_id").isNotNull(), F.lit("Verification failed"))
            .otherwise(F.lit("Unknown")),
        )
        .withColumn("fail_reason_label", _label_from_mapping(F.col("valid_fail_reason_id"), FAIL_REASON_LABELS, "Reason"))
        .withColumn(
            "is_explicit_failure",
            F.when(
                (F.col("verified_status_id") == F.lit(4)) | F.col("valid_fail_reason_id").isNotNull(),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn("is_under_review", F.when(F.col("verified_status_id") == F.lit(2), F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "is_verified",
            F.when(
                (F.col("verified_status_id") == F.lit(3)) | (F.col("succeed_flag") == F.lit(1)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn("is_expired", F.when(F.col("verified_status_id") == F.lit(7), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("is_black_list", F.when(F.col("verified_status_id") == F.lit(5), F.lit(1)).otherwise(F.lit(0)))
    )
    return events


def _build_verification_events():
    current_events = _verification_events_from_df(
        spark.table(SILVER_USER_VERIFICATION),
        "user_verification",
        status_history_source=False,
    )
    log_df = _optional_table(SILVER_USER_VERIFICATION_LOG)
    log_events = (
        _verification_events_from_df(
            log_df,
            "user_verification_log",
            status_history_source=True,
        )
        if log_df is not None
        else None
    )

    if current_events is None:
        raise ValueError(
            f"{SILVER_USER_VERIFICATION} does not contain the required user, source, timestamp, and verify_type columns."
        )

    if log_events is None:
        return current_events, "user_verification attempt outcomes"

    log_users = log_events.select("user_id", "source_system").dropDuplicates()
    current_without_log = current_events.join(log_users, on=["user_id", "source_system"], how="left_anti")
    return (
        (
            log_events
            .unionByName(current_without_log, allowMissingColumns=True)
            .dropDuplicates(
                [
                    "user_id",
                    "source_system",
                    "verification_attempt_ts",
                    "verification_event_ts",
                "verify_type_id",
                "verified_status_id",
                "fail_reason_id",
                "valid_fail_reason_id",
            ]
        )
        ),
        "user_verification_log status history with user_verification fallback",
    )


def _build_deposit_base():
    deposit_df = spark.table(SILVER_WALLET_DEPOSIT)
    user_id_col = _pick_col(deposit_df, ["user_id"], "deposit user_id")
    source_col = _pick_col(deposit_df, ["source_system"], "deposit source_system")
    deposit_id_col = _pick_col(deposit_df, ["id"], "deposit id")
    status_col = _pick_col(deposit_df, ["deposit_status"], "deposit status", required=False)
    channel_col = _pick_col(deposit_df, ["deposit_channel_type"], "deposit channel", required=False)
    attempt_ts_col = _pick_col(
        deposit_df,
        ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
        "deposit attempt timestamp",
    )
    success_ts_col = _pick_col(
        deposit_df,
        ["success_time_uae", "success_time"],
        "deposit success timestamp",
        required=False,
    )

    deposit_base = (
        deposit_df
        .select(
            F.col(user_id_col).cast("long").alias("user_id"),
            F.col(source_col).cast("string").alias("source_system"),
            F.col(deposit_id_col).cast("string").alias("deposit_id"),
            F.col(status_col).cast("int").alias("deposit_status_id") if status_col else F.lit(None).cast("int").alias("deposit_status_id"),
            F.col(channel_col).cast("int").alias("deposit_channel_type") if channel_col else F.lit(None).cast("int").alias("deposit_channel_type"),
            _safe_to_timestamp(attempt_ts_col).alias("ftd_attempt_ts"),
            _safe_to_timestamp(success_ts_col).alias("ftd_success_ts") if success_ts_col else F.lit(None).cast("timestamp").alias("ftd_success_ts"),
        )
        .filter(F.col("ftd_attempt_ts").isNotNull())
    )

    if channel_col:
        deposit_base = deposit_base.filter(
            (F.col("deposit_channel_type") < F.lit(100))
            | (F.col("deposit_channel_type") >= F.lit(200))
            | F.col("deposit_channel_type").isNull()
        )

    return (
        deposit_base
        .withColumn(
            "is_ftd_success",
            F.when(
                (F.col("deposit_status_id") == F.lit(3)) | F.col("ftd_success_ts").isNotNull(),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "is_ftd_failure",
            F.when(F.col("deposit_status_id").isin(4, 5), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "is_ftd_pending",
            F.when(F.col("deposit_status_id").isin(1, 2), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("ftd_status", _label_from_mapping(F.col("deposit_status_id"), DEPOSIT_STATUS_LABELS, "Status"))
        .withColumn(
            "ftd_success_event_ts",
            F.when(
                F.col("is_ftd_success") == F.lit(1),
                F.coalesce(F.col("ftd_success_ts"), F.col("ftd_attempt_ts")),
            ),
        )
    )


def _build_coupon_base():
    coupon_df = _optional_table(SILVER_WALLET_COUPON)
    if coupon_df is None:
        return None

    user_id_col = _pick_col(coupon_df, ["uid", "user_id"], "coupon user_id", required=False)
    source_col = _pick_col(coupon_df, ["source_system"], "coupon source system", required=False)
    coupon_id_col = _pick_col(coupon_df, ["id"], "coupon id", required=False)
    timestamp_col = _pick_col(
        coupon_df,
        ["created_time_uae", "created_time", "valid_start", "updated_time_uae", "updated_time"],
        "coupon timestamp",
        required=False,
    )
    amount_col = _pick_col(coupon_df, ["amount"], "coupon amount", required=False)
    status_col = _pick_col(coupon_df, ["status"], "coupon status", required=False)
    type_col = _pick_col(coupon_df, ["type"], "coupon type", required=False)

    if not user_id_col or not source_col or not timestamp_col:
        return None

    return (
        coupon_df
        .select(
            F.col(user_id_col).cast("long").alias("user_id"),
            F.col(source_col).cast("string").alias("source_system"),
            F.col(coupon_id_col).cast("string").alias("coupon_id") if coupon_id_col else F.lit("").alias("coupon_id"),
            _safe_to_timestamp(timestamp_col).alias("coupon_ts"),
            F.col(amount_col).cast("double").alias("coupon_amount") if amount_col else F.lit(None).cast("double").alias("coupon_amount"),
            F.col(status_col).cast("int").alias("coupon_status_id") if status_col else F.lit(None).cast("int").alias("coupon_status_id"),
            F.col(type_col).cast("int").alias("coupon_type_id") if type_col else F.lit(None).cast("int").alias("coupon_type_id"),
        )
        .filter(F.col("coupon_ts").isNotNull())
    )


verification_events, verification_history_source = _build_verification_events()
deposit_base = _build_deposit_base()
coupon_base = _build_coupon_base()

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    feat_df = read_current_table(
        spark=spark,
        table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}",
    )

    cohort = (
        feat_df
        .select(
            F.col("unique_user_id").cast("string").alias("unique_user_id"),
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").cast("string").alias("source_system"),
            F.col("registration_ts").cast("timestamp").alias("registration_ts"),
            F.col("registration_month").cast("date").alias("registration_month"),
            F.col("observation_complete").cast("boolean").alias("observation_complete"),
        )
        .dropDuplicates(["unique_user_id"])
        .withColumn(
            "journey_end_ts",
            F.col("registration_ts") + F.expr(f"INTERVAL {int(journey_window_days)} DAYS"),
        )
    )

    deposit_events_all = (
        cohort.alias("c")
        .join(
            deposit_base.alias("d"),
            (F.col("c.user_id") == F.col("d.user_id"))
            & (F.col("c.source_system") == F.col("d.source_system"))
            & (F.col("d.ftd_attempt_ts") >= F.col("c.registration_ts"))
            & (F.col("d.ftd_attempt_ts") < F.col("c.journey_end_ts")),
            "inner",
        )
        .select("c.unique_user_id", "d.*")
    )

    first_ftd_success = (
        deposit_events_all
        .filter(F.col("is_ftd_success") == F.lit(1))
        .groupBy("unique_user_id")
        .agg(F.min("ftd_success_event_ts").alias("_first_ftd_success_cutoff_ts"))
    )
    deposit_events = (
        deposit_events_all
        .join(first_ftd_success, on="unique_user_id", how="left")
        .filter(
            F.col("_first_ftd_success_cutoff_ts").isNull()
            | (F.col("ftd_attempt_ts") <= F.col("_first_ftd_success_cutoff_ts"))
        )
        .drop("_first_ftd_success_cutoff_ts")
    )

    deposit_agg = (
        deposit_events
        .groupBy("unique_user_id")
        .agg(
            F.countDistinct("deposit_id").cast("long").alias("num_ftd_attempts_30d"),
            F.sum("is_ftd_success").cast("long").alias("num_ftd_successes_30d"),
            F.sum("is_ftd_failure").cast("long").alias("num_ftd_failures_30d"),
            F.sum("is_ftd_pending").cast("long").alias("num_ftd_pending_events_30d"),
            F.min("ftd_attempt_ts").alias("first_ftd_attempt_ts"),
            F.min("ftd_success_event_ts").alias("first_ftd_success_ts"),
            F.min(F.when(F.col("is_ftd_failure") == F.lit(1), F.col("ftd_attempt_ts"))).alias("first_ftd_failure_ts"),
            F.min(F.when(F.col("is_ftd_pending") == F.lit(1), F.col("ftd_attempt_ts"))).alias("first_ftd_pending_ts"),
        )
        .withColumn("has_ftd_attempt_30d", F.when(F.col("num_ftd_attempts_30d") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("has_ftd_success_30d", F.when(F.col("num_ftd_successes_30d") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("has_ftd_failure_30d", F.when(F.col("num_ftd_failures_30d") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("has_ftd_pending_30d", F.when(F.col("num_ftd_pending_events_30d") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "ftd_recovered_after_failure",
            F.when(
                F.col("first_ftd_failure_ts").isNotNull()
                & F.col("first_ftd_success_ts").isNotNull()
                & (F.col("first_ftd_success_ts") > F.col("first_ftd_failure_ts")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "ftd_recovery_hours_after_failure",
            F.when(
                F.col("ftd_recovered_after_failure") == F.lit(1),
                (F.col("first_ftd_success_ts").cast("long") - F.col("first_ftd_failure_ts").cast("long")) / F.lit(3600.0),
            ),
        )
    )

    deposit_order = Window.partitionBy("unique_user_id").orderBy(
        F.col("ftd_attempt_ts").asc(),
        F.col("deposit_id").asc(),
    )
    deposit_reverse_order = Window.partitionBy("unique_user_id").orderBy(
        F.col("ftd_attempt_ts").desc(),
        F.col("deposit_id").desc(),
    )
    first_deposit_status = (
        deposit_events
        .withColumn("_rn", F.row_number().over(deposit_order))
        .filter(F.col("_rn") == F.lit(1))
        .select("unique_user_id", F.col("ftd_status").alias("first_ftd_status"))
    )
    latest_deposit_status = (
        deposit_events
        .withColumn("_rn", F.row_number().over(deposit_reverse_order))
        .filter(F.col("_rn") == F.lit(1))
        .select("unique_user_id", F.col("ftd_status").alias("latest_ftd_status"))
    )
    deposit_features = (
        deposit_agg
        .join(first_deposit_status, on="unique_user_id", how="left")
        .join(latest_deposit_status, on="unique_user_id", how="left")
    )

    cohort_with_ftd = (
        cohort
        .join(deposit_features, on="unique_user_id", how="left")
        .fillna(
            {
                "num_ftd_attempts_30d": 0,
                "num_ftd_successes_30d": 0,
                "num_ftd_failures_30d": 0,
                "num_ftd_pending_events_30d": 0,
                "has_ftd_attempt_30d": 0,
                "has_ftd_success_30d": 0,
                "has_ftd_failure_30d": 0,
                "has_ftd_pending_30d": 0,
                "ftd_recovered_after_failure": 0,
            }
        )
        .withColumn(
            "verification_cutoff_ts",
            F.least(
                F.col("journey_end_ts"),
                F.coalesce(F.col("first_ftd_success_ts"), F.col("journey_end_ts")),
            ),
        )
        .withColumn(
            "has_ftd_failure_unrecovered",
            F.when(
                (F.col("has_ftd_failure_30d") == F.lit(1))
                & (F.col("has_ftd_success_30d") == F.lit(0)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "has_ftd_pending_unresolved",
            F.when(
                F.col("latest_ftd_status").isin("Pending", "Handling")
                & (F.col("has_ftd_success_30d") == F.lit(0)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    )

    if verification_events is not None:
        ekyc_events = (
            cohort_with_ftd.alias("c")
            .join(
                verification_events.alias("v"),
                (F.col("c.user_id") == F.col("v.user_id"))
                & (F.col("c.source_system") == F.col("v.source_system"))
                & (F.col("v.verification_event_ts") >= F.col("c.registration_ts"))
                & (F.col("v.verification_event_ts") < F.col("c.verification_cutoff_ts")),
                "inner",
            )
            .select("c.unique_user_id", "v.*")
        )

        ekyc_agg = (
            ekyc_events
            .groupBy("unique_user_id")
            .agg(
                F.count("*").cast("long").alias("num_ekyc_events_30d"),
                F.countDistinct("verified_status_id").cast("long").alias("num_ekyc_distinct_statuses_30d"),
                F.max("is_verified").cast("int").alias("has_ekyc_verified_30d"),
                F.max("is_explicit_failure").cast("int").alias("has_ekyc_failure_30d"),
                F.max("is_under_review").cast("int").alias("has_ekyc_under_review_30d"),
                F.max("is_expired").cast("int").alias("has_ekyc_expired_30d"),
                F.max("is_black_list").cast("int").alias("has_ekyc_black_list_30d"),
                F.min("verification_attempt_ts").alias("first_ekyc_event_ts"),
                F.max("verification_event_ts").alias("last_ekyc_event_ts"),
                F.min(F.when(F.col("is_verified") == F.lit(1), F.col("verification_event_ts"))).alias("first_ekyc_verified_ts"),
                F.min(F.when(F.col("is_explicit_failure") == F.lit(1), F.col("verification_event_ts"))).alias("first_ekyc_failure_ts"),
                F.min(F.when(F.col("is_under_review") == F.lit(1), F.col("verification_event_ts"))).alias("first_ekyc_under_review_ts"),
                F.countDistinct(F.col("valid_fail_reason_id")).cast("long").alias("num_ekyc_fail_reasons_30d"),
            )
            .withColumn("has_ekyc_event_30d", F.when(F.col("num_ekyc_events_30d") > F.lit(0), F.lit(1)).otherwise(F.lit(0)))
        )

        ekyc_order = Window.partitionBy("unique_user_id").orderBy(
            F.col("verification_event_ts").asc(),
            F.col("verification_event_id").asc(),
        )
        ekyc_reverse_order = Window.partitionBy("unique_user_id").orderBy(
            F.col("verification_event_ts").desc(),
            F.col("verification_event_id").desc(),
        )
        latest_ekyc_status = (
            ekyc_events
            .withColumn("_rn", F.row_number().over(ekyc_reverse_order))
            .filter(F.col("_rn") == F.lit(1))
            .select("unique_user_id", F.col("ekyc_status").alias("latest_ekyc_status"))
        )
        first_failure_reason = (
            ekyc_events
            .filter(F.col("is_explicit_failure") == F.lit(1))
            .withColumn("_rn", F.row_number().over(ekyc_order))
            .filter(F.col("_rn") == F.lit(1))
            .select("unique_user_id", F.col("fail_reason_label").alias("first_ekyc_fail_reason"))
        )
        status_changes = (
            ekyc_events
            .withColumn("_previous_status", F.lag("ekyc_status").over(ekyc_order))
            .filter(F.col("_previous_status").isNull() | (F.col("ekyc_status") != F.col("_previous_status")))
        )
        status_sequences = (
            status_changes
            .groupBy("unique_user_id")
            .agg(
                F.count("*").cast("long").alias("num_ekyc_status_changes_30d"),
                F.expr(
                    "concat_ws(' → ', transform("
                    "sort_array(collect_list(named_struct("
                    "'event_ts', verification_event_ts, "
                    "'event_id', verification_event_id, "
                    "'status', ekyc_status"
                    "))), x -> x.status))"
                ).alias("ekyc_status_sequence"),
            )
        )
        recovered_after_failure = (
            ekyc_events.alias("e")
            .join(
                ekyc_agg.select("unique_user_id", "first_ekyc_failure_ts").alias("a"),
                on="unique_user_id",
                how="inner",
            )
            .filter(F.col("e.is_verified") == F.lit(1))
            .filter(F.col("e.verification_event_ts") > F.col("a.first_ekyc_failure_ts"))
            .groupBy("unique_user_id")
            .agg(F.min(F.col("e.verification_event_ts")).alias("first_ekyc_verified_after_failure_ts"))
        )
        recovered_after_review = (
            ekyc_events.alias("e")
            .join(
                ekyc_agg.select("unique_user_id", "first_ekyc_under_review_ts").alias("a"),
                on="unique_user_id",
                how="inner",
            )
            .filter(F.col("e.is_verified") == F.lit(1))
            .filter(F.col("e.verification_event_ts") > F.col("a.first_ekyc_under_review_ts"))
            .groupBy("unique_user_id")
            .agg(F.min(F.col("e.verification_event_ts")).alias("first_ekyc_verified_after_review_ts"))
        )
        ekyc_features = (
            ekyc_agg
            .join(latest_ekyc_status, on="unique_user_id", how="left")
            .join(first_failure_reason, on="unique_user_id", how="left")
            .join(status_sequences, on="unique_user_id", how="left")
            .join(recovered_after_failure, on="unique_user_id", how="left")
            .join(recovered_after_review, on="unique_user_id", how="left")
            .withColumn(
                "ekyc_recovered_after_failure",
                F.when(F.col("first_ekyc_verified_after_failure_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
            )
            .withColumn(
                "ekyc_recovered_after_review",
                F.when(F.col("first_ekyc_verified_after_review_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
            )
            .withColumn(
                "ekyc_recovery_hours_after_failure",
                F.when(
                    F.col("first_ekyc_verified_after_failure_ts").isNotNull(),
                    (
                        F.col("first_ekyc_verified_after_failure_ts").cast("long")
                        - F.col("first_ekyc_failure_ts").cast("long")
                    )
                    / F.lit(3600.0),
                ),
            )
            .withColumn(
                "ekyc_review_to_verified_hours",
                F.when(
                    F.col("first_ekyc_verified_after_review_ts").isNotNull(),
                    (
                        F.col("first_ekyc_verified_after_review_ts").cast("long")
                        - F.col("first_ekyc_under_review_ts").cast("long")
                    )
                    / F.lit(3600.0),
                ),
            )
        )
    else:
        ekyc_features = (
            cohort
            .select("unique_user_id")
            .withColumn("num_ekyc_events_30d", F.lit(0).cast("long"))
            .withColumn("num_ekyc_distinct_statuses_30d", F.lit(0).cast("long"))
            .withColumn("has_ekyc_event_30d", F.lit(0))
            .withColumn("has_ekyc_verified_30d", F.lit(0))
            .withColumn("has_ekyc_failure_30d", F.lit(0))
            .withColumn("has_ekyc_under_review_30d", F.lit(0))
            .withColumn("has_ekyc_expired_30d", F.lit(0))
            .withColumn("has_ekyc_black_list_30d", F.lit(0))
            .withColumn("ekyc_recovered_after_failure", F.lit(0))
            .withColumn("ekyc_recovered_after_review", F.lit(0))
            .withColumn("num_ekyc_fail_reasons_30d", F.lit(0).cast("long"))
            .withColumn("num_ekyc_status_changes_30d", F.lit(0).cast("long"))
            .withColumn("first_ekyc_event_ts", F.lit(None).cast("timestamp"))
            .withColumn("last_ekyc_event_ts", F.lit(None).cast("timestamp"))
            .withColumn("first_ekyc_verified_ts", F.lit(None).cast("timestamp"))
            .withColumn("first_ekyc_failure_ts", F.lit(None).cast("timestamp"))
            .withColumn("first_ekyc_under_review_ts", F.lit(None).cast("timestamp"))
            .withColumn("first_ekyc_fail_reason", F.lit(None).cast("string"))
            .withColumn("latest_ekyc_status", F.lit(None).cast("string"))
            .withColumn("ekyc_status_sequence", F.lit(None).cast("string"))
            .withColumn("first_ekyc_verified_after_failure_ts", F.lit(None).cast("timestamp"))
            .withColumn("first_ekyc_verified_after_review_ts", F.lit(None).cast("timestamp"))
            .withColumn("ekyc_recovery_hours_after_failure", F.lit(None).cast("double"))
            .withColumn("ekyc_review_to_verified_hours", F.lit(None).cast("double"))
        )

    if coupon_base is not None:
        coupon_features = (
            cohort_with_ftd.alias("c")
            .join(
                coupon_base.alias("p"),
                (F.col("c.user_id") == F.col("p.user_id"))
                & (F.col("c.source_system") == F.col("p.source_system"))
                & (F.col("p.coupon_ts") >= F.col("c.registration_ts"))
                & (
                    F.col("p.coupon_ts")
                    < F.coalesce(F.col("c.first_ftd_attempt_ts"), F.col("c.journey_end_ts"))
                ),
                "inner",
            )
            .groupBy("c.unique_user_id")
            .agg(
                F.count("*").cast("long").alias("num_coupons_before_first_ftd"),
                F.countDistinct("p.coupon_type_id").cast("long").alias("num_coupon_types_before_first_ftd"),
                F.sum(F.coalesce(F.col("p.coupon_amount"), F.lit(0.0))).cast("double").alias("coupon_amount_before_first_ftd"),
                F.min(F.col("p.coupon_ts")).alias("first_coupon_before_ftd_ts"),
            )
            .withColumn(
                "has_coupon_before_first_ftd",
                F.when(F.col("num_coupons_before_first_ftd") > F.lit(0), F.lit(1)).otherwise(F.lit(0)),
            )
        )
    else:
        coupon_features = (
            cohort
            .select("unique_user_id")
            .withColumn("num_coupons_before_first_ftd", F.lit(0).cast("long"))
            .withColumn("num_coupon_types_before_first_ftd", F.lit(0).cast("long"))
            .withColumn("coupon_amount_before_first_ftd", F.lit(0.0))
            .withColumn("has_coupon_before_first_ftd", F.lit(0))
        )

    journey = (
        cohort_with_ftd
        .join(ekyc_features, on="unique_user_id", how="left")
        .join(coupon_features, on="unique_user_id", how="left")
        .fillna(
            {
                "num_ekyc_events_30d": 0,
                "num_ekyc_distinct_statuses_30d": 0,
                "num_ekyc_fail_reasons_30d": 0,
                "num_ekyc_status_changes_30d": 0,
                "has_ekyc_event_30d": 0,
                "has_ekyc_verified_30d": 0,
                "has_ekyc_failure_30d": 0,
                "has_ekyc_under_review_30d": 0,
                "has_ekyc_expired_30d": 0,
                "has_ekyc_black_list_30d": 0,
                "ekyc_recovered_after_failure": 0,
                "ekyc_recovered_after_review": 0,
                "num_coupons_before_first_ftd": 0,
                "num_coupon_types_before_first_ftd": 0,
                "coupon_amount_before_first_ftd": 0.0,
                "has_coupon_before_first_ftd": 0,
            }
        )
        .withColumn(
            "first_ekyc_event_delay_hours_30d",
            F.when(
                F.col("first_ekyc_event_ts").isNotNull(),
                (F.col("first_ekyc_event_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "first_ekyc_verified_delay_hours_30d",
            F.when(
                F.col("first_ekyc_verified_ts").isNotNull(),
                (F.col("first_ekyc_verified_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "first_ekyc_failure_delay_hours_30d",
            F.when(
                F.col("first_ekyc_failure_ts").isNotNull(),
                (F.col("first_ekyc_failure_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "first_ekyc_under_review_delay_hours_30d",
            F.when(
                F.col("first_ekyc_under_review_ts").isNotNull(),
                (F.col("first_ekyc_under_review_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "first_ftd_attempt_delay_hours_30d",
            F.when(
                F.col("first_ftd_attempt_ts").isNotNull(),
                (F.col("first_ftd_attempt_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "first_ftd_success_delay_hours_30d",
            F.when(
                F.col("first_ftd_success_ts").isNotNull(),
                (F.col("first_ftd_success_ts").cast("long") - F.col("registration_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "has_ekyc_before_first_ftd_attempt",
            F.when(
                F.col("first_ekyc_event_ts").isNotNull()
                & F.col("first_ftd_attempt_ts").isNotNull()
                & (F.col("first_ekyc_event_ts") <= F.col("first_ftd_attempt_ts")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "hours_ekyc_verified_to_first_ftd_attempt",
            F.when(
                F.col("first_ekyc_verified_ts").isNotNull()
                & F.col("first_ftd_attempt_ts").isNotNull()
                & (F.col("first_ftd_attempt_ts") >= F.col("first_ekyc_verified_ts")),
                (F.col("first_ftd_attempt_ts").cast("long") - F.col("first_ekyc_verified_ts").cast("long")) / F.lit(3600.0),
            ),
        )
        .withColumn(
            "journey_outcome",
            F.when(F.col("has_ftd_success_30d") == F.lit(1), F.lit("FTD converted"))
            .when(F.col("has_ftd_pending_unresolved") == F.lit(1), F.lit("FTD pending / handling"))
            .when(F.col("has_ftd_failure_unrecovered") == F.lit(1), F.lit("FTD failed, not recovered"))
            .when(F.col("has_ftd_attempt_30d") == F.lit(1), F.lit("FTD attempted, no success"))
            .when(F.col("ekyc_recovered_after_failure") == F.lit(1), F.lit("eKYC recovered, no FTD attempt"))
            .when(F.col("latest_ekyc_status") == F.lit("Verified"), F.lit("eKYC verified, no FTD attempt"))
            .when(F.col("latest_ekyc_status") == F.lit("Under review"), F.lit("eKYC under review, no FTD attempt"))
            .when(F.col("latest_ekyc_status") == F.lit("Verification expired"), F.lit("eKYC expired, no FTD attempt"))
            .when(F.col("latest_ekyc_status") == F.lit("Black list"), F.lit("eKYC black list, no FTD attempt"))
            .when(
                (F.col("has_ekyc_failure_30d") == F.lit(1))
                & (F.col("ekyc_recovered_after_failure") == F.lit(0)),
                F.lit("eKYC failed, not recovered"),
            )
            .when(F.col("has_ekyc_event_30d") == F.lit(1), F.lit("eKYC started, no FTD attempt"))
            .otherwise(F.lit("Registered, no eKYC / FTD")),
        )
        .withColumn(
            "journey_path",
            F.concat_ws(
                " → ",
                F.array(
                    F.lit("Registered"),
                    F.when(F.col("has_ekyc_event_30d") == F.lit(1), F.lit("eKYC started")),
                    F.when(F.col("has_ekyc_failure_30d") == F.lit(1), F.lit("eKYC failure")),
                    F.when(F.col("ekyc_recovered_after_failure") == F.lit(1), F.lit("eKYC recovered")),
                    F.when(F.col("has_ekyc_verified_30d") == F.lit(1), F.lit("eKYC verified")),
                    F.when(F.col("has_coupon_before_first_ftd") == F.lit(1), F.lit("Coupon before FTD")),
                    F.when(F.col("has_ftd_attempt_30d") == F.lit(1), F.lit("FTD attempted")),
                    F.when(F.col("has_ftd_success_30d") == F.lit(1), F.lit("FTD converted")),
                ),
            ),
        )
        .withColumn("journey_window_days", F.lit(int(journey_window_days)))
        .withColumn("verification_history_source", F.lit(verification_history_source))
        .withColumn("coupon_source_available", F.lit(coupon_base is not None))
        .withColumn("created_at", F.current_timestamp())
    )

    journey_publish = publish_table(
        spark=spark,
        df=journey,
        table_base=f"{T_FUNNEL_REPORT_PREFIX}_ekyc_journey_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["unique_user_id", "run_ts"],
        timeseries_column="run_ts",
        append_merge_schema=True,
    )

    print(
        {
            "analysis": analysis_name,
            "journey_window_days": int(journey_window_days),
            "verification_history_source": verification_history_source,
            "coupon_table_used": bool(coupon_base is not None),
            "journey_table": journey_publish["current_table"],
        }
    )
