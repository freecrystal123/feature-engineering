# Databricks notebook source
# MAGIC %md
# MAGIC ## 10 - User remark behavior explorer (converted vs non-converted)
# MAGIC
# MAGIC Goal:
# MAGIC - Understand behavior patterns in `prod_silver.user.user_remark`
# MAGIC - Compare users that converted (`label_deposit_success_30d = 1`) vs non-converted (`= 0`)
# MAGIC - Inspect event categories, operation types, raw remark texts, and remark flow sequences after registration

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_deposit

# COMMAND ----------

# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "analysis_2")
dbutils.widgets.text("source_system", "TRUEWIN")
dbutils.widgets.text("remark_window_days", str(LABEL_WINDOW_DAYS))
dbutils.widgets.text("sequence_steps", "4")
dbutils.widgets.text("top_n", "30")
dbutils.widgets.text("only_observation_complete", "true")
dbutils.widgets.text("show_all_non_converted_flows", "true")
dbutils.widgets.text("collapse_consecutive_steps", "true")
dbutils.widgets.text("feature_diag_bins", "10")
dbutils.widgets.text("feature_diag_min_users", "30")

analysis_id = dbutils.widgets.get("analysis_id").strip()
source_system_filter = dbutils.widgets.get("source_system").strip()
remark_window_days = int(dbutils.widgets.get("remark_window_days").strip() or str(LABEL_WINDOW_DAYS))
sequence_steps = int(dbutils.widgets.get("sequence_steps").strip() or "4")
top_n = int(dbutils.widgets.get("top_n").strip() or "30")
only_observation_complete = dbutils.widgets.get("only_observation_complete").strip().lower() in {"1", "true", "yes", "y", "on"}
show_all_non_converted_flows = dbutils.widgets.get("show_all_non_converted_flows").strip().lower() in {"1", "true", "yes", "y", "on"}
collapse_consecutive_steps = dbutils.widgets.get("collapse_consecutive_steps").strip().lower() in {"1", "true", "yes", "y", "on"}
feature_diag_bins = int(dbutils.widgets.get("feature_diag_bins").strip() or "10")
feature_diag_min_users = int(dbutils.widgets.get("feature_diag_min_users").strip() or "30")


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def _pick_col(df, candidates: list[str], col_desc: str, required: bool = True):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Could not find {col_desc}. Tried columns: {candidates}")
    return None


def _safe_to_timestamp(col_name: str):
    q = f"`{col_name}`"
    return F.coalesce(
        F.expr(f"try_to_timestamp({q})"),
        F.expr(f"try_to_timestamp({q}, 'yyyy-MM-dd''T''HH:mm:ss.SSSXXX')"),
        F.expr(f"try_to_timestamp({q}, 'yyyy-MM-dd HH:mm:ss')"),
    )


def _clean_remark_text_col(text_col):
    t = F.coalesce(text_col.cast("string"), F.lit(""))
    t = F.lower(t)
    t = F.regexp_replace(t, r"<[^>]+>", " ")
    t = F.regexp_replace(t, r"&nbsp;", " ")
    t = F.regexp_replace(t, r"\s+", " ")
    t = F.trim(t)
    return t


def _mask_pii_in_remark_col(text_col):
    t = _clean_remark_text_col(text_col)
    t = F.regexp_replace(t, r"[a-z0-9._%+\-*]+@[a-z0-9.\-*]+\.[a-z]{2,}", "<email>")
    t = F.regexp_replace(t, r"\b[\d\*]{6,}\b", "<masked_number>")
    t = F.regexp_replace(t, r"\s+", " ")
    t = F.trim(t)
    return t


def _remark_category_col(text_col):
    t = _clean_remark_text_col(text_col)
    return (
        F.when(t == "", F.lit("empty"))
        .when(t.rlike(r"(report by pb|report pb|pb report|scam reported by pb)"), F.lit("report_by_pb"))
        .when(t.rlike(r"(chargeback|fraud ban|fraud|ato attack|duplicate accounts)"), F.lit("fraud_chargeback"))
        .when(t.rlike(r"(suspic|suspit|suspec|suspect|under investigation|inconsistent bank info)"), F.lit("risk_investigation"))
        .when(t.rlike(r"user exclusion type self-exclusion"), F.lit("self_exclusion"))
        .when(t.rlike(r"user exclusion type deposit limit"), F.lit("deposit_limit"))
        .when(t.rlike(r"user exclusion type bet limit"), F.lit("bet_limit"))
        .when(t.rlike(r"user exclusion type wager limit"), F.lit("wager_limit"))
        .when(t.rlike(r"(manual_verify: verified|system_face: verified|system: verified|ekyc information modification: verified|kyc information modification: verified)"), F.lit("verification_success"))
        .when(t.rlike(r"(verification failed|not verified|manual_verify: black list|manual_verify: under review|system: under review|system_face: not verified)"), F.lit("verification_failed_or_review"))
        .when(t.rlike(r"(mail account:|mail: .* - |phone number: .* - |phone number: .* changed|phone number changed|change password|add address)"), F.lit("profile_or_contact_change"))
        .when(t.rlike(r"(account grouping)"), F.lit("account_grouping"))
        .when(t.rlike(r"(customer request|customer requested|user requested|requested to|requesting|account deletion|close the account|account suspension|unsuspend)"), F.lit("customer_request"))
        .when(t.rlike(r"(blocked|block|lock|locked|frozen|suspend|suspended|black list|blacklist|ban)"), F.lit("lock_suspend_ban"))
        .when(t.rlike(r"(withdrawal holder name doesn't match|deposit holder name doesn't match|bank info)"), F.lit("payment_kyc_mismatch"))
        .when(t.rlike(r"(first deposit|first withdrawal|deposit|withdraw|wallet|payment|bank|card)"), F.lit("payment_wallet"))
        .when(t.rlike(r"^(creation account|test|testmm|batchdisable|enable|a|22|pb)$"), F.lit("system_or_noise"))
        .when(t.rlike(r"(kyc|ekyc|verify|verification|id card|real_name|sign_id_card)"), F.lit("kyc_verification_generic"))
        .when(t.rlike(r"(password|login|otp|security)"), F.lit("security_login"))
        .when(t.rlike(r"(address|email|phone|name|profile|birth)"), F.lit("profile_update_generic"))
        .otherwise(F.lit("other"))
    )


def _remark_normalized_col(text_col):
    t = _clean_remark_text_col(text_col)
    t_masked = _mask_pii_in_remark_col(text_col)
    return (
        F.when(t == "", F.lit("Empty"))
        .when(t.rlike(r"(report by pb|report pb|pb report|scam reported by pb)"), F.lit("Report by PB"))
        .when(t.rlike(r"(chargeback|fraud ban|fraud|ato attack|duplicate accounts)"), F.lit("Fraud or chargeback flag"))
        .when(t.rlike(r"(suspic|suspit|suspec|suspect|under investigation|inconsistent bank info)"), F.lit("Suspicious or risk investigation"))
        .when(t.rlike(r"user exclusion type self-exclusion"), F.lit("Self-exclusion setting"))
        .when(t.rlike(r"user exclusion type deposit limit"), F.lit("Deposit limit setting"))
        .when(t.rlike(r"user exclusion type bet limit"), F.lit("Bet limit setting"))
        .when(t.rlike(r"user exclusion type wager limit"), F.lit("Wager limit setting"))
        .when(t.rlike(r"(manual_verify: verified|system_face: verified|system: verified|ekyc information modification: verified|kyc information modification: verified)"), F.lit("Verification success"))
        .when(t.rlike(r"(verification failed|not verified|manual_verify: black list|manual_verify: under review|system: under review|system_face: not verified)"), F.lit("Verification failed or under review"))
        .when(t.rlike(r"(address .* changed to|add address|address .* is changed to)"), F.lit("Address change"))
        .when(t.rlike(r"(initial email address: null|mail account: null is changed to|mail: null -)"), F.lit("New email address"))
        .when(t.rlike(r"(mail account: .* is changed to|mail: .* - .*)"), F.lit("Changed email address"))
        .when(t.rlike(r"(phone number: .* is changed to|phone number: .* - .*|phone number changed|mobile number update)"), F.lit("Changed phone number"))
        .when(t.rlike(r"(change password)"), F.lit("Password change"))
        .when(t.rlike(r"(account grouping)"), F.lit("Account grouping change"))
        .when(t.rlike(r"(customer request|customer requested|user requested|requested to|requesting|account deletion|close the account|account suspension|unsuspend)"), F.lit("Customer request"))
        .when(t.rlike(r"(blocked|block|lock|locked|frozen|suspend|suspended|black list|blacklist|ban)"), F.lit("Account lock/suspend/ban"))
        .when(t.rlike(r"(withdrawal holder name doesn't match|deposit holder name doesn't match|bank info)"), F.lit("Payment/KYC mismatch"))
        .when(t.rlike(r"(first deposit)"), F.lit("First deposit marker"))
        .when(t.rlike(r"(first withdrawal)"), F.lit("First withdrawal marker"))
        .when(t.rlike(r"(creation account)"), F.lit("Account creation note"))
        .when(t.rlike(r"^(test|testmm|batchdisable|enable|a|22|pb)$"), F.lit("System/test note"))
        .otherwise(F.substring(t_masked, 1, 200))
    )


def _remark_flow_step_col(text_col, op_col):
    t = _clean_remark_text_col(text_col)
    op = op_col.cast("int")
    return (
        F.when(t == "", F.lit("Other remark"))
        .when(t.rlike(r"^creation account$"), F.lit("Account created"))
        .when(t.rlike(r"(ekyc.*verification failed|ekyc information modification: verification failed)"), F.lit("EKYC: Verification failed"))
        .when(t.rlike(r"(ekyc.*not verified|ekyc information modification: not verified)"), F.lit("EKYC: Not verified"))
        .when(t.rlike(r"(ekyc.*black list|ekyc information modification: black list)"), F.lit("EKYC: Black list"))
        .when(t.rlike(r"(ekyc.*under review)"), F.lit("EKYC: Under review"))
        .when(t.rlike(r"(ekyc.*verified|ekyc information modification: verified)"), F.lit("EKYC: Verified"))
        .when(t.rlike(r"(kyc.*verification failed|kyc information modification: verification failed)"), F.lit("KYC: Verification failed"))
        .when(t.rlike(r"(kyc.*not verified|kyc information modification: not verified)"), F.lit("KYC: Not verified"))
        .when(t.rlike(r"(kyc.*black list|kyc information modification: black list)"), F.lit("KYC: Black list"))
        .when(t.rlike(r"(kyc.*under review)"), F.lit("KYC: Under review"))
        .when(t.rlike(r"(kyc.*verified|kyc information modification: verified)"), F.lit("KYC: Verified"))
        .when(t.rlike(r"(system_face: verification failed)"), F.lit("SYSTEM_FACE: Verification failed"))
        .when(t.rlike(r"(system_face: not verified)"), F.lit("SYSTEM_FACE: Not verified"))
        .when(t.rlike(r"(system_face: black list)"), F.lit("SYSTEM_FACE: Black list"))
        .when(t.rlike(r"(system_face: under review)"), F.lit("SYSTEM_FACE: Under review"))
        .when(t.rlike(r"(system_face: verified)"), F.lit("SYSTEM_FACE: Verified"))
        .when(t.rlike(r"(manual_verify: verification failed)"), F.lit("MANUAL_VERIFY: Verification failed"))
        .when(t.rlike(r"(manual_verify: not verified)"), F.lit("MANUAL_VERIFY: Not verified"))
        .when(t.rlike(r"(manual_verify: black list)"), F.lit("MANUAL_VERIFY: Black list"))
        .when(t.rlike(r"(manual_verify: under review)"), F.lit("MANUAL_VERIFY: Under review"))
        .when(t.rlike(r"(manual_verify: verified)"), F.lit("MANUAL_VERIFY: Verified"))
        .when(t.rlike(r"(system: verification failed)"), F.lit("SYSTEM: Verification failed"))
        .when(t.rlike(r"(system: not verified)"), F.lit("SYSTEM: Not verified"))
        .when(t.rlike(r"(system: black list)"), F.lit("SYSTEM: Black list"))
        .when(t.rlike(r"(system: under review)"), F.lit("SYSTEM: Under review"))
        .when(t.rlike(r"(system: verified)"), F.lit("SYSTEM: Verified"))
        .when(t.rlike(r"(address .* changed to|address .* is changed to)"), F.lit("Address changed"))
        .when(t.rlike(r"(add address)"), F.lit("Address added"))
        .when(t.rlike(r"(change password)"), F.lit("Password changed"))
        .when(t.rlike(r"(^mail: null -|initial email address: null new email address|mail account: null is changed to)"), F.lit("Email added"))
        .when(t.rlike(r"(^mail: .* - .*|mail account: .* is changed to|initial email address: .* new email address)"), F.lit("Email changed"))
        .when(t.rlike(r"(phone number: .* is changed to|phone number changed|mobile number update)"), F.lit("Phone changed"))
        .when(t.rlike(r"(phone number: .* - .*)"), F.lit("Phone updated"))
        .when(t.rlike(r"first deposit"), F.lit("First deposit event"))
        .when(t.rlike(r"first withdrawal"), F.lit("First withdrawal event"))
        .when(t.rlike(r"user exclusion type self-exclusion"), F.lit("Self-exclusion change"))
        .when(t.rlike(r"user exclusion type deposit limit"), F.lit("Deposit limit change"))
        .when(t.rlike(r"user exclusion type bet limit"), F.lit("Bet limit change"))
        .when(t.rlike(r"user exclusion type wager limit"), F.lit("Wager limit change"))
        .when(t.rlike(r"(report by pb|report pb|pb report|scam reported by pb)"), F.lit("Reported by PB"))
        .when(t.rlike(r"(fraud|chargeback|ato attack|duplicate accounts|suspic|suspit|suspec|suspect|under investigation)"), F.lit("Risk/Fraud investigation"))
        .when(t.rlike(r"(account status: normal - restricted)"), F.lit("Account restricted"))
        .when(t.rlike(r"(account status: restricted - normal)"), F.lit("Account reopened"))
        .when(t.rlike(r"(account grouping add: deposit alert day|added to deposit alert day)"), F.lit("Deposit alert grouping"))
        .when(t.rlike(r"(account grouping)"), F.lit("Account grouping change"))
        .when(t.rlike(r"(customer request|customer requested|user requested|requested to|requesting|account deletion|close the account|account suspension|unsuspend)"), F.lit("Customer request"))
        .when(t.rlike(r"(blocked|block|lock|locked|frozen|suspend|suspended|black list|blacklist|ban)"), F.lit("Lock/Suspend/Ban"))
        .when(op == F.lit(17), F.lit("Address update event"))
        .when(op == F.lit(7), F.lit("Password event"))
        .when(op == F.lit(0), F.lit("General note event"))
        .otherwise(F.lit("Other remark"))
    )


def _remark_flow_priority_col(flow_step_col):
    return (
        F.when(flow_step_col == F.lit("Account created"), F.lit(0))
        .when(flow_step_col.rlike(r"^EKYC:"), F.lit(1))
        .when(flow_step_col.rlike(r"^KYC:"), F.lit(2))
        .when(flow_step_col.rlike(r"^SYSTEM_FACE:"), F.lit(3))
        .when(flow_step_col.rlike(r"^MANUAL_VERIFY:"), F.lit(4))
        .when(flow_step_col.rlike(r"^SYSTEM:"), F.lit(5))
        .otherwise(F.lit(10))
    )


def _class_label_col(flag_col):
    return F.when(flag_col == 1, F.lit("Converted")).otherwise(F.lit("Not converted"))


def _show_header(title: str):
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def _get_remark_feature_columns(feat_df):
    preferred = [
        "num_user_remarks_7d",
        "has_user_remark_7d",
        "num_distinct_remark_flow_steps_7d",
        "first_user_remark_delay_hours_7d",
    ]
    dynamic_prefixes = ("num_remark_", "has_remark_", "first_remark_")

    cols = []
    for c in preferred:
        if c in feat_df.columns:
            cols.append(c)

    dynamic_cols = sorted(
        [
            c
            for c in feat_df.columns
            if c.startswith(dynamic_prefixes)
        ]
    )
    for c in dynamic_cols:
        if c not in cols:
            cols.append(c)
    return cols


def _get_wallet_tx_feature_columns(feat_df):
    preferred = [
        "num_wallet_txn_rows_predeposit_30d",
    ]
    dynamic_prefixes = (
        "num_wallet_txn_trans_type_",
        "has_wallet_txn_trans_type_",
    )

    cols = []
    for c in preferred:
        if c in feat_df.columns:
            cols.append(c)

    dynamic_cols = sorted([c for c in feat_df.columns if c.startswith(dynamic_prefixes)])
    for c in dynamic_cols:
        if c not in cols:
            cols.append(c)
    return cols


def _get_training_feature_columns(feat_df, *, analysis_id_ctx: str | None = None, analysis_name_ctx: str | None = None):
    # Reuse the same feature-selection rule as model training.
    return get_funnel_feature_columns(
        feat_df,
        analysis_id=analysis_id_ctx,
        analysis_name=analysis_name_ctx,
    )


def _build_numeric_segment_column(df, feature_col: str, n_bins: int):
    clean = (
        df
        .select(
            F.col("converted").cast("int").alias("converted"),
            F.col(feature_col).cast("double").alias("feature_value"),
        )
    )

    quantiles = [i / float(max(1, n_bins)) for i in range(max(1, n_bins) + 1)]
    raw_breaks = clean.approxQuantile("feature_value", quantiles, 0.01)
    breaks = []
    for b in raw_breaks:
        if b is None:
            continue
        if not breaks or b > breaks[-1]:
            breaks.append(float(b))

    if len(breaks) <= 2:
        segmented = (
            clean
            .withColumn(
                "segment",
                F.when(F.col("feature_value").isNull(), F.lit("MISSING"))
                .otherwise(F.concat(F.lit("VALUE="), F.format_number(F.col("feature_value"), 6))),
            )
            .withColumn("segment_order", F.when(F.col("feature_value").isNull(), F.lit(-1)).otherwise(F.col("feature_value")))
        )
        return segmented, breaks

    segment_expr = F.when(F.col("feature_value").isNull(), F.lit("MISSING"))
    order_expr = F.when(F.col("feature_value").isNull(), F.lit(-1.0))
    for i in range(len(breaks) - 1):
        left = breaks[i]
        right = breaks[i + 1]
        if i < len(breaks) - 2:
            cond = (F.col("feature_value") >= F.lit(left)) & (F.col("feature_value") < F.lit(right))
            label = f"[{left:.6g}, {right:.6g})"
        else:
            cond = (F.col("feature_value") >= F.lit(left)) & (F.col("feature_value") <= F.lit(right))
            label = f"[{left:.6g}, {right:.6g}]"
        segment_expr = segment_expr.when(cond, F.lit(label))
        order_expr = order_expr.when(cond, F.lit(float(i + 1)))

    segmented = (
        clean
        .withColumn("segment", segment_expr.otherwise(F.lit("OUT_OF_RANGE")))
        .withColumn("segment_order", order_expr.otherwise(F.lit(float(len(breaks)))))
    )
    return segmented, breaks


# COMMAND ----------

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    table_base = f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}"
    feat_df = read_current_table(spark=spark, table_base=table_base)

    required_cols = {"unique_user_id", "user_id", "source_system", "registration_ts", "label_deposit_success_30d"}
    missing_required = [c for c in required_cols if c not in feat_df.columns]
    if missing_required:
        raise ValueError(
            f"Feature table '{current_table_name(table_base)}' missing required columns: {missing_required}"
        )

    if only_observation_complete and "observation_complete" in feat_df.columns:
        feat_df = feat_df.filter(F.col("observation_complete") == F.lit(True))

    if source_system_filter.lower() != "all":
        feat_df = feat_df.filter(F.upper(F.col("source_system")) == F.lit(source_system_filter.upper()))

    cohort_users = (
        feat_df
        .select(
            F.col("unique_user_id"),
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system"),
            F.col("registration_ts").cast("timestamp").alias("registration_ts"),
            F.col("label_deposit_success_30d").cast("int").alias("converted"),
        )
        .filter(F.col("unique_user_id").isNotNull())
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("registration_ts").isNotNull())
        .dropDuplicates(["unique_user_id"])
    )

    cohort_users = cohort_users.persist()
    cohort_total = cohort_users.count()
    if cohort_total == 0:
        print(f"[{analysis_name}] No users after filters.")
        continue

    _show_header(f"Analysis: {analysis_name}")
    print(
        {
            "analysis_name": analysis_name,
            "source_system_filter": source_system_filter,
            "cohort_users": int(cohort_total),
            "remark_window_days": int(remark_window_days),
            "sequence_steps": int(sequence_steps),
            "top_n": int(top_n),
            "only_observation_complete": bool(only_observation_complete),
            "show_all_non_converted_flows": bool(show_all_non_converted_flows),
            "collapse_consecutive_steps": bool(collapse_consecutive_steps),
        }
    )

    _show_header("Cohort class balance")
    class_balance = (
        cohort_users
        .groupBy("converted")
        .agg(F.count("*").alias("users"))
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .withColumn("share", F.col("users") / F.lit(float(cohort_total)))
        .select("class_label", "converted", "users", "share")
        .orderBy(F.col("converted").asc())
    )
    display(class_balance)

    remarks_df = spark.table(SILVER_USER_REMARK)
    remark_ts_col = _pick_col(
        remarks_df,
        ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
        "remark timestamp",
        required=True,
    )
    remark_text_col = _pick_col(remarks_df, ["remark"], "remark text", required=False)
    remark_operation_col = _pick_col(remarks_df, ["operation_type"], "remark operation_type", required=False)
    remark_id_col = _pick_col(remarks_df, ["id"], "remark id", required=False)

    _show_header("Presence check: are all feature users in user_remark?")
    remarks_user_keys = (
        remarks_df
        .select(
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
        )
        .filter(F.col("user_id").isNotNull())
    )
    if source_system_filter.lower() != "all":
        remarks_user_keys = remarks_user_keys.filter(F.upper(F.col("source_system")) == F.lit(source_system_filter.upper()))
    remarks_user_keys = remarks_user_keys.dropDuplicates(["user_id", "source_system"])

    cohort_presence_all_time = (
        cohort_users.alias("u")
        .join(
            remarks_user_keys.alias("r"),
            (F.col("u.user_id") == F.col("r.user_id")) & (F.col("u.source_system") == F.col("r.source_system")),
            "left",
        )
        .withColumn("is_in_user_remark", F.when(F.col("r.user_id").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .select(
            F.col("u.unique_user_id"),
            F.col("u.user_id"),
            F.col("u.source_system"),
            F.col("u.converted"),
            F.col("is_in_user_remark"),
        )
    )

    all_time_presence_summary = (
        cohort_presence_all_time
        .groupBy("converted", "is_in_user_remark")
        .agg(F.count("*").alias("users"))
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .orderBy(F.col("converted").asc(), F.col("is_in_user_remark").desc())
    )
    display(all_time_presence_summary)

    all_time_presence_totals = (
        cohort_presence_all_time
        .agg(
            F.count("*").alias("cohort_users"),
            F.sum(F.col("is_in_user_remark")).alias("users_present_in_user_remark"),
        )
        .withColumn("users_missing_in_user_remark", F.col("cohort_users") - F.col("users_present_in_user_remark"))
        .withColumn("presence_rate", F.col("users_present_in_user_remark") / F.col("cohort_users"))
        .collect()[0]
    )
    print(
        {
            "cohort_users": int(all_time_presence_totals["cohort_users"]),
            "users_present_in_user_remark": int(all_time_presence_totals["users_present_in_user_remark"]),
            "users_missing_in_user_remark": int(all_time_presence_totals["users_missing_in_user_remark"]),
            "presence_rate": float(all_time_presence_totals["presence_rate"] or 0.0),
        }
    )

    _show_header("Sample users missing in user_remark (all-time)")
    missing_in_user_remark = (
        cohort_presence_all_time
        .filter(F.col("is_in_user_remark") == F.lit(0))
        .select("unique_user_id", "user_id", "source_system", "converted")
        .orderBy(F.col("source_system").asc(), F.col("user_id").asc())
    )
    display(missing_in_user_remark.limit(100))

    min_max = cohort_users.agg(
        F.min("registration_ts").alias("min_registration_ts"),
        F.max("registration_ts").alias("max_registration_ts"),
    ).collect()[0]
    min_registration_ts = min_max["min_registration_ts"]
    max_registration_ts = min_max["max_registration_ts"]

    remarks_base = (
        remarks_df
        .select(
            F.col(remark_id_col).cast("long").alias("remark_id") if remark_id_col else F.lit(None).cast("long").alias("remark_id"),
            F.col("user_id").cast("long").alias("user_id"),
            F.col("source_system").alias("source_system"),
            _safe_to_timestamp(remark_ts_col).alias("remark_ts"),
            F.col(remark_operation_col).cast("int").alias("remark_operation_type") if remark_operation_col else F.lit(None).cast("int").alias("remark_operation_type"),
            F.trim(F.col(remark_text_col).cast("string")).alias("remark_text_raw") if remark_text_col else F.lit(None).cast("string").alias("remark_text_raw"),
        )
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("remark_ts").isNotNull())
    )

    if source_system_filter.lower() != "all":
        remarks_base = remarks_base.filter(F.upper(F.col("source_system")) == F.lit(source_system_filter.upper()))

    if min_registration_ts is not None and max_registration_ts is not None:
        remarks_base = (
            remarks_base
            .filter(F.col("remark_ts") >= F.lit(min_registration_ts))
            .filter(F.col("remark_ts") < (F.lit(max_registration_ts) + F.expr(f"INTERVAL {int(remark_window_days)} DAYS")))
        )

    remark_events = (
        cohort_users.alias("u")
        .join(
            remarks_base.alias("r"),
            (F.col("u.user_id") == F.col("r.user_id"))
            & (F.col("u.source_system") == F.col("r.source_system"))
            & (F.col("r.remark_ts") >= F.col("u.registration_ts"))
            & (F.col("r.remark_ts") < (F.col("u.registration_ts") + F.expr(f"INTERVAL {int(remark_window_days)} DAYS"))),
            "inner",
        )
        .select(
            F.col("u.unique_user_id"),
            F.col("u.converted"),
            F.col("u.registration_ts"),
            F.col("r.remark_id"),
            F.col("r.remark_ts"),
            F.col("r.remark_operation_type"),
            F.col("r.remark_text_raw"),
            _remark_normalized_col(F.col("r.remark_text_raw")).alias("remark_normalized"),
            _remark_category_col(F.col("r.remark_text_raw")).alias("remark_category"),
            _remark_flow_step_col(F.col("r.remark_text_raw"), F.col("r.remark_operation_type")).alias("remark_flow_step"),
            ((F.col("r.remark_ts").cast("long") - F.col("u.registration_ts").cast("long")) / 3600.0).alias("hours_from_registration"),
        )
    )

    remark_events = remark_events.persist()
    event_total = remark_events.count()
    users_with_remark = remark_events.select("unique_user_id").distinct().count()

    _show_header("Remark coverage in post-registration window")
    coverage = (
        class_balance.alias("c")
        .join(
            remark_events.groupBy("converted").agg(
                F.countDistinct("unique_user_id").alias("users_with_remarks"),
                F.count("*").alias("remark_events"),
            ).alias("r"),
            on="converted",
            how="left",
        )
        .fillna(0, subset=["users_with_remarks", "remark_events"])
        .withColumn("remark_user_rate", F.col("users_with_remarks") / F.col("users"))
        .withColumn("events_per_user", F.col("remark_events") / F.col("users"))
        .select(
            "class_label",
            "converted",
            "users",
            "users_with_remarks",
            "remark_user_rate",
            "remark_events",
            "events_per_user",
        )
        .orderBy(F.col("converted").desc())
    )
    display(coverage)
    print(
        {
            "cohort_users": int(cohort_total),
            "users_with_remark": int(users_with_remark),
            "remark_events": int(event_total),
            "remark_user_coverage": float(users_with_remark) / float(cohort_total) if cohort_total else 0.0,
        }
    )

    _show_header("User-level remark count distribution")
    user_event_counts = (
        remark_events
        .groupBy("unique_user_id", "converted")
        .agg(F.count("*").alias("num_remarks"))
    )
    user_event_dist = (
        user_event_counts
        .groupBy("converted")
        .agg(
            F.count("*").alias("users_with_remarks"),
            F.avg("num_remarks").alias("avg_remarks_per_remarked_user"),
            F.expr("percentile_approx(num_remarks, 0.5)").alias("p50"),
            F.expr("percentile_approx(num_remarks, 0.9)").alias("p90"),
            F.max("num_remarks").alias("max_remarks"),
        )
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .orderBy(F.col("converted").asc())
    )
    display(user_event_dist)

    _show_header("Top remark categories by class")
    class_sizes = cohort_users.groupBy("converted").agg(F.count("*").alias("class_users"))
    cat_stats = (
        remark_events
        .groupBy("converted", "remark_category")
        .agg(
            F.count("*").alias("remark_events"),
            F.countDistinct("unique_user_id").alias("users_with_category"),
            F.avg("hours_from_registration").alias("avg_hours_from_registration"),
        )
        .join(class_sizes, on="converted", how="left")
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .withColumn("share_users_in_class", F.col("users_with_category") / F.col("class_users"))
        .orderBy(F.col("converted").asc(), F.col("users_with_category").desc(), F.col("remark_category").asc())
    )
    display(cat_stats.limit(int(top_n)))

    _show_header("Top operation_type by class")
    op_stats = (
        remark_events
        .groupBy("converted", "remark_operation_type")
        .agg(
            F.count("*").alias("remark_events"),
            F.countDistinct("unique_user_id").alias("users_with_operation_type"),
            F.avg("hours_from_registration").alias("avg_hours_from_registration"),
        )
        .join(class_sizes, on="converted", how="left")
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .withColumn("share_users_in_class", F.col("users_with_operation_type") / F.col("class_users"))
        .orderBy(F.col("converted").asc(), F.col("users_with_operation_type").desc(), F.col("remark_operation_type").asc_nulls_last())
    )
    display(op_stats.limit(int(top_n)))

    _show_header("Top raw remark texts by class")
    top_texts = (
        remark_events
        .withColumn("remark_text_clean", F.coalesce(F.col("remark_text_raw"), F.lit("")))
        .groupBy("converted", "remark_text_clean", "remark_category")
        .agg(
            F.count("*").alias("remark_events"),
            F.countDistinct("unique_user_id").alias("users_with_text"),
        )
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .orderBy(F.col("converted").asc(), F.col("users_with_text").desc(), F.col("remark_text_clean").asc())
    )
    display(top_texts.limit(int(top_n)))

    _show_header("All normalized remarks for non-converted users")
    non_converted_row = (
        class_sizes
        .filter(F.col("converted") == F.lit(0))
        .select("class_users")
        .limit(1)
        .collect()
    )
    non_converted_users_total = int(non_converted_row[0]["class_users"]) if non_converted_row else 0
    non_converted_remarks = (
        remark_events
        .filter(F.col("converted") == F.lit(0))
        .groupBy("remark_normalized", "remark_category")
        .agg(
            F.count("*").alias("remark_events"),
            F.countDistinct("unique_user_id").alias("users_with_remark"),
            F.avg("hours_from_registration").alias("avg_hours_from_registration"),
            F.expr("percentile_approx(hours_from_registration, 0.5)").alias("p50_hours_from_registration"),
            F.first("remark_text_raw", ignorenulls=True).alias("example_raw_remark"),
        )
        .withColumn(
            "share_non_converted_users",
            F.col("users_with_remark") / F.lit(float(non_converted_users_total if non_converted_users_total > 0 else 1)),
        )
        .orderBy(F.col("users_with_remark").desc(), F.col("remark_events").desc(), F.col("remark_normalized").asc())
    )
    print({"non_converted_users_total": int(non_converted_users_total)})
    display(non_converted_remarks)

    _show_header("Top normalized remarks for non-converted users")
    display(non_converted_remarks.limit(int(top_n)))

    _show_header("Top flow steps for non-converted users")
    non_converted_flow_steps = (
        remark_events
        .filter(F.col("converted") == F.lit(0))
        .groupBy("remark_flow_step")
        .agg(
            F.count("*").alias("remark_events"),
            F.countDistinct("unique_user_id").alias("users_with_step"),
            F.avg("hours_from_registration").alias("avg_hours_from_registration"),
            F.expr("percentile_approx(hours_from_registration, 0.5)").alias("p50_hours_from_registration"),
        )
        .withColumn(
            "share_non_converted_users",
            F.col("users_with_step") / F.lit(float(non_converted_users_total if non_converted_users_total > 0 else 1)),
        )
        .orderBy(F.col("users_with_step").desc(), F.col("remark_events").desc(), F.col("remark_flow_step").asc())
    )
    display(non_converted_flow_steps)

    _show_header("Flow sequences (first N remark flow steps per user)")
    sequence_source = (
        remark_events
        .withColumn("flow_step", F.col("remark_flow_step"))
        .withColumn("flow_step_priority", _remark_flow_priority_col(F.col("remark_flow_step")))
    )
    w = Window.partitionBy("unique_user_id").orderBy(
        F.col("remark_ts").asc(),
        F.col("flow_step_priority").asc(),
        F.col("remark_id").asc_nulls_last(),
    )
    if collapse_consecutive_steps:
        sequence_source = (
            sequence_source
            .withColumn("prev_flow_step", F.lag("flow_step").over(w))
            .filter(F.col("prev_flow_step").isNull() | (F.col("flow_step") != F.col("prev_flow_step")))
            .drop("prev_flow_step")
        )
    sequence_base = (
        sequence_source
        .withColumn("step_no", F.row_number().over(w))
        .filter(F.col("step_no") <= F.lit(int(sequence_steps)))
    )

    sequence_per_user = (
        sequence_base
        .groupBy("unique_user_id", "converted")
        .agg(F.sort_array(F.collect_list(F.struct("step_no", "flow_step"))).alias("steps"))
        .withColumn("sequence", F.expr("concat_ws(' > ', transform(steps, x -> x.flow_step))"))
    )

    top_sequences = (
        sequence_per_user
        .groupBy("converted", "sequence")
        .agg(F.count("*").alias("users"))
        .join(class_sizes, on="converted", how="left")
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .withColumn("share_in_class", F.col("users") / F.col("class_users"))
        .orderBy(F.col("converted").asc(), F.col("users").desc(), F.col("sequence").asc())
    )
    if show_all_non_converted_flows:
        _show_header("All flow sequences for non-converted users (converted=0)")
        non_converted_sequences = (
            top_sequences
            .filter(F.col("converted") == F.lit(0))
            .orderBy(F.col("users").desc(), F.col("sequence").asc())
        )
        non_converted_flow_count = non_converted_sequences.count()
        print({"non_converted_distinct_flows": int(non_converted_flow_count)})
        display(non_converted_sequences)
        _show_header("Top flow sequences for converted users (reference)")
        display(
            top_sequences
            .filter(F.col("converted") == F.lit(1))
            .orderBy(F.col("users").desc(), F.col("sequence").asc())
            .limit(int(top_n))
        )
    else:
        display(top_sequences.limit(int(top_n)))

    _show_header("Step-to-step transition patterns")
    w_step = Window.partitionBy("unique_user_id").orderBy(F.col("step_no").asc())
    transitions = (
        sequence_base
        .withColumn("next_category", F.lead("remark_category").over(w_step))
        .withColumn("next_step_no", F.lead("step_no").over(w_step))
        .filter(F.col("next_category").isNotNull())
    )
    transition_stats = (
        transitions
        .groupBy("converted", "step_no", "remark_category", "next_category")
        .agg(
            F.count("*").alias("transitions"),
            F.countDistinct("unique_user_id").alias("users"),
        )
        .join(class_sizes, on="converted", how="left")
        .withColumn("class_label", _class_label_col(F.col("converted")))
        .withColumn("share_users_in_class", F.col("users") / F.col("class_users"))
        .orderBy(F.col("converted").asc(), F.col("step_no").asc(), F.col("users").desc())
    )
    display(transition_stats.limit(int(top_n)))

    _show_header("Quick check: conversion by remark category presence")
    category_presence = (
        remark_events
        .select("unique_user_id", "converted", "remark_category")
        .dropDuplicates(["unique_user_id", "remark_category"])
        .groupBy("remark_category")
        .agg(
            F.countDistinct("unique_user_id").alias("users_with_category"),
            F.avg("converted").alias("conversion_rate_in_category"),
        )
    )
    global_conversion = cohort_users.agg(F.avg("converted").alias("global_conversion_rate")).collect()[0]["global_conversion_rate"]
    category_presence = (
        category_presence
        .withColumn("global_conversion_rate", F.lit(float(global_conversion or 0.0)))
        .withColumn("delta_vs_global_pp", (F.col("conversion_rate_in_category") - F.col("global_conversion_rate")) * 100.0)
        .orderBy(F.col("users_with_category").desc(), F.col("remark_category").asc())
    )
    display(category_presence.limit(int(top_n)))

    _show_header("Wallet pre-deposit exploration (wallet_key + coupon)")
    wallet_table = SILVER_WALLET_WALLET
    deposit_table = SILVER_WALLET_DEPOSIT
    coupon_table = f"{READ_SILVER_DB}.wallet.coupon"

    if not spark.catalog.tableExists(wallet_table):
        print({"wallet_table_found": False, "wallet_table": wallet_table})
    else:
        print(
            {
                "wallet_table_found": True,
                "wallet_table": wallet_table,
                "deposit_table_found": bool(spark.catalog.tableExists(deposit_table)),
                "coupon_table_found": bool(spark.catalog.tableExists(coupon_table)),
                "wallet_cutoff_rule": "wallet rows by created_time from registration_ts until first deposit attempt_ts; if no attempt, until registration_ts + 30d",
            }
        )

        # Build per-user cutoff: before first deposit attempt in 30d (or full 30d if no attempt)
        cutoff_users = cohort_users.select(
            "unique_user_id",
            "user_id",
            "source_system",
            "registration_ts",
            "converted",
        )

        if spark.catalog.tableExists(deposit_table):
            deposit_df = spark.table(deposit_table)
            dep_ts_col = _pick_col(
                deposit_df,
                ["created_time_uae", "created_time", "updated_time_uae", "updated_time"],
                "deposit attempt timestamp",
                required=False,
            )
            dep_user_col = _pick_col(deposit_df, ["user_id"], "deposit user_id", required=False)
            dep_source_col = _pick_col(deposit_df, ["source_system"], "deposit source_system", required=False)

            if dep_ts_col and dep_user_col and dep_source_col:
                deposit_base = (
                    deposit_df
                    .select(
                        F.col(dep_user_col).cast("long").alias("user_id"),
                        F.col(dep_source_col).alias("source_system"),
                        _safe_to_timestamp(dep_ts_col).alias("attempt_ts"),
                    )
                    .filter(F.col("attempt_ts").isNotNull())
                )
                if source_system_filter.lower() != "all":
                    deposit_base = deposit_base.filter(F.upper(F.col("source_system")) == F.lit(source_system_filter.upper()))

                first_attempt = (
                    cutoff_users.alias("u")
                    .join(
                        deposit_base.alias("d"),
                        (F.col("u.user_id") == F.col("d.user_id"))
                        & (F.col("u.source_system") == F.col("d.source_system"))
                        & (F.col("d.attempt_ts") >= F.col("u.registration_ts"))
                        & (F.col("d.attempt_ts") < (F.col("u.registration_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"))),
                        "left",
                    )
                    .groupBy("u.unique_user_id")
                    .agg(F.min("d.attempt_ts").alias("first_deposit_attempt_ts"))
                )
            else:
                first_attempt = cutoff_users.select("unique_user_id").withColumn("first_deposit_attempt_ts", F.lit(None).cast("timestamp"))
        else:
            first_attempt = cutoff_users.select("unique_user_id").withColumn("first_deposit_attempt_ts", F.lit(None).cast("timestamp"))

        cutoff_users = (
            cutoff_users.alias("u")
            .join(first_attempt.alias("d"), on="unique_user_id", how="left")
            .withColumn(
                "wallet_cutoff_ts",
                F.coalesce(
                    F.col("d.first_deposit_attempt_ts"),
                    F.col("u.registration_ts") + F.expr(f"INTERVAL {int(LABEL_WINDOW_DAYS)} DAYS"),
                ),
            )
            .withColumn("has_deposit_attempt_30d", F.when(F.col("d.first_deposit_attempt_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
            .select("unique_user_id", "user_id", "source_system", "registration_ts", "converted", "wallet_cutoff_ts", "has_deposit_attempt_30d")
        ).persist()

        wallet_df = spark.table(wallet_table)
        wallet_ts_col = _pick_col(
            wallet_df,
            # For structure analysis, use wallet creation timestamp first.
            # Using updated_time as primary can miss rows that were created in-window
            # but updated later.
            ["created_time_uae", "created_time", "updated_time_uae", "updated_time", "latest_effective_time_uae", "latest_effective_time"],
            "wallet timestamp",
            required=False,
        )
        wallet_user_col = _pick_col(wallet_df, ["user_id"], "wallet user_id", required=False)
        wallet_source_col = _pick_col(wallet_df, ["source_system"], "wallet source_system", required=False)
        wallet_key_col = _pick_col(wallet_df, ["wallet_key"], "wallet key", required=False)
        wallet_balance_col = _pick_col(wallet_df, ["balance"], "wallet balance", required=False)
        wallet_ineffective_col = _pick_col(wallet_df, ["ineffective_balance"], "wallet ineffective balance", required=False)
        wallet_expire_col = _pick_col(wallet_df, ["expire_balance"], "wallet expire balance", required=False)
        wallet_frozen_col = _pick_col(wallet_df, ["frozen"], "wallet frozen", required=False)

        if wallet_ts_col and wallet_user_col and wallet_source_col and wallet_key_col:
            print({"wallet_timestamp_column_used": wallet_ts_col})
            wallet_base = (
                wallet_df
                .select(
                    F.col(wallet_user_col).cast("long").alias("user_id"),
                    F.col(wallet_source_col).alias("source_system"),
                    F.trim(F.lower(F.col(wallet_key_col).cast("string"))).alias("wallet_key"),
                    _safe_to_timestamp(wallet_ts_col).alias("wallet_event_ts"),
                    F.coalesce(F.col(wallet_balance_col).cast("double"), F.lit(0.0)).alias("balance") if wallet_balance_col else F.lit(0.0).alias("balance"),
                    F.coalesce(F.col(wallet_ineffective_col).cast("double"), F.lit(0.0)).alias("ineffective_balance") if wallet_ineffective_col else F.lit(0.0).alias("ineffective_balance"),
                    F.coalesce(F.col(wallet_expire_col).cast("double"), F.lit(0.0)).alias("expire_balance") if wallet_expire_col else F.lit(0.0).alias("expire_balance"),
                    F.coalesce(F.col(wallet_frozen_col).cast("double"), F.lit(0.0)).alias("frozen") if wallet_frozen_col else F.lit(0.0).alias("frozen"),
                )
                .filter(F.col("wallet_event_ts").isNotNull())
                .filter(F.col("wallet_key").isNotNull())
            )
            if source_system_filter.lower() != "all":
                wallet_base = wallet_base.filter(F.upper(F.col("source_system")) == F.lit(source_system_filter.upper()))

            wallet_pre = (
                cutoff_users.alias("u")
                .join(
                    wallet_base.alias("w"),
                    (F.col("u.user_id") == F.col("w.user_id"))
                    & (F.col("u.source_system") == F.col("w.source_system"))
                    & (F.col("w.wallet_event_ts") >= F.col("u.registration_ts"))
                    & (F.col("w.wallet_event_ts") < F.col("u.wallet_cutoff_ts")),
                    "left",
                )
                .select(
                    F.col("u.unique_user_id"),
                    F.col("u.converted"),
                    F.col("u.has_deposit_attempt_30d"),
                    F.col("u.registration_ts"),
                    F.col("u.wallet_cutoff_ts"),
                    F.col("w.wallet_key"),
                    F.col("w.wallet_event_ts"),
                    F.col("w.balance"),
                    F.col("w.ineffective_balance"),
                    F.col("w.expire_balance"),
                    F.col("w.frozen"),
                )
                .filter(F.col("wallet_event_ts").isNotNull())
            ).persist()

            _show_header("Distinct wallet_key values pre-deposit")
            wallet_keys = wallet_pre.select("wallet_key").distinct().orderBy("wallet_key")
            display(wallet_keys)

            _show_header("wallet_key frequency pre-deposit")
            key_freq = (
                wallet_pre
                .groupBy("wallet_key")
                .agg(
                    F.count("*").alias("wallet_rows"),
                    F.countDistinct("unique_user_id").alias("users_with_key"),
                    F.avg("balance").alias("avg_balance"),
                    F.expr("percentile_approx(balance, 0.5)").alias("p50_balance"),
                    F.max("balance").alias("max_balance"),
                )
                .withColumn("user_share", F.col("users_with_key") / F.lit(float(cohort_total)))
                .orderBy(F.col("users_with_key").desc(), F.col("wallet_key").asc())
            )
            display(key_freq)

            _show_header("Wallet structure per user before cutoff")
            per_user_wallet = (
                wallet_pre
                .groupBy("unique_user_id", "converted")
                .agg(
                    F.count("*").alias("num_wallet_rows_pre_cutoff"),
                    F.countDistinct("wallet_key").alias("num_wallet_keys_pre_cutoff"),
                    F.concat_ws(",", F.sort_array(F.collect_set("wallet_key"))).alias("wallet_key_set"),
                    F.sum(F.when(F.col("wallet_key") == F.lit("bonus"), F.lit(1)).otherwise(F.lit(0))).alias("num_bonus_rows_pre_cutoff"),
                    F.max(F.when(F.col("wallet_key") == F.lit("bonus"), F.lit(1)).otherwise(F.lit(0))).alias("has_bonus_wallet_pre_cutoff"),
                )
            )
            wallet_pattern = (
                per_user_wallet
                .groupBy("converted", "wallet_key_set")
                .agg(F.count("*").alias("users"))
                .join(class_sizes, on="converted", how="left")
                .withColumn("class_label", _class_label_col(F.col("converted")))
                .withColumn("share_in_class", F.col("users") / F.col("class_users"))
                .orderBy(F.col("converted").asc(), F.col("users").desc())
            )
            display(wallet_pattern.limit(int(top_n)))

            wallet_count_summary = (
                per_user_wallet
                .groupBy("converted")
                .agg(
                    F.count("*").alias("users_with_wallet_rows"),
                    F.avg("num_wallet_rows_pre_cutoff").alias("avg_wallet_rows"),
                    F.expr("percentile_approx(num_wallet_rows_pre_cutoff, 0.5)").alias("p50_wallet_rows"),
                    F.expr("percentile_approx(num_wallet_rows_pre_cutoff, 0.9)").alias("p90_wallet_rows"),
                    F.avg("num_wallet_keys_pre_cutoff").alias("avg_wallet_keys"),
                    F.expr("percentile_approx(num_wallet_keys_pre_cutoff, 0.5)").alias("p50_wallet_keys"),
                    F.avg("has_bonus_wallet_pre_cutoff").alias("bonus_presence_rate"),
                )
                .withColumn("class_label", _class_label_col(F.col("converted")))
                .orderBy("converted")
            )
            display(wallet_count_summary)

            _show_header("Conversion by presence of each wallet_key pre-deposit")
            key_presence = (
                wallet_pre
                .select("unique_user_id", "converted", "wallet_key")
                .dropDuplicates(["unique_user_id", "wallet_key"])
                .groupBy("wallet_key")
                .agg(
                    F.countDistinct("unique_user_id").alias("users_with_key"),
                    F.avg("converted").alias("conversion_rate_with_key"),
                )
                .withColumn("population_pct", F.col("users_with_key") / F.lit(float(cohort_total)))
                .withColumn("delta_vs_global_pp", (F.col("conversion_rate_with_key") - F.lit(float(global_conversion or 0.0))) * F.lit(100.0))
                .orderBy(F.col("users_with_key").desc(), F.col("wallet_key").asc())
            )
            display(key_presence)

            # Coupon linkage with bonus wallet events
            _show_header("Bonus wallet key vs coupon linkage")
            if spark.catalog.tableExists(coupon_table):
                coupon_df = spark.table(coupon_table)
                coupon_user_col = _pick_col(coupon_df, ["user_id", "uid"], "coupon user_id", required=False)
                coupon_source_col = _pick_col(coupon_df, ["source_system"], "coupon source_system", required=False)
                coupon_ts_col = _pick_col(
                    coupon_df,
                    ["created_time_uae", "created_time", "updated_time_uae", "updated_time", "valid_start"],
                    "coupon timestamp",
                    required=False,
                )
                coupon_amount_col = _pick_col(coupon_df, ["amount"], "coupon amount", required=False)
                coupon_status_col = _pick_col(coupon_df, ["status"], "coupon status", required=False)
                coupon_type_col = _pick_col(coupon_df, ["type"], "coupon type", required=False)

                if coupon_user_col and coupon_source_col and coupon_ts_col:
                    coupon_base = (
                        coupon_df
                        .select(
                            F.col(coupon_user_col).cast("long").alias("user_id"),
                            F.col(coupon_source_col).alias("source_system"),
                            _safe_to_timestamp(coupon_ts_col).alias("coupon_ts"),
                            F.col(coupon_amount_col).cast("double").alias("coupon_amount") if coupon_amount_col else F.lit(None).cast("double").alias("coupon_amount"),
                            F.col(coupon_status_col).cast("int").alias("coupon_status") if coupon_status_col else F.lit(None).cast("int").alias("coupon_status"),
                            F.col(coupon_type_col).cast("int").alias("coupon_type") if coupon_type_col else F.lit(None).cast("int").alias("coupon_type"),
                        )
                        .filter(F.col("coupon_ts").isNotNull())
                    )
                    if source_system_filter.lower() != "all":
                        coupon_base = coupon_base.filter(F.upper(F.col("source_system")) == F.lit(source_system_filter.upper()))

                    coupon_pre = (
                        cutoff_users.alias("u")
                        .join(
                            coupon_base.alias("c"),
                            (F.col("u.user_id") == F.col("c.user_id"))
                            & (F.col("u.source_system") == F.col("c.source_system"))
                            & (F.col("c.coupon_ts") >= F.col("u.registration_ts"))
                            & (F.col("c.coupon_ts") < F.col("u.wallet_cutoff_ts")),
                            "left",
                        )
                        .select(
                            F.col("u.unique_user_id"),
                            F.col("u.converted"),
                            F.col("c.coupon_ts"),
                            F.col("c.coupon_amount"),
                            F.col("c.coupon_status"),
                            F.col("c.coupon_type"),
                        )
                        .filter(F.col("coupon_ts").isNotNull())
                    ).persist()

                    coupon_user_features = (
                        coupon_pre
                        .groupBy("unique_user_id")
                        .agg(
                            F.count("*").alias("num_coupon_events_pre_cutoff"),
                            F.max(F.lit(1)).alias("has_coupon_pre_cutoff"),
                            F.min("coupon_ts").alias("first_coupon_ts"),
                            F.sum(F.coalesce(F.col("coupon_amount"), F.lit(0.0))).alias("sum_coupon_amount_pre_cutoff"),
                            F.countDistinct("coupon_type").alias("num_coupon_types_pre_cutoff"),
                            F.countDistinct("coupon_status").alias("num_coupon_status_pre_cutoff"),
                        )
                    )

                    bonus_user_features = (
                        per_user_wallet
                        .select(
                            "unique_user_id",
                            F.col("num_bonus_rows_pre_cutoff"),
                            F.col("has_bonus_wallet_pre_cutoff"),
                        )
                    )

                    bonus_coupon = (
                        cutoff_users
                        .select("unique_user_id", "converted")
                        .join(bonus_user_features, on="unique_user_id", how="left")
                        .join(coupon_user_features, on="unique_user_id", how="left")
                        .fillna(
                            {
                                "num_bonus_rows_pre_cutoff": 0,
                                "has_bonus_wallet_pre_cutoff": 0,
                                "num_coupon_events_pre_cutoff": 0,
                                "has_coupon_pre_cutoff": 0,
                                "sum_coupon_amount_pre_cutoff": 0.0,
                                "num_coupon_types_pre_cutoff": 0,
                                "num_coupon_status_pre_cutoff": 0,
                            }
                        )
                        .withColumn(
                            "bonus_coupon_relation",
                            F.when((F.col("has_bonus_wallet_pre_cutoff") == 1) & (F.col("has_coupon_pre_cutoff") == 1), F.lit("bonus_and_coupon"))
                            .when((F.col("has_bonus_wallet_pre_cutoff") == 1) & (F.col("has_coupon_pre_cutoff") == 0), F.lit("bonus_without_coupon"))
                            .when((F.col("has_bonus_wallet_pre_cutoff") == 0) & (F.col("has_coupon_pre_cutoff") == 1), F.lit("coupon_without_bonus"))
                            .otherwise(F.lit("no_bonus_no_coupon"))
                        )
                    )

                    relation_summary = (
                        bonus_coupon
                        .groupBy("converted", "bonus_coupon_relation")
                        .agg(
                            F.count("*").alias("users"),
                            F.avg("num_bonus_rows_pre_cutoff").alias("avg_bonus_rows"),
                            F.avg("num_coupon_events_pre_cutoff").alias("avg_coupon_events"),
                        )
                        .join(class_sizes, on="converted", how="left")
                        .withColumn("class_label", _class_label_col(F.col("converted")))
                        .withColumn("share_in_class", F.col("users") / F.col("class_users"))
                        .orderBy(F.col("converted").asc(), F.col("users").desc())
                    )
                    display(relation_summary)

                    # Proximity between bonus-wallet and coupon events (within 24h)
                    bonus_events = (
                        wallet_pre
                        .filter(F.col("wallet_key") == F.lit("bonus"))
                        .select("unique_user_id", "wallet_event_ts")
                        .withColumnRenamed("wallet_event_ts", "bonus_ts")
                    )
                    if bonus_events.limit(1).count() > 0:
                        bonus_coupon_24h = (
                            bonus_events.alias("b")
                            .join(
                                coupon_pre.select("unique_user_id", "coupon_ts").alias("c"),
                                (F.col("b.unique_user_id") == F.col("c.unique_user_id"))
                                & (F.abs(F.col("b.bonus_ts").cast("long") - F.col("c.coupon_ts").cast("long")) <= F.lit(24 * 3600)),
                                "left",
                            )
                            .groupBy("b.unique_user_id", "b.bonus_ts")
                            .agg(F.max(F.when(F.col("c.coupon_ts").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("has_coupon_within_24h"))
                        )
                        bonus_coupon_proximity = (
                            bonus_coupon_24h
                            .agg(
                                F.count("*").alias("bonus_events"),
                                F.sum("has_coupon_within_24h").alias("bonus_events_with_coupon_24h"),
                            )
                            .withColumn("share_bonus_events_with_coupon_24h", F.col("bonus_events_with_coupon_24h") / F.col("bonus_events"))
                        )
                        display(bonus_coupon_proximity)

                    coupon_pre_summary = (
                        coupon_pre
                        .groupBy("converted")
                        .agg(
                            F.count("*").alias("coupon_events"),
                            F.countDistinct("unique_user_id").alias("users_with_coupon_events"),
                            F.avg("coupon_amount").alias("avg_coupon_amount"),
                        )
                        .join(class_sizes, on="converted", how="left")
                        .withColumn("class_label", _class_label_col(F.col("converted")))
                        .withColumn("share_users_in_class", F.col("users_with_coupon_events") / F.col("class_users"))
                        .orderBy("converted")
                    )
                    display(coupon_pre_summary)

                    coupon_pre.unpersist()
                else:
                    print(
                        {
                            "coupon_join_skipped": True,
                            "reason": "missing required coupon columns (user/source/timestamp)",
                            "coupon_table": coupon_table,
                        }
                    )
            else:
                print({"coupon_table_found": False, "coupon_table": coupon_table})

            _show_header("Candidate wallet features to consider")
            wallet_candidates_all = _get_wallet_tx_feature_columns(feat_df)
            training_features = set(
                _get_training_feature_columns(
                    feat_df,
                    analysis_id_ctx=analysis,
                    analysis_name_ctx=analysis_name,
                )
            )
            wallet_candidates_training = [c for c in wallet_candidates_all if c in training_features]
            print(
                {
                    "wallet_candidates_all": wallet_candidates_all,
                    "wallet_candidates_used_in_training": wallet_candidates_training,
                }
            )

            wallet_pre.unpersist()
        else:
            print(
                {
                    "wallet_exploration_skipped": True,
                    "reason": "missing required wallet columns (user/source/timestamp/wallet_key)",
                    "wallet_table": wallet_table,
                }
            )

        cutoff_users.unpersist()

    _show_header("Remark feature validation (missing, percentiles, conversion by segment)")
    remark_feature_cols = _get_remark_feature_columns(feat_df)
    print(
        {
            "remark_features_found": len(remark_feature_cols),
            "feature_diag_bins": int(feature_diag_bins),
            "feature_diag_min_users": int(feature_diag_min_users),
        }
    )
    if remark_feature_cols:
        print("Remark features:", remark_feature_cols)
        feature_diag_df = (
            feat_df
            .select(
                F.col("label_deposit_success_30d").cast("int").alias("converted"),
                *[F.col(c) for c in remark_feature_cols],
            )
        ).persist()

        diag_total = feature_diag_df.count()
        global_conv = (
            feature_diag_df
            .agg(F.avg(F.col("converted").cast("double")).alias("global_conversion_rate"))
            .collect()[0]["global_conversion_rate"]
        )
        global_conv = float(global_conv or 0.0)

        summary_rows = []
        segment_stats_df = None
        for feature_col in remark_feature_cols:
            agg = (
                feature_diag_df
                .agg(
                    F.count("*").alias("n_rows"),
                    F.sum(F.when(F.col(feature_col).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("missing_count"),
                    F.sum(F.when(F.col(feature_col).cast("double") == F.lit(0.0), F.lit(1)).otherwise(F.lit(0))).alias("zero_count"),
                    F.countDistinct(F.col(feature_col)).alias("n_unique"),
                    F.avg(F.col(feature_col).cast("double")).alias("mean"),
                    F.stddev(F.col(feature_col).cast("double")).alias("std"),
                    F.expr(
                        f"percentile_approx(cast({feature_col} as double), array(0.01,0.05,0.25,0.50,0.75,0.95,0.99), 10000)"
                    ).alias("percentiles"),
                )
                .collect()[0]
            )

            percentiles = agg["percentiles"] or [None] * 7
            summary_rows.append(
                {
                    "feature_name": feature_col,
                    "n_rows": int(agg["n_rows"] or 0),
                    "missing_count": int(agg["missing_count"] or 0),
                    "missing_rate": (float(agg["missing_count"] or 0.0) / float(diag_total)) if diag_total else 0.0,
                    "zero_count": int(agg["zero_count"] or 0),
                    "zero_rate": (float(agg["zero_count"] or 0.0) / float(diag_total)) if diag_total else 0.0,
                    "n_unique": int(agg["n_unique"] or 0),
                    "mean": float(agg["mean"]) if agg["mean"] is not None else None,
                    "std": float(agg["std"]) if agg["std"] is not None else None,
                    "p01": float(percentiles[0]) if len(percentiles) > 0 and percentiles[0] is not None else None,
                    "p05": float(percentiles[1]) if len(percentiles) > 1 and percentiles[1] is not None else None,
                    "p25": float(percentiles[2]) if len(percentiles) > 2 and percentiles[2] is not None else None,
                    "p50": float(percentiles[3]) if len(percentiles) > 3 and percentiles[3] is not None else None,
                    "p75": float(percentiles[4]) if len(percentiles) > 4 and percentiles[4] is not None else None,
                    "p95": float(percentiles[5]) if len(percentiles) > 5 and percentiles[5] is not None else None,
                    "p99": float(percentiles[6]) if len(percentiles) > 6 and percentiles[6] is not None else None,
                }
            )

            segmented, breaks = _build_numeric_segment_column(feature_diag_df, feature_col, int(feature_diag_bins))
            breaks_text = ", ".join([f"{b:.6g}" for b in breaks]) if breaks else ""
            seg_stats = (
                segmented
                .groupBy("segment", "segment_order")
                .agg(
                    F.count("*").alias("users"),
                    F.sum(F.col("converted").cast("double")).alias("converted_users"),
                )
                .withColumn("conversion_rate", F.col("converted_users") / F.col("users"))
                .withColumn("non_conversion_rate", F.lit(1.0) - F.col("conversion_rate"))
                .withColumn("feature_name", F.lit(feature_col))
                .withColumn("global_conversion_rate", F.lit(global_conv))
                .withColumn("delta_vs_global_pp", (F.col("conversion_rate") - F.col("global_conversion_rate")) * F.lit(100.0))
                .withColumn("quantile_breaks", F.lit(breaks_text))
                .filter((F.col("users") >= F.lit(int(feature_diag_min_users))) | (F.col("segment") == F.lit("MISSING")))
                .select(
                    "feature_name",
                    "segment",
                    "segment_order",
                    "users",
                    "converted_users",
                    "conversion_rate",
                    "non_conversion_rate",
                    "global_conversion_rate",
                    "delta_vs_global_pp",
                    "quantile_breaks",
                )
            )
            segment_stats_df = seg_stats if segment_stats_df is None else segment_stats_df.unionByName(seg_stats)

        summary_df = spark.createDataFrame(summary_rows)
        display(
            summary_df.orderBy(
                F.col("missing_rate").desc(),
                F.col("n_unique").desc(),
                F.col("feature_name").asc(),
            )
        )

        if segment_stats_df is not None:
            _show_header("Conversion by segment for remark features")
            display(
                segment_stats_df.orderBy(
                    F.col("feature_name").asc(),
                    F.col("segment_order").asc(),
                    F.col("segment").asc(),
                )
            )
            _show_header("Lowest-conversion segments (remark features)")
            display(
                segment_stats_df.orderBy(
                    F.col("delta_vs_global_pp").asc(),
                    F.col("users").desc(),
                ).limit(int(top_n))
            )

        feature_diag_df.unpersist()
    else:
        print("No remark features found in current feature table for this analysis.")

    _show_header("Wallet transaction feature validation (missing, percentiles, conversion by segment)")
    wallet_feature_cols = _get_wallet_tx_feature_columns(feat_df)
    print(
        {
            "wallet_tx_features_found": len(wallet_feature_cols),
            "feature_diag_bins": int(feature_diag_bins),
            "feature_diag_min_users": int(feature_diag_min_users),
        }
    )
    if wallet_feature_cols:
        print("Wallet transaction features:", wallet_feature_cols)
        wallet_diag_df = (
            feat_df
            .select(
                F.col("label_deposit_success_30d").cast("int").alias("converted"),
                *[F.col(c) for c in wallet_feature_cols],
            )
        ).persist()

        wallet_diag_total = wallet_diag_df.count()
        wallet_global_conv = (
            wallet_diag_df
            .agg(F.avg(F.col("converted").cast("double")).alias("global_conversion_rate"))
            .collect()[0]["global_conversion_rate"]
        )
        wallet_global_conv = float(wallet_global_conv or 0.0)

        wallet_summary_rows = []
        wallet_segment_stats_df = None
        for feature_col in wallet_feature_cols:
            agg = (
                wallet_diag_df
                .agg(
                    F.count("*").alias("n_rows"),
                    F.sum(F.when(F.col(feature_col).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("missing_count"),
                    F.sum(F.when(F.col(feature_col).cast("double") == F.lit(0.0), F.lit(1)).otherwise(F.lit(0))).alias("zero_count"),
                    F.countDistinct(F.col(feature_col)).alias("n_unique"),
                    F.avg(F.col(feature_col).cast("double")).alias("mean"),
                    F.stddev(F.col(feature_col).cast("double")).alias("std"),
                    F.expr(
                        f"percentile_approx(cast({feature_col} as double), array(0.01,0.05,0.25,0.50,0.75,0.95,0.99), 10000)"
                    ).alias("percentiles"),
                )
                .collect()[0]
            )

            percentiles = agg["percentiles"] or [None] * 7
            wallet_summary_rows.append(
                {
                    "feature_name": feature_col,
                    "n_rows": int(agg["n_rows"] or 0),
                    "missing_count": int(agg["missing_count"] or 0),
                    "missing_rate": (float(agg["missing_count"] or 0.0) / float(wallet_diag_total)) if wallet_diag_total else 0.0,
                    "zero_count": int(agg["zero_count"] or 0),
                    "zero_rate": (float(agg["zero_count"] or 0.0) / float(wallet_diag_total)) if wallet_diag_total else 0.0,
                    "n_unique": int(agg["n_unique"] or 0),
                    "mean": float(agg["mean"]) if agg["mean"] is not None else None,
                    "std": float(agg["std"]) if agg["std"] is not None else None,
                    "p01": float(percentiles[0]) if len(percentiles) > 0 and percentiles[0] is not None else None,
                    "p05": float(percentiles[1]) if len(percentiles) > 1 and percentiles[1] is not None else None,
                    "p25": float(percentiles[2]) if len(percentiles) > 2 and percentiles[2] is not None else None,
                    "p50": float(percentiles[3]) if len(percentiles) > 3 and percentiles[3] is not None else None,
                    "p75": float(percentiles[4]) if len(percentiles) > 4 and percentiles[4] is not None else None,
                    "p95": float(percentiles[5]) if len(percentiles) > 5 and percentiles[5] is not None else None,
                    "p99": float(percentiles[6]) if len(percentiles) > 6 and percentiles[6] is not None else None,
                }
            )

            segmented, breaks = _build_numeric_segment_column(wallet_diag_df, feature_col, int(feature_diag_bins))
            breaks_text = ", ".join([f"{b:.6g}" for b in breaks]) if breaks else ""
            seg_stats = (
                segmented
                .groupBy("segment", "segment_order")
                .agg(
                    F.count("*").alias("users"),
                    F.sum(F.col("converted").cast("double")).alias("converted_users"),
                )
                .withColumn("conversion_rate", F.col("converted_users") / F.col("users"))
                .withColumn("non_conversion_rate", F.lit(1.0) - F.col("conversion_rate"))
                .withColumn("feature_name", F.lit(feature_col))
                .withColumn("global_conversion_rate", F.lit(wallet_global_conv))
                .withColumn("delta_vs_global_pp", (F.col("conversion_rate") - F.col("global_conversion_rate")) * F.lit(100.0))
                .withColumn("quantile_breaks", F.lit(breaks_text))
                .filter((F.col("users") >= F.lit(int(feature_diag_min_users))) | (F.col("segment") == F.lit("MISSING")))
                .select(
                    "feature_name",
                    "segment",
                    "segment_order",
                    "users",
                    "converted_users",
                    "conversion_rate",
                    "non_conversion_rate",
                    "global_conversion_rate",
                    "delta_vs_global_pp",
                    "quantile_breaks",
                )
            )
            wallet_segment_stats_df = seg_stats if wallet_segment_stats_df is None else wallet_segment_stats_df.unionByName(seg_stats)

        wallet_summary_df = spark.createDataFrame(wallet_summary_rows)
        display(
            wallet_summary_df.orderBy(
                F.col("missing_rate").desc(),
                F.col("n_unique").desc(),
                F.col("feature_name").asc(),
            )
        )

        if wallet_segment_stats_df is not None:
            _show_header("Conversion by segment for wallet transaction features")
            display(
                wallet_segment_stats_df.orderBy(
                    F.col("feature_name").asc(),
                    F.col("segment_order").asc(),
                    F.col("segment").asc(),
                )
            )
            _show_header("Lowest-conversion segments (wallet transaction features)")
            display(
                wallet_segment_stats_df.orderBy(
                    F.col("delta_vs_global_pp").asc(),
                    F.col("users").desc(),
                ).limit(int(top_n))
            )

        wallet_diag_df.unpersist()
    else:
        print("No wallet transaction features found in current feature table for this analysis.")

    _show_header("Training feature validation (all model features)")
    training_feature_cols = _get_training_feature_columns(
        feat_df,
        analysis_id_ctx=analysis,
        analysis_name_ctx=analysis_name,
    )
    print(
        {
            "training_features_found": len(training_feature_cols),
            "feature_diag_bins": int(feature_diag_bins),
            "feature_diag_min_users": int(feature_diag_min_users),
        }
    )
    if training_feature_cols:
        print("Training features:", training_feature_cols)
        train_diag_df = (
            feat_df
            .select(
                F.col("label_deposit_success_30d").cast("int").alias("converted"),
                *[F.col(c) for c in training_feature_cols],
            )
        ).persist()

        train_diag_total = train_diag_df.count()
        train_global_conv = (
            train_diag_df
            .agg(F.avg(F.col("converted").cast("double")).alias("global_conversion_rate"))
            .collect()[0]["global_conversion_rate"]
        )
        train_global_conv = float(train_global_conv or 0.0)

        train_summary_rows = []
        train_segment_stats_df = None
        for feature_col in training_feature_cols:
            agg = (
                train_diag_df
                .agg(
                    F.count("*").alias("n_rows"),
                    F.sum(F.when(F.col(feature_col).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("missing_count"),
                    F.sum(F.when(F.col(feature_col).cast("double") == F.lit(0.0), F.lit(1)).otherwise(F.lit(0))).alias("zero_count"),
                    F.countDistinct(F.col(feature_col)).alias("n_unique"),
                    F.avg(F.col(feature_col).cast("double")).alias("mean"),
                    F.stddev(F.col(feature_col).cast("double")).alias("std"),
                    F.expr(
                        f"percentile_approx(cast({feature_col} as double), array(0.01,0.05,0.25,0.50,0.75,0.95,0.99), 10000)"
                    ).alias("percentiles"),
                )
                .collect()[0]
            )

            percentiles = agg["percentiles"] or [None] * 7
            train_summary_rows.append(
                {
                    "feature_name": feature_col,
                    "n_rows": int(agg["n_rows"] or 0),
                    "missing_count": int(agg["missing_count"] or 0),
                    "missing_rate": (float(agg["missing_count"] or 0.0) / float(train_diag_total)) if train_diag_total else 0.0,
                    "zero_count": int(agg["zero_count"] or 0),
                    "zero_rate": (float(agg["zero_count"] or 0.0) / float(train_diag_total)) if train_diag_total else 0.0,
                    "n_unique": int(agg["n_unique"] or 0),
                    "mean": float(agg["mean"]) if agg["mean"] is not None else None,
                    "std": float(agg["std"]) if agg["std"] is not None else None,
                    "p01": float(percentiles[0]) if len(percentiles) > 0 and percentiles[0] is not None else None,
                    "p05": float(percentiles[1]) if len(percentiles) > 1 and percentiles[1] is not None else None,
                    "p25": float(percentiles[2]) if len(percentiles) > 2 and percentiles[2] is not None else None,
                    "p50": float(percentiles[3]) if len(percentiles) > 3 and percentiles[3] is not None else None,
                    "p75": float(percentiles[4]) if len(percentiles) > 4 and percentiles[4] is not None else None,
                    "p95": float(percentiles[5]) if len(percentiles) > 5 and percentiles[5] is not None else None,
                    "p99": float(percentiles[6]) if len(percentiles) > 6 and percentiles[6] is not None else None,
                }
            )

            segmented, breaks = _build_numeric_segment_column(train_diag_df, feature_col, int(feature_diag_bins))
            breaks_text = ", ".join([f"{b:.6g}" for b in breaks]) if breaks else ""
            seg_stats = (
                segmented
                .groupBy("segment", "segment_order")
                .agg(
                    F.count("*").alias("users"),
                    F.sum(F.col("converted").cast("double")).alias("converted_users"),
                )
                .withColumn("conversion_rate", F.col("converted_users") / F.col("users"))
                .withColumn("non_conversion_rate", F.lit(1.0) - F.col("conversion_rate"))
                .withColumn("feature_name", F.lit(feature_col))
                .withColumn("global_conversion_rate", F.lit(train_global_conv))
                .withColumn("delta_vs_global_pp", (F.col("conversion_rate") - F.col("global_conversion_rate")) * F.lit(100.0))
                .withColumn("quantile_breaks", F.lit(breaks_text))
                .filter((F.col("users") >= F.lit(int(feature_diag_min_users))) | (F.col("segment") == F.lit("MISSING")))
                .select(
                    "feature_name",
                    "segment",
                    "segment_order",
                    "users",
                    "converted_users",
                    "conversion_rate",
                    "non_conversion_rate",
                    "global_conversion_rate",
                    "delta_vs_global_pp",
                    "quantile_breaks",
                )
            )
            train_segment_stats_df = seg_stats if train_segment_stats_df is None else train_segment_stats_df.unionByName(seg_stats)

        train_summary_df = spark.createDataFrame(train_summary_rows)
        display(
            train_summary_df.orderBy(
                F.col("missing_rate").desc(),
                F.col("zero_rate").desc(),
                F.col("n_unique").asc(),
                F.col("feature_name").asc(),
            )
        )

        if train_segment_stats_df is not None:
            _show_header("Conversion by segment for all training features")
            display(
                train_segment_stats_df.orderBy(
                    F.col("feature_name").asc(),
                    F.col("segment_order").asc(),
                    F.col("segment").asc(),
                )
            )
            _show_header("Lowest-conversion segments (all training features)")
            display(
                train_segment_stats_df.orderBy(
                    F.col("delta_vs_global_pp").asc(),
                    F.col("users").desc(),
                ).limit(int(top_n))
            )

        _show_header("Potentially weak features (all training features)")
        display(
            train_summary_df
            .withColumn(
                "drop_reason",
                F.when(F.col("missing_rate") >= F.lit(0.95), F.lit("high_missing"))
                .when(F.col("zero_rate") >= F.lit(0.995), F.lit("near_all_zero"))
                .when(F.col("n_unique") <= F.lit(1), F.lit("constant"))
                .otherwise(F.lit(None)),
            )
            .filter(F.col("drop_reason").isNotNull())
            .orderBy(F.col("drop_reason").asc(), F.col("feature_name").asc())
        )

        train_diag_df.unpersist()
    else:
        print("No training features found in current feature table for this analysis.")

    cohort_users.unpersist()
    remark_events.unpersist()
