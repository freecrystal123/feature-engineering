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

analysis_id = dbutils.widgets.get("analysis_id").strip()
source_system_filter = dbutils.widgets.get("source_system").strip()
remark_window_days = int(dbutils.widgets.get("remark_window_days").strip() or str(LABEL_WINDOW_DAYS))
sequence_steps = int(dbutils.widgets.get("sequence_steps").strip() or "4")
top_n = int(dbutils.widgets.get("top_n").strip() or "30")
only_observation_complete = dbutils.widgets.get("only_observation_complete").strip().lower() in {"1", "true", "yes", "y", "on"}
show_all_non_converted_flows = dbutils.widgets.get("show_all_non_converted_flows").strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _class_label_col(flag_col):
    return F.when(flag_col == 1, F.lit("Converted")).otherwise(F.lit("Not converted"))


def _show_header(title: str):
    print(f"\n{'=' * 24} {title} {'=' * 24}")


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

    _show_header("Flow sequences (first N remark categories per user)")
    w = Window.partitionBy("unique_user_id").orderBy(F.col("remark_ts").asc(), F.col("remark_id").asc_nulls_last())
    sequence_base = (
        remark_events
        .withColumn("step_no", F.row_number().over(w))
        .filter(F.col("step_no") <= F.lit(int(sequence_steps)))
    )

    sequence_per_user = (
        sequence_base
        .groupBy("unique_user_id", "converted")
        .agg(F.sort_array(F.collect_list(F.struct("step_no", "remark_category"))).alias("steps"))
        .withColumn("sequence", F.expr("concat_ws(' > ', transform(steps, x -> x.remark_category))"))
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

    cohort_users.unpersist()
    remark_events.unpersist()
