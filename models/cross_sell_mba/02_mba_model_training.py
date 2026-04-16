# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 - MBA model training (FP-Growth with Apriori fallback)
# MAGIC
# MAGIC Goal:
# MAGIC - Learn association rules from user baskets.
# MAGIC - Focus on cross-platform links (Casino <-> Sports).
# MAGIC - Run for multiple windows (30d and 365d).
# MAGIC - Highlight examples:
# MAGIC   - American Football bettors playing Blackjack
# MAGIC   - Horse Racing bettors playing Roulette

# COMMAND ----------

# MAGIC %run ./00_config_cross_sell_mba

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------
# MAGIC %run ../../core/uc_feature_store

# COMMAND ----------

dbutils.widgets.text("candidate_alias", "challenger")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
import mlflow
import mlflow.spark

# COMMAND ----------

analysis_name = ANALYSIS_NAMES[MBA_ANALYSIS]
candidate_alias = dbutils.widgets.get("candidate_alias").strip()
mlflow.start_run(run_name=f"mba_rules_{analysis_name}")
mlflow.log_params(
    {
        "subject": SUBJECT_NAME,
        "analysis_id": MBA_ANALYSIS,
        "analysis_name": analysis_name,
        "lookback_days_list": ",".join(str(x) for x in MBA_LOOKBACK_DAYS_LIST),
        "min_support": float(MBA_MIN_SUPPORT),
        "min_confidence": float(MBA_MIN_CONFIDENCE),
        "min_lift": float(MBA_MIN_LIFT),
        "top_rules": int(MBA_TOP_RULES),
    }
)
transactions_base = f"{T_MBA_TRANSACTIONS_PREFIX}_{analysis_name}"
transactions_feature_table = snapshot_table_name(transactions_base)
transactions_entity_source_table = current_table_name(transactions_base)

mba_training_set_result = create_training_set_from_feature_tables(
    spark=spark,
    primary_table=transactions_feature_table,
    entity_source_table=transactions_entity_source_table,
    entity_keys=["unique_user_id", "lookback_days"],
    primary_feature_names=[
        "items",
        "lookback_days",
        "num_items",
        "casino_items_count",
        "sports_items_count",
        "is_cross_platform_user",
    ],
    secondary_lookups=[],
    label=None,
    exclude_columns=[],
    entity_timestamp_col="run_ts",
)

transactions = mba_training_set_result["training_set"].load_df()

# COMMAND ----------

def _prepare_rules_dataframe(rules_df, lookback_days: int):
    norm_item_expr = "trim(regexp_replace(lower(x), '_', ' '))"

    rules_prepared = (
        rules_df
        .withColumn("antecedent_str", F.concat_ws(" + ", F.col("antecedent")))
        .withColumn("consequent_str", F.concat_ws(" + ", F.col("consequent")))
        .withColumn("antecedent_has_sports", F.expr("exists(antecedent, x -> x like 'sports::%')"))
        .withColumn("antecedent_has_casino", F.expr("exists(antecedent, x -> x like 'casino::%')"))
        .withColumn("consequent_has_sports", F.expr("exists(consequent, x -> x like 'sports::%')"))
        .withColumn("consequent_has_casino", F.expr("exists(consequent, x -> x like 'casino::%')"))
        .withColumn(
            "is_cross_platform_rule",
            (
                (F.col("antecedent_has_sports") & F.col("consequent_has_casino")) |
                (F.col("antecedent_has_casino") & F.col("consequent_has_sports"))
            ).cast("int"),
        )
        .withColumn(
            "is_focus_american_football_blackjack",
            (
                (
                    F.expr(f"exists(antecedent, x -> {norm_item_expr} = 'sports::american football')") &
                    F.expr(f"exists(consequent, x -> {norm_item_expr} = 'casino::blackjack')")
                ) |
                (
                    F.expr(f"exists(antecedent, x -> {norm_item_expr} = 'casino::blackjack')") &
                    F.expr(f"exists(consequent, x -> {norm_item_expr} = 'sports::american football')")
                )
            ).cast("int"),
        )
        .withColumn(
            "is_focus_horse_racing_roulette",
            (
                (
                    F.expr(f"exists(antecedent, x -> {norm_item_expr} = 'sports::horse racing')") &
                    F.expr(f"exists(consequent, x -> {norm_item_expr} = 'casino::roulette')")
                ) |
                (
                    F.expr(f"exists(antecedent, x -> {norm_item_expr} = 'casino::roulette')") &
                    F.expr(f"exists(consequent, x -> {norm_item_expr} = 'sports::horse racing')")
                )
            ).cast("int"),
        )
    )

    rules_with_meta = (
        rules_prepared
        .withColumn("analysis_id", F.lit(MBA_ANALYSIS))
        .withColumn("analysis_name", F.lit(analysis_name))
        .withColumn("lookback_days", F.lit(int(lookback_days)))
        .withColumn("created_at", F.current_timestamp())
    )

    rules_ranked = rules_with_meta.filter(F.col("lift") >= F.lit(float(MBA_MIN_LIFT)))

    cross_rules = (
        rules_ranked
        .filter(F.col("is_cross_platform_rule") == F.lit(1))
        .orderBy(F.col("lift").desc_nulls_last(), F.col("confidence").desc())
    )

    focus_rules = (
        rules_with_meta
        .filter(
            (F.col("is_focus_american_football_blackjack") == F.lit(1)) |
            (F.col("is_focus_horse_racing_roulette") == F.lit(1))
        )
        .orderBy(F.col("confidence").desc())
    )

    return rules_ranked, cross_rules, focus_rules


def _prepare_fallback_train_df(train_df, lookback_days: int):
    # 1) Keep only the most frequent items to control one-hot width.
    n_users = train_df.select("unique_user_id").distinct().count()
    if n_users == 0:
        return None

    min_item_users = max(2, int(n_users * float(MBA_FALLBACK_MIN_SUPPORT)))
    item_stats = (
        train_df
        .select("unique_user_id", F.explode("items").alias("item"))
        .groupBy("item")
        .agg(F.countDistinct("unique_user_id").alias("num_users"))
        .filter(F.col("num_users") >= F.lit(int(min_item_users)))
        .orderBy(F.col("num_users").desc(), F.col("item").asc())
        .limit(int(MBA_FALLBACK_TOP_ITEMS))
    )
    keep_items = [r["item"] for r in item_stats.select("item").collect()]
    if len(keep_items) < 2:
        print(
            f"[lookback={lookback_days}] Fallback prefilter kept <2 items "
            f"(n_users={n_users}, min_item_users={min_item_users})."
        )
        return None

    keep_array_expr = F.array(*[F.lit(x) for x in keep_items])
    filtered_df = (
        train_df
        .withColumn("items_filtered", F.array_intersect(F.col("items"), keep_array_expr))
        .filter(F.size("items_filtered") >= F.lit(2))
        .select("unique_user_id", F.col("items_filtered").alias("items"))
    )

    # 2) Cap rows sent to pandas to avoid driver OOM.
    total_rows = filtered_df.count()
    if total_rows == 0:
        return None
    if total_rows > int(MBA_FALLBACK_MAX_USERS):
        frac = float(MBA_FALLBACK_MAX_USERS) / float(total_rows)
        filtered_df = filtered_df.sample(withReplacement=False, fraction=frac, seed=int(MBA_FALLBACK_SAMPLE_SEED))
        filtered_df = filtered_df.limit(int(MBA_FALLBACK_MAX_USERS))
        print(
            f"[lookback={lookback_days}] Fallback sampled users: "
            f"{total_rows} -> ~{MBA_FALLBACK_MAX_USERS} (fraction={frac:.4f})."
        )

    return filtered_df


def _run_window_apriori_mlxtend(train_df, lookback_days: int):
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules
        import pandas as pd
    except Exception as e:
        raise ImportError(
            "Spark FPGrowth is unavailable in this cluster mode and mlxtend is not installed. "
            "Run `%pip install mlxtend` and retry the notebook."
        ) from e

    safe_train_df = _prepare_fallback_train_df(train_df, lookback_days)
    if safe_train_df is None:
        return None, None, None

    pdf = safe_train_df.select("items").toPandas()
    baskets = pdf["items"].tolist()

    if len(baskets) == 0:
        return None, None, None

    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    onehot_df = pd.DataFrame(te_array, columns=te.columns_)

    fallback_min_support = max(float(MBA_MIN_SUPPORT), float(MBA_FALLBACK_MIN_SUPPORT))
    fallback_min_confidence = max(float(MBA_MIN_CONFIDENCE), float(MBA_FALLBACK_MIN_CONFIDENCE))

    freq_itemsets = apriori(onehot_df, min_support=fallback_min_support, use_colnames=True)
    if freq_itemsets.empty:
        return None, None, None

    rules_pd = association_rules(freq_itemsets, metric="confidence", min_threshold=fallback_min_confidence)
    if rules_pd.empty:
        return None, None, None

    rules_pd = rules_pd[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules_pd["antecedent"] = rules_pd["antecedents"].apply(lambda s: sorted(list(s)))
    rules_pd["consequent"] = rules_pd["consequents"].apply(lambda s: sorted(list(s)))
    rules_pd = rules_pd.drop(columns=["antecedents", "consequents"])

    rules_spark = spark.createDataFrame(rules_pd)
    return _prepare_rules_dataframe(rules_spark, lookback_days)


def _run_window_pairwise_spark(train_df, lookback_days: int):
    # Fully Spark fallback: build directed pair rules A -> B from user baskets.
    tx_df = (
        train_df
        .withColumn("items", F.array_distinct(F.col("items")))
        .withColumn("items", F.sort_array(F.col("items")))
        .withColumn("items", F.slice(F.col("items"), 1, F.lit(int(MBA_PAIRWISE_MAX_ITEMS_PER_USER))))
        .filter(F.size("items") >= F.lit(2))
    )

    n_tx = tx_df.select("unique_user_id").distinct().count()
    if n_tx == 0:
        return None, None, None

    # 1-item support (for confidence/lift denominator)
    item_users = (
        tx_df
        .select("unique_user_id", F.explode("items").alias("item"))
        .groupBy("item")
        .agg(F.countDistinct("unique_user_id").alias("item_users"))
    )

    # Undirected pair count per user (i < j), then duplicate into directed rules.
    pair_users = (
        tx_df
        .select("unique_user_id", F.posexplode("items").alias("i", "item_i"), F.col("items"))
        .select("unique_user_id", "i", "item_i", F.posexplode("items").alias("j", "item_j"))
        .filter(F.col("i") < F.col("j"))
        .groupBy("item_i", "item_j")
        .agg(F.countDistinct("unique_user_id").alias("pair_users"))
    )

    directed_pairs = (
        pair_users
        .select(
            F.col("item_i").alias("antecedent_item"),
            F.col("item_j").alias("consequent_item"),
            "pair_users",
        )
        .unionByName(
            pair_users.select(
                F.col("item_j").alias("antecedent_item"),
                F.col("item_i").alias("consequent_item"),
                "pair_users",
            ),
            allowMissingColumns=True,
        )
    )

    rules = (
        directed_pairs.alias("r")
        .join(item_users.alias("a"), F.col("r.antecedent_item") == F.col("a.item"), "inner")
        .join(item_users.alias("c"), F.col("r.consequent_item") == F.col("c.item"), "inner")
        .withColumn("support", F.col("r.pair_users") / F.lit(float(n_tx)))
        .withColumn("confidence", F.col("r.pair_users") / F.col("a.item_users"))
        .withColumn("consequent_support", F.col("c.item_users") / F.lit(float(n_tx)))
        .withColumn("lift", F.col("confidence") / F.col("consequent_support"))
        .filter(F.col("support") >= F.lit(float(MBA_MIN_SUPPORT)))
        .filter(F.col("confidence") >= F.lit(float(MBA_MIN_CONFIDENCE)))
        .select(
            F.array(F.col("r.antecedent_item")).alias("antecedent"),
            F.array(F.col("r.consequent_item")).alias("consequent"),
            F.col("support").cast("double").alias("support"),
            F.col("confidence").cast("double").alias("confidence"),
            F.col("lift").cast("double").alias("lift"),
        )
    )

    return _prepare_rules_dataframe(rules, lookback_days)


def run_window_mba(tx_df, lookback_days: int):
    train_df = (
        tx_df
        .filter(F.col("lookback_days") == F.lit(int(lookback_days)))
        .select("unique_user_id", "items")
        .filter(F.size("items") >= F.lit(2))
    )

    if train_df.limit(1).count() == 0:
        return None, None, None, None

    try:
        fp = FPGrowth(
            itemsCol="items",
            minSupport=float(MBA_MIN_SUPPORT),
            minConfidence=float(MBA_MIN_CONFIDENCE),
        )
        model = fp.fit(train_df)
        rules = model.associationRules.select("antecedent", "consequent", "support", "confidence", "lift")
        rules_ranked, cross_rules, focus_rules = _prepare_rules_dataframe(rules, lookback_days)
        return rules_ranked, cross_rules, focus_rules, model
    except Exception as e:
        err = str(e)
        if ("CONNECT_ML.UNSUPPORTED_EXCEPTION" in err) or ("FPGrowth algorithm is not supported" in err):
            print(
                f"FPGrowth unsupported for lookback={lookback_days} in Spark Connect mode. "
                "Falling back to Spark pairwise rules (A -> B)."
            )
            rules_ranked, cross_rules, focus_rules = _run_window_pairwise_spark(train_df, lookback_days)
            return rules_ranked, cross_rules, focus_rules, None
        raise

# COMMAND ----------

all_rules = None
all_cross_rules = None
all_focus_rules = None
model_for_registry = None

for d in MBA_LOOKBACK_DAYS_LIST:
    rules_df, cross_df, focus_df, fp_model = run_window_mba(transactions, int(d))

    if rules_df is None:
        print(f"No rules generated for lookback={d} days. Check data volume/thresholds.")
        continue

    print(f"Computed rules for lookback={d} days")

    if all_rules is None:
        all_rules = rules_df
        all_cross_rules = cross_df
        all_focus_rules = focus_df
    else:
        all_rules = all_rules.unionByName(rules_df, allowMissingColumns=True)
        all_cross_rules = all_cross_rules.unionByName(cross_df, allowMissingColumns=True)
        all_focus_rules = all_focus_rules.unionByName(focus_df, allowMissingColumns=True)

    # Prefer registering the model from the largest lookback window.
    if (fp_model is not None) and (int(d) == int(max(MBA_LOOKBACK_DAYS_LIST))):
        model_for_registry = fp_model

if all_rules is None:
    raise ValueError("No MBA rules were generated. Check mba_transactions input and thresholds.")

# COMMAND ----------

rules_base = f"{T_MBA_RULES_PREFIX}_{analysis_name}"
cross_base = f"{T_MBA_RULES_CROSS_PREFIX}_{analysis_name}"
focus_base = f"{T_MBA_RULES_FOCUS_PREFIX}_{analysis_name}"

rules_publish = publish_table(
    spark=spark,
    df=all_rules,
    table_base=rules_base,
    analysis_name=analysis_name,
)
cross_publish = publish_table(
    spark=spark,
    df=all_cross_rules,
    table_base=cross_base,
    analysis_name=analysis_name,
    run_id=rules_publish["run_id"],
)
focus_publish = publish_table(
    spark=spark,
    df=all_focus_rules,
    table_base=focus_base,
    analysis_name=analysis_name,
    run_id=rules_publish["run_id"],
)

print(
    f"Published rules tables (lookbacks={MBA_LOOKBACK_DAYS_LIST}, run_id={rules_publish['run_id']}): "
    f"{rules_publish['current_table']}, {cross_publish['current_table']}, {focus_publish['current_table']}"
)
mlflow.log_params(
    {
        "published_rules_current": rules_publish["current_table"],
        "published_cross_current": cross_publish["current_table"],
        "published_focus_current": focus_publish["current_table"],
    }
)
mlflow.log_metrics(
    {
        "rules_rows": float(all_rules.count()),
        "cross_rules_rows": float(all_cross_rules.count()),
        "focus_rules_rows": float(all_focus_rules.count()),
    }
)

if model_for_registry is not None:
    model_name = build_model_name(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=WRITE_CATALOG,
    )
    log_model_with_feature_lineage(
        fe_client=get_feature_engineering_client(),
        model=model_for_registry,
        training_set=mba_training_set_result["training_set"],
        registered_model_name=model_name,
        flavor=mlflow.spark,
        artifact_path="model",
    )

    run_id = mlflow.active_run().info.run_id
    candidate = register_candidate(
        model_name=model_name,
        run_id=run_id,
        alias=candidate_alias,
        tags={
            "analysis": analysis_name,
            "governance_status": "candidate",
            "subject": SUBJECT_NAME,
        },
    )
    mlflow.log_params(
        {
            "registered_model_name": model_name,
            "candidate_alias": candidate_alias,
            "candidate_version": int(candidate["version"]),
        }
    )
else:
    mlflow.log_params(
        {
            "registered_model_name": "skipped",
            "model_registration_reason": "fpgrowth_unavailable_in_runtime",
        }
    )

mlflow.end_run()

# COMMAND ----------

# Top cross-platform rules by window
for d in MBA_LOOKBACK_DAYS_LIST:
    print(f"Top cross-platform rules | lookback={d} days")
    display(
        read_current_table(
            spark=spark,
            table_base=f"{T_MBA_RULES_CROSS_PREFIX}_{analysis_name}",
        )
        .select("antecedent","consequent","support","confidence","lift")
        .filter(F.col("lookback_days") == F.lit(int(d)))
        .orderBy(F.col("lift").desc_nulls_last(), F.col("confidence").desc())
        .limit(MBA_TOP_RULES)
    )

# COMMAND ----------

# Focus checks requested by business
for d in MBA_LOOKBACK_DAYS_LIST:
    print(f"Focus rules | lookback={d} days")
    display(
        read_current_table(
            spark=spark,
            table_base=f"{T_MBA_RULES_FOCUS_PREFIX}_{analysis_name}",
        )
        .filter(F.col("lookback_days") == F.lit(int(d)))
        .orderBy(F.col("confidence").desc())
    )
