# Databricks notebook source
# MAGIC %md
# MAGIC ## 6 - Conversion funnel demographic deep dive (SHAP-driven)

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_deposit

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("top_k_features", str(DEMOGRAPHIC_TOP_FEATURES_DEFAULT))
dbutils.widgets.text("max_segments", str(DEMOGRAPHIC_MAX_SEGMENTS))
dbutils.widgets.text("numeric_bins", str(DEMOGRAPHIC_NUMERIC_BINS))

analysis_id = dbutils.widgets.get("analysis_id").strip()


def _as_positive_int(raw_value: str, default_value: int, minimum: int = 1) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except Exception:
        return int(default_value)
    return max(int(minimum), parsed)


top_k_features = _as_positive_int(dbutils.widgets.get("top_k_features"), int(DEMOGRAPHIC_TOP_FEATURES_DEFAULT), minimum=1)
max_segments = _as_positive_int(dbutils.widgets.get("max_segments"), int(DEMOGRAPHIC_MAX_SEGMENTS), minimum=2)
numeric_bins = _as_positive_int(dbutils.widgets.get("numeric_bins"), int(DEMOGRAPHIC_NUMERIC_BINS), minimum=2)
force_discrete_features = set([str(c) for c in DEMOGRAPHIC_FORCE_DISCRETE_FEATURES])


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def _read_if_exists(table_base: str):
    current_name = current_table_name(table_base)
    snapshot_name = snapshot_table_name(table_base)
    if spark.catalog.tableExists(current_name):
        return spark.table(current_name)
    if spark.catalog.tableExists(snapshot_name):
        return spark.table(snapshot_name)
    return None


def _is_numeric_dtype(dtype_name: str) -> bool:
    t = str(dtype_name).lower().strip()
    return t in {"byte", "short", "integer", "long", "float", "double", "int", "bigint", "smallint", "tinyint"} or t.startswith("decimal")


def _segment_columns_for_feature(df, feature_name: str, *, max_levels: int, n_bins: int, force_discrete: bool = False):
    dtype_name = next((f.dataType.typeName() for f in df.schema.fields if f.name == feature_name), "")
    is_numeric = _is_numeric_dtype(dtype_name)
    feature_col = F.col(feature_name)
    feature_as_str = feature_col.cast("string")

    if is_numeric:
        if force_discrete:
            segment_value = F.when(feature_col.isNull(), F.lit("__MISSING__")).otherwise(feature_as_str)
            segment_order = F.when(feature_col.isNull(), F.lit(-1.0)).otherwise(feature_col.cast("double"))
            return segment_value, segment_order, "numeric_value"

        approx_unique = (
            df
            .agg(F.approx_count_distinct(feature_col).alias("u"))
            .collect()[0]["u"]
        )
        approx_unique = int(approx_unique or 0)

        if approx_unique <= max_levels:
            segment_value = F.when(feature_col.isNull(), F.lit("__MISSING__")).otherwise(feature_as_str)
            segment_order = F.when(feature_col.isNull(), F.lit(-1.0)).otherwise(feature_col.cast("double"))
            return segment_value, segment_order, "numeric_value"

        quantile_probs = [float(i) / float(n_bins) for i in range(n_bins + 1)]
        raw_bounds = df.where(feature_col.isNotNull()).approxQuantile(feature_name, quantile_probs, 0.01)
        bounds = sorted(set([float(v) for v in raw_bounds if v is not None]))

        if len(bounds) <= 1:
            segment_value = F.when(feature_col.isNull(), F.lit("__MISSING__")).otherwise(feature_as_str)
            segment_order = F.when(feature_col.isNull(), F.lit(-1.0)).otherwise(F.lit(1.0))
            return segment_value, segment_order, "numeric_value"

        segment_value = F.when(feature_col.isNull(), F.lit("__MISSING__"))
        segment_order = F.when(feature_col.isNull(), F.lit(-1.0))

        for idx in range(len(bounds) - 1):
            left = float(bounds[idx])
            right = float(bounds[idx + 1])
            if right <= left:
                continue
            is_last = idx == len(bounds) - 2
            label = f"[{left:.6g}, {right:.6g}{']' if is_last else ')'}"
            condition = (feature_col >= F.lit(left)) & ((feature_col <= F.lit(right)) if is_last else (feature_col < F.lit(right)))
            segment_value = segment_value.when(condition, F.lit(label))
            segment_order = segment_order.when(condition, F.lit(float(idx + 1)))

        segment_value = segment_value.otherwise(F.lit("__OUTSIDE__"))
        segment_order = segment_order.otherwise(F.lit(999999.0))
        return segment_value, segment_order, "numeric_bin"

    top_values = [
        r["v"]
        for r in (
            df.where(feature_col.isNotNull())
            .groupBy(feature_as_str.alias("v"))
            .count()
            .orderBy(F.col("count").desc(), F.col("v").asc())
            .limit(max_levels - 1)
            .collect()
        )
    ]
    top_values = [v for v in top_values if v is not None]

    segment_value = (
        F.when(feature_col.isNull(), F.lit("__MISSING__"))
        .when(feature_as_str.isin(top_values), feature_as_str)
        .otherwise(F.lit("__OTHER__"))
    )

    segment_order = F.when(feature_col.isNull(), F.lit(-1.0))
    for idx, value in enumerate(top_values):
        segment_order = segment_order.when(feature_as_str == F.lit(value), F.lit(float(idx + 1)))
    segment_order = segment_order.otherwise(F.lit(999999.0))

    return segment_value, segment_order, "categorical"


summary_schema = T.StructType(
    [
        T.StructField("analysis_name", T.StringType(), True),
        T.StructField("feature_name", T.StringType(), False),
        T.StructField("feature_rank", T.IntegerType(), True),
        T.StructField("mean_abs_shap", T.DoubleType(), True),
        T.StructField("mean_shap", T.DoubleType(), True),
        T.StructField("sample_size", T.LongType(), True),
        T.StructField("n_users", T.LongType(), True),
        T.StructField("converted_users", T.DoubleType(), True),
        T.StructField("conversion_rate", T.DoubleType(), True),
        T.StructField("missing_users", T.LongType(), True),
        T.StructField("missing_rate", T.DoubleType(), True),
        T.StructField("approx_unique", T.LongType(), True),
        T.StructField("feature_mean", T.DoubleType(), True),
        T.StructField("feature_median", T.DoubleType(), True),
        T.StructField("feature_std", T.DoubleType(), True),
    ]
)

segment_schema = T.StructType(
    [
        T.StructField("analysis_name", T.StringType(), True),
        T.StructField("feature_name", T.StringType(), False),
        T.StructField("feature_rank", T.IntegerType(), True),
        T.StructField("mean_abs_shap", T.DoubleType(), True),
        T.StructField("mean_shap", T.DoubleType(), True),
        T.StructField("sample_size", T.LongType(), True),
        T.StructField("segment_kind", T.StringType(), True),
        T.StructField("segment_value", T.StringType(), True),
        T.StructField("segment_order", T.DoubleType(), True),
        T.StructField("users", T.LongType(), True),
        T.StructField("converted_users", T.DoubleType(), True),
        T.StructField("conversion_rate", T.DoubleType(), True),
        T.StructField("non_conversion_rate", T.DoubleType(), True),
        T.StructField("population_pct", T.DoubleType(), True),
        T.StructField("is_missing_segment", T.IntegerType(), True),
    ]
)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {WRITE_SCHEMA}")

for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    feat_df = read_current_table(spark=spark, table_base=f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}")
    shap_df = _read_if_exists(f"{T_FUNNEL_SHAP_PREFIX}_{analysis_name}")

    if "observation_complete" in feat_df.columns:
        feat_df = feat_df.filter(F.col("observation_complete") == F.lit(True))

    if "label_deposit_success_30d" not in feat_df.columns:
        raise ValueError(
            f"Table for analysis '{analysis_name}' does not contain label_deposit_success_30d."
        )

    feature_candidates = set(
        get_funnel_feature_columns(
            feat_df,
            analysis_id=analysis,
            analysis_name=analysis_name,
        )
    )
    if not feature_candidates:
        raise ValueError(f"No model-ready features found for analysis '{analysis_name}'.")

    if shap_df is not None and "feature_name" in shap_df.columns:
        shap_top = (
            shap_df
            .select(
                F.col("feature_name").cast("string").alias("feature_name"),
                F.col("feature_rank").cast("int").alias("feature_rank"),
                F.col("mean_abs_shap").cast("double").alias("mean_abs_shap"),
                F.col("mean_shap").cast("double").alias("mean_shap"),
                F.col("sample_size").cast("long").alias("sample_size"),
            )
            .filter(F.col("feature_name").isin(list(feature_candidates)))
            .orderBy(F.col("mean_abs_shap").desc_nulls_last(), F.col("feature_rank").asc_nulls_last())
            .limit(int(top_k_features))
            .collect()
        )
    else:
        fallback_features = sorted(list(feature_candidates))[:int(top_k_features)]
        shap_top = [
            {
                "feature_name": name,
                "feature_rank": idx + 1,
                "mean_abs_shap": None,
                "mean_shap": None,
                "sample_size": None,
            }
            for idx, name in enumerate(fallback_features)
        ]

    if not shap_top:
        raise ValueError(
            f"No SHAP-driven features available for demographic report in analysis '{analysis_name}'."
        )

    shap_selected_features = [str(r["feature_name"]) for r in shap_top]
    available_columns = set(feat_df.columns)
    classic_features = [
        f for f in DEMOGRAPHIC_CLASSIC_FEATURES
        if f in available_columns
    ]
    selected_features = []
    for feature_name in shap_selected_features + classic_features:
        if feature_name not in selected_features:
            selected_features.append(feature_name)

    shap_lookup = {
        str(r["feature_name"]): {
            "feature_rank": int(r["feature_rank"]) if r["feature_rank"] is not None else None,
            "mean_abs_shap": float(r["mean_abs_shap"]) if r["mean_abs_shap"] is not None else None,
            "mean_shap": float(r["mean_shap"]) if r["mean_shap"] is not None else None,
            "sample_size": int(r["sample_size"]) if r["sample_size"] is not None else None,
        }
        for r in shap_top
    }

    base_cols = ["unique_user_id", "label_deposit_success_30d"] + selected_features
    base_df = feat_df.select(*[c for c in base_cols if c in feat_df.columns])
    feature_dtype_lookup = {f.name: f.dataType.typeName() for f in base_df.schema.fields}
    total_users = int(base_df.count())

    summary_rows = []
    segment_dfs = []

    for feature_name in selected_features:
        info = shap_lookup.get(feature_name, {})
        feature_df = base_df.select(
            F.col("label_deposit_success_30d").cast("double").alias("label"),
            F.col(feature_name).alias(feature_name),
        )
        is_numeric_feature = _is_numeric_dtype(feature_dtype_lookup.get(feature_name, ""))

        summary_agg_exprs = [
            F.count("*").alias("n_users"),
            F.sum(F.col("label")).alias("converted_users"),
            F.sum(F.when(F.col(feature_name).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("missing_users"),
            F.approx_count_distinct(F.col(feature_name)).alias("approx_unique"),
        ]
        if is_numeric_feature:
            summary_agg_exprs.extend(
                [
                    F.avg(F.col(feature_name).cast("double")).alias("feature_mean"),
                    F.expr(f"percentile_approx(CAST(`{feature_name}` AS DOUBLE), 0.5)").alias("feature_median"),
                    F.stddev_samp(F.col(feature_name).cast("double")).alias("feature_std"),
                ]
            )
        else:
            summary_agg_exprs.extend(
                [
                    F.lit(None).cast("double").alias("feature_mean"),
                    F.lit(None).cast("double").alias("feature_median"),
                    F.lit(None).cast("double").alias("feature_std"),
                ]
            )

        summary_metrics = (
            feature_df
            .agg(*summary_agg_exprs)
            .collect()[0]
        )

        n_users = int(summary_metrics["n_users"] or 0)
        converted_users = float(summary_metrics["converted_users"] or 0.0)
        missing_users = int(summary_metrics["missing_users"] or 0)
        conversion_rate = (converted_users / float(n_users)) if n_users > 0 else 0.0
        missing_rate = (float(missing_users) / float(n_users)) if n_users > 0 else 0.0

        summary_rows.append(
            {
                "analysis_name": analysis_name,
                "feature_name": feature_name,
                "feature_rank": info.get("feature_rank"),
                "mean_abs_shap": info.get("mean_abs_shap"),
                "mean_shap": info.get("mean_shap"),
                "sample_size": info.get("sample_size"),
                "n_users": n_users,
                "converted_users": converted_users,
                "conversion_rate": conversion_rate,
                "missing_users": missing_users,
                "missing_rate": missing_rate,
                "approx_unique": int(summary_metrics["approx_unique"] or 0),
                "feature_mean": float(summary_metrics["feature_mean"]) if summary_metrics["feature_mean"] is not None else None,
                "feature_median": float(summary_metrics["feature_median"]) if summary_metrics["feature_median"] is not None else None,
                "feature_std": float(summary_metrics["feature_std"]) if summary_metrics["feature_std"] is not None else None,
            }
        )

        segment_value_col, segment_order_col, segment_kind = _segment_columns_for_feature(
            feature_df,
            feature_name,
            max_levels=int(max_segments),
            n_bins=int(numeric_bins),
            force_discrete=(feature_name in force_discrete_features),
        )

        seg_df = (
            feature_df
            .withColumn("segment_value", segment_value_col)
            .withColumn("segment_order", segment_order_col.cast("double"))
            .groupBy("segment_value", "segment_order")
            .agg(
                F.count("*").alias("users"),
                F.sum(F.col("label")).alias("converted_users"),
            )
            .withColumn(
                "conversion_rate",
                F.when(F.col("users") > 0, F.col("converted_users") / F.col("users")).otherwise(F.lit(0.0)),
            )
            .withColumn("non_conversion_rate", F.lit(1.0) - F.col("conversion_rate"))
            .withColumn(
                "population_pct",
                F.when(F.lit(total_users) > 0, F.col("users") / F.lit(float(total_users))).otherwise(F.lit(0.0)),
            )
            .withColumn("is_missing_segment", F.when(F.col("segment_value") == F.lit("__MISSING__"), F.lit(1)).otherwise(F.lit(0)))
            .withColumn("analysis_name", F.lit(analysis_name))
            .withColumn("feature_name", F.lit(feature_name))
            .withColumn("feature_rank", F.lit(info.get("feature_rank")).cast("int"))
            .withColumn("mean_abs_shap", F.lit(info.get("mean_abs_shap")).cast("double"))
            .withColumn("mean_shap", F.lit(info.get("mean_shap")).cast("double"))
            .withColumn("sample_size", F.lit(info.get("sample_size")).cast("long"))
            .withColumn("segment_kind", F.lit(segment_kind))
            .select(*[f.name for f in segment_schema.fields])
        )

        segment_dfs.append(seg_df)

    summary_out_df = spark.createDataFrame(summary_rows, schema=summary_schema)

    if segment_dfs:
        segment_out_df = segment_dfs[0]
        for extra_df in segment_dfs[1:]:
            segment_out_df = segment_out_df.unionByName(extra_df, allowMissingColumns=True)
    else:
        segment_out_df = spark.createDataFrame([], schema=segment_schema)

    summary_publish = publish_table(
        spark=spark,
        df=summary_out_df.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_FUNNEL_REPORT_PREFIX}_demographic_summary_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["feature_name", "run_ts"],
        timeseries_column="run_ts",
    )

    segment_publish = publish_table(
        spark=spark,
        df=segment_out_df.withColumn("created_at", F.current_timestamp()),
        table_base=f"{T_FUNNEL_REPORT_PREFIX}_demographic_segments_{analysis_name}",
        analysis_name=analysis_name,
        current_mode="view",
        primary_keys=["feature_name", "segment_value", "run_ts"],
        timeseries_column="run_ts",
    )

    print(
        {
            "analysis": analysis_name,
            "top_k_features": int(top_k_features),
            "max_segments": int(max_segments),
            "numeric_bins": int(numeric_bins),
            "force_discrete_features": sorted(list(force_discrete_features)),
            "classic_features": classic_features,
            "selected_features": selected_features,
            "summary_table": summary_publish["current_table"],
            "segments_table": segment_publish["current_table"],
        }
    )
