# Databricks notebook source
# MAGIC %md
# MAGIC ## 5 - Feature distribution explorer
# MAGIC
# MAGIC Reads the latest feature table and plots one distribution chart per feature.

# COMMAND ----------

# MAGIC %run ./00_config_conversion_funnel_deposit

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("feature_scope", "all_numeric")
dbutils.widgets.text("feature_regex", ".*")
dbutils.widgets.text("sample_rows", "150000")
dbutils.widgets.text("max_features", "0")
dbutils.widgets.text("bins", "40")
dbutils.widgets.text("clip_quantile", "0.01")
dbutils.widgets.text("only_observation_complete", "true")

analysis_id = dbutils.widgets.get("analysis_id").strip()
feature_scope = dbutils.widgets.get("feature_scope").strip().lower()
feature_regex = dbutils.widgets.get("feature_regex").strip() or ".*"
sample_rows = int(dbutils.widgets.get("sample_rows").strip() or "150000")
max_features = int(dbutils.widgets.get("max_features").strip() or "0")
bins = int(dbutils.widgets.get("bins").strip() or "40")
clip_quantile = float(dbutils.widgets.get("clip_quantile").strip() or "0.01")
only_observation_complete = dbutils.widgets.get("only_observation_complete").strip().lower() in {"1", "true", "yes", "y", "on"}


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]


def _numeric_cols_from_schema(df, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    numeric_types = {"byte", "short", "integer", "long", "float", "double", "decimal", "boolean"}
    cols = []
    for f in df.schema.fields:
        if f.name in excluded:
            continue
        if f.dataType.typeName() in numeric_types:
            cols.append(f.name)
    return cols


def _select_feature_columns(
    df,
    scope: str,
    regex_pattern: str,
    *,
    analysis_id_ctx: str | None = None,
    analysis_name_ctx: str | None = None,
) -> list[str]:
    if scope not in {"all_numeric", "model_features"}:
        raise ValueError("feature_scope must be 'all_numeric' or 'model_features'.")

    technical_excluded = {
        "run_id", "run_ts", "ref_date", "analysis_name", "created_at",
        "unique_user_id", "user_id", "source_system",
    }

    if scope == "model_features":
        base_cols = get_funnel_feature_columns(
            df,
            analysis_id=analysis_id_ctx,
            analysis_name=analysis_name_ctx,
        )
    else:
        base_cols = _numeric_cols_from_schema(df, excluded=technical_excluded)

    pattern = re.compile(regex_pattern)
    filtered = [c for c in base_cols if pattern.search(c)]
    if max_features > 0:
        filtered = filtered[:max_features]
    return filtered


def _plot_distribution(series: pd.Series, feature_name: str, n_bins: int, clip_q: float):
    s = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_bool_dtype(s):
        s = s.astype("int64")
    total_n = int(len(s))
    valid = s.dropna()
    if pd.api.types.is_bool_dtype(valid):
        valid = valid.astype("int64")
    valid_num = pd.to_numeric(valid, errors="coerce").astype("float64")
    valid_n = int(len(valid))
    missing_n = total_n - valid_n
    missing_rate = (missing_n / total_n) if total_n > 0 else 0.0
    unique_n = int(valid.nunique())

    fig, ax = plt.subplots(figsize=(10, 4.2))

    clipped_note = ""
    if unique_n <= 15:
        counts = valid.value_counts(dropna=False).sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#0e7490", alpha=0.9)
        ax.set_ylabel("count")
        ax.set_xlabel("value")
    else:
        plot_values = valid_num
        if clip_q > 0 and unique_n > 30:
            lo = float(valid_num.quantile(clip_q))
            hi = float(valid_num.quantile(1.0 - clip_q))
            if lo < hi:
                plot_values = valid_num.clip(lower=lo, upper=hi)
                clipped_note = f" | clipped q[{clip_q:.2%}, {1.0-clip_q:.2%}]"

        ax.hist(plot_values, bins=max(5, n_bins), color="#0e7490", alpha=0.9, edgecolor="white")
        ax.set_ylabel("count")
        ax.set_xlabel("value")

    mean_val = float(valid_num.mean()) if valid_n > 0 else np.nan
    median_val = float(valid_num.median()) if valid_n > 0 else np.nan
    std_val = float(valid_num.std()) if valid_n > 1 else np.nan

    ax.set_title(
        f"{feature_name} | n={valid_n:,} | missing={missing_n:,} ({missing_rate:.1%}) | "
        f"unique={unique_n:,} | mean={mean_val:.4g} | median={median_val:.4g} | std={std_val:.4g}{clipped_note}"
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    try:
        display(fig)
    except Exception:
        plt.show()
    finally:
        plt.close(fig)

    return {
        "feature_name": feature_name,
        "n_total": total_n,
        "n_valid": valid_n,
        "n_missing": missing_n,
        "missing_rate": missing_rate,
        "n_unique": unique_n,
        "mean": mean_val,
        "std": std_val,
        "min": float(valid_num.min()) if valid_n > 0 else np.nan,
        "p25": float(valid_num.quantile(0.25)) if valid_n > 0 else np.nan,
        "p50": median_val,
        "p75": float(valid_num.quantile(0.75)) if valid_n > 0 else np.nan,
        "p95": float(valid_num.quantile(0.95)) if valid_n > 0 else np.nan,
        "max": float(valid_num.max()) if valid_n > 0 else np.nan,
    }


for analysis in get_analyses_to_run():
    analysis_name = ANALYSIS_NAMES[analysis]
    table_base = f"{T_FUNNEL_FEATURES_PREFIX}_{analysis_name}"
    feat_df = read_current_table(spark=spark, table_base=table_base)

    if only_observation_complete and "observation_complete" in feat_df.columns:
        feat_df = feat_df.filter(F.col("observation_complete") == F.lit(True))

    feature_cols = _select_feature_columns(
        feat_df,
        scope=feature_scope,
        regex_pattern=feature_regex,
        analysis_id_ctx=analysis,
        analysis_name_ctx=analysis_name,
    )
    if not feature_cols:
        raise ValueError(
            f"No columns selected. Check feature_scope='{feature_scope}', feature_regex='{feature_regex}', "
            f"max_features={max_features}."
        )

    select_cols = feature_cols
    base_df = feat_df.select(*select_cols)

    total_rows = base_df.count()
    if sample_rows > 0 and total_rows > sample_rows:
        fraction = min(1.0, float(sample_rows) / float(total_rows))
        sampled_df = base_df.sample(withReplacement=False, fraction=fraction, seed=int(DEFAULT_RANDOM_SEED)).limit(sample_rows)
        sampled_rows = sampled_df.count()
    else:
        sampled_df = base_df
        sampled_rows = total_rows

    print(
        {
            "analysis": analysis_name,
            "feature_scope": feature_scope,
            "feature_regex": feature_regex,
            "num_features": len(feature_cols),
            "total_rows": int(total_rows),
            "sampled_rows": int(sampled_rows),
            "only_observation_complete": bool(only_observation_complete),
        }
    )

    pdf = sampled_df.toPandas()
    summaries = []
    for c in feature_cols:
        summaries.append(_plot_distribution(pdf[c], c, n_bins=bins, clip_q=clip_quantile))

    summary_pdf = pd.DataFrame(summaries).sort_values(["missing_rate", "n_unique"], ascending=[False, False])
    print("Feature summary table")
    display(summary_pdf)
