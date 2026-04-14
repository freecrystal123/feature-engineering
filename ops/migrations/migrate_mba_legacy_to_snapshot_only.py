# Databricks notebook source
# COMMAND ----------
# MAGIC %run ../../models/utils

# COMMAND ----------

from uuid import uuid4
import pandas as pd
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("catalog", "uat_analytics")
dbutils.widgets.text("source_schema", "user_segmentation")
dbutils.widgets.text("target_schema", "cross_sell_mba")
dbutils.widgets.text("analysis_name", "ols")
dbutils.widgets.text("ref_date", "2026-03-19")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("dry_run", "true")

# COMMAND ----------

def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


catalog = dbutils.widgets.get("catalog").strip()
source_schema = dbutils.widgets.get("source_schema").strip()
target_schema = dbutils.widgets.get("target_schema").strip()
analysis_name = dbutils.widgets.get("analysis_name").strip()
ref_date = dbutils.widgets.get("ref_date").strip()
run_id_input = dbutils.widgets.get("run_id").strip()
dry_run = _to_bool(dbutils.widgets.get("dry_run"), default=True)

if not catalog:
    raise ValueError("catalog cannot be empty")
if not source_schema:
    raise ValueError("source_schema cannot be empty")
if not target_schema:
    raise ValueError("target_schema cannot be empty")
if not analysis_name:
    raise ValueError("analysis_name cannot be empty")
if not ref_date:
    raise ValueError("ref_date cannot be empty")

run_id_value = run_id_input or str(uuid4())

# COMMAND ----------

table_prefixes = [
    "mba_user_item_features",
    "mba_transactions",
    "mba_association_rules",
    "mba_association_rules_cross",
    "mba_association_rules_focus",
]

source_bases = [f"{catalog}.{source_schema}.{p}_{analysis_name}" for p in table_prefixes]
target_bases = [f"{catalog}.{target_schema}.{p}_{analysis_name}" for p in table_prefixes]

print("MBA migration plan (legacy base -> target snapshot only):")
for s, t in zip(source_bases, target_bases):
    print(f"- {s} -> {snapshot_table_name(t)}")
print(f"dry_run={dry_run}, run_id={run_id_value}, ref_date={ref_date}")
print("Note: *_current will NOT be modified.")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{target_schema}")

results = []
metadata_cols = {"run_id", "run_ts", "ref_date", "analysis_name"}

for source_base, target_base in zip(source_bases, target_bases):
    source_exists = bool(spark.catalog.tableExists(source_base))
    target_snapshot = snapshot_table_name(target_base)
    item = {
        "source_base": source_base,
        "target_snapshot": target_snapshot,
        "source_exists": source_exists,
        "dry_run": dry_run,
        "published": False,
        "run_id": run_id_value,
        "ref_date": ref_date,
    }

    if not source_exists:
        item["status"] = "skipped_source_not_found"
        results.append(item)
        continue

    df = spark.table(source_base)
    data_cols = [c for c in df.columns if c not in metadata_cols]
    out_df = (
        df.select(*data_cols)
        .withColumn("run_id", F.lit(run_id_value))
        .withColumn("run_ts", F.current_timestamp())
        .withColumn("ref_date", F.to_date(F.lit(ref_date)))
        .withColumn("analysis_name", F.lit(analysis_name))
    )

    item["source_rows"] = df.count()
    item["source_cols"] = len(df.columns)

    if not dry_run:
        out_df.write.format("delta").mode("append").saveAsTable(target_snapshot)
        item["published"] = True
        item["status"] = "published_to_snapshot"
    else:
        item["status"] = "dry_run"

    results.append(item)

# COMMAND ----------

display(pd.DataFrame(results))

# COMMAND ----------

import json

print(json.dumps({"results": results}, indent=2, default=str))
