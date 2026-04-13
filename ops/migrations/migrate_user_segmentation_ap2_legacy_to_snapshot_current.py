# Databricks notebook source
# COMMAND ----------
# MAGIC %run ../../models/00_config

# COMMAND ----------
# MAGIC %run ../../models/utils

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

dbutils.widgets.text("catalog", "uat_analytics")
dbutils.widgets.text("schema", "user_segmentation")
dbutils.widgets.text("analysis_name", "online_live_casino_ap2")
dbutils.widgets.text("include_suggestion", "false")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("dry_run", "true")

# COMMAND ----------

def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
analysis_name = dbutils.widgets.get("analysis_name").strip()
include_suggestion = _to_bool(dbutils.widgets.get("include_suggestion"), default=False)
run_id_value = dbutils.widgets.get("run_id").strip() or None
dry_run = _to_bool(dbutils.widgets.get("dry_run"), default=True)

if not catalog:
    raise ValueError("catalog cannot be empty")
if not schema:
    raise ValueError("schema cannot be empty")
if not analysis_name:
    raise ValueError("analysis_name cannot be empty")

# COMMAND ----------

table_templates = [
    "feat_{analysis}",
    "pca_{analysis}",
    "pred_{analysis}",
    "cluster_{analysis}",
]
if include_suggestion:
    table_templates.append("suggestion_{analysis}")

table_bases = [f"{catalog}.{schema}.{tpl.format(analysis=analysis_name)}" for tpl in table_templates]

print("Migration plan (legacy base -> snapshot/current):")
for tb in table_bases:
    print(f"- {tb}")
print(f"dry_run={dry_run}, run_id={run_id_value}")

# COMMAND ----------

results = []

for table_base in table_bases:
    source_exists = bool(spark.catalog.tableExists(table_base))

    item = {
        "table_base": table_base,
        "source_exists": source_exists,
        "snapshot_table": snapshot_table_name(table_base),
        "current_table": current_table_name(table_base),
        "published": False,
        "run_id": run_id_value,
    }

    if not source_exists:
        item["status"] = "skipped_source_not_found"
        results.append(item)
        continue

    df = spark.table(table_base)
    item["source_rows"] = df.count()
    item["source_cols"] = len(df.columns)

    if not dry_run:
        publish_result = publish_table(
            spark=spark,
            df=df,
            table_base=table_base,
            analysis_name=analysis_name,
            run_id=run_id_value,
            current_mode="table",
        )
        item["published"] = True
        item["run_id"] = publish_result["run_id"]
        item["status"] = "published"
    else:
        item["status"] = "dry_run"

    results.append(item)

# COMMAND ----------

display(pd.DataFrame(results))

# COMMAND ----------

summary = {
    "catalog": catalog,
    "schema": schema,
    "analysis_name": analysis_name,
    "dry_run": dry_run,
    "results": results,
}

import json
print(json.dumps(summary, indent=2, default=str))
