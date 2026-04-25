# Databricks notebook source
# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------

from __future__ import annotations

import json
from typing import Any


# COMMAND ----------

dbutils.widgets.text("dry_run", "true")
dbutils.widgets.text("clone_mode", "DEEP")
dbutils.widgets.text("copy_reports", "true")
dbutils.widgets.text("copy_models", "true")


# COMMAND ----------

def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


dry_run = _to_bool(dbutils.widgets.get("dry_run"), default=True)
clone_mode = dbutils.widgets.get("clone_mode").strip().upper() or "DEEP"
copy_reports = _to_bool(dbutils.widgets.get("copy_reports"), default=True)
copy_models = _to_bool(dbutils.widgets.get("copy_models"), default=True)

if clone_mode not in {"DEEP", "SHALLOW"}:
    raise ValueError("clone_mode must be DEEP or SHALLOW")


# COMMAND ----------

USER_SEGMENTATION_BASE_TABLES_UAT = [
    "cluster_live_casino",
    "cluster_ols",
    "cluster_online_casino",
    "cluster_online_live_casino",
    "cluster_sports",
    "feat_live_casino",
    "feat_ols",
    "feat_online_casino",
    "feat_online_live_casino",
    "feat_sports",
    "pca_live_casino",
    "pca_ols",
    "pca_online_casino",
    "pca_online_live_casino",
    "pca_sports",
    "pred_live_casino",
    "pred_ols",
    "pred_online_casino",
    "pred_online_live_casino",
    "pred_sports",
]

SCHEMA_PLANS = [
    {
        "source_catalog": "uat_analytics",
        "target_catalog": "ml_train",
        "schema": "user_segmentation",
        "mode": "selected_snapshot_current",
        "base_tables": USER_SEGMENTATION_BASE_TABLES_UAT,
        "direct_tables": ["eval_score"],
    },
    {
        "source_catalog": "uat_analytics",
        "target_catalog": "ml_train",
        "schema": "cross_sell_mba",
        "mode": "all_snapshot_current",
        "base_tables": [],
        "direct_tables": [],
    },
    {
        "source_catalog": "prod_analytics",
        "target_catalog": "ml_prod",
        "schema": "user_segmentation",
        "mode": "all_snapshot_current",
        "base_tables": [],
        "direct_tables": [],
    },
    {
        "source_catalog": "prod_analytics",
        "target_catalog": "ml_prod",
        "schema": "cross_sell_mba",
        "mode": "all_snapshot_current",
        "base_tables": [],
        "direct_tables": [],
    },
]


# COMMAND ----------

def _table_exists(name: str) -> bool:
    return bool(spark.catalog.tableExists(name))


def _ensure_schema(schema_name: str):
    if not dry_run:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")


def _clone_table(source_table: str, target_table: str) -> dict[str, Any]:
    exists = _table_exists(source_table)
    out = {
        "type": "table_clone",
        "source": source_table,
        "target": target_table,
        "source_exists": exists,
        "executed": False,
    }
    if not exists:
        out["status"] = "skipped_source_not_found"
        return out

    if not dry_run:
        spark.sql(f"CREATE OR REPLACE TABLE {target_table} {clone_mode} CLONE {source_table}")
        out["executed"] = True
    out["status"] = "ok"
    return out


def _recreate_current_view(source_current: str, target_snapshot: str, target_current: str) -> dict[str, Any]:
    exists = _table_exists(source_current)
    out = {
        "type": "current_view",
        "source": source_current,
        "target": target_current,
        "source_exists": exists,
        "executed": False,
    }
    if not exists:
        out["status"] = "skipped_source_not_found"
        return out

    run_id_value = None
    try:
        rows = spark.table(source_current).select("run_id").distinct().limit(1).collect()
        if rows:
            run_id_value = rows[0]["run_id"]
    except Exception:
        run_id_value = None

    if not dry_run:
        if run_id_value:
            safe_run_id = str(run_id_value).replace("'", "''")
            spark.sql(
                f"""
                CREATE OR REPLACE VIEW {target_current} AS
                SELECT *
                FROM {target_snapshot}
                WHERE run_id = '{safe_run_id}'
                """
            )
        else:
            spark.sql(
                f"""
                CREATE OR REPLACE VIEW {target_current} AS
                SELECT *
                FROM {target_snapshot}
                """
            )
        out["executed"] = True

    out["status"] = "ok"
    out["run_id"] = run_id_value
    return out


def _copy_snapshot_current_pair(source_base: str, target_base: str) -> list[dict[str, Any]]:
    source_snapshot = snapshot_table_name(source_base)
    source_current = current_table_name(source_base)
    target_snapshot = snapshot_table_name(target_base)
    target_current = current_table_name(target_base)

    result = [_clone_table(source_snapshot, target_snapshot)]
    result.append(_recreate_current_view(source_current, target_snapshot, target_current))
    return result


def _list_schema_objects(catalog: str, schema: str) -> list[dict[str, Any]]:
    info_table = f"{catalog}.information_schema.tables"
    rows = spark.sql(
        f"""
        SELECT table_name, table_type
        FROM {info_table}
        WHERE table_schema = '{schema}'
        """
    ).collect()
    return [{"table_name": r["table_name"], "table_type": r["table_type"]} for r in rows]


def _copy_schema_reports(source_catalog: str, target_catalog: str, schema: str) -> dict[str, Any]:
    source = f"dbfs:/Volumes/{source_catalog}/{schema}/reports"
    target = f"dbfs:/Volumes/{target_catalog}/{schema}/reports"
    out = {
        "type": "reports",
        "source": source,
        "target": target,
        "executed": False,
    }

    try:
        _ = dbutils.fs.ls(source)
    except Exception as e:
        out["status"] = "skipped_source_not_found"
        out["reason"] = str(e)
        return out

    if not dry_run:
        dbutils.fs.mkdirs(target)
        dbutils.fs.cp(source, target, True)
        out["executed"] = True

    out["status"] = "ok"
    return out


def _copy_schema_models(source_catalog: str, target_catalog: str, schema: str) -> dict[str, Any]:
    from mlflow import MlflowClient

    client = MlflowClient()
    source_schema = f"{source_catalog}.{schema}"
    target_schema = f"{target_catalog}.{schema}"

    out = {
        "type": "models",
        "source_schema": source_schema,
        "target_schema": target_schema,
        "items": [],
    }

    model_rows = spark.sql(f"SHOW MODELS IN {source_schema}").collect()
    for row in model_rows:
        src_model = row["modelName"]
        tgt_model = src_model.replace(f"{source_catalog}.", f"{target_catalog}.", 1)
        item = {
            "source_model": src_model,
            "target_model": tgt_model,
            "copied_versions": [],
            "aliases": {},
        }

        if not dry_run:
            try:
                client.get_registered_model(tgt_model)
            except Exception:
                client.create_registered_model(tgt_model)

        src_versions = client.search_model_versions(f"name='{src_model}'")
        version_map: dict[int, int] = {}
        for v in src_versions:
            src_uri = f"models:/{src_model}/{int(v.version)}"
            if not dry_run:
                copied = client.copy_model_version(src_uri, tgt_model)
                item["copied_versions"].append(int(copied.version))
                version_map[int(v.version)] = int(copied.version)
            else:
                item["copied_versions"].append(int(v.version))
                version_map[int(v.version)] = int(v.version)

        try:
            src_rm = client.get_registered_model(src_model)
            aliases = dict(getattr(src_rm, "aliases", {}) or {})
        except Exception:
            aliases = {}

        item["aliases"] = aliases
        if aliases and (not dry_run):
            for alias, src_ver in aliases.items():
                src_ver_int = int(src_ver)
                if src_ver_int in version_map:
                    client.set_registered_model_alias(tgt_model, alias, version_map[src_ver_int])

        out["items"].append(item)

    return out


# COMMAND ----------

results: list[dict[str, Any]] = []

for plan in SCHEMA_PLANS:
    source_catalog = plan["source_catalog"]
    target_catalog = plan["target_catalog"]
    schema = plan["schema"]
    mode = plan["mode"]
    source_schema = f"{source_catalog}.{schema}"
    target_schema = f"{target_catalog}.{schema}"

    _ensure_schema(target_schema)

    plan_result = {
        "source_schema": source_schema,
        "target_schema": target_schema,
        "mode": mode,
        "items": [],
    }

    if mode == "selected_snapshot_current":
        for base in plan["base_tables"]:
            source_base = f"{source_schema}.{base}"
            target_base = f"{target_schema}.{base}"
            plan_result["items"].extend(_copy_snapshot_current_pair(source_base, target_base))

        for table_name in plan["direct_tables"]:
            src = f"{source_schema}.{table_name}"
            tgt = f"{target_schema}.{table_name}"
            plan_result["items"].append(_clone_table(src, tgt))

    elif mode == "all_snapshot_current":
        objects = _list_schema_objects(source_catalog, schema)
        table_names = {o["table_name"] for o in objects}

        bases = sorted({n[: -len("_snapshot")] for n in table_names if n.endswith("_snapshot")})
        handled = set()
        for base in bases:
            source_base = f"{source_schema}.{base}"
            target_base = f"{target_schema}.{base}"
            plan_result["items"].extend(_copy_snapshot_current_pair(source_base, target_base))
            handled.add(f"{base}_snapshot")
            handled.add(f"{base}_current")

        extras = sorted(table_names - handled)
        for extra in extras:
            src = f"{source_schema}.{extra}"
            tgt = f"{target_schema}.{extra}"
            plan_result["items"].append(_clone_table(src, tgt))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if copy_reports:
        plan_result["items"].append(_copy_schema_reports(source_catalog, target_catalog, schema))

    if copy_models:
        plan_result["items"].append(_copy_schema_models(source_catalog, target_catalog, schema))

    results.append(plan_result)


# COMMAND ----------

output = {
    "dry_run": dry_run,
    "clone_mode": clone_mode,
    "copy_reports": copy_reports,
    "copy_models": copy_models,
    "results": results,
}

print(json.dumps(output, indent=2, default=str))
