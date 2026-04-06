from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any
import os
from uuid import uuid4


@dataclass(frozen=True)
class SubjectSpec:
    schema: str
    report_folder: str
    table_templates: tuple[str, ...]
    model_template: str | None = None
    report_templates: tuple[str, ...] = ()


SUBJECT_SPECS: dict[str, SubjectSpec] = {
    "user_segmentation": SubjectSpec(
        schema="user_segmentation",
        report_folder="reports",
        table_templates=(
            "feat_{analysis}",
            "pca_{analysis}",
            "pred_{analysis}",
            "cluster_{analysis}",
            "suggestion_{analysis}",
        ),
        model_template="mdl_{analysis}",
        report_templates=(
            "clusters_report_{analysis}.html",
            "cluster_report_detailed_{analysis}.html",
            "cluster_detailed_report_{analysis}.html",
        ),
    ),
    "cross_sell_mba": SubjectSpec(
        schema="cross_sell_mba",
        report_folder="reports",
        table_templates=(
            "mba_user_item_features_{analysis}",
            "mba_transactions_{analysis}",
            "mba_association_rules_{analysis}",
            "mba_association_rules_cross_{analysis}",
            "mba_association_rules_focus_{analysis}",
        ),
        model_template=None,
        report_templates=(
            "mba_rules_report_{analysis}.html",
        ),
    ),
}


def _mlflow_client():
    from mlflow import MlflowClient

    return MlflowClient()


def snapshot_table_name(table_base: str) -> str:
    return f"{table_base}_snapshot"


def current_table_name(table_base: str) -> str:
    return f"{table_base}_current"


def _table_schema_name(table_name: str) -> str:
    parts = table_name.split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected fully-qualified table name, got: {table_name}")
    return ".".join(parts[:-1])


def _ensure_schema(spark, table_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_table_schema_name(table_name)}")


def _with_col_if_missing(df, col_name: str, expr):
    if col_name in df.columns:
        return df
    return df.withColumn(col_name, expr)


def publish_table(
    *,
    spark,
    df,
    table_base: str,
    analysis_name: str | None = None,
    ref_date: str | None = None,
    run_id: str | None = None,
    current_mode: str = "view",
) -> dict[str, Any]:
    """
    Publish a dataframe using snapshot/current convention.
    - Snapshot: append-only historical table
    - Current: latest approved run as a view (default) or physical table
    """
    snapshot_name = snapshot_table_name(table_base)
    current_name = current_table_name(table_base)
    current_mode = current_mode.lower().strip()
    if current_mode not in {"view", "table"}:
        raise ValueError("current_mode must be 'view' or 'table'")

    _ensure_schema(spark, snapshot_name)
    _ensure_schema(spark, current_name)

    rid = run_id or str(uuid4())
    from pyspark.sql import functions as F

    out_df = df
    out_df = _with_col_if_missing(out_df, "run_id", F.lit(rid))
    out_df = _with_col_if_missing(out_df, "run_ts", F.current_timestamp())
    out_df = _with_col_if_missing(
        out_df,
        "ref_date",
        F.to_date(F.lit(ref_date)) if ref_date is not None else F.current_date(),
    )
    if analysis_name is not None:
        out_df = _with_col_if_missing(out_df, "analysis_name", F.lit(analysis_name))

    out_df.write.format("delta").mode("append").saveAsTable(snapshot_name)

    if current_mode == "view":
        spark.sql(
            f"""
            CREATE OR REPLACE VIEW {current_name} AS
            SELECT *
            FROM {snapshot_name}
            WHERE run_id = '{rid}'
            """
        )
    else:
        out_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(current_name)

    return {
        "table_base": table_base,
        "snapshot_table": snapshot_name,
        "current_table": current_name,
        "run_id": rid,
        "current_mode": current_mode,
    }


def read_current_table(*, spark, table_base: str):
    current_name = current_table_name(table_base)
    snapshot_name = snapshot_table_name(table_base)

    if spark.catalog.tableExists(current_name):
        return spark.table(current_name)

    if spark.catalog.tableExists(snapshot_name):
        return spark.sql(
            f"""
            SELECT *
            FROM {snapshot_name}
            WHERE run_ts = (
              SELECT MAX(run_ts) FROM {snapshot_name}
            )
            """
        )

    if spark.catalog.tableExists(table_base):
        return spark.table(table_base)

    raise ValueError(
        f"Table not found for base '{table_base}'. "
        f"Tried: '{current_name}', '{snapshot_name}', '{table_base}'."
    )


def build_model_name(subject: str, analysis_name: str, catalog: str = "uat_analytics") -> str:
    subject = subject.strip()
    if subject not in SUBJECT_SPECS:
        raise ValueError(f"Unsupported subject '{subject}'. Supported: {list(SUBJECT_SPECS)}")
    return f"{catalog}.{SUBJECT_SPECS[subject].schema}.mdl_{analysis_name}"


def model_uri(model_name: str, *, version: int | None = None, alias: str | None = None) -> str:
    if version is not None:
        return f"models:/{model_name}/{int(version)}"
    if alias:
        return f"models:/{model_name}@{alias}"
    return f"models:/{model_name}"


def load_model(subject: str, analysis_name: str, *, catalog: str = "uat_analytics", version: int | None = None, alias: str | None = None):
    from mlflow import sklearn as mlflow_sklearn

    mname = build_model_name(subject=subject, analysis_name=analysis_name, catalog=catalog)
    muri = model_uri(mname, version=version, alias=alias)
    return mlflow_sklearn.load_model(muri)


def get_model(
    subject: str,
    analysis_name: str,
    *,
    catalog: str = "uat_analytics",
    version: int | None = None,
    alias: str | None = None,
):
    """
    Load a model with sklearn flavor first and fallback to pyfunc wrapped sklearn model.
    """
    import mlflow

    mname = build_model_name(subject=subject, analysis_name=analysis_name, catalog=catalog)
    muri = model_uri(mname, version=version, alias=alias)

    try:
        return mlflow.sklearn.load_model(muri)
    except Exception:
        pyfunc_model = mlflow.pyfunc.load_model(muri)
        return pyfunc_model._model_impl.python_model.model


def list_model_versions(model_name: str):
    client = _mlflow_client()
    return client.search_model_versions(f"name='{model_name}'")


def set_model_alias(model_name: str, alias: str, version: int):
    client = _mlflow_client()
    client.set_registered_model_alias(model_name, alias, int(version))


def _resolve_model_version(model_name: str, *, version: int | None = None, alias: str | None = None, run_id: str | None = None) -> int:
    client = _mlflow_client()

    if version is not None:
        return int(version)

    if alias:
        mv = client.get_model_version_by_alias(model_name, alias)
        return int(mv.version)

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'.")

    if run_id:
        by_run = [v for v in versions if getattr(v, "run_id", None) == run_id]
        if not by_run:
            raise ValueError(f"No model version found for model '{model_name}' and run_id '{run_id}'.")
        return int(max(by_run, key=lambda v: int(v.version)).version)

    return int(max(versions, key=lambda v: int(v.version)).version)


def register_candidate(
    model_name: str,
    *,
    version: int | None = None,
    run_id: str | None = None,
    alias: str = "challenger",
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Register/select model candidate version and assign candidate alias.
    """
    client = _mlflow_client()
    resolved_version = _resolve_model_version(model_name, version=version, run_id=run_id)
    client.set_registered_model_alias(model_name, alias, resolved_version)

    if tags:
        for k, v in tags.items():
            client.set_model_version_tag(model_name, str(resolved_version), str(k), str(v))

    return {
        "model_name": model_name,
        "version": resolved_version,
        "alias": alias,
        "run_id": run_id,
        "registered": True,
    }


def validate_candidate(
    model_name: str,
    *,
    version: int | None = None,
    alias: str | None = "challenger",
    required_signature: bool = True,
    sample_input=None,
) -> dict[str, Any]:
    """
    Validate candidate model availability, signature, and optional sample prediction.
    """
    import mlflow

    resolved_version = _resolve_model_version(model_name, version=version, alias=alias)
    uri = model_uri(model_name, version=resolved_version)
    pyfunc_model = mlflow.pyfunc.load_model(uri)
    metadata = getattr(pyfunc_model, "metadata", None)
    signature = getattr(metadata, "signature", None)

    checks = {
        "model_load_ok": True,
        "signature_ok": (signature is not None) if required_signature else True,
        "sample_predict_ok": True,
    }

    sample_error = None
    if sample_input is not None:
        try:
            _ = pyfunc_model.predict(sample_input)
        except Exception as e:
            checks["sample_predict_ok"] = False
            sample_error = str(e)

    passed = all(checks.values())

    return {
        "model_name": model_name,
        "version": resolved_version,
        "alias": alias,
        "uri": uri,
        "passed": passed,
        "checks": checks,
        "sample_error": sample_error,
    }


def promote_alias(
    model_name: str,
    *,
    source_alias: str = "challenger",
    target_alias: str = "champion",
    version: int | None = None,
) -> dict[str, Any]:
    """
    Promote model alias (e.g. challenger -> champion) within the same registered model.
    """
    client = _mlflow_client()
    resolved_version = _resolve_model_version(model_name, version=version, alias=source_alias if version is None else None)
    client.set_registered_model_alias(model_name, target_alias, resolved_version)
    return {
        "model_name": model_name,
        "version": resolved_version,
        "source_alias": source_alias,
        "target_alias": target_alias,
        "promoted": True,
    }


def promote_model_version(source_model: str, target_model: str, *, version: int | None = None, alias: str | None = "champion"):
    """Copy one model version from source to target and optionally set alias on target."""
    result = _deploy_model(
        source_model=source_model,
        target_model=target_model,
        model_version=version,
        model_alias=alias,
        dry_run=False,
    )
    return result


def smoke_test_table(
    *,
    spark,
    table_name: str,
    required_columns: list[str] | None = None,
    min_rows: int = 1,
) -> dict[str, Any]:
    exists = bool(spark.catalog.tableExists(table_name))
    row_count = None
    missing_columns: list[str] = []
    passed = False
    error = None

    try:
        if exists:
            df = spark.table(table_name)
            cols = set(df.columns)
            if required_columns:
                missing_columns = [c for c in required_columns if c not in cols]
            row_count = df.limit(max(min_rows, 1)).count() if min_rows > 0 else 0
            enough_rows = (row_count >= min_rows) if min_rows > 0 else True
            passed = enough_rows and (len(missing_columns) == 0)
        else:
            passed = False
    except Exception as e:
        error = str(e)
        passed = False

    return {
        "type": "table",
        "target": table_name,
        "exists": exists,
        "row_count_sample": row_count,
        "missing_columns": missing_columns,
        "min_rows": min_rows,
        "passed": passed,
        "error": error,
    }


def smoke_test_model(
    *,
    model_name: str,
    version: int | None = None,
    alias: str | None = "champion",
    required_signature: bool = True,
    sample_input=None,
) -> dict[str, Any]:
    try:
        validation = validate_candidate(
            model_name,
            version=version,
            alias=alias,
            required_signature=required_signature,
            sample_input=sample_input,
        )
        return {
            "type": "model",
            "target": model_name,
            "version": validation["version"],
            "alias": alias,
            "passed": bool(validation["passed"]),
            "checks": validation["checks"],
            "error": validation.get("sample_error"),
        }
    except Exception as e:
        return {
            "type": "model",
            "target": model_name,
            "version": version,
            "alias": alias,
            "passed": False,
            "checks": {},
            "error": str(e),
        }


def smoke_test_report(
    *,
    report_path: str,
    min_size_bytes: int = 1,
) -> dict[str, Any]:
    exists = os.path.exists(report_path)
    size = os.path.getsize(report_path) if exists else 0
    passed = exists and (size >= int(min_size_bytes))
    return {
        "type": "report",
        "target": report_path,
        "exists": exists,
        "size_bytes": size,
        "min_size_bytes": min_size_bytes,
        "passed": passed,
        "error": None if passed else "report missing or below minimum size",
    }


def run_deploy_smoke_tests(
    *,
    spark,
    deploy_plan: dict[str, Any],
    required_columns_by_table: dict[str, list[str]] | None = None,
    min_rows_by_table: dict[str, int] | None = None,
    model_alias: str = "champion",
    report_min_size_bytes: int = 1,
) -> dict[str, Any]:
    required_columns_by_table = required_columns_by_table or {}
    min_rows_by_table = min_rows_by_table or {}

    results: list[dict[str, Any]] = []

    for t in deploy_plan.get("tables", []):
        target = t["target"]
        results.append(
            smoke_test_table(
                spark=spark,
                table_name=target,
                required_columns=required_columns_by_table.get(target),
                min_rows=min_rows_by_table.get(target, 1),
            )
        )

    for r in deploy_plan.get("reports", []):
        results.append(
            smoke_test_report(
                report_path=r["target"],
                min_size_bytes=report_min_size_bytes,
            )
        )

    model_plan = deploy_plan.get("model")
    if model_plan and model_plan.get("target_model"):
        results.append(
            smoke_test_model(
                model_name=model_plan["target_model"],
                alias=model_alias,
                required_signature=True,
            )
        )

    passed = all(bool(x.get("passed")) for x in results) if results else True
    return {
        "passed": passed,
        "results": results,
    }


def _copy_report_if_exists(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _deploy_model(
    source_model: str,
    target_model: str,
    model_version: int | None,
    model_alias: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_model": source_model,
        "target_model": target_model,
        "copied": False,
        "reason": None,
    }

    try:
        from mlflow import MlflowClient
    except Exception as e:  # pragma: no cover
        result["reason"] = f"mlflow unavailable: {e}"
        return result

    client = MlflowClient()

    try:
        client.get_registered_model(source_model)
    except Exception as e:
        result["reason"] = f"source model not found: {e}"
        return result

    if dry_run:
        result["copied"] = True
        result["reason"] = "dry_run"
        return result

    try:
        client.get_registered_model(target_model)
    except Exception:
        client.create_registered_model(target_model)

    if model_version is not None:
        src_uri = f"models:/{source_model}/{int(model_version)}"
    elif model_alias:
        src_uri = f"models:/{source_model}@{model_alias}"
    else:
        versions = client.search_model_versions(f"name='{source_model}'")
        if not versions:
            result["reason"] = "source model has no versions"
            return result
        latest = max(versions, key=lambda v: int(v.version))
        src_uri = f"models:/{source_model}/{latest.version}"

    copied = client.copy_model_version(src_uri, target_model)
    result["copied"] = True
    result["source_uri"] = src_uri
    result["target_version"] = getattr(copied, "version", None)

    if model_alias and result["target_version"] is not None:
        try:
            client.set_registered_model_alias(target_model, model_alias, result["target_version"])
            result["alias_set"] = model_alias
        except Exception as e:
            result["alias_set"] = f"failed: {e}"

    return result


def deploy(
    subject: str,
    analysis_name: str,
    *,
    spark,
    source_catalog: str = "uat_analytics",
    target_catalog: str = "prod_analytics",
    dry_run: bool = True,
    clone_mode: str = "DEEP",
    model_version: int | None = None,
    model_alias: str | None = "champion",
    run_smoke_tests: bool = False,
    required_columns_by_table: dict[str, list[str]] | None = None,
    min_rows_by_table: dict[str, int] | None = None,
    report_min_size_bytes: int = 1,
) -> dict[str, Any]:
    """
    Promote assets for a given subject + analysis.

    Parameters
    ----------
    subject: str
        "user_segmentation" or "cross_sell_mba".
    analysis_name: str
        Analysis suffix used in tables/models/reports (e.g., "ols").
    spark:
        Active SparkSession.
    source_catalog/target_catalog: str
        Catalogs for source and target environments.
    dry_run: bool
        If True, only builds the plan and validates availability.
    clone_mode: str
        "DEEP" or "SHALLOW" for Delta clone. (deep is everything, shallow is metadata-only)
    model_version/model_alias:
        Optional model selector. If version is provided, it takes precedence.
    """
    subject = subject.strip()
    if subject not in SUBJECT_SPECS:
        raise ValueError(f"Unsupported subject '{subject}'. Supported: {list(SUBJECT_SPECS)}")

    spec = SUBJECT_SPECS[subject]
    clone_mode = clone_mode.upper().strip()
    if clone_mode not in {"DEEP", "SHALLOW"}:
        raise ValueError("clone_mode must be DEEP or SHALLOW")

    source_schema = f"{source_catalog}.{spec.schema}"
    target_schema = f"{target_catalog}.{spec.schema}"

    plan: dict[str, Any] = {
        "subject": subject,
        "analysis_name": analysis_name,
        "source_schema": source_schema,
        "target_schema": target_schema,
        "dry_run": dry_run,
        "tables": [],
        "reports": [],
        "model": None,
    }

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")

    # Tables
    for template in spec.table_templates:
        src_table = f"{source_schema}.{template.format(analysis=analysis_name)}"
        dst_table = f"{target_schema}.{template.format(analysis=analysis_name)}"
        exists = bool(spark.catalog.tableExists(src_table))

        item = {
            "source": src_table,
            "target": dst_table,
            "exists": exists,
            "executed": False,
        }

        if exists and not dry_run:
            spark.sql(f"CREATE OR REPLACE TABLE {dst_table} {clone_mode} CLONE {src_table}")
            item["executed"] = True

        plan["tables"].append(item)

    # Reports
    src_report_dir = Path(f"/Volumes/{source_catalog}/{spec.schema}/{spec.report_folder}")
    dst_report_dir = Path(f"/Volumes/{target_catalog}/{spec.schema}/{spec.report_folder}")

    for report_tpl in spec.report_templates:
        report_name = report_tpl.format(analysis=analysis_name)
        src = src_report_dir / report_name
        dst = dst_report_dir / report_name
        copied_or_planned = _copy_report_if_exists(src, dst, dry_run=dry_run)
        plan["reports"].append(
            {
                "source": str(src),
                "target": str(dst),
                "exists": src.exists(),
                "executed": copied_or_planned and (not dry_run),
                "planned": copied_or_planned and dry_run,
            }
        )

    # Model
    if spec.model_template:
        model_name = spec.model_template.format(analysis=analysis_name)
        source_model = f"{source_schema}.{model_name}"
        target_model = f"{target_schema}.{model_name}"
        plan["model"] = _deploy_model(
            source_model=source_model,
            target_model=target_model,
            model_version=model_version,
            model_alias=model_alias,
            dry_run=dry_run,
        )

    if run_smoke_tests and (not dry_run):
        plan["smoke_tests"] = run_deploy_smoke_tests(
            spark=spark,
            deploy_plan=plan,
            required_columns_by_table=required_columns_by_table,
            min_rows_by_table=min_rows_by_table,
            model_alias=model_alias or "champion",
            report_min_size_bytes=report_min_size_bytes,
        )

    return plan
