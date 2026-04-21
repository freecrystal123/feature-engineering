# Databricks notebook source

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
import os
import re
from uuid import uuid4


# COMMAND ----------

# MAGIC %run ../models/models_structure


# COMMAND ----------

def _mlflow_client():
    from mlflow import MlflowClient

    return MlflowClient()


# COMMAND ----------

def normalize_environment(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if v in {"dev", "train", "prod"}:
        return v
    m = re.search(r"/(dev|train|prod)(?:/|$)", v)
    if m:
        return m.group(1)
    return None


# COMMAND ----------

def detect_environment(default: str = "train") -> str:
    # 1) Prefer explicit ENVIRONMENT variable (e.g., passed in Databricks Jobs).
    env = normalize_environment(os.getenv("ENVIRONMENT"))
    if env:
        return env

    # 2) Detect from Databricks notebook path (workspace root_path contains /train or /prod).
    try:
        dbutils_obj = globals().get("dbutils")
        if dbutils_obj is not None:
            notebook_path = (
                dbutils_obj.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .notebookPath()
                .get()
            )
            env = normalize_environment(notebook_path)
            if env:
                return env
    except Exception:
        pass

    # 3) Fallback to process working directory path.
    env = normalize_environment(os.getcwd())
    if env:
        return env

    return default


# COMMAND ----------

def snapshot_table_name(table_base: str) -> str:
    return f"{table_base}_snapshot"


# COMMAND ----------

def current_table_name(table_base: str) -> str:
    return f"{table_base}_current"


# COMMAND ----------

def _table_schema_name(table_name: str) -> str:
    parts = table_name.split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected fully-qualified table name, got: {table_name}")
    return ".".join(parts[:-1])


# COMMAND ----------

def _ensure_schema(spark, table_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_table_schema_name(table_name)}")


# COMMAND ----------

def _with_col_if_missing(df, col_name: str, expr):
    if col_name in df.columns:
        return df
    return df.withColumn(col_name, expr)


# COMMAND ----------

def _normalize_cols(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


# COMMAND ----------

def _ensure_uc_primary_key_constraints(
    *,
    spark,
    table_name: str,
    primary_keys: str | list[str] | tuple[str, ...] | None = None,
    timeseries_column: str | None = None,
) -> dict[str, Any]:
    """
    Ensure UC informational PRIMARY KEY exists for feature-table usage.
    """
    pk_cols = _normalize_cols(primary_keys)
    if not pk_cols:
        return {"applied": False, "reason": "no_primary_keys"}

    cols = spark.table(table_name).columns
    missing = [c for c in pk_cols if c not in cols]
    if missing:
        raise ValueError(f"Primary key columns not found in {table_name}: {missing}")

    if timeseries_column is not None and timeseries_column not in cols:
        raise ValueError(
            f"timeseries_column='{timeseries_column}' was not found in table '{table_name}'."
        )
    if (timeseries_column is not None) and (timeseries_column not in pk_cols):
        pk_cols.append(timeseries_column)

    # UC PK columns must be NOT NULL.
    for c in pk_cols:
        null_count = (
            spark.sql(
                f"""
                SELECT COUNT(*) AS c
                FROM {table_name}
                WHERE `{c}` IS NULL
                """
            )
            .collect()[0]["c"]
        )
        if int(null_count) > 0:
            raise ValueError(
                f"Cannot apply PRIMARY KEY on {table_name}: column '{c}' contains {int(null_count)} NULL values."
            )

        try:
            spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN `{c}` SET NOT NULL")
        except Exception as e:
            err = str(e).lower()
            already_not_null = (
                "already" in err
                and ("not null" in err or "nullable" in err)
            )
            if not already_not_null:
                raise

    pk_expr_parts: list[str] = []
    for c in pk_cols:
        if timeseries_column is not None and c == timeseries_column:
            pk_expr_parts.append(f"`{c}` TIMESERIES")
        else:
            pk_expr_parts.append(f"`{c}`")
    pk_expr = ", ".join(pk_expr_parts)
    constraint_name = f"pk_{table_name.split('.')[-1]}".replace("-", "_")

    try:
        spark.sql(
            f"""
            ALTER TABLE {table_name}
            ADD CONSTRAINT {constraint_name}
            PRIMARY KEY ({pk_expr})
            """
        )
        return {
            "applied": True,
            "constraint_name": constraint_name,
            "primary_keys": pk_cols,
            "timeseries_column": timeseries_column,
        }
    except Exception as e:
        # UC raises when the PK already exists; treat as idempotent.
        err = str(e)
        lower_err = err.lower()
        already_exists = (
            ("already exists" in lower_err)
            or ("duplicate constraint" in lower_err)
            or ("resource_already_exists.uc_child_columns_already_exist_in_primary_key_or_unique_constraint" in lower_err)
            or ("already has a primary key constraint" in lower_err)
        )
        if not already_exists:
            raise
        return {
            "applied": False,
            "constraint_name": constraint_name,
            "primary_keys": pk_cols,
            "timeseries_column": timeseries_column,
            "reason": err,
        }


# COMMAND ----------

def publish_table(
    *,
    spark,
    df,
    table_base: str,
    analysis_name: str | None = None,
    ref_date: str | None = None,
    run_id: str | None = None,
    current_mode: str = "view",
    primary_keys: str | list[str] | tuple[str, ...] | None = None,
    timeseries_column: str | None = None,
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
    snapshot_constraints = _ensure_uc_primary_key_constraints(
        spark=spark,
        table_name=snapshot_name,
        primary_keys=primary_keys,
        timeseries_column=timeseries_column,
    )

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
        _ensure_uc_primary_key_constraints(
            spark=spark,
            table_name=current_name,
            primary_keys=primary_keys,
            timeseries_column=timeseries_column,
        )

    return {
        "table_base": table_base,
        "snapshot_table": snapshot_name,
        "current_table": current_name,
        "run_id": rid,
        "current_mode": current_mode,
        "primary_keys": _normalize_cols(primary_keys),
        "timeseries_column": timeseries_column,
        "snapshot_constraints": snapshot_constraints,
    }


# COMMAND ----------

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


# COMMAND ----------

def get_feature_engineering_client():
    from databricks.feature_engineering import FeatureEngineeringClient

    return FeatureEngineeringClient()


# COMMAND ----------

def _normalize_lookup_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    required = {"table_name", "lookup_key"}
    missing = [k for k in required if not out.get(k)]
    if missing:
        raise ValueError(f"Feature lookup spec missing required keys: {missing}. Spec={spec}")
    return out


# COMMAND ----------

def build_feature_lookups(
    *,
    primary_table: str,
    primary_lookup_key: str | list[str] | tuple[str, ...],
    primary_feature_names: list[str] | None = None,
    secondary_lookups: list[dict[str, Any]] | None = None,
    primary_timestamp_lookup_key: str | None = None,
):
    """
    Build FeatureLookup list from one primary feature table and optional secondary tables.
    """
    from databricks.feature_engineering import FeatureLookup

    lookups = [
        FeatureLookup(
            table_name=primary_table,
            lookup_key=primary_lookup_key,
            feature_names=primary_feature_names,
            timestamp_lookup_key=primary_timestamp_lookup_key,
        )
    ]

    for raw in (secondary_lookups or []):
        spec = _normalize_lookup_spec(raw)
        lookups.append(
            FeatureLookup(
                table_name=spec["table_name"],
                lookup_key=spec["lookup_key"],
                feature_names=spec.get("feature_names"),
                rename_outputs=spec.get("rename_outputs"),
                timestamp_lookup_key=spec.get("timestamp_lookup_key"),
            )
        )
    return lookups


# COMMAND ----------

def create_feature_training_set(
    *,
    fe_client,
    entity_df,
    primary_table: str,
    primary_lookup_key: str | list[str] | tuple[str, ...],
    primary_feature_names: list[str] | None = None,
    secondary_lookups: list[dict[str, Any]] | None = None,
    label: str | None = None,
    exclude_columns: list[str] | None = None,
    timestamp_lookup_key: str | None = None,
):
    lookups = build_feature_lookups(
        primary_table=primary_table,
        primary_lookup_key=primary_lookup_key,
        primary_feature_names=primary_feature_names,
        secondary_lookups=secondary_lookups,
        primary_timestamp_lookup_key=timestamp_lookup_key,
    )
    return fe_client.create_training_set(
        df=entity_df,
        feature_lookups=lookups,
        label=label,
        exclude_columns=exclude_columns or [],
    )


# COMMAND ----------

def log_model_with_feature_lineage(
    *,
    fe_client,
    model,
    training_set,
    registered_model_name: str,
    flavor,
    artifact_path: str = "model",
    signature=None,
    input_example=None,
    extra_kwargs: dict[str, Any] | None = None,
):
    kwargs = dict(extra_kwargs or {})
    return fe_client.log_model(
        model=model,
        artifact_path=artifact_path,
        flavor=flavor,
        training_set=training_set,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        **kwargs,
    )


# COMMAND ----------

def build_model_name(subject: str, analysis_name: str, catalog: str = "ml_train") -> str:
    subject = subject.strip()
    if subject not in SUBJECT_SPECS:
        raise ValueError(f"Unsupported subject '{subject}'. Supported: {list(SUBJECT_SPECS)}")
    return f"{catalog}.{SUBJECT_SPECS[subject].schema}.mdl_{analysis_name}"


# COMMAND ----------

def model_uri(model_name: str, *, version: int | None = None, alias: str | None = None) -> str:
    if version is not None:
        return f"models:/{model_name}/{int(version)}"
    if alias:
        return f"models:/{model_name}@{alias}"
    return f"models:/{model_name}"


# COMMAND ----------

def load_model(subject: str, analysis_name: str, *, catalog: str = "ml_train", version: int | None = None, alias: str | None = None):
    from mlflow import sklearn as mlflow_sklearn

    mname = build_model_name(subject=subject, analysis_name=analysis_name, catalog=catalog)
    muri = model_uri(mname, version=version, alias=alias)
    return mlflow_sklearn.load_model(muri)


# COMMAND ----------

def get_model(
    subject: str,
    analysis_name: str,
    *,
    catalog: str = "ml_train",
    version: int | None = None,
    alias: str | None = None,
):
    """
    Load a model with sklearn flavor first and fallback to pyfunc wrappers.
    """
    import mlflow
    import inspect

    mname = build_model_name(subject=subject, analysis_name=analysis_name, catalog=catalog)
    muri = model_uri(mname, version=version, alias=alias)

    def _is_sklearn_like(obj) -> bool:
        mod = getattr(getattr(obj, "__class__", None), "__module__", "") or ""
        return mod.startswith("sklearn.")

    def _can_predict(obj) -> bool:
        return callable(getattr(obj, "predict", None))

    def _unwrap_model_object(obj):
        if obj is None:
            return None

        visited: set[int] = set()
        queue = [obj]
        candidate_attrs = (
            "model",
            "_model",
            "wrapped_model",
            "sk_model",
            "_sk_model",
            "sklearn_model",
            "prediction_model",
            "python_model",
        )

        while queue:
            cur = queue.pop(0)
            if cur is None:
                continue
            cur_id = id(cur)
            if cur_id in visited:
                continue
            visited.add(cur_id)

            if _is_sklearn_like(cur) and _can_predict(cur):
                return cur

            for attr in candidate_attrs:
                nxt = getattr(cur, attr, None)
                if nxt is not None and id(nxt) not in visited:
                    queue.append(nxt)

        return obj

    try:
        model = mlflow.sklearn.load_model(muri)
        return _unwrap_model_object(model)
    except Exception:
        pyfunc_model = mlflow.pyfunc.load_model(muri)
        impl = getattr(pyfunc_model, "_model_impl", None)
        python_model = getattr(impl, "python_model", None) if impl is not None else None

        for candidate in (
            pyfunc_model,
            impl,
            python_model,
            getattr(python_model, "model", None) if python_model is not None else None,
            getattr(impl, "model", None) if impl is not None else None,
        ):
            unwrapped = _unwrap_model_object(candidate)
            if _is_sklearn_like(unwrapped) and _can_predict(unwrapped):
                return unwrapped

        if _can_predict(pyfunc_model):
            sig = inspect.signature(pyfunc_model.predict)
            if "params" in sig.parameters:
                class _PyfuncCompat:
                    def __init__(self, m):
                        self._m = m

                    def predict(self, X):
                        return self._m.predict(X, params=None)

                return _PyfuncCompat(pyfunc_model)
        return pyfunc_model


# COMMAND ----------

def list_model_versions(model_name: str):
    client = _mlflow_client()
    return client.search_model_versions(f"name='{model_name}'")


# COMMAND ----------

def set_model_alias(model_name: str, alias: str, version: int):
    client = _mlflow_client()
    client.set_registered_model_alias(model_name, alias, int(version))


# COMMAND ----------

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


# COMMAND ----------

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


# COMMAND ----------

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


# COMMAND ----------

def promote_alias(
    model_name: str,
    *,
    source_alias: str = "challenger",
    target_alias: str = "champion",
    version: int | None = None,
    remove_source_alias: bool = True,
) -> dict[str, Any]:
    """
    Promote model alias (e.g. challenger -> champion) within the same registered model.
    """
    client = _mlflow_client()
    resolved_version = _resolve_model_version(model_name, version=version, alias=source_alias if version is None else None)
    client.set_registered_model_alias(model_name, target_alias, resolved_version)

    source_alias_removed = False
    if remove_source_alias and source_alias and (source_alias != target_alias):
        try:
            src_mv = client.get_model_version_by_alias(model_name, source_alias)
            if int(src_mv.version) == int(resolved_version):
                client.delete_registered_model_alias(model_name, source_alias)
                source_alias_removed = True
        except Exception:
            # If source alias does not exist or cannot be removed, keep promotion successful.
            source_alias_removed = False

    return {
        "model_name": model_name,
        "version": resolved_version,
        "source_alias": source_alias,
        "target_alias": target_alias,
        "source_alias_removed": source_alias_removed,
        "promoted": True,
    }


# COMMAND ----------

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


# COMMAND ----------

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


# COMMAND ----------

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


# COMMAND ----------

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


# COMMAND ----------

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
    if model_plan and (model_plan.get("target_model") or model_plan.get("smoke_model")):
        model_name_for_smoke = model_plan.get("smoke_model") or model_plan.get("target_model")
        results.append(
            smoke_test_model(
                model_name=model_name_for_smoke,
                alias=model_alias,
                required_signature=True,
            )
        )

    passed = all(bool(x.get("passed")) for x in results) if results else True
    return {
        "passed": passed,
        "results": results,
    }


# COMMAND ----------

def _copy_report_if_exists(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dry_run:
        return True
    shutil.copy2(src, dst)
    return True


# COMMAND ----------

def write_report_with_single_backup(report_path: str, content: str) -> dict[str, str]:
    """
    Write report content, keeping a single backup copy of the previous version.
    Backup path is '<report_path>.bkp' and is overwritten each save.
    """
    path = Path(report_path)
    backup = Path(f"{report_path}.bkp")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        shutil.copy2(path, backup)

    path.write_text(content, encoding="utf-8")
    return {"report_path": str(path), "backup_path": str(backup)}


# COMMAND ----------

def _resolve_source_model_uri(
    *,
    source_model: str,
    model_version: int | None,
    model_alias: str | None,
):
    client = _mlflow_client()
    if model_version is not None:
        resolved_version = int(model_version)
    else:
        resolved_version = _resolve_model_version(
            source_model,
            alias=model_alias,
        )
    return {
        "source_uri": model_uri(source_model, version=resolved_version),
        "resolved_version": resolved_version,
        "client": client,
    }


# COMMAND ----------

def _import_mlflow_flavor(flavor_name: str):
    name = str(flavor_name).strip().lower()
    if name == "sklearn":
        from mlflow import sklearn as mlflow_flavor
        loader = mlflow_flavor.load_model
        return mlflow_flavor, loader
    if name == "spark":
        from mlflow import spark as mlflow_flavor
        loader = mlflow_flavor.load_model
        return mlflow_flavor, loader
    raise ValueError(f"Unsupported flavor '{flavor_name}'. Supported: sklearn, spark")


# COMMAND ----------

def _load_model_for_feature_rebind(*, source_uri: str, preferred_flavor: str):
    """
    Load source model object for rebind, tolerating FE-wrapped pyfunc-only registrations.
    Returns: (mlflow_flavor_module, model_object, resolved_flavor_name)
    """
    import mlflow

    preferred = str(preferred_flavor).strip().lower()

    if preferred == "spark":
        from mlflow import spark as mlflow_spark
        return mlflow_spark, mlflow_spark.load_model(source_uri), "spark"

    # Preferred path for sklearn-native models.
    from mlflow import sklearn as mlflow_sklearn
    try:
        return mlflow_sklearn, mlflow_sklearn.load_model(source_uri), "sklearn"
    except Exception:
        pass

    # Fallback: unwrap sklearn estimator from FE/pyfunc wrappers.
    pyfunc_model = mlflow.pyfunc.load_model(source_uri)
    impl = getattr(pyfunc_model, "_model_impl", None)
    python_model = getattr(impl, "python_model", None) if impl is not None else None

    def _is_sklearn_like(obj) -> bool:
        mod = getattr(getattr(obj, "__class__", None), "__module__", "") or ""
        return mod.startswith("sklearn.")

    def _can_predict(obj) -> bool:
        return callable(getattr(obj, "predict", None))

    visited: set[int] = set()
    queue = [
        impl,
        python_model,
        getattr(python_model, "model", None) if python_model is not None else None,
        getattr(impl, "model", None) if impl is not None else None,
        getattr(impl, "sklearn_model", None) if impl is not None else None,
    ]
    candidate_attrs = (
        "model",
        "_model",
        "wrapped_model",
        "sk_model",
        "_sk_model",
        "sklearn_model",
        "prediction_model",
        "python_model",
    )

    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)

        if _is_sklearn_like(cur) and _can_predict(cur):
            return mlflow_sklearn, cur, "sklearn_unwrapped"

        for attr in candidate_attrs:
            nxt = getattr(cur, attr, None)
            if nxt is not None and id(nxt) not in visited:
                queue.append(nxt)

    raise ValueError(
        "Feature-rebind deploy could not load source model as native sklearn object. "
        f"source_uri={source_uri}, preferred_flavor={preferred_flavor}"
    )


# COMMAND ----------

def _as_lookup_key(keys: tuple[str, ...]):
    if len(keys) == 1:
        return keys[0]
    return list(keys)


# COMMAND ----------

def _feature_rebind_primary_feature_names(*, spark, table_name: str, spec) -> list[str]:
    if spec.feature_names:
        return list(spec.feature_names)

    key_cols = set(spec.entity_keys)
    metadata_cols = set(spec.metadata_columns)
    excluded = key_cols | metadata_cols
    if spec.label_col:
        excluded.add(spec.label_col)

    candidate_cols = []
    for f in spark.table(table_name).schema.fields:
        if f.name in excluded:
            continue
        candidate_cols.append(f.name)
    return candidate_cols


# COMMAND ----------

def _relog_model_with_feature_rebinding(
    *,
    spark,
    source_model: str,
    target_model: str,
    model_version: int | None,
    model_alias: str | None,
    analysis_name: str,
    target_schema: str,
    feature_rebind_spec,
    dry_run: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_model": source_model,
        "target_model": target_model,
        "copied": False,
        "rebound_feature_lineage": True,
        "reason": None,
    }

    source_info = _resolve_source_model_uri(
        source_model=source_model,
        model_version=model_version,
        model_alias=model_alias,
    )
    result["source_uri"] = source_info["source_uri"]
    result["source_version"] = int(source_info["resolved_version"])

    if dry_run:
        result["copied"] = True
        result["reason"] = "dry_run"
        result["target_feature_table"] = snapshot_table_name(
            f"{target_schema}.{feature_rebind_spec.feature_table_template.format(analysis=analysis_name)}"
        )
        return result

    client = source_info["client"]
    try:
        client.get_registered_model(target_model)
    except Exception:
        client.create_registered_model(target_model)

    target_feature_base = f"{target_schema}.{feature_rebind_spec.feature_table_template.format(analysis=analysis_name)}"
    target_primary_table = snapshot_table_name(target_feature_base)
    target_entity_source = (
        current_table_name(target_feature_base)
        if bool(feature_rebind_spec.use_current_as_entity_source)
        else target_primary_table
    )
    if not spark.catalog.tableExists(target_primary_table):
        raise ValueError(
            f"Feature-rebind deploy failed: target feature table not found: '{target_primary_table}'. "
            "Run feature engineering in target environment first."
        )
    if not spark.catalog.tableExists(target_entity_source):
        raise ValueError(
            f"Feature-rebind deploy failed: entity source table not found: '{target_entity_source}'. "
            "Run feature engineering in target environment first."
        )

    source_entity_df = spark.table(target_entity_source)
    source_entity_cols = set(source_entity_df.columns)

    entity_cols = list(feature_rebind_spec.entity_keys)
    dedup_cols = list(feature_rebind_spec.entity_keys)
    if feature_rebind_spec.entity_timestamp_col and feature_rebind_spec.entity_timestamp_col not in entity_cols:
        entity_cols.append(feature_rebind_spec.entity_timestamp_col)
        dedup_cols.append(feature_rebind_spec.entity_timestamp_col)

    # Label is optional during rebind: include it only when available.
    label_for_rebind = None
    if feature_rebind_spec.label_col and feature_rebind_spec.label_col in source_entity_cols:
        label_for_rebind = feature_rebind_spec.label_col
        if label_for_rebind not in entity_cols:
            entity_cols.append(label_for_rebind)

    entity_df = (
        source_entity_df
        .select(*entity_cols)
        .dropDuplicates(dedup_cols)
    )

    primary_feature_names = _feature_rebind_primary_feature_names(
        spark=spark,
        table_name=target_primary_table,
        spec=feature_rebind_spec,
    )
    if not primary_feature_names:
        raise ValueError(
            f"No feature columns were inferred for target feature table '{target_primary_table}'."
        )

    fe_client = get_feature_engineering_client()
    training_set = create_feature_training_set(
        fe_client=fe_client,
        entity_df=entity_df,
        primary_table=target_primary_table,
        primary_lookup_key=_as_lookup_key(tuple(feature_rebind_spec.entity_keys)),
        primary_feature_names=primary_feature_names,
        secondary_lookups=[],
        label=label_for_rebind,
        exclude_columns=[],
        timestamp_lookup_key=feature_rebind_spec.entity_timestamp_col,
    )

    mlflow_flavor, source_model_obj, resolved_flavor_name = _load_model_for_feature_rebind(
        source_uri=source_info["source_uri"],
        preferred_flavor=feature_rebind_spec.flavor,
    )

    import mlflow
    source_pyfunc = mlflow.pyfunc.load_model(source_info["source_uri"])
    source_signature = getattr(getattr(source_pyfunc, "metadata", None), "signature", None)

    with mlflow.start_run(run_name=f"deploy_rebind_{analysis_name}_{target_model.split('.')[-1]}"):
        mlflow.log_params(
            {
                "deploy_mode": "relog_with_feature_rebinding",
                "source_model": source_model,
                "target_model": target_model,
                "source_model_uri": source_info["source_uri"],
                "target_primary_feature_table": target_primary_table,
                "target_entity_source_table": target_entity_source,
                "feature_rebind_flavor": feature_rebind_spec.flavor,
                "feature_rebind_resolved_flavor": resolved_flavor_name,
                "feature_count": len(primary_feature_names),
                "feature_rebind_label": str(label_for_rebind),
            }
        )

        model_info = log_model_with_feature_lineage(
            fe_client=fe_client,
            model=source_model_obj,
            training_set=training_set,
            registered_model_name=target_model,
            flavor=mlflow_flavor,
            artifact_path="model",
            signature=source_signature,
            input_example=None,
            extra_kwargs={},
        )
        run_id = mlflow.active_run().info.run_id

    target_version = _resolve_model_version(target_model, run_id=run_id)
    result["copied"] = True
    result["target_version"] = int(target_version)
    result["target_feature_table"] = target_primary_table
    result["target_entity_source_table"] = target_entity_source
    result["model_info"] = str(model_info)

    if model_alias and result["target_version"] is not None:
        client.set_registered_model_alias(target_model, model_alias, int(result["target_version"]))
        result["alias_set"] = model_alias

    return result


# COMMAND ----------

def _deploy_model(
    source_model: str,
    target_model: str,
    model_version: int | None,
    model_alias: str | None,
    dry_run: bool,
    spark=None,
    analysis_name: str | None = None,
    target_schema: str | None = None,
    feature_rebind_spec=None,
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

    if feature_rebind_spec is not None:
        if spark is None or analysis_name is None or target_schema is None:
            raise ValueError(
                "Feature-rebind deploy requires spark, analysis_name and target_schema."
            )
        return _relog_model_with_feature_rebinding(
            spark=spark,
            source_model=source_model,
            target_model=target_model,
            model_version=model_version,
            model_alias=model_alias,
            analysis_name=analysis_name,
            target_schema=target_schema,
            feature_rebind_spec=feature_rebind_spec,
            dry_run=dry_run,
        )

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


# COMMAND ----------

def deploy(
    subject: str,
    analysis_name: str,
    *,
    spark,
    source_catalog: str = "ml_train",
    target_catalog: str = "ml_prod",
    dry_run: bool = True,
    clone_mode: str = "DEEP",
    model_version: int | None = None,
    model_alias: str | None = "champion",
    deploy_tables: bool = False,
    deploy_reports: bool = False,
    deploy_model: bool = True,
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
    deploy_tables/deploy_reports/deploy_model:
        Select which assets to promote. Default promotes model only.
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
        "selected_assets": {
            "tables": bool(deploy_tables),
            "reports": bool(deploy_reports),
            "model": bool(deploy_model),
        },
        "tables": [],
        "reports": [],
        "model": None,
    }

    if (deploy_tables or deploy_reports or deploy_model) and (not dry_run):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")

    # Tables
    if deploy_tables:
        for template in spec.table_templates:
            base_name = f"{template.format(analysis=analysis_name)}"
            src_base = f"{source_schema}.{base_name}"
            dst_base = f"{target_schema}.{base_name}"

            src_snapshot = snapshot_table_name(src_base)
            dst_snapshot = snapshot_table_name(dst_base)
            src_current = current_table_name(src_base)
            dst_current = current_table_name(dst_base)

            # 1) Snapshot table clone
            snapshot_exists = bool(spark.catalog.tableExists(src_snapshot))
            snapshot_item = {
                "source": src_snapshot,
                "target": dst_snapshot,
                "exists": snapshot_exists,
                "executed": False,
                "kind": "snapshot",
            }

            if snapshot_exists and not dry_run:
                spark.sql(f"CREATE OR REPLACE TABLE {dst_snapshot} {clone_mode} CLONE {src_snapshot}")
                snapshot_item["executed"] = True

            plan["tables"].append(snapshot_item)

            # 2) Current object recreation (as view on top of target snapshot)
            current_exists = bool(spark.catalog.tableExists(src_current))
            current_item = {
                "source": src_current,
                "target": dst_current,
                "exists": current_exists and snapshot_exists,
                "executed": False,
                "kind": "current",
            }

            if current_exists and snapshot_exists and not dry_run:
                run_id_value = None
                try:
                    rows = spark.table(src_current).select("run_id").distinct().limit(1).collect()
                    if rows:
                        run_id_value = rows[0]["run_id"]
                except Exception:
                    run_id_value = None

                if run_id_value:
                    safe_run_id = str(run_id_value).replace("'", "''")
                    spark.sql(
                        f"""
                        CREATE OR REPLACE VIEW {dst_current} AS
                        SELECT *
                        FROM {dst_snapshot}
                        WHERE run_id = '{safe_run_id}'
                        """
                    )
                else:
                    # Fallback: expose latest snapshot state when source current has no run_id.
                    spark.sql(
                        f"""
                        CREATE OR REPLACE VIEW {dst_current} AS
                        SELECT *
                        FROM {dst_snapshot}
                        """
                    )
                current_item["executed"] = True

            plan["tables"].append(current_item)

    # Reports
    if deploy_reports:
        if not dry_run:
            spark.sql(f"CREATE VOLUME IF NOT EXISTS {target_schema}.{spec.report_folder}")

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
    if deploy_model and spec.model_template:
        model_name = spec.model_template.format(analysis=analysis_name)
        source_model = f"{source_schema}.{model_name}"
        target_model = f"{target_schema}.{model_name}"
        # New architecture: do not copy/relog models across catalogs.
        # Promotion (challenger -> champion) is handled on source catalog by evaluator.
        if source_catalog != target_catalog:
            plan["model"] = {
                "mode": "alias_promotion_only",
                "source_model": source_model,
                "target_model": target_model,
                "smoke_model": source_model,
                "copied": False,
                "executed": False,
                "reason": "cross-catalog model deployment disabled by architecture",
            }
        else:
            plan["model"] = _deploy_model(
                source_model=source_model,
                target_model=target_model,
                model_version=model_version,
                model_alias=model_alias,
                dry_run=dry_run,
                spark=spark,
                analysis_name=analysis_name,
                target_schema=target_schema,
                feature_rebind_spec=spec.feature_rebind,
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
