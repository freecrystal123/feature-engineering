from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any


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

    return plan
