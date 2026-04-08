# Databricks notebook source
# COMMAND ----------
dbutils.widgets.text("subject", "user_segmentation")
dbutils.widgets.text("analysis_name", "analysis_1")
dbutils.widgets.text("source_catalog", "uat_analytics")
dbutils.widgets.text("target_catalog", "prod_analytics")
dbutils.widgets.text("source_alias", "challenger")
dbutils.widgets.text("target_alias", "champion")
dbutils.widgets.text("promote_first", "true")
dbutils.widgets.text("dry_run", "false")
dbutils.widgets.text("deploy_model", "true")
dbutils.widgets.text("deploy_tables", "false")
dbutils.widgets.text("deploy_reports", "false")
dbutils.widgets.text("run_smoke_tests", "true")

# COMMAND ----------
import json

from models.utils import build_model_name, deploy, promote_alias


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


subject = dbutils.widgets.get("subject").strip()
analysis_name = dbutils.widgets.get("analysis_name").strip()
source_catalog = dbutils.widgets.get("source_catalog").strip()
target_catalog = dbutils.widgets.get("target_catalog").strip()
source_alias = dbutils.widgets.get("source_alias").strip()
target_alias = dbutils.widgets.get("target_alias").strip()
promote_first = _to_bool(dbutils.widgets.get("promote_first"), default=True)
dry_run = _to_bool(dbutils.widgets.get("dry_run"), default=False)
deploy_model = _to_bool(dbutils.widgets.get("deploy_model"), default=True)
deploy_tables = _to_bool(dbutils.widgets.get("deploy_tables"), default=False)
deploy_reports = _to_bool(dbutils.widgets.get("deploy_reports"), default=False)
run_smoke_tests = _to_bool(dbutils.widgets.get("run_smoke_tests"), default=True)

source_model_name = build_model_name(
    subject=subject,
    analysis_name=analysis_name,
    catalog=source_catalog,
)

promotion = {
    "skipped": True,
    "reason": "promote_first=false",
}
if promote_first:
    if dry_run:
        promotion = {
            "skipped": True,
            "reason": "dry_run=true (promotion skipped)",
        }
    else:
        promotion = promote_alias(
            model_name=source_model_name,
            source_alias=source_alias,
            target_alias=target_alias,
        )

plan = deploy(
    subject=subject,
    analysis_name=analysis_name,
    spark=spark,
    source_catalog=source_catalog,
    target_catalog=target_catalog,
    dry_run=dry_run,
    model_alias=target_alias,
    deploy_model=deploy_model,
    deploy_tables=deploy_tables,
    deploy_reports=deploy_reports,
    run_smoke_tests=run_smoke_tests,
)

output = {
    "subject": subject,
    "analysis_name": analysis_name,
    "source_model_name": source_model_name,
    "promotion": promotion,
    "deploy_plan": plan,
}
print(json.dumps(output, indent=2, default=str))

smoke = plan.get("smoke_tests")
if smoke and (not smoke.get("passed", False)):
    raise ValueError(f"Smoke tests failed: {json.dumps(smoke, default=str)}")
