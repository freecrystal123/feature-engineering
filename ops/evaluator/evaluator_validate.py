# Databricks notebook source
# COMMAND ----------
# MAGIC %run ../../models/utils

# COMMAND ----------
dbutils.widgets.text("subject", "user_segmentation")
dbutils.widgets.text("analysis_name", "analysis_1")
dbutils.widgets.text("model_catalog", "uat_analytics")
dbutils.widgets.text("candidate_alias", "challenger")
dbutils.widgets.text("required_signature", "true")

# COMMAND ----------
import json


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


subject = dbutils.widgets.get("subject").strip()
analysis_name = dbutils.widgets.get("analysis_name").strip()
model_catalog = dbutils.widgets.get("model_catalog").strip()
candidate_alias = dbutils.widgets.get("candidate_alias").strip()
required_signature = _to_bool(dbutils.widgets.get("required_signature"), default=True)

model_name = build_model_name(
    subject=subject,
    analysis_name=analysis_name,
    catalog=model_catalog,
)

validation = validate_candidate(
    model_name=model_name,
    alias=candidate_alias,
    required_signature=required_signature,
)

output = {
    "subject": subject,
    "analysis_name": analysis_name,
    "model_catalog": model_catalog,
    "model_name": model_name,
    "candidate_alias": candidate_alias,
    "validation": validation,
}

print(json.dumps(output, indent=2, default=str))

dbutils.jobs.taskValues.set(key="model_name", value=model_name)
dbutils.jobs.taskValues.set(key="model_version", value=str(validation["version"]))
dbutils.jobs.taskValues.set(key="validation_passed", value=str(bool(validation["passed"])).lower())

if not validation["passed"]:
    raise ValueError(
        f"Validation failed for model '{model_name}' (alias={candidate_alias}, version={validation['version']})."
    )
