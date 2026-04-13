# Databricks notebook source
# COMMAND ----------

from dataclasses import dataclass


# COMMAND ----------

@dataclass(frozen=True)
class SubjectSpec:
    schema: str
    report_folder: str
    table_templates: tuple[str, ...]
    model_template: str | None = None
    report_templates: tuple[str, ...] = ()


# COMMAND ----------

SUBJECT_SPECS: dict[str, SubjectSpec] = {
    "user_segmentation": SubjectSpec(
        schema="user_segmentation",
        report_folder="reports",
        table_templates=(
            "feat_{analysis}",
            "pca_{analysis}",
            "pred_{analysis}",
            "cluster_{analysis}",
        ),
        model_template="mdl_{analysis}",
        report_templates=(
            "general_information_{analysis}.html",
            "detailed_information_{analysis}.html",
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
