# Databricks notebook source
# COMMAND ----------

from dataclasses import dataclass


# COMMAND ----------

@dataclass(frozen=True)
class FeatureRebindSpec:
    feature_table_template: str
    entity_keys: tuple[str, ...]
    entity_timestamp_col: str | None = None
    label_col: str | None = None
    flavor: str = "sklearn"
    use_current_as_entity_source: bool = True
    metadata_columns: tuple[str, ...] = (
        "run_id",
        "run_ts",
        "ref_date",
        "analysis_name",
        "created_at",
    )
    feature_names: tuple[str, ...] | None = None


# COMMAND ----------

@dataclass(frozen=True)
class SubjectSpec:
    schema: str
    report_folder: str
    table_templates: tuple[str, ...]
    model_template: str | None = None
    report_templates: tuple[str, ...] = ()
    feature_rebind: FeatureRebindSpec | None = None


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
        model_template="mdl_{analysis}",
        report_templates=(
            "mba_rules_report_{analysis}.html",
        ),
    ),
    "reference_ml_template": SubjectSpec(
        schema="reference_ml_template",
        report_folder="reports",
        table_templates=(
            "feat_{analysis}",
            "pred_{analysis}",
        ),
        model_template="mdl_{analysis}",
        report_templates=(
            "reference_report_{analysis}.html",
        ),
        feature_rebind=FeatureRebindSpec(
            feature_table_template="feat_{analysis}",
            entity_keys=("unique_user_id",),
            entity_timestamp_col="run_ts",
            label_col="label_active_7d",
            flavor="sklearn",
            use_current_as_entity_source=True,
        ),
    ),
    "conversion_funnel_deposit": SubjectSpec(
        schema="conversion_funnel_deposit",
        report_folder="reports",
        table_templates=(
            "feat_{analysis}",
            "pred_{analysis}",
            "eval_{analysis}",
            "shap_{analysis}",
            "rpt_funnel_{analysis}",
            "rpt_shap_{analysis}",
            "rpt_eval_{analysis}",
        ),
        model_template="mdl_{analysis}",
        report_templates=(),
        feature_rebind=FeatureRebindSpec(
            feature_table_template="feat_{analysis}",
            entity_keys=("unique_user_id",),
            entity_timestamp_col="run_ts",
            label_col="label_deposit_success_30d",
            flavor="sklearn",
            use_current_as_entity_source=True,
        ),
    ),
}
