# Databricks notebook source
# MAGIC %md
# MAGIC ## UC Feature Store Bridge
# MAGIC
# MAGIC Reusable bridge between ML notebooks and Unity Catalog Feature Engineering.

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

from typing import Any


# COMMAND ----------

def build_entity_df_from_table(
    *,
    spark,
    table_name: str,
    entity_keys: str | list[str] | tuple[str, ...],
    entity_timestamp_col: str | None = None,
    entity_extra_cols: str | list[str] | tuple[str, ...] | None = None,
):
    """
    Build the base entity dataframe used by create_training_set.
    """
    keys = _normalize_cols(entity_keys)
    if not keys:
        raise ValueError("entity_keys cannot be empty")

    select_cols = list(keys)
    if entity_timestamp_col is not None and entity_timestamp_col not in select_cols:
        select_cols.append(entity_timestamp_col)
    for c in _normalize_cols(entity_extra_cols):
        if c not in select_cols:
            select_cols.append(c)

    base_df = spark.table(table_name).select(*select_cols)
    return base_df.dropDuplicates(select_cols)


# COMMAND ----------

def create_training_set_from_feature_tables(
    *,
    spark,
    primary_table: str,
    entity_source_table: str | None = None,
    entity_keys: str | list[str] | tuple[str, ...],
    primary_feature_names: list[str] | None = None,
    secondary_lookups: list[dict[str, Any]] | None = None,
    label: str | None = None,
    exclude_columns: list[str] | None = None,
    entity_timestamp_col: str | None = None,
    entity_extra_cols: str | list[str] | tuple[str, ...] | None = None,
):
    """
    Standardized UC Feature Engineering training_set creation.
    """
    fe = get_feature_engineering_client()

    merged_entity_extra_cols = _normalize_cols(entity_extra_cols)
    if label is not None and label not in merged_entity_extra_cols:
        merged_entity_extra_cols.append(label)

    source_table = entity_source_table or primary_table
    entity_df = build_entity_df_from_table(
        spark=spark,
        table_name=source_table,
        entity_keys=entity_keys,
        entity_timestamp_col=entity_timestamp_col,
        entity_extra_cols=merged_entity_extra_cols,
    )

    training_set = create_feature_training_set(
        fe_client=fe,
        entity_df=entity_df,
        primary_table=primary_table,
        primary_lookup_key=entity_keys,
        primary_feature_names=primary_feature_names,
        secondary_lookups=secondary_lookups,
        label=label,
        exclude_columns=exclude_columns or [],
        timestamp_lookup_key=entity_timestamp_col,
    )

    return {
        "fe_client": fe,
        "entity_df": entity_df,
        "training_set": training_set,
    }


# COMMAND ----------

def register_model_with_uc_feature_lineage(
    *,
    model,
    training_set,
    registered_model_name: str,
    flavor,
    artifact_path: str = "model",
    signature=None,
    input_example=None,
    extra_kwargs: dict[str, Any] | None = None,
):
    """
    Canonical model registration path with UC feature lineage.
    """
    fe = get_feature_engineering_client()
    return log_model_with_feature_lineage(
        fe_client=fe,
        model=model,
        training_set=training_set,
        registered_model_name=registered_model_name,
        flavor=flavor,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
        extra_kwargs=extra_kwargs,
    )


# COMMAND ----------

def create_training_set_and_register_model(
    *,
    spark,
    model,
    registered_model_name: str,
    flavor,
    primary_table: str,
    entity_source_table: str | None = None,
    entity_keys: str | list[str] | tuple[str, ...],
    primary_feature_names: list[str] | None = None,
    secondary_lookups: list[dict[str, Any]] | None = None,
    label: str | None = None,
    exclude_columns: list[str] | None = None,
    entity_timestamp_col: str | None = None,
    entity_extra_cols: str | list[str] | tuple[str, ...] | None = None,
    artifact_path: str = "model",
    signature=None,
    input_example=None,
    extra_kwargs: dict[str, Any] | None = None,
):
    """
    One-shot API: create_training_set + log_model (with feature lineage).
    """
    ts_result = create_training_set_from_feature_tables(
        spark=spark,
        primary_table=primary_table,
        entity_source_table=entity_source_table,
        entity_keys=entity_keys,
        primary_feature_names=primary_feature_names,
        secondary_lookups=secondary_lookups,
        label=label,
        exclude_columns=exclude_columns,
        entity_timestamp_col=entity_timestamp_col,
        entity_extra_cols=entity_extra_cols,
    )

    model_info = register_model_with_uc_feature_lineage(
        model=model,
        training_set=ts_result["training_set"],
        registered_model_name=registered_model_name,
        flavor=flavor,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
        extra_kwargs=extra_kwargs,
    )

    return {
        "entity_df": ts_result["entity_df"],
        "training_set": ts_result["training_set"],
        "model_info": model_info,
    }
