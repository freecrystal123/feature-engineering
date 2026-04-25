# Databricks notebook source
# MAGIC %md
# MAGIC ## 2 - Reference model training (with UC feature lineage)

# COMMAND ----------

# MAGIC %run ./00_config_reference_ml_template

# COMMAND ----------
# MAGIC %run ../../core/utils

# COMMAND ----------
# MAGIC %run ../../core/uc_feature_store

# COMMAND ----------

dbutils.widgets.text("analysis_id", "all")
dbutils.widgets.text("candidate_alias", DEFAULT_CANDIDATE_ALIAS)

# COMMAND ----------

import json
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

analysis_id = dbutils.widgets.get("analysis_id").strip()
candidate_alias = dbutils.widgets.get("candidate_alias").strip()


def get_analyses_to_run():
    if analysis_id.lower() == "all":
        return list(ANALYSIS_NAMES.keys())
    if analysis_id not in ANALYSIS_NAMES:
        raise ValueError(f"Invalid analysis_id '{analysis_id}'. Valid: {list(ANALYSIS_NAMES)} or 'all'.")
    return [analysis_id]

def create_model_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    random_state=int(DEFAULT_RANDOM_SEED),
                    max_iter=300,
                ),
            ),
        ]
    )


def train_and_register(analysis: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    table_base = f"{T_REFERENCE_FEATURES_PREFIX}_{analysis_name}"
    primary_table = snapshot_table_name(table_base)
    entity_source_table = current_table_name(table_base)

    table_df = spark.table(primary_table)
    feature_cols = get_reference_feature_columns(table_df)
    if not feature_cols:
        raise ValueError(f"No numeric feature columns inferred from table '{primary_table}'.")

    ts_result = create_training_set_from_feature_tables(
        spark=spark,
        primary_table=primary_table,
        entity_source_table=entity_source_table,
        entity_keys="unique_user_id",
        primary_feature_names=feature_cols,
        secondary_lookups=[],
        label="label_active_7d",
        exclude_columns=["unique_user_id", "run_ts"],
        entity_timestamp_col="run_ts",
    )

    training_set = ts_result["training_set"]
    train_pdf = training_set.load_df().orderBy("unique_user_id").toPandas()

    X = prepare_reference_model_input(train_pdf, feature_cols)
    y = train_pdf["label_active_7d"].astype("int64")

    model = create_model_pipeline()
    model.fit(X, y)

    model_name = build_model_name(
        subject=SUBJECT_NAME,
        analysis_name=analysis_name,
        catalog=WRITE_CATALOG,
    )

    with mlflow.start_run(run_name=f"train_{analysis_name}"):
        mlflow.log_params(
            {
                "subject": SUBJECT_NAME,
                "analysis_id": analysis,
                "analysis_name": analysis_name,
                "write_catalog": WRITE_CATALOG,
                "n_rows": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "candidate_alias": candidate_alias,
            }
        )
        mlflow.log_text("\n".join(feature_cols), "features.txt")

        log_model_with_feature_lineage(
            fe_client=get_feature_engineering_client(),
            model=model,
            training_set=training_set,
            registered_model_name=model_name,
            flavor=mlflow.sklearn,
            artifact_path="model",
            signature=None,
            input_example=None,
            extra_kwargs={},
        )

        run_id = mlflow.active_run().info.run_id
        candidate = register_candidate(
            model_name=model_name,
            run_id=run_id,
            alias=candidate_alias,
            tags={
                "analysis": analysis_name,
                "governance_status": "candidate",
                "subject": SUBJECT_NAME,
            },
        )

        return {
            "analysis": analysis,
            "analysis_name": analysis_name,
            "model_name": model_name,
            "candidate_alias": candidate_alias,
            "candidate_version": candidate["version"],
            "auto_promoted": False,
            "champion_alias": DEFAULT_CHAMPION_ALIAS,
        }


results = [train_and_register(a) for a in get_analyses_to_run()]
print(json.dumps({"results": results}, indent=2, default=str))
