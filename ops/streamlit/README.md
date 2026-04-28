# ML Reports App

Unified Databricks App for report browsing in two environments:

- `ml_train` (training)
- `ml_prod` (production)

## Directory structure

- `app.py`: main catalog UI (single-page)
- `run_app.py`: Streamlit launcher using `DATABRICKS_APP_PORT`
- `app.yaml`: Databricks Apps startup config
- `core/`: core discovery and Databricks query modules
- `reports/`: interactive Streamlit reports rendered inside catalog

## What the app serves

Core files discovered automatically from Unity Catalog volumes:

- `/Volumes/ml_train/<schema>/reports/**`
- `/Volumes/ml_prod/<schema>/reports/**`

Interactive reports discovered automatically from local modules:

- `ops/streamlit/reports/*.py` (except `__init__.py`)

## Environment variables

- `REPORTS_CATALOGS` (default: `ml_train,ml_prod`)
- `REPORTS_VOLUME_NAME` (default: `reports`)

## Required privileges for app service principal

Grant on both catalogs:

- `USE CATALOG`
- `USE SCHEMA`
- `READ VOLUME`

Example:

```sql
GRANT USE CATALOG ON CATALOG ml_train TO `<app-service-principal-id>`;
GRANT USE CATALOG ON CATALOG ml_prod TO `<app-service-principal-id>`;
GRANT USE SCHEMA ON CATALOG ml_train TO `<app-service-principal-id>`;
GRANT USE SCHEMA ON CATALOG ml_prod TO `<app-service-principal-id>`;
GRANT READ VOLUME ON CATALOG ml_train TO `<app-service-principal-id>`;
GRANT READ VOLUME ON CATALOG ml_prod TO `<app-service-principal-id>`;
```

## Deploy

1. Sync source code:
   - `databricks sync --watch . /Workspace/Users/<user>/falaj-machine-learning/ops/streamlit`
2. Deploy app:
   - `databricks apps deploy ml-reports-catalog --source-code-path /Workspace/Users/<user>/falaj-machine-learning/ops/streamlit`
