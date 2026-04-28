APP_DIR := ops/reports
WORKSPACE_PATH := /Workspace/Users/aneto@mcorp.ae/test-catalog
APP_NAME := cross-sell-analysis

.PHONY: sync-watch deploy

sync-watch:
	cd $(APP_DIR) && databricks sync --watch . $(WORKSPACE_PATH)

deploy:
	cd $(APP_DIR) && databricks apps deploy $(APP_NAME) --source-code-path $(WORKSPACE_PATH)

