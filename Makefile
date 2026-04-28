APP_DIR := ops/reports
WORKSPACE_PATH := /Workspace/Users/aneto@mcorp.ae/test-catalog
APP_NAME := machine-learning-reports

.PHONY: sync sync-watch deploy deploy-app

sync:
	cd $(APP_DIR) && databricks sync . $(WORKSPACE_PATH)

sync-watch:
	cd $(APP_DIR) && databricks sync --watch . $(WORKSPACE_PATH)

deploy:
	cd $(APP_DIR) && databricks apps deploy $(APP_NAME) --source-code-path $(WORKSPACE_PATH)

deploy-app:
	$(MAKE) sync
	$(MAKE) deploy
