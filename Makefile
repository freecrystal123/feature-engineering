APP_DIR := streamlit
WORKSPACE_PATH := /Workspace/feature-engineering/prod
APP_NAME := machine-learning-reports
APP_WS_PATH := $(WORKSPACE_PATH)/files/$(APP_DIR)


.PHONY: sync sync-watch deploy-app sync-and-deploy

sync:
	cd $(APP_DIR) && databricks sync . $(APP_WS_PATH)

sync-watch:
	cd $(APP_DIR) && databricks sync --watch . $(APP_WS_PATH)

deploy-app:
	cd $(APP_DIR) && databricks apps deploy $(APP_NAME) --source-code-path $(APP_WS_PATH)

sync-and-deploy-app:
	$(MAKE) sync
	$(MAKE) deploy-app

