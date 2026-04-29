from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import pandas as pd
from databricks import sql
from databricks.sdk.core import Config

WAREHOUSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


def resolve_sql_http_path(explicit_http_path: str | None = None) -> str:
    raw = str(explicit_http_path or "").strip()
    if raw:
        if raw.startswith("/"):
            return raw
        if raw.startswith("sql/"):
            return f"/{raw}"
        if WAREHOUSE_ID_PATTERN.fullmatch(raw):
            return f"/sql/1.0/warehouses/{raw}"
        return raw

    for env_name in ("DATABRICKS_SQL_HTTP_PATH", "DATABRICKS_HTTP_PATH"):
        value = str(os.getenv(env_name, "")).strip()
        if value:
            return resolve_sql_http_path(value)

    for env_name in ("DATABRICKS_SQL_WAREHOUSE_ID", "WAREHOUSE_ID"):
        warehouse_id = str(os.getenv(env_name, "")).strip()
        if warehouse_id:
            return resolve_sql_http_path(warehouse_id)

    return ""


def _resolve_server_hostname(config_host: str) -> str:
    raw_host = str(config_host).strip()
    if "://" not in raw_host:
        raw_host = f"https://{raw_host}"
    parsed = urlparse(raw_host)
    host = (parsed.netloc or parsed.path).strip().strip("/")
    if "/" in host:
        host = host.split("/", 1)[0]
    if not host:
        raise ValueError("Unable to resolve Databricks server hostname from Config.host.")
    return host


class DatabricksDataSelector:
    def __init__(self, http_path: str | None = None):
        self.http_path = resolve_sql_http_path(http_path)
        if not self.http_path:
            raise ValueError(
                "Missing SQL warehouse HTTP path. Set DATABRICKS_SQL_HTTP_PATH (or DATABRICKS_HTTP_PATH)."
            )

        cfg = Config()
        self.server_hostname = _resolve_server_hostname(str(cfg.host))
        self._credentials_provider = lambda: cfg.authenticate

    def data_select(self, query: str) -> pd.DataFrame:
        statement = str(query or "").strip()
        if not statement:
            raise ValueError("Query is empty.")

        with sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            credentials_provider=self._credentials_provider,
            _use_arrow_native_complex_types=False,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(statement)
                return cursor.fetchall_arrow().to_pandas()


def data_select(query: str, http_path: str | None = None) -> pd.DataFrame:
    return DatabricksDataSelector(http_path=http_path).data_select(query)
