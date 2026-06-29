from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import streamlit as st


def parse_catalogs(raw: str) -> list[str]:
    parsed = [c.strip() for c in str(raw).split(",") if c.strip()]
    if not parsed:
        parsed = ["ml_train", "ml_prod"]
    return list(dict.fromkeys(parsed))


def environment_for_catalog(catalog_name: str) -> str:
    name = str(catalog_name).strip().lower()
    if name == "ml_train" or name.endswith("_train"):
        return "training"
    if name == "ml_prod" or name.endswith("_prod"):
        return "production"
    return "other"


@st.cache_resource(show_spinner="Loading...")
def workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _schema_short_name(schema_name: str) -> str:
    return schema_name.split(".")[-1]


def _extract_schema_from_volume_path(volume_path: str) -> str:
    parts = PurePosixPath(volume_path).parts
    if len(parts) >= 5 and parts[1] == "Volumes":
        return parts[3]
    tokens = str(volume_path).strip("/").split("/")
    if len(tokens) >= 4 and tokens[0] == "Volumes":
        return tokens[2]
    return "unknown_schema"


def _format_last_modified(raw_value) -> str:
    if raw_value is None:
        return "-"
    if isinstance(raw_value, (int, float)):
        epoch = float(raw_value)
        if epoch > 10_000_000_000:
            epoch /= 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(raw_value)


def _discover_report_volume_paths(
    catalog_name: str,
    reports_volume_name: str,
) -> tuple[list[str], list[str], int]:
    warnings: list[str] = []
    paths: list[str] = []
    schemas_scanned = 0

    try:
        client = workspace_client()
    except Exception as exc:
        warnings.append(f"Failed to initialize Databricks SDK client: {exc}")
        return paths, warnings, schemas_scanned

    try:
        schemas = list(client.schemas.list(catalog_name=catalog_name))
    except Exception as exc:
        warnings.append(
            f"Failed to list schemas in catalog '{catalog_name}'. "
            f"Check app authorization and UC permissions. Error: {exc}"
        )
        return paths, warnings, schemas_scanned

    for schema in schemas:
        schema_name = _schema_short_name(str(getattr(schema, "name", "")))
        if (not schema_name) or (schema_name.lower() == "information_schema"):
            continue
        schemas_scanned += 1

        try:
            volumes = list(client.volumes.list(catalog_name=catalog_name, schema_name=schema_name))
        except Exception as exc:
            warnings.append(f"Failed to list volumes for {catalog_name}.{schema_name}: {exc}")
            continue

        for volume in volumes:
            full_name = str(getattr(volume, "full_name", "") or "")
            short_name = str(getattr(volume, "name", "") or "")
            if (not short_name) and full_name:
                short_name = full_name.split(".")[-1]
            if short_name != reports_volume_name:
                continue

            volume_path = str(getattr(volume, "volume_path", "") or "").strip()
            if not volume_path:
                volume_path = f"/Volumes/{catalog_name}/{schema_name}/{reports_volume_name}"
            paths.append(volume_path.rstrip("/"))

    unique_paths = sorted(set(paths))
    return unique_paths, warnings, schemas_scanned


def _list_files_recursive(directory_path: str) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    files: list[dict] = []

    try:
        client = workspace_client()
    except Exception as exc:
        warnings.append(f"Failed to initialize Databricks SDK client: {exc}")
        return files, warnings

    stack = [directory_path.rstrip("/")]
    visited_dirs: set[str] = set()

    while stack:
        current_dir = stack.pop()
        if current_dir in visited_dirs:
            continue
        visited_dirs.add(current_dir)

        try:
            entries = list(client.files.list_directory_contents(current_dir))
        except Exception as exc:
            warnings.append(f"Failed to list directory {current_dir}: {exc}")
            continue

        for entry in entries:
            raw_path = str(getattr(entry, "path", "") or "").strip()
            if not raw_path:
                continue
            normalized_path = raw_path.rstrip("/")

            is_dir = getattr(entry, "is_directory", None)
            if is_dir is None:
                is_dir = raw_path.endswith("/")
            if is_dir is None:
                try:
                    client.files.get_directory_metadata(normalized_path)
                    is_dir = True
                except Exception:
                    is_dir = False

            if bool(is_dir):
                stack.append(normalized_path)
                continue

            files.append(
                {
                    "file_path": normalized_path,
                    "size_bytes": int(getattr(entry, "file_size", 0) or 0),
                    "updated_at": _format_last_modified(getattr(entry, "last_modified", None)),
                }
            )

    return files, warnings


@st.cache_data(ttl=60, show_spinner="Loading...")
def discover_core_reports(catalog_names: tuple[str, ...], reports_volume_name: str) -> tuple[list[dict], list[str], dict]:
    rows: list[dict] = []
    warnings: list[str] = []
    by_catalog: dict[str, dict] = {}

    for catalog_name in catalog_names:
        report_roots, discover_warnings, schemas_scanned = _discover_report_volume_paths(
            catalog_name=catalog_name,
            reports_volume_name=reports_volume_name,
        )
        warnings.extend(discover_warnings)

        by_catalog[catalog_name] = {
            "environment": environment_for_catalog(catalog_name),
            "schemas_scanned": schemas_scanned,
            "report_volumes_found": len(report_roots),
            "files_found": 0,
        }

        for reports_dir in report_roots:
            schema_name = _extract_schema_from_volume_path(reports_dir)
            file_rows, file_warnings = _list_files_recursive(reports_dir)
            warnings.extend(file_warnings)

            root_prefix = reports_dir.rstrip("/") + "/"
            for file_row in file_rows:
                file_path = str(file_row["file_path"])
                relative_path = (
                    file_path[len(root_prefix):]
                    if file_path.startswith(root_prefix)
                    else PurePosixPath(file_path).name
                )
                rows.append(
                    {
                        "source_type": "core_file",
                        "environment": environment_for_catalog(catalog_name),
                        "catalog": catalog_name,
                        "schema": schema_name,
                        "relative_path": relative_path,
                        "report_title": Path(relative_path).name,
                        "extension": Path(file_path).suffix.lower() or "(none)",
                        "size_bytes": int(file_row["size_bytes"]),
                        "updated_at": str(file_row["updated_at"]),
                        "file_path": file_path,
                    }
                )
                by_catalog[catalog_name]["files_found"] += 1

    stats = {
        "catalogs_scanned": len(catalog_names),
        "report_volumes_found": sum(v["report_volumes_found"] for v in by_catalog.values()),
        "files_found": len(rows),
        "by_catalog": by_catalog,
    }
    return rows, warnings, stats


def download_file_bytes(file_path: str) -> bytes:
    client = workspace_client()
    response = client.files.download(file_path)
    with response.contents as stream:
        return stream.read()


def read_text_preview(file_path: str, limit_bytes: int) -> tuple[str, bool]:
    client = workspace_client()
    response = client.files.download(file_path)
    with response.contents as stream:
        raw = stream.read(limit_bytes + 1)
    truncated = len(raw) > limit_bytes
    raw = raw[:limit_bytes]
    return raw.decode("utf-8", errors="replace"), truncated
