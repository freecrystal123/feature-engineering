from __future__ import annotations

import base64
import importlib
import mimetypes
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core.catalog_service import (
    discover_core_reports,
    download_file_bytes,
    parse_catalogs,
    read_text_preview,
)

st.set_page_config(page_title="ML Reports Catalog", page_icon=":bar_chart:", layout="wide")

REPORTS_VOLUME_NAME = os.getenv("REPORTS_VOLUME_NAME", "reports")
CATALOGS_ENV = os.getenv("REPORTS_CATALOGS", "ml_train,ml_prod")
CATALOG_NAMES = parse_catalogs(CATALOGS_ENV)
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log"}
HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_TEXT_PREVIEW_BYTES = 300_000
NO_SELECTION_KEY = "__none__"
SIDEBAR_REPORT_KEY = "sidebar_report_radio"
SIDEBAR_ENV_KEY = "sidebar_reports_environment"
SIDEBAR_EMPTY_LABEL = "__select_report__"


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

          html, body, [class*="st-"], [class*="css"] {
            font-family: "IBM Plex Sans", system-ui, sans-serif;
          }

          [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(900px 460px at -15% -20%, rgba(255, 111, 76, 0.22), transparent 72%),
              radial-gradient(900px 460px at 120% -25%, rgba(255, 139, 63, 0.16), transparent 74%),
              linear-gradient(180deg, #0a111f 0%, #0c1526 52%, #0b1321 100%);
            color: #e7edf8;
          }

          [data-testid="stHeader"] {
            background: transparent;
          }

          section[data-testid="stSidebar"] {
            border-right: 1px solid #22334f;
            background: linear-gradient(180deg, #0b1425 0%, #0a1220 100%);
          }

          .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 2px 0 56px 0;
          }

          .sidebar-brand-icon {
            font-size: 38px;
            line-height: 1;
            filter: grayscale(1) saturate(0) contrast(1.2) brightness(1.35);
          }

          .sidebar-brand-text {
            font-family: "Space Grotesk", sans-serif;
            font-size: 30px;
            line-height: 1.05;
            font-weight: 700;
            color: #f3f7ff;
            letter-spacing: 0.2px;
          }

          .main .block-container {
            max-width: none;
            width: 100%;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
          }

          .hero {
            border: 1px solid #223652;
            background: linear-gradient(132deg, rgba(17, 28, 45, 0.95) 0%, rgba(18, 32, 52, 0.96) 65%, rgba(23, 36, 54, 0.96) 100%);
            border-radius: 18px;
            padding: 24px 26px;
            margin-bottom: 16px;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.34);
          }

          .hero h1 {
            margin: 0;
            font-size: 34px;
            line-height: 1.08;
            color: #f5f8ff;
            font-family: "Space Grotesk", sans-serif;
            font-weight: 700;
            letter-spacing: 0.1px;
          }

          .hero p {
            margin: 8px 0 0 0;
            color: #b8c7de;
            font-size: 14px;
            max-width: 900px;
          }

          .stDownloadButton > button, .stButton > button {
            border-radius: 10px;
            border: 1px solid #35557a;
            color: #ebf2ff;
            background: linear-gradient(130deg, #1b2c45 0%, #1b2f4b 100%);
          }

          .stDownloadButton > button:hover, .stButton > button:hover {
            border-color: #ff6f4c;
            color: #ffffff;
          }

          [data-testid="stSidebar"] .stRadio > label {
            font-weight: 600;
            color: #f1f6ff;
            margin-top: 8px;
          }

          [data-testid="stSidebar"] div[role="radiogroup"] > label {
            border: 1px solid #233957;
            background: #101a2b;
            border-radius: 10px;
            padding: 7px 10px;
            margin-bottom: 8px;
            transition: border-color 0.16s ease;
          }

          [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            border-color: #ff6f4c;
          }

          [data-testid="stSidebar"] div[role="radiogroup"] > label p {
            color: #d5e1f4;
            font-size: 13px;
          }

          [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            border-color: #ff6f4c;
            background: linear-gradient(130deg, rgba(255, 111, 76, 0.17) 0%, rgba(255, 143, 62, 0.12) 100%);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mime_for(file_path: str) -> str:
    detected, _ = mimetypes.guess_type(Path(file_path).name)
    return detected or "application/octet-stream"


def _render_core_file(file_path: str, extension: str) -> bool:
    if extension in HTML_EXTENSIONS:
        html_content = download_file_bytes(file_path).decode("utf-8", errors="replace")
        components.html(html_content, height=1150, scrolling=True)
        return True

    if extension == ".pdf":
        data = download_file_bytes(file_path)
        b64 = base64.b64encode(data).decode("utf-8")
        iframe = (
            "<iframe width='100%' height='1150' style='border:none' "
            f"src='data:application/pdf;base64,{b64}'></iframe>"
        )
        st.markdown(iframe, unsafe_allow_html=True)
        return True

    if extension in IMAGE_EXTENSIONS:
        st.image(download_file_bytes(file_path), use_container_width=True)
        return True

    if extension in TEXT_EXTENSIONS:
        preview, truncated = read_text_preview(file_path, limit_bytes=MAX_TEXT_PREVIEW_BYTES)
        st.code(preview, language="text")
        if truncated:
            st.caption(f"Preview truncated to {MAX_TEXT_PREVIEW_BYTES:,} bytes.")
        return True

    st.info("Inline preview is not available for this file type.")
    return False


def _humanize_file_title(file_name: str) -> str:
    stem = Path(file_name).stem
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(word.capitalize() for word in cleaned.split()) if cleaned else file_name


def _humanize_identifier(value: str) -> str:
    text = str(value).replace("_", " ").replace("-", " ").strip()
    return " ".join(word.capitalize() for word in text.split()) if text else "Unknown"


def _default_report_description(schema: str) -> str:
    return f"Operational report for {_humanize_identifier(schema)}."


def _environment_label(environment: str) -> str:
    mapping = {
        "training": "Training",
        "production": "Production",
        "other": "Other",
    }
    return mapping.get(str(environment).lower(), str(environment).capitalize())


def _discover_streamlit_rows() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    reports_dir = Path(__file__).resolve().parent / "reports"

    if not reports_dir.exists():
        return rows, [f"Streamlit reports directory not found: {reports_dir}"]

    for report_file in sorted(reports_dir.glob("*.py")):
        if report_file.name == "__init__.py" or report_file.name.startswith("_"):
            continue

        module_path = f"reports.{report_file.stem}"
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            warnings.append(f"Failed to import module {module_path}: {exc}")
            continue

        render_fn = getattr(module, "render", None)
        if not callable(render_fn):
            warnings.append(f"Module {module_path} ignored: missing render(context) function.")
            continue

        metadata = getattr(module, "REPORT_META", {})
        if not isinstance(metadata, dict):
            metadata = {}

        schema = str(metadata.get("schema") or report_file.stem).strip().lower()
        title = str(metadata.get("title") or _humanize_identifier(report_file.stem))
        description = str(metadata.get("description") or _default_report_description(schema))

        raw_environments = metadata.get("environments", ("training", "production"))
        if isinstance(raw_environments, str):
            environments = (raw_environments,)
        else:
            environments = tuple(raw_environments)
        normalized_environments = tuple(
            str(env).strip().lower() for env in environments if str(env).strip()
        ) or ("production",)

        for environment in normalized_environments:
            rows.append(
                {
                    "source_type": "streamlit",
                    "environment": environment,
                    "catalog": "internal",
                    "schema": schema,
                    "relative_path": report_file.stem,
                    "report_title": title,
                    "extension": ".py",
                    "size_bytes": 0,
                    "updated_at": "-",
                    "file_path": "",
                    "description": description,
                    "module_path": module_path,
                }
            )
    return rows, warnings


def _render_streamlit_module(module_path: str, context: dict) -> None:
    module = importlib.import_module(module_path)
    render_fn = getattr(module, "render", None)
    if not callable(render_fn):
        raise ValueError(f"Module '{module_path}' has no callable 'render(context)' function.")
    render_fn(context)


def _build_catalog_entries(core_rows: list[dict], streamlit_rows: list[dict]) -> list[dict]:
    entries: list[dict] = []

    for row in core_rows:
        file_path = str(row.get("file_path", ""))
        file_name = str(row.get("report_title", ""))
        if file_path.lower().endswith(".bkp") or file_name.lower().endswith(".bkp"):
            continue
        schema = str(row.get("schema", "")).strip().lower()
        environment = str(row.get("environment", "other")).strip().lower()
        title = _humanize_file_title(file_name)
        key = f"core::{environment}::{schema}::{file_path}"
        entries.append(
            {
                "key": key,
                "title": title,
                "description": _default_report_description(schema),
                "schema": schema,
                "environment": environment,
                "environment_label": _environment_label(environment),
                "source_type": "core_file",
                "extension": str(row.get("extension", "")).lower(),
                "file_path": file_path,
                "updated_at": str(row.get("updated_at", "-")),
                "catalog": str(row.get("catalog", "")),
            }
        )

    for row in streamlit_rows:
        module_path = str(row.get("module_path", ""))
        schema = str(row.get("schema", "")).strip().lower()
        environment = str(row.get("environment", "other")).strip().lower()
        title = str(row.get("report_title", "")).strip() or _humanize_identifier(schema)
        key = f"streamlit::{environment}::{schema}::{module_path}"
        entries.append(
            {
                "key": key,
                "title": title,
                "description": str(row.get("description", "")).strip() or _default_report_description(schema),
                "schema": schema,
                "environment": environment,
                "environment_label": _environment_label(environment),
                "source_type": "streamlit",
                "module_path": module_path,
            }
        )

    entries.sort(
        key=lambda item: (
            _humanize_identifier(item["schema"]).lower(),
            item["title"].lower(),
            item["environment"].lower(),
            item["source_type"].lower(),
        )
    )
    return entries


def _render_sidebar_menu(entries: list[dict]) -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <span class="sidebar-brand-icon">🧠</span>
          <span class="sidebar-brand-text">ML Viewer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    env_options = ["production", "training"]
    env_labels = {
        "production": "Production Reports",
        "training": "Training Reports",
    }
    if SIDEBAR_ENV_KEY not in st.session_state:
        st.session_state[SIDEBAR_ENV_KEY] = "production"
    if st.session_state[SIDEBAR_ENV_KEY] not in env_options:
        st.session_state[SIDEBAR_ENV_KEY] = "production"

    selected_environment = st.sidebar.selectbox(
        "Reports",
        options=env_options,
        index=env_options.index(str(st.session_state[SIDEBAR_ENV_KEY])),
        format_func=lambda env: env_labels[str(env)],
        key=SIDEBAR_ENV_KEY,
    )
    filtered_entries = [e for e in entries if str(e.get("environment", "")).lower() == selected_environment]

    options = [item["key"] for item in filtered_entries]
    labels = {
        item["key"]: (
            item["title"]
        )
        for item in filtered_entries
    }

    if not options:
        st.sidebar.info("No reports available for this environment.")
        st.session_state[SIDEBAR_REPORT_KEY] = SIDEBAR_EMPTY_LABEL
        return NO_SELECTION_KEY

    sidebar_options = [SIDEBAR_EMPTY_LABEL] + options
    labels[SIDEBAR_EMPTY_LABEL] = "Home"

    if SIDEBAR_REPORT_KEY not in st.session_state:
        st.session_state[SIDEBAR_REPORT_KEY] = SIDEBAR_EMPTY_LABEL

    current_value = st.session_state.get(SIDEBAR_REPORT_KEY)
    if current_value not in sidebar_options:
        st.session_state[SIDEBAR_REPORT_KEY] = SIDEBAR_EMPTY_LABEL

    selected = st.sidebar.radio(
        "Available reports",
        options=sidebar_options,
        index=sidebar_options.index(str(st.session_state[SIDEBAR_REPORT_KEY])),
        format_func=lambda key: labels[str(key)],
        key=SIDEBAR_REPORT_KEY,
        label_visibility="collapsed",
    )

    if selected == SIDEBAR_EMPTY_LABEL:
        return NO_SELECTION_KEY
    return str(selected)


def _render_report_view(entry: dict) -> None:
    if entry["source_type"] == "streamlit":
        _render_streamlit_module(
            module_path=str(entry["module_path"]),
            context={
                "environment": entry["environment"],
                "schema": entry["schema"],
                "catalogs": CATALOG_NAMES,
            },
        )
        return

    selected_path = str(entry["file_path"])
    has_inline_preview = _render_core_file(selected_path, str(entry.get("extension", "")))
    if not has_inline_preview:
        payload = download_file_bytes(selected_path)
        st.download_button(
            label="Download report",
            data=payload,
            file_name=Path(selected_path).name,
            mime=_mime_for(selected_path),
        )


_inject_css()

core_rows, _, _ = discover_core_reports(tuple(CATALOG_NAMES), REPORTS_VOLUME_NAME)
streamlit_rows, streamlit_warnings = _discover_streamlit_rows()
catalog_entries = _build_catalog_entries(core_rows, streamlit_rows)

if not catalog_entries:
    st.warning("No accessible reports were found.")
    st.stop()

entry_by_key = {item["key"]: item for item in catalog_entries}
selected_key = _render_sidebar_menu(catalog_entries)
if selected_key == NO_SELECTION_KEY:
    st.markdown(
        """
        <div class="hero">
          <h1>Machine Learning Reports</h1>
          <p>
            Unified report hub for machine learning analyses across training and production
            environments. Open any report from the left menu to view the latest published output.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    selected_entry = entry_by_key.get(selected_key)
    if selected_entry:
        try:
            _render_report_view(selected_entry)
        except Exception as exc:
            st.error(f"Failed to open selected report: {exc}")

if streamlit_warnings:
    with st.sidebar.expander("Technical warnings", expanded=False):
        for warning in streamlit_warnings:
            st.write(f"- {warning}")
