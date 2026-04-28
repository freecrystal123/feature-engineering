from __future__ import annotations

import base64
import importlib
import mimetypes
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core_reports.analysis_metadata import analysis_description, analysis_title
from core_reports.catalog_service import (
    discover_core_reports,
    download_file_bytes,
    parse_catalogs,
    read_text_preview,
)
from core_reports.streamlit_registry import STREAMLIT_REPORTS

st.set_page_config(page_title="ML Reports Catalog", page_icon=":bar_chart:", layout="wide")

REPORTS_VOLUME_NAME = os.getenv("REPORTS_VOLUME_NAME", "reports")
CATALOGS_ENV = os.getenv("REPORTS_CATALOGS", "ml_train,ml_prod")
CATALOG_NAMES = parse_catalogs(CATALOGS_ENV)
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log"}
HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_TEXT_PREVIEW_BYTES = 300_000
ENVIRONMENTS = ("training", "production")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
          html, body, [class*="st-"], [class*="css"] {
            font-family: "Manrope", system-ui, sans-serif;
          }
          .main .block-container {
            max-width: 100%;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
          }
          .hero {
            border: 1px solid #d6dfec;
            background: linear-gradient(120deg, #f0f6ff 0%, #ecfff7 100%);
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 18px;
          }
          .hero h1 {
            margin: 0;
            font-size: 30px;
            line-height: 1.15;
            color: #0f172a;
            letter-spacing: 0;
          }
          .hero p {
            margin: 8px 0 0 0;
            color: #334155;
            font-size: 15px;
          }
          .analysis-head {
            border: 1px solid #d6dfec;
            background: #ffffff;
            border-radius: 10px;
            padding: 16px 18px;
            margin: 8px 0 18px 0;
          }
          .analysis-title {
            margin: 0;
            font-size: 24px;
            color: #0f172a;
            font-weight: 800;
            letter-spacing: 0;
          }
          .analysis-sub {
            margin: 7px 0 0 0;
            color: #475569;
            font-size: 14px;
          }
          .chip {
            display: inline-block;
            margin: 10px 8px 0 0;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid #d8e3f3;
            color: #0f172a;
            background: #f8fbff;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mime_for(file_path: str) -> str:
    detected, _ = mimetypes.guess_type(Path(file_path).name)
    return detected or "application/octet-stream"


def _render_core_file(file_path: str, extension: str) -> None:
    if extension in HTML_EXTENSIONS:
        html = download_file_bytes(file_path).decode("utf-8", errors="replace")
        components.html(html, height=1150, scrolling=True)
        return

    if extension == ".pdf":
        data = download_file_bytes(file_path)
        b64 = base64.b64encode(data).decode("utf-8")
        iframe = (
            "<iframe width='100%' height='1150' style='border:none' "
            f"src='data:application/pdf;base64,{b64}'></iframe>"
        )
        st.markdown(iframe, unsafe_allow_html=True)
        return

    if extension in IMAGE_EXTENSIONS:
        st.image(download_file_bytes(file_path), use_container_width=True)
        return

    if extension in TEXT_EXTENSIONS:
        preview, truncated = read_text_preview(file_path, limit_bytes=MAX_TEXT_PREVIEW_BYTES)
        st.code(preview, language="text")
        if truncated:
            st.caption(f"Preview truncated to {MAX_TEXT_PREVIEW_BYTES:,} bytes.")
        return

    st.info("Inline preview is not available for this file type. Use download.")


def _build_streamlit_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in STREAMLIT_REPORTS:
        for env in spec.environments:
            rows.append(
                {
                    "source_type": "streamlit",
                    "environment": env,
                    "catalog": "internal",
                    "schema": spec.schema,
                    "relative_path": spec.report_id,
                    "report_title": spec.title,
                    "extension": ".py",
                    "size_bytes": 0,
                    "updated_at": "-",
                    "file_path": "",
                    "description": spec.description,
                    "module_path": spec.module_path,
                }
            )
    return rows


def _format_analysis_option(schema_name: str) -> str:
    return f"{analysis_title(schema_name)} ({schema_name})"


def _analysis_counts_for_environment(all_rows: list[dict], environment: str) -> tuple[int, int, int]:
    filtered = [row for row in all_rows if row["environment"] == environment]
    analyses = len({row["schema"] for row in filtered})
    core_files = len([row for row in filtered if row["source_type"] == "core_file"])
    streamlit_reports = len([row for row in filtered if row["source_type"] == "streamlit"])
    return analyses, core_files, streamlit_reports


def _render_streamlit_module(module_path: str, context: dict) -> None:
    module = importlib.import_module(module_path)
    render_fn = getattr(module, "render", None)
    if not callable(render_fn):
        raise ValueError(f"Module '{module_path}' has no callable 'render(context)' function.")
    render_fn(context)


_inject_css()

st.markdown(
    f"""
    <div class="hero">
      <h1>Machine Learning Reports</h1>
      <p>
        Unified catalog across training and production environments, organized by analysis
        schema and ready to open HTML, PDF, text, image and Streamlit interactive reports.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

toolbar_col1, toolbar_col2 = st.columns([3, 1])
with toolbar_col2:
    if st.button("Refresh catalog", use_container_width=True):
        st.cache_data.clear()

core_rows, scan_warnings, scan_stats = discover_core_reports(tuple(CATALOG_NAMES), REPORTS_VOLUME_NAME)
streamlit_rows = _build_streamlit_rows()
all_rows = core_rows + streamlit_rows

train_analyses, train_core, train_streamlit = _analysis_counts_for_environment(all_rows, "training")
prod_analyses, prod_core, prod_streamlit = _analysis_counts_for_environment(all_rows, "production")

card1, card2, card3, card4 = st.columns(4)
card1.metric("Training analyses", f"{train_analyses:,}")
card2.metric("Production analyses", f"{prod_analyses:,}")
card3.metric("Core files found", f"{len(core_rows):,}")
card4.metric("Interactive reports", f"{len(streamlit_rows):,}")

if scan_warnings:
    with st.expander("Scan warnings"):
        for warning in scan_warnings:
            st.write(f"- {warning}")

with st.expander("Catalog scan summary"):
    summary_rows = []
    for catalog_name, values in scan_stats["by_catalog"].items():
        summary_rows.append(
            {
                "catalog": catalog_name,
                "environment": values["environment"],
                "schemas_scanned": values["schemas_scanned"],
                "report_volumes_found": values["report_volumes_found"],
                "files_found": values["files_found"],
            }
        )
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No catalog details to display.")

if not all_rows:
    st.warning("No reports found in catalogs or local Streamlit registry.")
    st.stop()

st.markdown("#### Environment")
selected_environment = st.radio(
    label="Environment",
    options=ENVIRONMENTS,
    index=1,
    horizontal=True,
    label_visibility="collapsed",
)

environment_rows = [row for row in all_rows if row["environment"] == selected_environment]
analysis_options = sorted({row["schema"] for row in environment_rows})

if not analysis_options:
    st.info(f"No analyses found for `{selected_environment}`.")
    st.stop()

selected_schema = st.selectbox(
    "Analysis",
    options=analysis_options,
    format_func=_format_analysis_option,
)

analysis_rows = [row for row in environment_rows if row["schema"] == selected_schema]
analysis_core_rows = [row for row in analysis_rows if row["source_type"] == "core_file"]
analysis_streamlit_rows = [row for row in analysis_rows if row["source_type"] == "streamlit"]

analysis_label = analysis_title(selected_schema)
analysis_about = analysis_description(selected_schema)

st.markdown(
    (
        '<div class="analysis-head">'
        f'<h2 class="analysis-title">{analysis_label}</h2>'
        f'<p class="analysis-sub">{analysis_about}</p>'
        f'<span class="chip">Environment: {selected_environment}</span>'
        f'<span class="chip">Core reports: {len(analysis_core_rows)}</span>'
        f'<span class="chip">Interactive reports: {len(analysis_streamlit_rows)}</span>'
        '</div>'
    ),
    unsafe_allow_html=True,
)

tab_core, tab_streamlit = st.tabs(["Core Reports", "Streamlit Reports"])

with tab_core:
    if not analysis_core_rows:
        st.info("No core report file found for this analysis and environment.")
    else:
        core_options = [
            f"{row['report_title']}  ({row['extension']}, {row['catalog']})"
            for row in analysis_core_rows
        ]
        selected_core_option = st.selectbox("Report file", options=core_options, key="core_file_select")
        selected_core_idx = core_options.index(selected_core_option)
        selected_core = analysis_core_rows[selected_core_idx]
        selected_path = str(selected_core["file_path"])

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.caption(f"Schema: `{selected_core['schema']}`")
        info_col2.caption(f"Catalog: `{selected_core['catalog']}`")
        info_col3.caption(f"Updated: `{selected_core['updated_at']}`")
        st.caption(f"Path: `{selected_path}`")

        try:
            _render_core_file(selected_path, str(selected_core["extension"]))
        except Exception as exc:
            st.error(f"Failed to render file: {exc}")

        try:
            payload = download_file_bytes(selected_path)
            st.download_button(
                label="Download file",
                data=payload,
                file_name=Path(selected_path).name,
                mime=_mime_for(selected_path),
            )
        except Exception as exc:
            st.error(f"Failed to download file: {exc}")

with tab_streamlit:
    if not analysis_streamlit_rows:
        st.info("No Streamlit report registered for this analysis and environment.")
    else:
        streamlit_options = [row["report_title"] for row in analysis_streamlit_rows]
        selected_streamlit_title = st.selectbox(
            "Interactive report",
            options=streamlit_options,
            key="streamlit_report_select",
        )
        selected_streamlit = next(
            row for row in analysis_streamlit_rows if row["report_title"] == selected_streamlit_title
        )
        st.caption(selected_streamlit.get("description", ""))

        try:
            _render_streamlit_module(
                module_path=str(selected_streamlit["module_path"]),
                context={
                    "environment": selected_environment,
                    "schema": selected_schema,
                    "catalogs": CATALOG_NAMES,
                },
            )
        except Exception as exc:
            st.error(f"Failed to render Streamlit report module: {exc}")

