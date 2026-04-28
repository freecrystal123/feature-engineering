from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamlitReportSpec:
    report_id: str
    title: str
    schema: str
    environments: tuple[str, ...]
    module_path: str
    description: str


STREAMLIT_REPORTS: tuple[StreamlitReportSpec, ...] = (
    StreamlitReportSpec(
        report_id="user_segmentation_overview",
        title="User Segmentation Overview",
        schema="user_segmentation",
        environments=("training", "production"),
        module_path="streamlit_reports.user_segmentation_overview",
        description="Interactive segmentation health summary with trend exploration.",
    ),
    StreamlitReportSpec(
        report_id="cross_sell_mba_explorer",
        title="Cross-sell MBA Explorer",
        schema="cross_sell_mba",
        environments=("training", "production"),
        module_path="streamlit_reports.cross_sell_mba_explorer",
        description="Rule performance explorer with lift and confidence controls.",
    ),
)

