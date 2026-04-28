from __future__ import annotations

ANALYSIS_DESCRIPTIONS: dict[str, str] = {
    "user_segmentation": (
        "Behavior and value segmentation with retention and activation views "
        "for lifecycle actions."
    ),
    "cross_sell_mba": (
        "Cross-product association analysis focused on lift, confidence and "
        "audience opportunities."
    ),
    "reference_ml_template": (
        "Reference template reports for baseline model performance and "
        "operational outputs."
    ),
}


def analysis_title(schema_name: str) -> str:
    label = str(schema_name).replace("_", " ").replace("-", " ").strip()
    if not label:
        return "Unknown Analysis"
    return " ".join(word.capitalize() for word in label.split())


def analysis_description(schema_name: str) -> str:
    key = str(schema_name).strip().lower()
    return ANALYSIS_DESCRIPTIONS.get(
        key,
        "Operational reports for this analysis schema.",
    )

