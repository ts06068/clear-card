from __future__ import annotations

import pandas as pd

from clear_card.config import GSS, META_ANALYSIS, NON_CITABLE, ORA, REVIEW
from clear_card.data import add_manuscript_categories


DISPLAY_NAMES = {
    GSS: "GSS",
    META_ANALYSIS: "Meta-analysis",
    ORA: "ORA",
    REVIEW: "Review",
    NON_CITABLE: "Non-citable items",
}


def build_supplemental_category_counts(
    analysis: pd.DataFrame,
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        analysis[["article_id", "target_journal", "target_pub_date", "category"]]
        .merge(
            classifications[["article_id", "category_1", "category_2", "category_3"]],
            on="article_id",
            how="left",
        )
        .drop_duplicates("article_id")
    )
    merged = add_manuscript_categories(merged)
    merged["year"] = pd.to_datetime(merged["target_pub_date"]).dt.year
    merged = merged.query("2008 <= year <= 2022")

    group_columns = ["target_journal", "category_1", "category_2", "category_3"]
    counts = (
        merged.groupby(group_columns + ["manuscript_category"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for category in DISPLAY_NAMES:
        if category not in counts.columns:
            counts[category] = 0
    counts = counts[list(DISPLAY_NAMES)]
    counts["total"] = counts.sum(axis=1)
    counts = counts.loc[counts["total"].gt(0)].copy()
    counts["main_classification"] = (
        counts[list(DISPLAY_NAMES)].idxmax(axis=1).map(DISPLAY_NAMES)
    )

    result = counts.reset_index().rename(
        columns={
            "target_journal": "journal",
            **DISPLAY_NAMES,
        }
    )
    result = result[
        [
            "journal",
            "category_1",
            "category_2",
            "category_3",
            "main_classification",
            "total",
            "GSS",
            "Meta-analysis",
            "ORA",
            "Review",
            "Non-citable items",
        ]
    ].rename(
        columns={
            "GSS": "gss",
            "Meta-analysis": "meta_analysis",
            "ORA": "ora",
            "Review": "review",
            "Non-citable items": "non_citable_items",
        }
    )
    result.columns.name = None
    return result
