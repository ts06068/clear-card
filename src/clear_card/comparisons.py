from __future__ import annotations

import pandas as pd

from clear_card.config import (
    CITABLE_CATEGORIES,
    GSS,
    META_ANALYSIS,
    NON_CITABLE,
    ORA,
    REVIEW,
)

VALIDATION_CATEGORY_MAP = {
    "original": ORA,
    "meta": META_ANALYSIS,
    "review": REVIEW,
    "gss": GSS,
    "short": NON_CITABLE,
}


def normalize_validation_categories(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["manuscript_category"] = (
        result["category_binary"].str.lower().map(VALIDATION_CATEGORY_MAP)
    )
    return result


def prepare_wos_citation_comparison(
    wos_validation: pd.DataFrame,
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Select matched citable items for the Scopus-WoS comparison."""

    # Article IDs define the cross-database match; manuscript categories come from Scopus.
    return (
        wos_validation.merge(
            analysis[["article_id", "manuscript_category"]],
            on="article_id",
            how="left",
        )
        .loc[lambda data: data["manuscript_category"].isin(CITABLE_CATEGORIES)]
        .drop_duplicates("article_id")
        .copy()
    )


def prepare_wos_publication_comparison(
    analysis: pd.DataFrame,
    wos_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Join annual Scopus citable-item counts to WoS publication counts."""

    scopus_counts = (
        analysis.loc[analysis["manuscript_category"].isin(CITABLE_CATEGORIES)]
        .groupby(["target_journal", "target_pub_year"])
        .size()
        .rename("scopus_count")
        .reset_index()
    )
    return scopus_counts.merge(
        wos_counts.rename(
            columns={
                "journal": "target_journal",
                "pub_year": "target_pub_year",
                "wos_count": "wos_count",
            }
        ),
        on=["target_journal", "target_pub_year"],
        how="inner",
    )
