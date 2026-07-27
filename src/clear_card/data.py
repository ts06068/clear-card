from __future__ import annotations

import pandas as pd

from clear_card.config import (
    GSS,
    JAMA_CARDIOLOGY,
    META_ANALYSIS,
    NON_CITABLE,
    ORA,
    REVIEW,
    STUDY,
    UNCLASSIFIED,
)
from clear_card.settings import RepositoryPaths


ANALYSIS_COLUMNS = {
    "article_id",
    "target_pub_date",
    "target_pub_year",
    "target_journal",
    "category",
    "lambda",
    "citation_within_2_year",
    "citation_within_5_year",
    "citation_within_9_year",
    "citation_within_10_year",
    "pct_predatory_5year",
    "source_journal_score_hindex_5yr_mean",
    "source_journal_score_sjr_5yr_mean",
    "source_journal_score_snip_5yr_mean",
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


# This translation table implements the mutually exclusive categories reported in the paper.
CATEGORY_MAP = {
    "ORIGINAL RESEARCH": ORA,
    "META ANALYSIS": META_ANALYSIS,
    "SYSTEMATIC REVIEW WITH META ANALYSIS": META_ANALYSIS,
    "SYSTEMATIC REVIEW": REVIEW,
    "SYSTEMATIC REVIEW WITHOUT META ANALYSIS": REVIEW,
    "NARRATIVE REVIEW": REVIEW,
    "REVIEW": REVIEW,
    "GUIDELINE": GSS,
    "STATEMENT": GSS,
    "STATISTICS": GSS,
    "BRIEF REPORT": NON_CITABLE,
    "CASE REPORT": NON_CITABLE,
    "COMMENTARY": NON_CITABLE,
    "EDITORIAL": NON_CITABLE,
    "BRIEF REPORT [UNCITABLE]": NON_CITABLE,
    "CASE REPORT [UNCITABLE]": NON_CITABLE,
    "COMMENTARY [UNCITABLE]": NON_CITABLE,
    "EDITORIAL [UNCITABLE]": NON_CITABLE,
}


def add_manuscript_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the mutually exclusive article definitions used in the manuscript."""

    result = frame.copy()
    result["manuscript_category"] = (
        result["category"].map(CATEGORY_MAP).fillna(UNCLASSIFIED)
    )
    return result


def select_study_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the manuscript's journal-specific publication windows."""

    _require_columns(frame, {"target_journal", "target_pub_year"}, "analysis data")
    years = pd.to_numeric(frame["target_pub_year"], errors="coerce")
    primary = frame["target_journal"].isin(STUDY.primary_journals) & years.between(
        STUDY.study_start_year, STUDY.study_end_year
    )
    jama = frame["target_journal"].eq(JAMA_CARDIOLOGY) & years.between(
        STUDY.jama_start_year, STUDY.study_end_year
    )
    return frame.loc[primary | jama].copy()


def load_analysis(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    frame = pd.read_csv(
        paths.processed_data / "manuscript_analysis.csv.gz", low_memory=False
    )
    _require_columns(frame, ANALYSIS_COLUMNS, "manuscript analysis data")
    frame["target_pub_date"] = pd.to_datetime(frame["target_pub_date"], errors="raise")
    frame = select_study_cohort(frame).drop_duplicates("article_id").copy()
    frame["pub_year"] = frame["target_pub_date"].dt.year
    return add_manuscript_categories(frame)


def load_journal_metrics(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(paths.journal_metrics)


def load_classifications(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(paths.processed_data / "article_classifications.csv.gz")


def load_scopus_validation(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(
        paths.processed_data / "scopus_validation.csv.gz",
        parse_dates=["target_pub_date"],
    )


def load_icite_validation(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(
        paths.processed_data / "icite_validation.csv.gz",
        parse_dates=["target_pub_date"],
    )


def load_wos_validation(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(paths.processed_data / "wos_scopus_validation.csv.gz")


def load_wos_publication_counts(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(paths.processed_data / "wos_publication_counts.csv")


def load_if_simulation(paths: RepositoryPaths | None = None) -> pd.DataFrame:
    paths = paths or RepositoryPaths()
    return pd.read_csv(paths.processed_data / "impact_factor_simulation.csv")
