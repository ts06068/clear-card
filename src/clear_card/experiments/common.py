from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from clear_card.comparisons import normalize_validation_categories
from clear_card.config import STUDY
from clear_card.data import (
    load_analysis,
    load_classifications,
    load_icite_validation,
    load_if_simulation,
    load_journal_metrics,
    load_scopus_validation,
    load_wos_publication_counts,
    load_wos_validation,
)
from clear_card.settings import RepositoryPaths


CORRELATION_PREDICTORS = {
    "mean_fitness": "ORA fitness",
    "mean_2ycit": "ORA 2-year citation count",
    "original_ratio": "Proportion of ORA publications (%)",
    "original_count": "Publication volume of ORA",
}


@dataclass(frozen=True)
class AnalysisInputs:
    analysis: pd.DataFrame
    classifications: pd.DataFrame
    journal_metrics: pd.DataFrame
    impact_factor_simulation: pd.DataFrame
    scopus_validation: pd.DataFrame
    icite_validation: pd.DataFrame
    wos_validation: pd.DataFrame
    wos_publication_counts: pd.DataFrame


def load_inputs(paths: RepositoryPaths) -> AnalysisInputs:
    """Load the frozen, de-identified inputs once for the full pipeline."""

    return AnalysisInputs(
        analysis=load_analysis(paths),
        classifications=load_classifications(paths),
        journal_metrics=load_journal_metrics(paths),
        impact_factor_simulation=load_if_simulation(paths),
        scopus_validation=normalize_validation_categories(
            load_scopus_validation(paths)
        ),
        icite_validation=normalize_validation_categories(load_icite_validation(paths)),
        wos_validation=load_wos_validation(paths),
        wos_publication_counts=load_wos_publication_counts(paths),
    )


def align_journal_metrics(
    journal_metrics: pd.DataFrame,
    *,
    lag_years: int = STUDY.journal_metric_lag_years,
) -> pd.DataFrame:
    """Pair publication years with journal metrics observed two years later."""

    wide = (
        journal_metrics.pivot(
            index=["target_journal", "metric_year"],
            columns="metric",
            values="value",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    wide["target_pub_year"] = wide["metric_year"] - lag_years
    return wide.sort_values(["target_journal", "target_pub_year"]).reset_index(
        drop=True
    )


def impact_factor_series(journal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return publication-year-aligned 2-year impact factors for trend plots."""

    return align_journal_metrics(journal_metrics).rename(
        columns={"impact_factor_2y": "impact_factor"}
    )[["target_journal", "target_pub_year", "impact_factor"]]
