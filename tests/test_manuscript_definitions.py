import pandas as pd

from clear_card.config import (
    GSS,
    META_ANALYSIS,
    NON_CITABLE,
    ORA,
    REVIEW,
    STUDY,
    UNCLASSIFIED,
)
from clear_card.data import add_manuscript_categories, select_study_cohort
from clear_card.experiments.common import align_journal_metrics


def test_article_types_match_the_manuscript() -> None:
    # This compact fixture documents every manuscript-level category boundary.
    frame = pd.DataFrame(
        {
            "category": [
                "ORIGINAL RESEARCH",
                "META ANALYSIS",
                "SYSTEMATIC REVIEW WITHOUT META ANALYSIS",
                "GUIDELINE",
                "EDITORIAL",
                "short",
                "OTHER",
            ]
        }
    )

    result = add_manuscript_categories(frame)

    assert result["manuscript_category"].tolist() == [
        ORA,
        META_ANALYSIS,
        REVIEW,
        GSS,
        NON_CITABLE,
        UNCLASSIFIED,
        UNCLASSIFIED,
    ]


def test_study_windows_match_the_manuscript() -> None:
    assert (STUDY.study_start_year, STUDY.study_end_year) == (2008, 2022)
    assert STUDY.fitness_end_year == 2019
    assert STUDY.jama_start_year == 2016
    assert STUDY.citation_snapshot == "2025-11"


def test_journal_metrics_use_a_two_year_lag() -> None:
    metrics = pd.DataFrame(
        {
            "target_journal": ["CIRCULATION"],
            "metric_year": [2010],
            "metric": ["impact_factor_2y"],
            "value": [14.432],
        }
    )

    aligned = align_journal_metrics(metrics)

    assert aligned.loc[0, "target_pub_year"] == 2008


def test_study_cohort_uses_journal_specific_start_years() -> None:
    frame = pd.DataFrame(
        {
            "target_journal": [
                "CIRCULATION",
                "JAMA CARDIOLOGY",
                "JAMA CARDIOLOGY",
                "OTHER JOURNAL",
            ],
            "target_pub_year": [2008, 2015, 2016, 2016],
        }
    )

    result = select_study_cohort(frame)

    assert result.index.tolist() == [0, 2]
