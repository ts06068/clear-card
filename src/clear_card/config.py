from __future__ import annotations

from dataclasses import dataclass


# Canonical journal names and study windows are shared across every analysis module.
CIRCULATION = "CIRCULATION"
EUROPEAN_HEART_JOURNAL = "EUROPEAN HEART JOURNAL"
JACC = "JOURNAL OF THE AMERICAN COLLEGE OF CARDIOLOGY"
JAMA_CARDIOLOGY = "JAMA CARDIOLOGY"

PRIMARY_JOURNALS = (CIRCULATION, EUROPEAN_HEART_JOURNAL, JACC)
ALL_JOURNALS = (*PRIMARY_JOURNALS, JAMA_CARDIOLOGY)
JOURNAL_LABELS = {
    CIRCULATION: "CIRC",
    EUROPEAN_HEART_JOURNAL: "EHJ",
    JACC: "JACC",
    JAMA_CARDIOLOGY: "JAMA Cardiology",
}

ORA = "ora"
META_ANALYSIS = "meta_analysis"
REVIEW = "review"
GSS = "gss"
NON_CITABLE = "non_citable"
UNCLASSIFIED = "unclassified"
NORA_CATEGORIES = (META_ANALYSIS, REVIEW, GSS)
CITABLE_CATEGORIES = (ORA, *NORA_CATEGORIES)


@dataclass(frozen=True)
class StudyConfig:
    """Manuscript cohort definitions and reproducibility settings."""

    study_start_year: int = 2008
    study_end_year: int = 2022
    citation_snapshot: str = "2025-11"
    fitness_end_year: int = 2019
    jama_start_year: int = 2016
    validation_end_year: int = 2014
    bootstrap_iterations: int = 10_000
    random_seed: int = 42
    significance_level: float = 0.05
    journal_metric_lag_years: int = 2
    primary_journals: tuple[str, ...] = PRIMARY_JOURNALS
    all_journals: tuple[str, ...] = ALL_JOURNALS


STUDY = StudyConfig()
