"""Builders for the 18 supplemental manuscript tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from clear_card.config import CITABLE_CATEGORIES, NON_CITABLE, STUDY
from clear_card.statistics import (
    SummaryFormat,
    composition_by_category,
    format_composition_table,
    require_columns,
    summarize_categories,
)

TABLE_FILE_TEMPLATE = "supplemental_table_{number:02d}.xlsx"
JOURNAL_METRIC_ORDER = (
    "impact_factor_2y",
    "impact_factor_5y",
    "citescore",
    "sjr",
    "snip",
    "eigenfactor",
    "article_influence_score",
    "immediacy_index",
)
EXCEL_COLUMN_LABELS = {
    "target_journal": "Journal",
    "pub_year": "Publication year",
    "metric_year": "Year",
    "ora": "ORA",
    "nora": "NORA Overall",
    "meta_analysis": "Meta-analysis",
    "review": "Review",
    "gss": "GSS",
    "non_citable": "Non-citable items",
    "impact_factor_2y": "Journal impact factor",
    "impact_factor_5y": "5-year impact factor",
    "citescore": "CiteScore",
    "sjr": "SJR",
    "snip": "SNIP",
    "eigenfactor": "Eigenfactor",
    "article_influence_score": "Article influence score",
    "immediacy_index": "Immediacy index",
}


def build_supplemental_table_01(
    journal_metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build ST1: yearly journal bibliometric measures."""

    required = {"target_journal", "metric_year", "metric", "value"}
    require_columns(journal_metrics, required, label="journal metrics")
    metrics = list(metric_columns or JOURNAL_METRIC_ORDER)
    unknown_metrics = set(journal_metrics["metric"].dropna()).difference(metrics)
    if metric_columns is None and unknown_metrics:
        names = ", ".join(sorted(unknown_metrics))
        raise ValueError(f"journal metrics contains unknown metrics: {names}")

    result = (
        journal_metrics.loc[journal_metrics["metric"].isin(metrics)]
        .pivot(
            index=["target_journal", "metric_year"],
            columns="metric",
            values="value",
        )
        .reindex(columns=metrics)
        .reset_index()
    )
    result.columns.name = None
    for column in metrics:
        numeric = pd.to_numeric(result[column], errors="coerce")
        result[column] = numeric.map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    return result.sort_values(["target_journal", "metric_year"]).reset_index(drop=True)


def build_supplemental_table_02(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST2: yearly mean fitness by manuscript category."""

    return _distribution_table(
        analysis,
        value_column="lambda",
        max_year=STUDY.fitness_end_year,
        summary_format="mean_sd",
        original_label="ora",
    )


def build_supplemental_table_03(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST3: yearly median fitness by manuscript category."""

    return _distribution_table(
        analysis,
        value_column="lambda",
        max_year=STUDY.fitness_end_year,
        summary_format="median_iqr",
        original_label="ora",
    )


def build_supplemental_table_04(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST4: yearly mean two-year citations by manuscript category."""

    return _distribution_table(
        analysis,
        value_column="citation_within_2_year",
        max_year=STUDY.study_end_year,
        summary_format="mean_sd",
        original_label="ora",
    )


def build_supplemental_table_05(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST5: yearly median two-year citations by manuscript category."""

    return _distribution_table(
        analysis,
        value_column="citation_within_2_year",
        max_year=STUDY.study_end_year,
        summary_format="median_iqr",
        original_label="ora",
    )


def build_supplemental_table_06(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST6: publication composition among ORA and NORA."""

    return _composition_table(
        analysis,
        value_column=None,
        include_non_citable=False,
    )


def build_supplemental_table_07(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST7: publication composition including non-citable items."""

    return _composition_table(
        analysis,
        value_column=None,
        include_non_citable=True,
    )


def build_supplemental_table_08(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST8: two-year citation composition among ORA and NORA."""

    return _composition_table(
        analysis,
        value_column="citation_within_2_year",
        include_non_citable=False,
    )


def build_supplemental_table_09(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST9: two-year citation composition including non-citable items."""

    return _composition_table(
        analysis,
        value_column="citation_within_2_year",
        include_non_citable=True,
    )


def build_supplemental_table_10(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST10: mean proportion of five-year predatory citations."""

    return _distribution_table(
        analysis,
        value_column="pct_predatory_5year",
        max_year=STUDY.fitness_end_year,
        summary_format="mean_sd",
        original_label="ora",
    )


def build_supplemental_table_11(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST11: median proportion of five-year predatory citations."""

    return _distribution_table(
        analysis,
        value_column="pct_predatory_5year",
        max_year=STUDY.fitness_end_year,
        summary_format="median_iqr",
        original_label="ora",
    )


def build_supplemental_table_12(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST12: mean H-index of journals citing within five years."""

    return _citing_journal_table(analysis, metric="hindex", summary_format="mean_sd")


def build_supplemental_table_13(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST13: median H-index of journals citing within five years."""

    return _citing_journal_table(analysis, metric="hindex", summary_format="median_iqr")


def build_supplemental_table_14(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST14: mean SJR of journals citing within five years."""

    return _citing_journal_table(analysis, metric="sjr", summary_format="mean_sd")


def build_supplemental_table_15(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST15: median SJR of journals citing within five years."""

    return _citing_journal_table(analysis, metric="sjr", summary_format="median_iqr")


def build_supplemental_table_16(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST16: mean SNIP of journals citing within five years."""

    return _citing_journal_table(analysis, metric="snip", summary_format="mean_sd")


def build_supplemental_table_17(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build ST17: median SNIP of journals citing within five years."""

    return _citing_journal_table(analysis, metric="snip", summary_format="median_iqr")


def build_supplemental_table_18(
    impact_factor_simulation: pd.DataFrame,
) -> pd.DataFrame:
    """Build ST18: simulated journal impact-factor components."""

    return impact_factor_simulation.copy()


def build_supplemental_tables(
    analysis: pd.DataFrame,
    journal_metrics: pd.DataFrame,
    impact_factor_simulation: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Build every supplemental table without writing files."""

    # Integer keys preserve manuscript numbering independently of output filenames.
    return {
        1: build_supplemental_table_01(journal_metrics),
        2: build_supplemental_table_02(analysis),
        3: build_supplemental_table_03(analysis),
        4: build_supplemental_table_04(analysis),
        5: build_supplemental_table_05(analysis),
        6: build_supplemental_table_06(analysis),
        7: build_supplemental_table_07(analysis),
        8: build_supplemental_table_08(analysis),
        9: build_supplemental_table_09(analysis),
        10: build_supplemental_table_10(analysis),
        11: build_supplemental_table_11(analysis),
        12: build_supplemental_table_12(analysis),
        13: build_supplemental_table_13(analysis),
        14: build_supplemental_table_14(analysis),
        15: build_supplemental_table_15(analysis),
        16: build_supplemental_table_16(analysis),
        17: build_supplemental_table_17(analysis),
        18: build_supplemental_table_18(impact_factor_simulation),
    }


def write_supplemental_table(
    table: pd.DataFrame,
    number: int,
    output_dir: Path,
    *,
    include_index: bool = False,
) -> Path:
    """Write one numbered supplemental table to an explicit directory."""

    if number not in range(1, 19):
        raise ValueError("number must be between 1 and 18")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / TABLE_FILE_TEMPLATE.format(number=number)
    display_table = table.rename(columns=EXCEL_COLUMN_LABELS)
    display_table.to_excel(output_path, index=include_index)
    return output_path


def write_supplemental_tables(
    tables: Mapping[int, pd.DataFrame],
    output_dir: Path,
    *,
    include_index: bool = False,
) -> list[Path]:
    """Write supplied supplemental tables to an explicit directory."""

    return [
        write_supplemental_table(
            tables[number],
            number,
            output_dir,
            include_index=include_index,
        )
        for number in sorted(tables)
    ]


def _distribution_table(
    analysis: pd.DataFrame,
    *,
    value_column: str,
    max_year: int,
    summary_format: SummaryFormat,
    original_label: str,
) -> pd.DataFrame:
    """Build a consistently formatted category-distribution table."""

    return summarize_categories(
        _study_journals(analysis),
        value_column,
        max_year=max_year,
        summary_format=summary_format,
        original_label=original_label,
    )


def _composition_table(
    analysis: pd.DataFrame,
    *,
    value_column: str | None,
    include_non_citable: bool,
) -> pd.DataFrame:
    """Build publication or citation composition with manuscript denominators."""

    categories = (
        (*CITABLE_CATEGORIES, NON_CITABLE)
        if include_non_citable
        else CITABLE_CATEGORIES
    )
    composition = composition_by_category(
        _study_journals(analysis),
        categories=categories,
        max_year=STUDY.study_end_year,
        value_column=value_column,
    )
    return format_composition_table(
        composition,
        include_non_citable=include_non_citable,
    )


def _citing_journal_table(
    analysis: pd.DataFrame,
    *,
    metric: str,
    summary_format: SummaryFormat,
) -> pd.DataFrame:
    """Build one five-year citing-journal-quality table."""

    metric_columns = {
        "hindex": "source_journal_score_hindex_5yr_mean",
        "sjr": "source_journal_score_sjr_5yr_mean",
        "snip": "source_journal_score_snip_5yr_mean",
    }
    if metric not in metric_columns:
        raise ValueError("metric must be 'hindex', 'sjr', or 'snip'")
    return _distribution_table(
        analysis,
        value_column=metric_columns[metric],
        max_year=STUDY.fitness_end_year,
        summary_format=summary_format,
        original_label="ora",
    )


def _study_journals(analysis: pd.DataFrame) -> pd.DataFrame:
    """Restrict a table input to the four journals in the study cohort."""

    require_columns(analysis, {"target_journal"}, label="analysis data")
    return analysis.loc[analysis["target_journal"].isin(STUDY.all_journals)]
