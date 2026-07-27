"""Grouped summaries and manuscript table formatting."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from clear_card.config import (
    GSS,
    META_ANALYSIS,
    NON_CITABLE,
    NORA_CATEGORIES,
    ORA,
    REVIEW,
    STUDY,
)

GROUP_COLUMNS = ("target_journal", "pub_year")
ATOMIC_CATEGORIES = (ORA, META_ANALYSIS, REVIEW, GSS, NON_CITABLE)
SummaryFormat = Literal["mean_sd", "median_iqr"]


def require_columns(
    frame: pd.DataFrame,
    columns: Collection[str],
    *,
    label: str = "data",
) -> None:
    """Raise a readable error when an input table lacks required columns."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def summarize_by_group(
    frame: pd.DataFrame,
    value_column: str,
    *,
    categories: Sequence[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    group_columns: Sequence[str] = GROUP_COLUMNS,
    category_column: str = "manuscript_category",
    year_column: str = "pub_year",
) -> pd.DataFrame:
    """Calculate the manuscript's sum, mean, SD, median, and quartiles."""

    required = {*group_columns, value_column}
    if categories is not None:
        required.add(category_column)
    if min_year is not None or max_year is not None:
        required.add(year_column)
    require_columns(frame, required)

    subset = frame
    if categories is not None:
        subset = subset.loc[subset[category_column].isin(categories)]
    if min_year is not None:
        subset = subset.loc[subset[year_column] >= min_year]
    if max_year is not None:
        subset = subset.loc[subset[year_column] <= max_year]

    return (
        subset.groupby(list(group_columns), dropna=False)[value_column]
        .agg(
            sum="sum",
            mean="mean",
            std="std",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def format_mean_sd(
    mean: pd.Series,
    standard_deviation: pd.Series,
    *,
    decimals: int = 2,
) -> pd.Series:
    """Format values as `mean (SD)` with the notebook's missing-SD behavior."""

    mean_text = mean.round(decimals).map(lambda value: f"{value:.{decimals}f}")
    sd_text = standard_deviation.round(decimals).map(
        lambda value: f"{value:.{decimals}f}"
    )
    return mean_text + " (" + sd_text + ")"


def format_median_iqr(
    median: pd.Series,
    q25: pd.Series,
    q75: pd.Series,
    *,
    decimals: int = 2,
) -> pd.Series:
    """Format values as `median (Q1-Q3)` using two decimal places."""

    median_text = median.round(decimals).map(lambda value: f"{value:.{decimals}f}")
    q25_text = q25.round(decimals).map(lambda value: f"{value:.{decimals}f}")
    q75_text = q75.round(decimals).map(lambda value: f"{value:.{decimals}f}")
    return median_text + " (" + q25_text + "-" + q75_text + ")"


def summarize_categories(
    frame: pd.DataFrame,
    value_column: str,
    *,
    min_year: int = STUDY.study_start_year,
    max_year: int,
    summary_format: SummaryFormat,
    original_label: str = "ora",
    category_column: str = "manuscript_category",
    group_columns: Sequence[str] = GROUP_COLUMNS,
    year_column: str = "pub_year",
) -> pd.DataFrame:
    """Build atomic-category and pooled-NORA summaries by journal and year."""

    require_columns(
        frame,
        {*group_columns, value_column, category_column, year_column},
    )
    subset = frame.loc[frame[year_column].between(min_year, max_year)].copy()
    subset[category_column] = subset[category_column].astype("string").str.lower()

    atomic = subset.loc[subset[category_column].isin(ATOMIC_CATEGORIES)]
    atomic_summary = _formatted_group_summary(
        atomic,
        value_column,
        summary_format=summary_format,
        group_columns=(*group_columns, category_column),
    )
    atomic_wide = (
        atomic_summary.pivot(
            index=list(group_columns),
            columns=category_column,
            values="formatted",
        )
        .reset_index()
        .rename(
            columns={
                ORA: original_label,
            }
        )
    )

    # Pool NORA records before summarizing instead of averaging its component summaries.
    nora = subset.loc[subset[category_column].isin(NORA_CATEGORIES)]
    nora_summary = _formatted_group_summary(
        nora,
        value_column,
        summary_format=summary_format,
        group_columns=group_columns,
    ).rename(columns={"formatted": "nora"})

    result = atomic_wide.merge(
        nora_summary[[*group_columns, "nora"]],
        on=list(group_columns),
        how="left",
    )
    output_columns = [
        *group_columns,
        original_label,
        "nora",
        META_ANALYSIS,
        REVIEW,
        GSS,
        NON_CITABLE,
    ]

    for column in output_columns:
        if column not in result:
            result[column] = np.nan
    result.columns.name = None
    return result[output_columns]


def _formatted_group_summary(
    frame: pd.DataFrame,
    value_column: str,
    *,
    summary_format: SummaryFormat,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate and format one value column for a supplied grouping."""

    grouped = frame.groupby(list(group_columns), dropna=False)[value_column]
    if summary_format == "mean_sd":
        result = grouped.agg(mean="mean", std="std").reset_index()
        result["formatted"] = format_mean_sd(result["mean"], result["std"])
    elif summary_format == "median_iqr":
        result = grouped.agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        ).reset_index()
        result["formatted"] = format_median_iqr(
            result["median"], result["q25"], result["q75"]
        )
    else:
        raise ValueError("summary_format must be 'mean_sd' or 'median_iqr'")
    return result


def composition_by_category(
    frame: pd.DataFrame,
    *,
    categories: Sequence[str],
    min_year: int = STUDY.study_start_year,
    max_year: int,
    value_column: str | None = None,
    category_column: str = "manuscript_category",
    group_columns: Sequence[str] = GROUP_COLUMNS,
    year_column: str = "pub_year",
) -> pd.DataFrame:
    """Aggregate publication counts or citation sums and their composition."""

    required = {*group_columns, category_column, year_column}
    if value_column is not None:
        required.add(value_column)
    require_columns(frame, required)

    subset = frame.loc[frame[year_column].between(min_year, max_year)].copy()
    subset[category_column] = subset[category_column].astype("string").str.lower()
    subset = subset.loc[subset[category_column].isin(categories)]
    grouped = subset.groupby([*group_columns, category_column], dropna=False)
    if value_column is None:
        totals = grouped.size().rename("value").reset_index()
    else:
        totals = grouped[value_column].sum().rename("value").reset_index()

    wide = totals.pivot(
        index=list(group_columns),
        columns=category_column,
        values="value",
    ).fillna(0)
    for category in categories:
        if category not in wide:
            wide[category] = 0

    wide["total"] = wide[list(categories)].sum(axis=1)
    nora_columns = [category for category in NORA_CATEGORIES if category in categories]
    wide["nora"] = wide[nora_columns].sum(axis=1)
    for category in categories:
        wide[f"{category}_proportion"] = wide[category] / wide["total"]
    wide["nora_proportion"] = wide["nora"] / wide["total"]

    result = wide.reset_index()
    result.columns.name = None
    return result


def format_sum_percentage(
    values: pd.Series,
    totals: pd.Series,
    *,
    decimals: int = 2,
) -> pd.Series:
    """Format counts as `sum (percentage)` without a percent sign."""

    numeric_values = values.fillna(0).astype(float)
    numeric_totals = totals.replace(0, np.nan).astype(float)
    percentages = numeric_values / numeric_totals * 100
    value_text = numeric_values.round(0).map(lambda value: f"{value:,.0f}")
    percentage_text = percentages.map(
        lambda value: "NA" if pd.isna(value) else f"{value:.{decimals}f}"
    )
    return value_text + " (" + percentage_text + ")"


def format_composition_table(
    composition: pd.DataFrame,
    *,
    include_non_citable: bool,
    group_columns: Sequence[str] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Select and format the manuscript's ORA/NORA composition columns."""

    required = {
        *group_columns,
        "total",
        ORA,
        META_ANALYSIS,
        REVIEW,
        GSS,
        "nora",
    }
    if include_non_citable:
        required.add(NON_CITABLE)
    require_columns(composition, required, label="composition data")

    result = composition[list(group_columns)].copy()
    source_columns = {
        "ora": ORA,
        "nora": "nora",
        META_ANALYSIS: META_ANALYSIS,
        REVIEW: REVIEW,
        GSS: GSS,
    }
    if include_non_citable:
        source_columns[NON_CITABLE] = NON_CITABLE
    for output_column, source_column in source_columns.items():
        result[output_column] = format_sum_percentage(
            composition[source_column], composition["total"]
        )
    return result
