"""Correlation, bootstrap, and journal-metric analysis."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr

from clear_card.config import CITABLE_CATEGORIES, ORA, STUDY
from clear_card.summaries import (
    GROUP_COLUMNS,
    composition_by_category,
    require_columns,
    summarize_by_group,
)

CorrelationMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class CorrelationDifference:
    """Bootstrap comparison of two absolute correlations with one outcome."""

    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


@dataclass(frozen=True)
class PairwiseCorrelationComparison:
    """Pairwise bootstrap results indexed by the compared metric names."""

    p_value: pd.DataFrame
    difference: pd.DataFrame
    ci_low: pd.DataFrame
    ci_high: pd.DataFrame
    n: pd.DataFrame


def format_p_value(p_value: float, *, threshold: float = 0.001) -> str:
    """Format a p-value for plot and table annotations."""

    if not np.isfinite(p_value):
        return "NA"
    if p_value < threshold:
        return f"<{threshold:.3f}"
    return f"{p_value:.3f}"


def safe_correlation(
    x: ArrayLike,
    y: ArrayLike,
    method: CorrelationMethod | str = "pearson",
) -> tuple[float, float]:
    """Return a finite-pair Pearson or Spearman correlation and p-value."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if len(x_values) < 2 or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return np.nan, np.nan

    normalized_method = method.lower()
    if normalized_method == "pearson":
        coefficient, p_value = pearsonr(x_values, y_values)
    elif normalized_method == "spearman":
        coefficient, p_value = spearmanr(x_values, y_values)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return float(coefficient), float(p_value)


def dual_correlation(x: ArrayLike, y: ArrayLike) -> dict[str, tuple[float, float]]:
    """Return both Pearson and Spearman results for one pair of variables."""

    return {
        "pearson": safe_correlation(x, y, "pearson"),
        "spearman": safe_correlation(x, y, "spearman"),
    }


def bootstrap_absolute_correlation_difference(
    x1: ArrayLike,
    x2: ArrayLike,
    y: ArrayLike,
    *,
    method: CorrelationMethod | str = "spearman",
    n_boot: int = 10_000,
    seed: int = 42,
) -> CorrelationDifference:
    """Compare two absolute correlations using paired bootstrap resampling."""

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    x1_values = np.asarray(x1, dtype=float)
    x2_values = np.asarray(x2, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x1_values.shape != x2_values.shape or x1_values.shape != y_values.shape:
        raise ValueError("x1, x2, and y must have the same shape")

    valid = np.isfinite(x1_values) & np.isfinite(x2_values) & np.isfinite(y_values)
    x1_values = x1_values[valid]
    x2_values = x2_values[valid]
    y_values = y_values[valid]
    sample_size = len(y_values)
    empty = CorrelationDifference(np.nan, np.nan, np.nan, np.nan, sample_size)
    if sample_size < 4:
        return empty

    first_correlation, _ = safe_correlation(x1_values, y_values, method)
    second_correlation, _ = safe_correlation(x2_values, y_values, method)
    if not np.isfinite(first_correlation) or not np.isfinite(second_correlation):
        return empty
    observed = abs(first_correlation) - abs(second_correlation)

    generator = np.random.default_rng(seed)
    differences = np.full(n_boot, np.nan, dtype=float)
    for index in range(n_boot):
        # Shared row indices preserve dependence between the two correlations.
        sampled = generator.integers(0, sample_size, sample_size)
        first_bootstrap, _ = safe_correlation(
            x1_values[sampled], y_values[sampled], method
        )
        second_bootstrap, _ = safe_correlation(
            x2_values[sampled], y_values[sampled], method
        )
        if np.isfinite(first_bootstrap) and np.isfinite(second_bootstrap):
            differences[index] = abs(first_bootstrap) - abs(second_bootstrap)

    differences = differences[np.isfinite(differences)]
    if len(differences) < max(100, int(0.2 * n_boot)):
        return CorrelationDifference(observed, np.nan, np.nan, np.nan, sample_size)

    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    lower_tail = np.mean(differences <= 0)
    upper_tail = np.mean(differences >= 0)
    p_value = float(np.clip(2 * min(lower_tail, upper_tail), 0, 1))
    return CorrelationDifference(
        float(observed), float(ci_low), float(ci_high), p_value, sample_size
    )


def pairwise_correlation_comparison(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
    outcome_column: str,
    *,
    method: CorrelationMethod | str = "spearman",
    n_boot: int = 10_000,
    seed: int = 42,
    log_columns: Collection[str] = (),
) -> PairwiseCorrelationComparison:
    """Compare each pair of metric-outcome correlations by bootstrap."""

    require_columns(frame, {*metric_columns, outcome_column})
    unknown_log_columns = set(log_columns).difference(metric_columns)
    if unknown_log_columns:
        names = ", ".join(sorted(unknown_log_columns))
        raise ValueError(f"log_columns are not metrics: {names}")

    size = len(metric_columns)
    matrices = {
        name: np.full((size, size), np.nan, dtype=float)
        for name in ("p_value", "difference", "ci_low", "ci_high", "n")
    }
    for first_index, first_column in enumerate(metric_columns):
        for second_index in range(first_index + 1, size):
            second_column = metric_columns[second_index]
            pair = frame[[first_column, second_column, outcome_column]].apply(
                pd.to_numeric, errors="coerce"
            )
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if first_column in log_columns:
                pair = pair.loc[pair[first_column] > 0].copy()
                pair[first_column] = np.log(pair[first_column])
            if second_column in log_columns:
                pair = pair.loc[pair[second_column] > 0].copy()
                pair[second_column] = np.log(pair[second_column])

            comparison = bootstrap_absolute_correlation_difference(
                pair[first_column].to_numpy(),
                pair[second_column].to_numpy(),
                pair[outcome_column].to_numpy(),
                method=method,
                n_boot=n_boot,
                seed=seed + first_index * 100 + second_index,
            )
            values = {
                "p_value": comparison.p_value,
                "difference": comparison.difference,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "n": comparison.n,
            }
            for name, value in values.items():
                matrices[name][first_index, second_index] = value
                matrices[name][second_index, first_index] = value

    labels = list(metric_columns)
    data_frames = {
        name: pd.DataFrame(values, index=labels, columns=labels)
        for name, values in matrices.items()
    }
    return PairwiseCorrelationComparison(**data_frames)


def build_correlation_dataset(
    analysis: pd.DataFrame,
    journal_metrics: pd.DataFrame,
    *,
    metric_year_column: str = "metric_year",
    metric_name_column: str = "metric",
    metric_value_column: str = "value",
    impact_factor_metric: str = "impact_factor_2y",
    metric_year_lag: int = STUDY.journal_metric_lag_years,
    exclude_journals: Collection[str] = ("JAMA CARDIOLOGY",),
) -> pd.DataFrame:
    """Assemble the annual ORA measures used in manuscript correlations."""

    require_columns(
        analysis,
        {
            *GROUP_COLUMNS,
            "manuscript_category",
            "lambda",
            "citation_within_2_year",
        },
        label="analysis data",
    )
    require_columns(
        journal_metrics,
        {
            "target_journal",
            metric_year_column,
            metric_name_column,
            metric_value_column,
        },
        label="journal metrics",
    )

    mean_fitness = summarize_by_group(
        analysis,
        "lambda",
        categories=(ORA,),
        min_year=STUDY.study_start_year,
        max_year=STUDY.fitness_end_year,
    )[[*GROUP_COLUMNS, "mean"]].rename(columns={"mean": "mean_fitness"})
    mean_citations = summarize_by_group(
        analysis,
        "citation_within_2_year",
        categories=(ORA,),
        min_year=STUDY.study_start_year,
        max_year=STUDY.study_end_year,
    )[[*GROUP_COLUMNS, "mean"]].rename(columns={"mean": "mean_2ycit"})

    publication_composition = composition_by_category(
        analysis,
        categories=CITABLE_CATEGORIES,
        max_year=STUDY.study_end_year,
    )
    publication_composition["original_count"] = publication_composition[ORA]
    publication_composition["original_ratio"] = (
        publication_composition[f"{ORA}_proportion"] * 100
    )

    metrics = journal_metrics.loc[
        journal_metrics[metric_name_column].eq(impact_factor_metric),
        ["target_journal", metric_year_column, metric_value_column],
    ].copy()
    metrics[metric_year_column] = pd.to_numeric(
        metrics[metric_year_column], errors="raise"
    )
    metrics[metric_value_column] = pd.to_numeric(
        metrics[metric_value_column], errors="coerce"
    )
    metric_keys = ["target_journal", metric_year_column]
    if metrics.duplicated(metric_keys).any():
        raise ValueError("journal metrics contains duplicate impact-factor rows")
    metrics["target_pub_year"] = metrics[metric_year_column] - metric_year_lag
    metrics = metrics.rename(
        columns={
            "target_pub_year": "pub_year",
            metric_value_column: "impact_factor",
        }
    )[[*GROUP_COLUMNS, "impact_factor"]]

    result = (
        publication_composition[[*GROUP_COLUMNS, "original_ratio", "original_count"]]
        .merge(mean_citations, on=list(GROUP_COLUMNS), how="left")
        .merge(mean_fitness, on=list(GROUP_COLUMNS), how="left")
        .merge(metrics, on=list(GROUP_COLUMNS), how="left")
    )
    result = result.loc[result["target_journal"].isin(STUDY.all_journals)]
    if exclude_journals:
        result = result.loc[~result["target_journal"].isin(exclude_journals)]
    result = result.sort_values(list(GROUP_COLUMNS)).reset_index(drop=True)
    return result.rename(columns={"pub_year": "target_pub_year"})
