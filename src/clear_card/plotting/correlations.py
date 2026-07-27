from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, NullLocator

from clear_card.plotting.style import (
    FigureConfig,
    _numeric,
    _require_columns,
    _selected_journals,
    manuscript_style,
    save_figure,
)
from clear_card.statistics import (
    PairwiseCorrelationComparison,
    format_p_value,
    pairwise_correlation_comparison,
    safe_correlation,
)


def _correlation_annotation(p_value: float) -> str:
    formatted = format_p_value(p_value)
    return f"p < {formatted[1:]}" if formatted.startswith("<") else f"p = {formatted}"


def _correlation_values(
    frame: pd.DataFrame,
    metric_column: str,
    outcome_column: str,
    *,
    log_metric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric = _numeric(frame[metric_column]).to_numpy(dtype=float)
    outcome = _numeric(frame[outcome_column]).to_numpy(dtype=float)
    valid = np.isfinite(metric) & np.isfinite(outcome)
    if log_metric:
        valid &= metric > 0
    plotted = metric[valid]
    # Preserve original units on the axis while correlating selected metrics on log scale.
    correlated = np.log(plotted) if log_metric else plotted
    return plotted, correlated, outcome[valid]


def _format_correlation_line(
    label: str,
    correlations: Mapping[str, tuple[float, float]],
    primary_method: str,
) -> str:
    if {"pearson", "spearman"}.issubset(correlations):
        pearson = correlations["pearson"][0]
        spearman = correlations["spearman"][0]
        return f"{label}: ρ = {spearman:.2f}, r = {pearson:.2f}"
    coefficient, p_value = correlations[primary_method]
    return f"{label}: {coefficient:.2f} ({_correlation_annotation(p_value)})"


def _draw_correlation_heatmap(
    axis: Axes,
    comparison: PairwiseCorrelationComparison,
    metric_columns: Sequence[str],
    metric_labels: Mapping[str, str],
    *,
    outcome_label: str,
    method: str,
) -> None:
    p_values = comparison.p_value.reindex(index=metric_columns, columns=metric_columns)
    matrix = p_values.to_numpy(dtype=float)
    display = matrix.copy()
    display[np.triu_indices(len(metric_columns), k=0)] = np.nan
    color_map = mpl.colormaps["Blues_r"].copy()
    color_map.set_bad(color="white")
    finite = matrix[np.isfinite(matrix)]
    maximum = max(float(np.max(finite)), 0.05) if finite.size else 0.05
    axis.imshow(
        display,
        cmap=color_map,
        vmin=0,
        vmax=maximum,
        interpolation="none",
        aspect="auto",
    )
    labels = [metric_labels[column] for column in metric_columns]
    axis.set_title(
        "P-value for pairwise comparison of absolute "
        f"{method.title()} correlation with {outcome_label.lower()}",
        fontsize=12,
        pad=8,
    )
    axis.set_xticks(np.arange(len(metric_columns)))
    axis.set_yticks(np.arange(len(metric_columns)))
    axis.set_xticklabels(
        labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9
    )
    axis.set_yticklabels(labels, fontsize=9)
    axis.tick_params(axis="both", which="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row in range(len(metric_columns)):
        for column in range(len(metric_columns)):
            value = matrix[row, column]
            text = format_p_value(value) if row > column and np.isfinite(value) else "—"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if np.isfinite(value) and value < 0.05 else "normal",
                color=(
                    "white"
                    if row > column and np.isfinite(value) and value < 0.05
                    else "#7a7a7a"
                    if text == "—"
                    else "black"
                ),
            )
    axis.hlines(
        np.arange(-0.5, len(metric_columns), 1),
        -0.5,
        len(metric_columns) - 0.5,
        colors="black",
        linewidth=0.8,
    )


def _prepare_correlation_analysis(
    frame: pd.DataFrame,
    config: FigureConfig,
    *,
    metric_labels: Mapping[str, str],
    outcome_column: str,
    method: Literal["pearson", "spearman"],
    secondary_method: Literal["pearson", "spearman"] | None,
    year_range: tuple[int, int],
    journals: Sequence[str] | None,
    journal_column: str,
    year_column: str,
    log_columns: Collection[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    pd.DataFrame,
    PairwiseCorrelationComparison,
    PairwiseCorrelationComparison | None,
]:
    metric_columns = tuple(metric_labels)
    _require_columns(
        frame,
        {journal_column, year_column, outcome_column, *metric_columns},
        "correlation data",
    )
    if not metric_columns:
        raise ValueError("At least one correlation metric is required.")
    selected = _selected_journals(config, journals or config.primary_journals)
    unknown_logs = set(log_columns).difference(metric_columns)
    if unknown_logs:
        raise ValueError(
            f"log_columns are not plotted metrics: {', '.join(sorted(unknown_logs))}"
        )
    if secondary_method == method:
        raise ValueError("secondary_method must differ from method.")

    data = frame.copy()
    data[year_column] = _numeric(data[year_column])
    data = data[
        data[year_column].between(*year_range) & data[journal_column].isin(selected)
    ].copy()
    if data.empty:
        raise ValueError("No observations remain for the requested correlation period.")

    comparison = pairwise_correlation_comparison(
        data,
        metric_columns,
        outcome_column,
        method=method,
        n_boot=bootstrap_iterations,
        seed=bootstrap_seed,
        log_columns=log_columns,
    )
    secondary_comparison = (
        pairwise_correlation_comparison(
            data,
            metric_columns,
            outcome_column,
            method=secondary_method,
            n_boot=bootstrap_iterations,
            seed=bootstrap_seed + 10_000,
            log_columns=log_columns,
        )
        if secondary_method is not None
        else None
    )
    return metric_columns, selected, data, comparison, secondary_comparison


def _create_correlation_layout(
    metric_count: int,
    *,
    panel_label: str | None,
    secondary_method: Literal["pearson", "spearman"] | None,
) -> tuple[Figure, list[Axes], Axes, list[Axes]]:
    figure = plt.figure(figsize=(14.8, 7.0))
    top = figure.add_gridspec(
        1,
        metric_count + 1,
        left=0.07,
        right=0.98,
        top=0.94 if panel_label else 0.96,
        bottom=0.34,
        width_ratios=[*[1.0] * metric_count, 0.55],
        wspace=0.14,
    )
    axes = [figure.add_subplot(top[0, 0])]
    axes.extend(
        figure.add_subplot(top[0, index], sharey=axes[0])
        for index in range(1, metric_count)
    )
    legend_axis = figure.add_subplot(top[0, -1])
    legend_axis.axis("off")
    heat_axes = (
        [figure.add_axes([0.37, 0.07, 0.26, 0.17])]
        if secondary_method is None
        else [
            figure.add_axes([0.18, 0.05, 0.27, 0.19]),
            figure.add_axes([0.56, 0.05, 0.27, 0.19]),
        ]
    )
    return figure, axes, legend_axis, heat_axes


def _draw_journal_correlation(
    axis: Axes,
    journal_data: pd.DataFrame,
    config: FigureConfig,
    *,
    journal: str,
    metric_column: str,
    outcome_column: str,
    year_column: str,
    log_metric: bool,
    method: Literal["pearson", "spearman"],
    secondary_method: Literal["pearson", "spearman"] | None,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        dict[str, tuple[float, float]],
    ]
    | None
):
    x_plot, x_correlation, outcome = _correlation_values(
        journal_data,
        metric_column,
        outcome_column,
        log_metric=log_metric,
    )
    if x_plot.size < 2:
        return None

    years = _numeric(journal_data[year_column]).to_numpy(dtype=float)
    raw_metric = _numeric(journal_data[metric_column]).to_numpy(dtype=float)
    raw_outcome = _numeric(journal_data[outcome_column]).to_numpy(dtype=float)
    valid = np.isfinite(years) & np.isfinite(raw_metric) & np.isfinite(raw_outcome)
    if log_metric:
        valid &= raw_metric > 0
    years = years[valid]
    year_min, year_max = float(years.min()), float(years.max())
    alpha = (
        0.25 + 0.75 * (years - year_min) / (year_max - year_min)
        if year_max > year_min
        else np.full(years.size, 0.8)
    )
    rgb = to_rgb(config.journal_colors[journal])
    colors = np.column_stack(
        [
            np.full(x_plot.size, rgb[0]),
            np.full(x_plot.size, rgb[1]),
            np.full(x_plot.size, rgb[2]),
            alpha,
        ]
    )
    axis.scatter(x_plot, outcome, s=24, color=colors, edgecolor="none", zorder=2)

    correlations = {method: safe_correlation(x_correlation, outcome, method)}
    if secondary_method is not None:
        correlations[secondary_method] = safe_correlation(
            x_correlation, outcome, secondary_method
        )
    return x_correlation, outcome, correlations


def _correlation_result_rows(
    metric_column: str,
    journal: str,
    correlations: Mapping[str, tuple[float, float]],
    *,
    sample_size: int,
    year_range: tuple[int, int],
) -> list[dict[str, object]]:
    return [
        {
            "metric": metric_column,
            "journal": journal,
            "coefficient": coefficient,
            "p_value": p_value,
            "sample_size": sample_size,
            "method": correlation_method,
            "start_year": year_range[0],
            "end_year": year_range[1],
        }
        for correlation_method, (coefficient, p_value) in correlations.items()
    ]


def _draw_correlation_annotations(
    axis: Axes,
    overall: Mapping[str, tuple[float, float]],
    journal_annotations: Sequence[tuple[str, Mapping[str, tuple[float, float]]]],
    *,
    method: str,
) -> None:
    text_y = 0.95
    if overall:
        axis.text(
            0.05,
            text_y,
            _format_correlation_line("Overall", overall, method),
            transform=axis.transAxes,
            va="top",
            fontsize=12.5,
            fontweight="bold",
        )
        text_y -= 0.11
    for label, correlations in journal_annotations:
        axis.text(
            0.05,
            text_y,
            _format_correlation_line(label, correlations, method),
            transform=axis.transAxes,
            va="top",
            fontsize=10,
        )
        text_y -= 0.085


def _style_correlation_axis(
    axis: Axes,
    metric_label: str,
    *,
    log_metric: bool,
) -> None:
    if log_metric:
        axis.set_xscale("log")
    axis.set_xlabel(metric_label, fontsize=13)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    axis.minorticks_off()


def _draw_metric_panel(
    axis: Axes,
    data: pd.DataFrame,
    config: FigureConfig,
    *,
    metric_column: str,
    metric_label: str,
    outcome_column: str,
    method: Literal["pearson", "spearman"],
    secondary_method: Literal["pearson", "spearman"] | None,
    selected_journals: Sequence[str],
    journal_column: str,
    year_column: str,
    log_metric: bool,
    year_range: tuple[int, int],
) -> list[dict[str, object]]:
    axis.set_box_aspect(1)
    overall_metric: list[np.ndarray] = []
    overall_outcome: list[np.ndarray] = []
    annotations: list[tuple[str, Mapping[str, tuple[float, float]]]] = []
    rows: list[dict[str, object]] = []

    for journal in selected_journals:
        result = _draw_journal_correlation(
            axis,
            data[data[journal_column].eq(journal)],
            config,
            journal=journal,
            metric_column=metric_column,
            outcome_column=outcome_column,
            year_column=year_column,
            log_metric=log_metric,
            method=method,
            secondary_method=secondary_method,
        )
        if result is None:
            continue
        x_correlation, outcome, correlations = result
        annotations.append((config.journal_labels[journal], correlations))
        overall_metric.append(x_correlation)
        overall_outcome.append(outcome)
        rows.extend(
            _correlation_result_rows(
                metric_column,
                journal,
                correlations,
                sample_size=int(x_correlation.size),
                year_range=year_range,
            )
        )

    overall_correlations: dict[str, tuple[float, float]] = {}
    if overall_metric:
        all_metric = np.concatenate(overall_metric)
        all_outcome = np.concatenate(overall_outcome)
        overall_correlations[method] = safe_correlation(all_metric, all_outcome, method)
        if secondary_method is not None:
            overall_correlations[secondary_method] = safe_correlation(
                all_metric, all_outcome, secondary_method
            )
        rows.extend(
            _correlation_result_rows(
                metric_column,
                "Overall",
                overall_correlations,
                sample_size=int(all_metric.size),
                year_range=year_range,
            )
        )

    _draw_correlation_annotations(
        axis,
        overall_correlations,
        annotations,
        method=method,
    )
    _style_correlation_axis(axis, metric_label, log_metric=log_metric)
    return rows


def _draw_correlation_legend(
    axis: Axes,
    config: FigureConfig,
    journals: Sequence[str],
) -> None:
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=config.journal_colors[journal],
            marker="o",
            linestyle="none",
            markersize=5.5,
            label=config.journal_labels[journal],
        )
        for journal in journals
    ]
    axis.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.5,
        fontsize=12,
    )


def plot_correlation_panels(
    frame: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
    *,
    metric_labels: Mapping[str, str],
    outcome_column: str = "impact_factor",
    outcome_label: str = "Journal impact factor",
    method: Literal["pearson", "spearman"] = "spearman",
    secondary_method: Literal["pearson", "spearman"] | None = None,
    year_range: tuple[int, int] = (2008, 2019),
    journals: Sequence[str] | None = None,
    journal_column: str = "target_journal",
    year_column: str = "target_pub_year",
    log_columns: Collection[str] = (),
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 42,
    panel_label: str | None = None,
) -> tuple[Figure, pd.DataFrame, PairwiseCorrelationComparison]:
    """Plot journal-metric correlations and paired bootstrap comparisons."""

    (
        metric_columns,
        selected,
        data,
        comparison,
        secondary_comparison,
    ) = _prepare_correlation_analysis(
        frame,
        config,
        metric_labels=metric_labels,
        outcome_column=outcome_column,
        method=method,
        secondary_method=secondary_method,
        year_range=year_range,
        journals=journals,
        journal_column=journal_column,
        year_column=year_column,
        log_columns=log_columns,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    with manuscript_style(config):
        figure, axes, legend_axis, heat_axes = _create_correlation_layout(
            len(metric_columns),
            panel_label=panel_label,
            secondary_method=secondary_method,
        )
        result_rows: list[dict[str, object]] = []
        for axis, metric_column in zip(axes, metric_columns, strict=True):
            result_rows.extend(
                _draw_metric_panel(
                    axis,
                    data,
                    config,
                    metric_column=metric_column,
                    metric_label=metric_labels[metric_column],
                    outcome_column=outcome_column,
                    method=method,
                    secondary_method=secondary_method,
                    selected_journals=selected,
                    journal_column=journal_column,
                    year_column=year_column,
                    log_metric=metric_column in log_columns,
                    year_range=year_range,
                )
            )

        axes[0].set_ylabel(outcome_label, fontsize=13, labelpad=8)
        _draw_correlation_legend(legend_axis, config, selected)
        _draw_correlation_heatmap(
            heat_axes[0],
            comparison,
            metric_columns,
            metric_labels,
            outcome_label=outcome_label,
            method=method,
        )
        if secondary_comparison is not None and secondary_method is not None:
            _draw_correlation_heatmap(
                heat_axes[1],
                secondary_comparison,
                metric_columns,
                metric_labels,
                outcome_label=outcome_label,
                method=secondary_method,
            )
        if panel_label:
            figure.suptitle(
                f"[{panel_label}] Publication years {year_range[0]}–{year_range[1]}",
                y=0.995,
            )
        save_figure(figure, output_path, config)
    return figure, pd.DataFrame(result_rows), comparison


def plot_dual_correlation_panels(
    frame: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
    *,
    metric_labels: Mapping[str, str],
    outcome_column: str,
    outcome_label: str,
    year_range: tuple[int, int],
    journals: Sequence[str],
    journal_column: str = "target_journal",
    year_column: str = "target_pub_year",
    log_columns: Collection[str] = (),
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 42,
    panel_label: str | None = None,
) -> tuple[Figure, pd.DataFrame, PairwiseCorrelationComparison]:
    """Plot Pearson and Spearman results together for a supplemental panel."""

    return plot_correlation_panels(
        frame,
        config,
        output_path,
        metric_labels=metric_labels,
        outcome_column=outcome_column,
        outcome_label=outcome_label,
        method="pearson",
        secondary_method="spearman",
        year_range=year_range,
        journals=journals,
        journal_column=journal_column,
        year_column=year_column,
        log_columns=log_columns,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        panel_label=panel_label,
    )
