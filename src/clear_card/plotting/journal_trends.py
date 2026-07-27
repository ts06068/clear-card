from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

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

SummaryStatistic = Literal["mean", "median"]


def summarize_journal_trends(
    frame: pd.DataFrame,
    *,
    value_column: str,
    category_values: Sequence[str],
    year_range: tuple[int, int],
    journals: Sequence[str],
    journal_column: str = "target_journal",
    year_column: str = "target_pub_year",
    category_column: str = "manuscript_category",
) -> pd.DataFrame:
    """Aggregate annual journal values using the manuscript summary statistics."""

    _require_columns(
        frame,
        {value_column, journal_column, year_column, category_column},
        "journal trend data",
    )
    data = frame.copy()
    data[year_column] = _numeric(data[year_column])
    data[value_column] = _numeric(data[value_column])
    categories = {str(value).lower() for value in category_values}
    mask = (
        data[year_column].between(*year_range)
        & data[journal_column].isin(journals)
        & data[category_column].astype("string").str.lower().isin(categories)
    )
    # Compute one annual summary for both estimates and intervals.
    return (
        data.loc[mask]
        .groupby([journal_column, year_column], observed=True)[value_column]
        .agg(
            total="sum",
            mean="mean",
            std="std",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
        .sort_values([journal_column, year_column])
        .reset_index(drop=True)
    )


def _lighten(color: str, factor: float = 0.8) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple((1.0 - factor) + factor * rgb)


def _draw_interval_series(
    axis: Axes,
    x: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str,
    y_limits: tuple[float, float],
    label: str,
) -> None:
    bottom, top = y_limits
    span = max(top - bottom, 1e-9)
    bottom_inset, top_inset = bottom + 0.02 * span, top - 0.02 * span
    for x_value, center_value, low_value, high_value in zip(
        x, center, lower, upper, strict=True
    ):
        if not np.all(np.isfinite([center_value, low_value, high_value])):
            continue
        if low_value >= bottom:
            axis.vlines(
                x_value, low_value, center_value, color=color, linewidth=0.8, alpha=0.35
            )
            axis.hlines(
                low_value,
                x_value - 0.05,
                x_value + 0.05,
                color=color,
                linewidth=0.8,
                alpha=0.35,
            )
        else:
            axis.annotate(
                "",
                xy=(x_value, bottom_inset),
                xytext=(x_value, center_value),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 0.8,
                    "alpha": 0.35,
                },
            )
        if high_value <= top:
            axis.vlines(
                x_value,
                center_value,
                high_value,
                color=color,
                linewidth=0.8,
                alpha=0.35,
            )
            axis.hlines(
                high_value,
                x_value - 0.05,
                x_value + 0.05,
                color=color,
                linewidth=0.8,
                alpha=0.35,
            )
        else:
            axis.annotate(
                "",
                xy=(x_value, top_inset),
                xytext=(x_value, center_value),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 0.8,
                    "alpha": 0.35,
                },
            )
    axis.plot(
        x,
        center,
        "-s",
        color=color,
        linewidth=1.4,
        markersize=5.5,
        markeredgewidth=0,
        label=label,
        zorder=2,
    )


def _set_year_axis(axis: Axes, year_range: tuple[int, int]) -> None:
    start, end = year_range
    years = np.arange(start, end + 1)
    axis.set_xticks(years)
    axis.set_xticklabels(
        [str(year) if year % 2 == 0 or year == end else "" for year in years]
    )
    axis.xaxis.set_minor_locator(NullLocator())


def plot_journal_trend(
    frame: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
    *,
    value_column: str,
    category_values: Sequence[str],
    year_range: tuple[int, int],
    statistic: SummaryStatistic,
    y_label: str,
    y_limits: tuple[float, float],
    journals: Sequence[str] | None = None,
    impact_factors: pd.DataFrame | None = None,
    impact_factor_column: str = "impact_factor",
    impact_factor_year_column: str = "target_pub_year",
    impact_factor_journal_column: str = "target_journal",
    impact_factor_limits: tuple[float, float] = (5.0, 50.0),
    y_decimals: int | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Plot annual mean±SD or median (IQR), optionally beside impact factor."""

    selected = _selected_journals(config, journals or config.primary_journals)
    summary = summarize_journal_trends(
        frame,
        value_column=value_column,
        category_values=category_values,
        year_range=year_range,
        journals=selected,
    )
    if summary.empty:
        raise ValueError("No observations remain for the requested journal trend.")

    with manuscript_style(config):
        figure, left_axis = plt.subplots(figsize=(6, 4))
        right_axis: Axes | None = None
        if impact_factors is not None:
            _require_columns(
                impact_factors,
                {
                    impact_factor_column,
                    impact_factor_year_column,
                    impact_factor_journal_column,
                },
                "impact factor data",
            )
            right_axis = left_axis.twinx()
            right_axis.spines["right"].set_visible(True)

        for journal in selected:
            journal_summary = summary[summary["target_journal"].eq(journal)]
            if journal_summary.empty:
                continue
            x = (
                journal_summary["target_pub_year"].to_numpy(dtype=float)
                + config.journal_offsets[journal]
            )
            if statistic == "mean":
                center = journal_summary["mean"].to_numpy(dtype=float)
                spread = journal_summary["std"].to_numpy(dtype=float)
                lower, upper = center - spread, center + spread
            elif statistic == "median":
                center = journal_summary["median"].to_numpy(dtype=float)
                lower = journal_summary["q25"].to_numpy(dtype=float)
                upper = journal_summary["q75"].to_numpy(dtype=float)
            else:
                raise ValueError("statistic must be 'mean' or 'median'.")
            _draw_interval_series(
                left_axis,
                x,
                center,
                lower,
                upper,
                color=config.journal_colors[journal],
                y_limits=y_limits,
                label=config.journal_labels[journal],
            )

            if right_axis is not None and impact_factors is not None:
                metric = impact_factors[
                    impact_factors[impact_factor_journal_column].eq(journal)
                    & _numeric(impact_factors[impact_factor_year_column]).between(
                        *year_range
                    )
                ].copy()
                metric[impact_factor_year_column] = _numeric(
                    metric[impact_factor_year_column]
                )
                metric[impact_factor_column] = _numeric(metric[impact_factor_column])
                metric = metric.dropna(
                    subset=[impact_factor_year_column, impact_factor_column]
                )
                metric = metric[metric[impact_factor_column].gt(0)].sort_values(
                    impact_factor_year_column
                )
                right_axis.plot(
                    metric[impact_factor_year_column].to_numpy(dtype=float)
                    + config.journal_offsets[journal],
                    metric[impact_factor_column].to_numpy(dtype=float),
                    linestyle="--",
                    linewidth=1.2,
                    color=_lighten(config.journal_colors[journal]),
                )

        left_axis.set_xlabel("Publication year")
        left_axis.set_ylabel(y_label)
        left_axis.set_ylim(*y_limits)
        left_axis.set_xlim(year_range[0] - 0.6, year_range[1] + 0.6)
        _set_year_axis(left_axis, year_range)
        if y_decimals is not None:
            left_axis.yaxis.set_major_formatter(FormatStrFormatter(f"%.{y_decimals}f"))
        left_axis.legend(loc="upper left", frameon=False, handlelength=1.6)

        if right_axis is not None:
            right_axis.set_ylabel("Journal Impact Factor")
            right_axis.set_ylim(*impact_factor_limits)
            right_axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            lag = config.impact_factor_lag
            secondary = left_axis.secondary_xaxis(
                "bottom", functions=(lambda year: year + lag, lambda year: year - lag)
            )
            secondary.spines["bottom"].set_position(("outward", 30))
            secondary.set_xlabel("Journal Impact Factor year")
            if_years = np.arange(year_range[0] + lag, year_range[1] + lag + 1)
            secondary.set_xticks(if_years)
            secondary.set_xticklabels(
                [
                    str(year) if year % 2 == 0 or year == if_years[-1] else ""
                    for year in if_years
                ]
            )
            secondary.xaxis.set_minor_locator(NullLocator())

        figure.tight_layout(pad=0.6)
        save_figure(figure, output_path, config)
    return figure, summary
