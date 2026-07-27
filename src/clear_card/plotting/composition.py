from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from clear_card.plotting.style import (
    FigureConfig,
    _numeric,
    _require_columns,
    _selected_journals,
    manuscript_style,
    save_figure,
)


def summarize_composition(
    frame: pd.DataFrame,
    *,
    categories: Sequence[str],
    journals: Sequence[str],
    year_range: tuple[int, int],
    weight_column: str | None = None,
    journal_column: str = "target_journal",
    year_column: str = "target_pub_year",
    category_column: str = "manuscript_category",
) -> pd.DataFrame:
    """Calculate journal-year category counts or weighted citation contributions."""

    required = {journal_column, year_column, category_column}
    if weight_column is not None:
        required.add(weight_column)
    _require_columns(frame, required, "composition data")
    category_order = tuple(str(category).lower() for category in categories)
    data = frame.copy()
    data[year_column] = _numeric(data[year_column])
    data[category_column] = data[category_column].astype("string").str.lower()
    data = data[
        data[year_column].between(*year_range)
        & data[journal_column].isin(journals)
        & data[category_column].isin(category_order)
    ].copy()
    if weight_column is None:
        grouped = data.groupby(
            [journal_column, year_column, category_column], observed=True
        ).size()
    else:
        data[weight_column] = _numeric(data[weight_column])
        grouped = data.groupby(
            [journal_column, year_column, category_column], observed=True
        )[weight_column].sum()

    # A complete grid keeps missing journal-years visible and aligned across panels.
    grid = pd.MultiIndex.from_product(
        [journals, range(year_range[0], year_range[1] + 1)],
        names=[journal_column, year_column],
    )
    wide = grouped.unstack(fill_value=0).reindex(
        index=grid, columns=category_order, fill_value=0
    )
    result = wide.rename(columns=lambda category: f"{category}_amount").reset_index()
    amount_columns = [f"{category}_amount" for category in category_order]
    result["total_amount"] = result[amount_columns].sum(axis=1)
    for category in category_order:
        result[f"{category}_share"] = (
            result[f"{category}_amount"].div(result["total_amount"].replace(0, np.nan))
            * 100.0
        )
    return result


def _plot_composition_lines(
    axis: Axes,
    summary: pd.DataFrame,
    config: FigureConfig,
    journals: Sequence[str],
    categories: Sequence[str],
    *,
    line_label: bool,
) -> np.ndarray:
    all_values: list[np.ndarray] = []
    for journal in journals:
        data = summary[summary["target_journal"].eq(journal)]
        values = sum(data[f"{category}_share"] for category in categories).to_numpy(
            dtype=float
        )
        all_values.append(values)
        axis.plot(
            data["target_pub_year"].to_numpy(dtype=float)
            + config.journal_offsets[journal],
            values,
            "-s",
            color=config.journal_colors[journal],
            linewidth=1.4 if line_label else 1.0,
            markersize=5.5 if line_label else 3.6,
            markeredgewidth=0,
            label=config.journal_labels[journal] if line_label else None,
        )
    return np.concatenate(all_values) if all_values else np.array([], dtype=float)


def plot_composition_trends(
    frame: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
    *,
    categories: Sequence[str],
    focus_categories: Sequence[str],
    year_range: tuple[int, int] = (2008, 2022),
    weight_column: str | None = None,
    journals: Sequence[str] | None = None,
    breakdown_categories: Sequence[str] = (),
    category_labels: Mapping[str, str] | None = None,
    y_label: str | None = None,
    y_limits: tuple[float, float] | None = None,
    legend_location: str = "best",
    show_volume_inset: bool = False,
) -> tuple[Figure, pd.DataFrame]:
    """Plot category shares of publication volume or 2-year citations over time."""

    selected = _selected_journals(config, journals or config.primary_journals)
    normalized_categories = tuple(str(category).lower() for category in categories)
    focus = tuple(str(category).lower() for category in focus_categories)
    breakdown = tuple(str(category).lower() for category in breakdown_categories)
    invalid = set(focus + breakdown).difference(normalized_categories)
    if invalid:
        raise ValueError(
            f"Composition categories are outside the denominator: {', '.join(sorted(invalid))}"
        )
    summary = summarize_composition(
        frame,
        categories=normalized_categories,
        journals=selected,
        year_range=year_range,
        weight_column=weight_column,
    )
    labels = dict(category_labels or {})

    with manuscript_style(config):
        if breakdown:
            rows = int(np.ceil(len(breakdown) / 2))
            figure = plt.figure(figsize=(12, max(4, rows * 2)))
            outer = figure.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.24)
            main_axis = figure.add_subplot(outer[0, 0])
            subgrid = outer[0, 1].subgridspec(rows, 2, hspace=0.5, wspace=0.25)
            detail_axes = [
                figure.add_subplot(subgrid[index // 2, index % 2])
                for index in range(len(breakdown))
            ]
        else:
            figure, main_axis = plt.subplots(figsize=(6, 4))
            detail_axes = []

        main_values = _plot_composition_lines(
            main_axis, summary, config, selected, focus, line_label=True
        )
        main_axis.set_xlabel("Publication year")
        main_axis.set_ylabel(
            y_label
            or (
                "Proportion of publications (%)"
                if weight_column is None
                else "Proportion of 2-year citations (%)"
            )
        )
        main_axis.set_xlim(year_range[0] - 0.6, year_range[1] + 0.6)
        if y_limits is None:
            top = (
                max(1.0, float(np.nanmax(main_values)) * 1.05)
                if main_values.size
                else 1.0
            )
            main_axis.set_ylim(0, top)
        else:
            main_axis.set_ylim(*y_limits)
        main_axis.set_xticks(np.arange(year_range[0], year_range[1] + 1, 2))
        main_axis.xaxis.set_minor_locator(MultipleLocator(1))
        main_axis.yaxis.set_minor_locator(NullLocator())
        main_axis.legend(loc=legend_location, frameon=False, handlelength=1.6)

        for axis, category in zip(detail_axes, breakdown, strict=True):
            values = _plot_composition_lines(
                axis, summary, config, selected, (category,), line_label=False
            )
            axis.set_title(
                labels.get(category, category.replace("_", " ").title()),
                fontsize=8,
                pad=2,
            )
            axis.set_xlim(year_range[0] - 0.6, year_range[1] + 0.6)
            top = max(1.0, float(np.nanmax(values)) * 1.05) if values.size else 1.0
            axis.set_ylim(0, top)
            major_years = np.array([2008, 2012, 2016, 2020, year_range[1]])
            axis.set_xticks(
                np.unique(
                    major_years[
                        (major_years >= year_range[0]) & (major_years <= year_range[1])
                    ]
                )
            )
            axis.xaxis.set_minor_locator(MultipleLocator(1))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.tick_params(labelsize=7, length=2.5)

        if show_volume_inset:
            volume_axis = inset_axes(
                main_axis,
                width="35%",
                height="30%",
                loc="lower left",
                bbox_to_anchor=(0.1, 0.1, 1, 1),
                bbox_transform=main_axis.transAxes,
                borderpad=0,
            )
            maximum = 0.0
            for journal in selected:
                data = summary[summary["target_journal"].eq(journal)]
                amount = sum(data[f"{category}_amount"] for category in focus).to_numpy(
                    dtype=float
                )
                maximum = max(maximum, float(np.nanmax(amount)))
                volume_axis.plot(
                    data["target_pub_year"].to_numpy(dtype=float)
                    + config.journal_offsets[journal],
                    amount,
                    "-s",
                    markersize=3.2,
                    linewidth=1.0,
                    color=config.journal_colors[journal],
                )
            volume_axis.set_ylabel(
                "Publication volume" if weight_column is None else "Citation volume",
                fontsize=7,
            )
            volume_axis.set_xlim(year_range[0] - 0.5, year_range[1] + 0.5)
            volume_axis.set_ylim(0, max(1.0, maximum * 1.12))
            volume_axis.set_xticks([2008, 2012, 2016, 2020, year_range[1]])
            volume_axis.xaxis.set_minor_locator(MultipleLocator(1))
            volume_axis.tick_params(labelsize=6, length=2)

        figure.tight_layout(pad=0.6)
        save_figure(figure, output_path, config)
    return figure, summary
