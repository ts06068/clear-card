from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator, StrMethodFormatter


from clear_card.plotting.style import (
    FigureConfig,
    _numeric,
    _require_columns,
    _selected_journals,
    manuscript_style,
    save_figure,
)


def prepare_validation_counts(
    scopus_frame: pd.DataFrame,
    icite_frame: pd.DataFrame,
    config: FigureConfig,
    *,
    category_values: Sequence[str],
    year_range: tuple[int, int] = (2008, 2022),
    id_column: str = "article_id",
    journal_column: str = "target_journal",
    year_column: str = "target_pub_year",
    category_column: str = "manuscript_category",
) -> pd.DataFrame:
    """Build Scopus, iCite, and overlap count tables for one category group."""

    required = {id_column, journal_column, year_column, category_column}
    _require_columns(scopus_frame, required, "Scopus validation data")
    _require_columns(icite_frame, required, "iCite validation data")
    categories = {str(category).lower() for category in category_values}

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        data[year_column] = _numeric(data[year_column])
        data[category_column] = data[category_column].astype("string").str.lower()
        return data[
            data[year_column].between(*year_range)
            & data[journal_column].isin(config.journals)
            & data[category_column].isin(categories)
        ].drop_duplicates(id_column)

    scopus = select(scopus_frame)
    icite = select(icite_frame)
    # Scopus supplies journal and year metadata for records present in both databases.
    overlap = (
        scopus[[id_column, journal_column, year_column]]
        .merge(icite[[id_column]], on=id_column, how="inner")
        .drop_duplicates(id_column)
    )
    grid = pd.MultiIndex.from_product(
        [range(year_range[0], year_range[1] + 1), config.journals],
        names=[year_column, journal_column],
    )
    tables: list[pd.DataFrame] = []
    for database, data in (("Scopus", scopus), ("iCite", icite), ("Overlap", overlap)):
        counts = (
            data.groupby([year_column, journal_column])
            .size()
            .reindex(grid, fill_value=0)
        )
        table = counts.rename("count").reset_index()
        table.insert(0, "database", database)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_validation_count_bars(
    counts: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
    *,
    volume_label: str,
    journals: Sequence[str] | None = None,
) -> Figure:
    """Plot journal-stacked annual counts for Scopus, iCite, and their overlap."""

    _require_columns(
        counts,
        {"database", "target_pub_year", "target_journal", "count"},
        "validation counts",
    )
    selected = _selected_journals(config, journals)
    database_order = ("Scopus", "iCite", "Overlap")
    years = np.sort(_numeric(counts["target_pub_year"]).dropna().unique()).astype(int)
    if years.size == 0:
        raise ValueError("Validation counts do not contain publication years.")

    with manuscript_style(config):
        figure, axes = plt.subplots(3, 1, figsize=(7.2, 6), sharex=True)
        x = np.arange(len(years))
        for axis, database in zip(axes, database_order, strict=True):
            bottom = np.zeros(len(years), dtype=float)
            subset = counts[counts["database"].eq(database)]
            for journal in selected:
                values = (
                    subset[subset["target_journal"].eq(journal)]
                    .set_index("target_pub_year")["count"]
                    .reindex(years, fill_value=0)
                    .to_numpy(dtype=float)
                )
                axis.bar(
                    x,
                    values,
                    bottom=bottom,
                    width=0.6,
                    color=config.journal_colors[journal],
                    label=config.journal_labels[journal],
                )
                bottom += values
            axis.set_title(database, fontsize=10, pad=4)
            axis.set_xlim(-0.6, len(x) - 0.4)
            axis.set_ylim(0, max(1.0, float(bottom.max())) * 1.12)
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_locator(NullLocator())

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([str(year) for year in years])
        axes[-1].set_xlabel("Publication year")
        figure.text(0.02, 0.5, volume_label, va="center", rotation="vertical")
        axes[0].legend(
            frameon=False,
            ncol=len(selected),
            loc="upper right",
            handlelength=1.2,
            bbox_to_anchor=(1.0, 1.35),
        )
        figure.tight_layout(pad=0.8, h_pad=0.6, rect=(0.05, 0.05, 1.0, 1.0))
        save_figure(figure, output_path, config)
    return figure


def prepare_citation_concordance(
    scopus_frame: pd.DataFrame,
    icite_frame: pd.DataFrame,
    config: FigureConfig,
    *,
    year_range: tuple[int, int] = (2008, 2022),
    id_column: str = "article_id",
    journal_column: str = "target_journal",
    year_column: str = "target_pub_year",
    category_column: str = "manuscript_category",
    citation_column: str = "citation_within_2_year",
) -> pd.DataFrame:
    """Join article-level Scopus and iCite 2-year citation counts."""

    scopus_required = {
        id_column,
        journal_column,
        year_column,
        category_column,
        citation_column,
    }
    icite_required = {id_column, citation_column}
    _require_columns(scopus_frame, scopus_required, "Scopus concordance data")
    _require_columns(icite_frame, icite_required, "iCite concordance data")
    scopus = scopus_frame[list(scopus_required)].drop_duplicates(id_column).copy()
    icite = icite_frame[[id_column, citation_column]].drop_duplicates(id_column).copy()
    concordance = scopus.merge(
        icite, on=id_column, how="inner", suffixes=("_scopus", "_icite")
    )
    concordance = concordance.rename(
        columns={
            f"{citation_column}_scopus": "citation_scopus",
            f"{citation_column}_icite": "citation_icite",
        }
    )
    concordance[year_column] = _numeric(concordance[year_column])
    concordance["citation_scopus"] = _numeric(concordance["citation_scopus"])
    concordance["citation_icite"] = _numeric(concordance["citation_icite"])
    concordance[category_column] = (
        concordance[category_column].astype("string").str.lower()
    )
    concordance = concordance[
        concordance[year_column].between(*year_range)
        & concordance[journal_column].isin(config.journals)
    ].copy()
    numerator = concordance["citation_scopus"].to_numpy(dtype=float)
    denominator = concordance["citation_icite"].to_numpy(dtype=float)
    concordance["bias_ratio"] = np.divide(
        numerator,
        denominator,
        out=np.full(len(concordance), np.nan, dtype=float),
        where=denominator > 1e-6,
    )
    concordance.loc[(denominator <= 1e-6) & (numerator <= 1e-6), "bias_ratio"] = 1.0
    return concordance.reset_index(drop=True)


def _origin_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_values, y_values = x[mask], y[mask]
    denominator = float(np.sum(np.square(x_values)))
    if x_values.size == 0 or denominator <= 0:
        return np.nan, np.nan, int(x_values.size)
    slope = float(np.sum(x_values * y_values) / denominator)
    if x_values.size <= 1:
        return slope, np.nan, int(x_values.size)
    residuals = y_values - slope * x_values
    variance = float(np.sum(np.square(residuals)) / (x_values.size - 1))
    return slope, float(np.sqrt(variance / denominator)), int(x_values.size)


def plot_citation_concordance(
    concordance: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
    *,
    category_groups: Mapping[str, Sequence[str]],
    journals: Sequence[str] | None = None,
    category_column: str = "manuscript_category",
    journal_column: str = "target_journal",
    scopus_column: str = "citation_scopus",
    icite_column: str = "citation_icite",
) -> tuple[Figure, pd.DataFrame]:
    """Plot Scopus–iCite concordance with identity and origin-regression lines."""

    _require_columns(
        concordance,
        {category_column, journal_column, scopus_column, icite_column},
        "citation concordance data",
    )
    selected = _selected_journals(config, journals or config.primary_journals)
    data = concordance.copy()
    data[category_column] = data[category_column].astype("string").str.lower()
    data[scopus_column] = _numeric(data[scopus_column])
    data[icite_column] = _numeric(data[icite_column])
    groups = [
        (label, {str(value).lower() for value in values})
        for label, values in category_groups.items()
    ]
    if not groups:
        raise ValueError(
            "At least one citation concordance category group is required."
        )

    with manuscript_style(config):
        figure, axes = plt.subplots(
            len(groups),
            len(selected),
            figsize=(7.2, 2.0 * len(groups)),
            squeeze=False,
        )
        result_rows: list[dict[str, object]] = []
        for row_index, (group_label, categories) in enumerate(groups):
            group = data[data[category_column].isin(categories)]
            finite_values = group[[icite_column, scopus_column]].to_numpy(dtype=float)
            row_max = float(np.nanmax(finite_values)) if finite_values.size else 0.0
            limit = max(1.0, row_max) * 1.05
            for column_index, journal in enumerate(selected):
                axis = axes[row_index, column_index]
                journal_data = group[group[journal_column].eq(journal)]
                x = journal_data[icite_column].to_numpy(dtype=float)
                y = journal_data[scopus_column].to_numpy(dtype=float)
                color = config.journal_colors[journal]
                axis.scatter(
                    x, y, s=10, alpha=0.75, color=color, edgecolors="none", zorder=2
                )
                axis.plot(
                    [0, limit], [0, limit], "--", linewidth=0.8, color="0.6", zorder=1
                )
                slope, standard_error, sample_size = _origin_slope(x, y)
                if np.isfinite(slope):
                    axis.plot(
                        [0, limit],
                        [0, slope * limit],
                        color=color,
                        linewidth=1.4,
                        zorder=3,
                    )
                annotation = (
                    f"Slope: {slope:.2f} (95% CI "
                    f"{slope - 1.96 * standard_error:.2f}–"
                    f"{slope + 1.96 * standard_error:.2f})"
                    if np.isfinite(slope) and np.isfinite(standard_error)
                    else "Slope: NA"
                )
                axis.text(
                    0.97,
                    0.1,
                    annotation,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.7",
                        "linewidth": 0.7,
                        "boxstyle": "round,pad=0.2",
                    },
                    zorder=4,
                )
                if row_index == 0:
                    axis.set_title(
                        f"[{ascii_uppercase[column_index]}] {config.journal_labels[journal]}",
                        fontsize=9,
                        pad=2,
                    )
                axis.set_xlim(0, limit)
                axis.set_ylim(0, limit)
                axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                if column_index > 0:
                    axis.tick_params(labelleft=False)
                result_rows.append(
                    {
                        "category_group": group_label,
                        "journal": journal,
                        "slope": slope,
                        "standard_error": standard_error,
                        "ci_low": slope - 1.96 * standard_error,
                        "ci_high": slope + 1.96 * standard_error,
                        "sample_size": sample_size,
                    }
                )

        figure.text(0.5, 0.02, "2-year citations in iCite", ha="center", fontsize=9)
        figure.text(
            0.02,
            0.5,
            "2-year citations in Scopus",
            va="center",
            rotation="vertical",
            fontsize=9,
        )
        figure.tight_layout(
            pad=0.3, rect=[0.1 if len(groups) > 1 else 0.06, 0.06, 1, 1]
        )
        if len(groups) > 1:
            for row_index, (group_label, _) in enumerate(groups):
                position = axes[row_index, 0].get_position()
                figure.text(
                    0.07,
                    0.5 * (position.y0 + position.y1),
                    group_label,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9,
                )
        save_figure(figure, output_path, config)
    return figure, pd.DataFrame(result_rows)
