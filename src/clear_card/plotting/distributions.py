from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clear_card.config import ALL_JOURNALS
from clear_card.plotting.style import (
    FigureConfig,
    manuscript_style,
    save_figure,
)


def plot_histogram_collection(
    frame: pd.DataFrame,
    *,
    value_column: str,
    category_groups: Mapping[str, Sequence[str]],
    value_range: tuple[float, float],
    bins: int,
    x_label: str,
    config: FigureConfig,
    output_path: Path,
):
    # Fix the eras so journal distributions remain directly comparable.
    periods = (
        ("2008–2011", 2008, 2011),
        ("2012–2015", 2012, 2015),
        ("2016–2019", 2016, 2019),
    )
    if value_column == "citation_within_2_year":
        periods = (*periods, ("2020–2022", 2020, 2022))
    with manuscript_style(config):
        figure, axes = plt.subplots(
            len(category_groups),
            len(ALL_JOURNALS),
            figsize=(4 * len(ALL_JOURNALS), 3.2 * len(category_groups)),
            squeeze=False,
            sharey="row",
        )
        for row, (group_label, categories) in enumerate(category_groups.items()):
            group = frame.loc[frame["manuscript_category"].isin(categories)]
            for column, journal in enumerate(ALL_JOURNALS):
                axis = axes[row, column]
                journal_data = group.loc[group["target_journal"].eq(journal)]
                for index, (period_label, start, end) in enumerate(periods):
                    values = (
                        journal_data.loc[
                            journal_data["target_pub_year"].between(start, end),
                            value_column,
                        ]
                        .dropna()
                        .clip(upper=value_range[1])
                    )
                    if values.empty:
                        continue
                    axis.hist(
                        values,
                        bins=bins,
                        range=value_range,
                        density=True,
                        histtype="step",
                        linewidth=1.2,
                        color=config.histogram_shades[journal][index],
                        label=period_label,
                    )
                    axis.axvline(
                        values.mean(),
                        color=config.histogram_shades[journal][index],
                        linestyle="--",
                        linewidth=1,
                    )
                if row == 0:
                    axis.set_title(config.journal_labels[journal])
                if column == 0:
                    axis.set_ylabel(f"{group_label}\nDensity")
                if row == len(category_groups) - 1:
                    axis.set_xlabel(x_label)
                axis.set_xlim(*value_range)
        axes[0, -1].legend(frameon=False, fontsize=7)
        figure.tight_layout()
        save_figure(figure, output_path, config)
    return figure
