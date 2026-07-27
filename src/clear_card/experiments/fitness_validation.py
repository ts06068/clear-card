from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clear_card.config import JOURNAL_LABELS, STUDY
from clear_card.figures import FigureConfig, manuscript_style, save_figure
from clear_card.modeling import (
    OUTCOME_COLUMNS,
    PREDICTOR_COLUMNS,
    estimate_roc_curves,
    prepare_fitness_validation,
)
from clear_card.statistics import safe_correlation


def generate_fitness_validation_figures(
    analysis: pd.DataFrame,
    output_dir: Path,
    *,
    config: FigureConfig,
) -> list[Path]:
    """Generate SF3–SF6 for long-term fitness validation."""

    cohort = prepare_fitness_validation(analysis)
    # Trimming is visual only; ROC models below retain the full eligible cohort.
    trimmed = cohort.loc[cohort["citation_within_10_year"].le(500)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for number, outcome_key, y_label in (
        (3, "delta_5_year_citations", "Citations during years 6–10"),
        (4, "citations_at_year_10", "Citations during year 10"),
    ):
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure = _plot_association_grid(
            trimmed,
            outcome_column=OUTCOME_COLUMNS[outcome_key],
            y_label=y_label,
            config=config,
            output_path=output,
        )
        plt.close(figure)
        outputs.append(output)

    for number, outcome_key, title in (
        (5, "delta_5_year_citations", "Citations during years 6–10"),
        (6, "citations_at_year_10", "Citations during year 10"),
    ):
        curves = estimate_roc_curves(
            cohort,
            outcome_column=OUTCOME_COLUMNS[outcome_key],
            random_seed=STUDY.random_seed,
        )
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure = _plot_roc_panels(curves, title, config, output)
        plt.close(figure)
        outputs.append(output)
    return outputs


def _plot_association_grid(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
    y_label: str,
    config: FigureConfig,
    output_path: Path,
):
    with manuscript_style(config):
        figure, axes = plt.subplots(
            len(PREDICTOR_COLUMNS),
            len(config.primary_journals),
            figsize=(9, 6),
            squeeze=False,
        )
        for row, (predictor_label, predictor_column) in enumerate(
            PREDICTOR_COLUMNS.items()
        ):
            for column, journal in enumerate(config.primary_journals):
                axis = axes[row, column]
                data = frame.loc[
                    frame["target_journal"].eq(journal),
                    [predictor_column, outcome_column],
                ].dropna()
                x = data[predictor_column].to_numpy(dtype=float)
                y = data[outcome_column].to_numpy(dtype=float)
                coefficient, _ = safe_correlation(x, y, method="pearson")
                axis.scatter(
                    x,
                    y,
                    s=12,
                    alpha=0.7,
                    color=config.journal_colors[journal],
                    edgecolors="none",
                )
                axis.text(
                    0.06,
                    0.94,
                    f"r = {coefficient:.2f}\nr² = {coefficient**2:.2f}",
                    transform=axis.transAxes,
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
                )
                if row == 0:
                    axis.set_title(f"[{chr(65 + column)}] {JOURNAL_LABELS[journal]}")
                if column == 0:
                    axis.set_ylabel(y_label)
                axis.set_xlabel(predictor_label.capitalize())
        figure.tight_layout()
        save_figure(figure, output_path, config)
    return figure


def _plot_roc_panels(
    curves: pd.DataFrame,
    title: str,
    config: FigureConfig,
    output_path: Path,
):
    quantile_labels = {0.90: "Top 10%", 0.75: "Top 25%", 0.50: "Top 50%"}
    colors = {"5-year citations": "#3333FF", "fitness": "#FF8000"}
    with manuscript_style(config):
        figure, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        for axis, quantile in zip(axes, quantile_labels, strict=True):
            panel = curves.loc[np.isclose(curves["quantile"], quantile)]
            for predictor, data in panel.groupby("predictor", sort=False):
                axis.plot(
                    data["false_positive_rate"],
                    data["true_positive_rate"],
                    color=colors[predictor],
                    label=f"{predictor} (AUC {data['auc'].iloc[0]:.2f})",
                )
            axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
            axis.set_title(quantile_labels[quantile])
            axis.set_xlabel("False positive rate")
            axis.set_aspect("equal", adjustable="box")
            axis.legend(loc="lower right", frameon=False)
        axes[0].set_ylabel("True positive rate")
        figure.suptitle(title, y=1.02)
        figure.tight_layout()
        save_figure(figure, output_path, config)
    return figure
