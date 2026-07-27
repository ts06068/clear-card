from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clear_card.config import CITABLE_CATEGORIES, ORA, STUDY
from clear_card.experiments.common import CORRELATION_PREDICTORS, impact_factor_series
from clear_card.figures import (
    FigureConfig,
    plot_composition_trends,
    plot_correlation_panels,
    plot_journal_trend,
)
from clear_card.statistics import build_correlation_dataset


def generate_main_figures(
    analysis: pd.DataFrame,
    journal_metrics: pd.DataFrame,
    output_dir: Path,
    *,
    config: FigureConfig | None = None,
) -> list[Path]:
    """Generate manuscript Figures 1–5 using canonical numbering."""

    figure_config = config or FigureConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    impact_factors = impact_factor_series(journal_metrics)
    outputs: list[Path] = []

    outputs.append(output_dir / "figure_01.pdf")
    figure, _ = plot_journal_trend(
        analysis,
        figure_config,
        outputs[-1],
        value_column="lambda",
        category_values=(ORA,),
        year_range=(STUDY.study_start_year, STUDY.fitness_end_year),
        statistic="mean",
        y_label="Fitness",
        y_limits=(0.5, 5.0),
        impact_factors=impact_factors,
    )
    plt.close(figure)

    outputs.append(output_dir / "figure_02.pdf")
    figure, _ = plot_journal_trend(
        analysis,
        figure_config,
        outputs[-1],
        value_column="citation_within_2_year",
        category_values=(ORA,),
        year_range=(STUDY.study_start_year, STUDY.study_end_year),
        statistic="mean",
        y_label="2-year citation count",
        y_limits=(0.0, 100.0),
        impact_factors=impact_factors,
    )
    plt.close(figure)

    outputs.append(output_dir / "figure_03a.pdf")
    figure, _ = plot_composition_trends(
        analysis,
        figure_config,
        outputs[-1],
        categories=CITABLE_CATEGORIES,
        focus_categories=(ORA,),
        weight_column="citation_within_2_year",
        y_label="Proportion of 2-year citations (%)",
        y_limits=(20.0, 80.0),
    )
    plt.close(figure)

    outputs.append(output_dir / "figure_03b.pdf")
    figure, _ = plot_composition_trends(
        analysis,
        figure_config,
        outputs[-1],
        categories=CITABLE_CATEGORIES,
        focus_categories=(ORA,),
        y_label="Proportion of publications (%)",
        y_limits=(40.0, 100.0),
        show_volume_inset=True,
    )
    plt.close(figure)

    correlation_data = build_correlation_dataset(analysis, journal_metrics)
    for panel, end_year in (("a", STUDY.fitness_end_year), ("b", STUDY.study_end_year)):
        outputs.append(output_dir / f"figure_04{panel}.pdf")
        figure, _, _ = plot_correlation_panels(
            correlation_data,
            figure_config,
            outputs[-1],
            metric_labels=CORRELATION_PREDICTORS,
            method="spearman",
            year_range=(STUDY.study_start_year, end_year),
            bootstrap_iterations=STUDY.bootstrap_iterations,
            bootstrap_seed=STUDY.random_seed,
            panel_label=panel.upper(),
        )
        plt.close(figure)

    outputs.append(output_dir / "figure_05.pdf")
    figure, _ = plot_journal_trend(
        analysis,
        figure_config,
        outputs[-1],
        value_column="source_journal_score_hindex_5yr_mean",
        category_values=(ORA,),
        year_range=(STUDY.study_start_year, STUDY.fitness_end_year),
        statistic="mean",
        y_label="Mean H-index of citing journals",
        y_limits=(100.0, 250.0),
    )
    plt.close(figure)
    return outputs
