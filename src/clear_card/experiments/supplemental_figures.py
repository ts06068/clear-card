from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import pandas as pd

from clear_card.config import (
    ALL_JOURNALS,
    CITABLE_CATEGORIES,
    GSS,
    META_ANALYSIS,
    NON_CITABLE,
    NORA_CATEGORIES,
    ORA,
    REVIEW,
    STUDY,
)
from clear_card.experiments.common import (
    CORRELATION_PREDICTORS,
    align_journal_metrics,
    impact_factor_series,
)
from clear_card.figures import (
    FigureConfig,
    plot_composition_trends,
    plot_dual_correlation_panels,
    plot_histogram_collection,
    plot_journal_trend,
)
from clear_card.statistics import build_correlation_dataset

NORA_GROUPS = {
    "NORA Overall": NORA_CATEGORIES,
    "Meta-analysis": (META_ANALYSIS,),
    "Review": (REVIEW,),
    "GSS": (GSS,),
}
JOURNAL_METRIC_LABELS = {
    "impact_factor_2y": "Journal impact factor",
    "impact_factor_5y": "5-year impact factor",
    "citescore": "CiteScore",
    "sjr": "SJR",
    "snip": "SNIP",
    "eigenfactor": "Eigenfactor",
    "article_influence_score": "Article influence score",
    "immediacy_index": "Immediacy index",
}


def generate_supplemental_analysis_figures(
    analysis: pd.DataFrame,
    journal_metrics: pd.DataFrame,
    output_dir: Path,
    *,
    config: FigureConfig,
) -> list[Path]:
    """Generate SF7–SF44 from compact, declarative figure specifications."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(_generate_histograms(analysis, output_dir, config))
    outputs.extend(_generate_trends(analysis, journal_metrics, output_dir, config))
    outputs.extend(_generate_compositions(analysis, output_dir, config))
    outputs.extend(
        _generate_journal_metric_correlations(
            analysis, journal_metrics, output_dir, config
        )
    )
    outputs.extend(_generate_citing_journal_trends(analysis, output_dir, config))
    return outputs


def _generate_histograms(
    analysis: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    # Declarative specifications keep manuscript numbering next to each cohort definition.
    specifications = (
        (7, "lambda", {"ORA": (ORA,)}, (0.0, 4.0), 20, "Fitness"),
        (8, "lambda", NORA_GROUPS, (0.0, 4.0), 20, "Fitness"),
        (
            9,
            "lambda",
            {"Non-citable items": (NON_CITABLE,)},
            (0.0, 4.0),
            20,
            "Fitness",
        ),
        (
            10,
            "citation_within_2_year",
            {"ORA": (ORA,)},
            (0.0, 150.0),
            20,
            "2-year citation count",
        ),
        (
            11,
            "citation_within_2_year",
            NORA_GROUPS,
            (0.0, 150.0),
            20,
            "2-year citation count",
        ),
        (
            12,
            "citation_within_2_year",
            {"Non-citable items": (NON_CITABLE,)},
            (0.0, 150.0),
            20,
            "2-year citation count",
        ),
    )
    outputs: list[Path] = []
    for number, value_column, groups, value_range, bins, x_label in specifications:
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure = plot_histogram_collection(
            analysis,
            value_column=value_column,
            category_groups=groups,
            value_range=value_range,
            bins=bins,
            x_label=x_label,
            config=config,
            output_path=output,
        )
        plt.close(figure)
        outputs.append(output)
    return outputs


def _generate_trends(
    analysis: pd.DataFrame,
    journal_metrics: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    impact_factors = impact_factor_series(journal_metrics)
    specifications = (
        (13, "lambda", (ORA,), 2019, "mean", "Mean ORA fitness", (0.5, 5.0)),
        (14, "lambda", (ORA,), 2019, "median", "Median ORA fitness", (0.0, 2.5)),
        (
            15,
            "citation_within_2_year",
            (ORA,),
            2022,
            "mean",
            "Mean ORA 2-year citation count",
            (0.0, 100.0),
        ),
        (
            16,
            "citation_within_2_year",
            (ORA,),
            2022,
            "median",
            "Median ORA 2-year citation count",
            (0.0, 50.0),
        ),
        (17, "lambda", NORA_CATEGORIES, 2019, "mean", "Mean NORA fitness", (0.0, 5.0)),
        (
            18,
            "lambda",
            NORA_CATEGORIES,
            2019,
            "median",
            "Median NORA fitness",
            (0.0, 2.5),
        ),
        (
            19,
            "citation_within_2_year",
            NORA_CATEGORIES,
            2022,
            "mean",
            "Mean NORA 2-year citation count",
            (0.0, 100.0),
        ),
        (
            20,
            "citation_within_2_year",
            NORA_CATEGORIES,
            2022,
            "median",
            "Median NORA 2-year citation count",
            (0.0, 50.0),
        ),
    )
    outputs: list[Path] = []
    for number, value, categories, end_year, statistic, label, limits in specifications:
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure, _ = plot_journal_trend(
            analysis,
            config,
            output,
            value_column=value,
            category_values=categories,
            year_range=(STUDY.study_start_year, end_year),
            statistic=statistic,
            y_label=label,
            y_limits=limits,
            journals=ALL_JOURNALS,
            impact_factors=impact_factors,
        )
        plt.close(figure)
        outputs.append(output)
    return outputs


def _generate_compositions(
    analysis: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    specifications = (
        (21, None, CITABLE_CATEGORIES, (ORA,), "ORA publication proportion (%)"),
        (
            22,
            None,
            CITABLE_CATEGORIES,
            NORA_CATEGORIES,
            "NORA publication proportion (%)",
        ),
        (
            23,
            None,
            (*CITABLE_CATEGORIES, NON_CITABLE),
            (ORA,),
            "ORA publication proportion (%)",
        ),
        (
            24,
            None,
            (*CITABLE_CATEGORIES, NON_CITABLE),
            NORA_CATEGORIES,
            "NORA publication proportion (%)",
        ),
        (
            25,
            "citation_within_2_year",
            CITABLE_CATEGORIES,
            (ORA,),
            "ORA share of 2-year citations (%)",
        ),
        (
            26,
            "citation_within_2_year",
            CITABLE_CATEGORIES,
            NORA_CATEGORIES,
            "NORA share of 2-year citations (%)",
        ),
        (
            27,
            "citation_within_2_year",
            (*CITABLE_CATEGORIES, NON_CITABLE),
            (ORA,),
            "ORA share of 2-year citations (%)",
        ),
        (
            28,
            "citation_within_2_year",
            (*CITABLE_CATEGORIES, NON_CITABLE),
            NORA_CATEGORIES,
            "NORA share of 2-year citations (%)",
        ),
    )
    outputs: list[Path] = []
    for number, weight, categories, focus, y_label in specifications:
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure, _ = plot_composition_trends(
            analysis,
            config,
            output,
            categories=categories,
            focus_categories=focus,
            weight_column=weight,
            journals=ALL_JOURNALS,
            breakdown_categories=NORA_CATEGORIES if len(focus) > 1 else (),
            y_label=y_label,
        )
        plt.close(figure)
        outputs.append(output)
    return outputs


def _generate_journal_metric_correlations(
    analysis: pd.DataFrame,
    journal_metrics: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    aligned_metrics = align_journal_metrics(journal_metrics)
    outputs: list[Path] = []
    for number, end_year, journals in (
        (29, STUDY.fitness_end_year, config.primary_journals),
        (30, STUDY.study_end_year, config.primary_journals),
        (31, STUDY.fitness_end_year, ALL_JOURNALS),
        (32, STUDY.study_end_year, ALL_JOURNALS),
    ):
        base = build_correlation_dataset(
            analysis,
            journal_metrics,
            exclude_journals=() if len(journals) == 4 else ("JAMA CARDIOLOGY",),
        )
        data = base.merge(
            aligned_metrics[
                ["target_journal", "target_pub_year", *JOURNAL_METRIC_LABELS]
            ],
            on=["target_journal", "target_pub_year"],
            how="left",
        )
        for index, (metric, label) in enumerate(JOURNAL_METRIC_LABELS.items()):
            letter = ascii_lowercase[index]
            output = output_dir / f"supplemental_figure_{number:02d}{letter}.pdf"
            figure, _, _ = plot_dual_correlation_panels(
                data,
                config,
                output,
                metric_labels=CORRELATION_PREDICTORS,
                outcome_column=metric,
                outcome_label=label,
                year_range=(STUDY.study_start_year, end_year),
                journals=journals,
                bootstrap_iterations=STUDY.bootstrap_iterations,
                bootstrap_seed=STUDY.random_seed,
                panel_label=letter.upper(),
            )
            plt.close(figure)
            outputs.append(output)
    return outputs


def _generate_citing_journal_trends(
    analysis: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    metrics = (
        ("source_journal_score_hindex_5yr_mean", "H-index", (0.0, 250.0)),
        ("source_journal_score_sjr_5yr_mean", "SJR", (0.0, 6.0)),
        ("source_journal_score_snip_5yr_mean", "SNIP", (0.0, 6.0)),
    )
    outputs: list[Path] = []
    number = 33
    for value_column, label, limits in metrics:
        for category_label, categories in (("ORA", (ORA,)), ("NORA", NORA_CATEGORIES)):
            for statistic in ("mean", "median"):
                output = output_dir / f"supplemental_figure_{number:02d}.pdf"
                figure, _ = plot_journal_trend(
                    analysis,
                    config,
                    output,
                    value_column=value_column,
                    category_values=categories,
                    year_range=(STUDY.study_start_year, STUDY.fitness_end_year),
                    statistic=statistic,
                    y_label=f"{statistic.title()} {label} of journals citing {category_label}",
                    y_limits=limits,
                    journals=ALL_JOURNALS,
                )
                plt.close(figure)
                outputs.append(output)
                number += 1
    return outputs
