from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from clear_card.comparisons import (
    prepare_wos_citation_comparison,
    prepare_wos_publication_comparison,
)
from clear_card.config import (
    CITABLE_CATEGORIES,
    GSS,
    META_ANALYSIS,
    NON_CITABLE,
    NORA_CATEGORIES,
    ORA,
    REVIEW,
)
from clear_card.figures import (
    FigureConfig,
    manuscript_style,
    plot_citation_concordance,
    plot_validation_count_bars,
    prepare_citation_concordance,
    prepare_validation_counts,
    save_figure,
)
from clear_card.statistics import dual_correlation


def generate_validation_figures(
    analysis: pd.DataFrame,
    scopus_validation: pd.DataFrame,
    icite_validation: pd.DataFrame,
    wos_validation: pd.DataFrame,
    wos_publication_counts: pd.DataFrame,
    output_dir: Path,
    *,
    config: FigureConfig,
) -> list[Path]:
    """Generate SF1–SF2 and SF45–SF54 validation figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _generate_flow_figures(
        analysis, scopus_validation, icite_validation, output_dir, config
    )
    outputs.extend(
        _generate_wos_figures(
            analysis,
            wos_validation,
            wos_publication_counts,
            output_dir,
            config,
        )
    )
    outputs.extend(
        _generate_icite_figures(scopus_validation, icite_validation, output_dir, config)
    )
    return outputs


def _generate_flow_figures(
    analysis: pd.DataFrame,
    scopus: pd.DataFrame,
    icite: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    # Flow counts are derived from the same filtered inputs used by validation panels.
    primary = analysis.loc[analysis["target_journal"].isin(config.primary_journals)]
    citable = primary.loc[primary["manuscript_category"].isin(CITABLE_CATEGORIES)]
    nora = citable.loc[citable["manuscript_category"].isin(NORA_CATEGORIES)]
    first_steps = (
        f"Records in three journals, 2008–2022\n(n = {len(primary):,})",
        f"Citable items\n(n = {len(citable):,})",
        (
            f"ORA (n = {(citable['manuscript_category'] == ORA).sum():,})\n"
            f"NORA (n = {len(nora):,})"
        ),
    )
    overlap = len(set(scopus["article_id"]).intersection(icite["article_id"]))
    second_steps = (
        f"Scopus validation records\n(n = {scopus['article_id'].nunique():,})",
        f"iCite validation records\n(n = {icite['article_id'].nunique():,})",
        f"Matched records\n(n = {overlap:,})",
    )
    outputs: list[Path] = []
    for number, steps in ((1, first_steps), (2, second_steps)):
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure = _plot_flow(steps, config, output)
        plt.close(figure)
        outputs.append(output)
    return outputs


def _plot_flow(steps: tuple[str, ...], config: FigureConfig, output_path: Path):
    with manuscript_style(config):
        figure, axis = plt.subplots(figsize=(7.2, 2.4))
        axis.axis("off")
        x_positions = np.linspace(0.15, 0.85, len(steps))
        for index, (x, text) in enumerate(zip(x_positions, steps, strict=True)):
            axis.text(
                x,
                0.5,
                text,
                ha="center",
                va="center",
                transform=axis.transAxes,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white"},
            )
            if index < len(steps) - 1:
                axis.annotate(
                    "",
                    xy=(x_positions[index + 1] - 0.12, 0.5),
                    xytext=(x + 0.12, 0.5),
                    xycoords=axis.transAxes,
                    arrowprops={"arrowstyle": "->", "linewidth": 1.2},
                )
        save_figure(figure, output_path, config)
    return figure


def _generate_wos_figures(
    analysis: pd.DataFrame,
    wos_validation: pd.DataFrame,
    wos_counts: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    citation_data = prepare_wos_citation_comparison(wos_validation, analysis)
    citation_output = output_dir / "supplemental_figure_45.pdf"
    figure = _plot_wos_citation_concordance(citation_data, config, citation_output)
    plt.close(figure)

    count_data = prepare_wos_publication_comparison(analysis, wos_counts)
    count_output = output_dir / "supplemental_figure_46.pdf"
    figure = _plot_wos_publication_counts(count_data, config, count_output)
    plt.close(figure)
    return [citation_output, count_output]


def _plot_wos_citation_concordance(
    data: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
):
    journals = config.journals
    with manuscript_style(config):
        figure, axes = plt.subplots(1, len(journals), figsize=(12, 3), squeeze=False)
        maximum = float(
            np.nanmax(data[["wos_total_citations", "scopus_total_citations"]])
        )
        limit = max(maximum, 1.0) * 1.05
        for axis, journal in zip(axes[0], journals, strict=True):
            subset = data.loc[data["journal"].eq(journal)]
            x = subset["scopus_total_citations"].clip(lower=1).to_numpy(dtype=float)
            y = subset["wos_total_citations"].clip(lower=1).to_numpy(dtype=float)
            correlations = dual_correlation(x, y)
            slope, ci_low, ci_high = _log_log_slope_ci(x, y)
            axis.scatter(
                x,
                y,
                s=10,
                alpha=0.55,
                color=config.journal_colors[journal],
                edgecolors="none",
            )
            axis.plot([1, limit], [1, limit], "--", color="0.6", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(1, limit)
            axis.set_ylim(1, limit)
            axis.set_title(config.journal_labels[journal])
            axis.text(
                0.04,
                0.96,
                (
                    f"Pearson r = {correlations['pearson'][0]:.2f}\n"
                    f"Spearman ρ = {correlations['spearman'][0]:.2f}\n"
                    f"Slope = {slope:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})"
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=7,
            )
        figure.supxlabel("Total citation count in Scopus")
        figure.supylabel("Total citation count in Web of Science")
        figure.tight_layout()
        save_figure(figure, output_path, config)
    return figure


def _log_log_slope_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    regression = linregress(np.log10(x), np.log10(y))
    slope = float(regression.slope)
    standard_error = float(regression.stderr)
    return slope, slope - 1.96 * standard_error, slope + 1.96 * standard_error


def _plot_wos_publication_counts(
    data: pd.DataFrame,
    config: FigureConfig,
    output_path: Path,
):
    with manuscript_style(config):
        figure, axes = plt.subplots(
            len(config.primary_journals), 1, figsize=(9, 8), sharex=True
        )
        for axis, journal in zip(axes, config.primary_journals, strict=True):
            subset = data.loc[data["target_journal"].eq(journal)].sort_values(
                "target_pub_year"
            )
            years = subset["target_pub_year"].to_numpy(dtype=int)
            x = np.arange(len(years))
            axis.bar(x - 0.19, subset["scopus_count"], width=0.38, label="Scopus")
            axis.bar(x + 0.19, subset["wos_count"], width=0.38, label="WoS")
            axis.set_title(config.journal_labels[journal])
            axis.set_xticks(x, years)
        axes[0].legend(frameon=False)
        axes[-1].set_xlabel("Publication year")
        figure.supylabel("Publication count")
        figure.tight_layout()
        save_figure(figure, output_path, config)
    return figure


def _generate_icite_figures(
    scopus: pd.DataFrame,
    icite: pd.DataFrame,
    output_dir: Path,
    config: FigureConfig,
) -> list[Path]:
    count_specs = (
        (47, (ORA,), "Publication volume of ORA"),
        (48, (META_ANALYSIS,), "Publication volume of meta-analysis"),
        (49, (REVIEW,), "Publication volume of review"),
        (50, (GSS,), "Publication volume of GSS"),
        (51, (NON_CITABLE,), "Publication volume of non-citable items"),
    )
    outputs: list[Path] = []
    for number, categories, label in count_specs:
        counts = prepare_validation_counts(
            scopus, icite, config, category_values=categories
        )
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure = plot_validation_count_bars(counts, config, output, volume_label=label)
        plt.close(figure)
        outputs.append(output)

    concordance = prepare_citation_concordance(scopus, icite, config)
    concordance_specs = (
        (52, {"ORA": (ORA,)}),
        (
            53,
            {
                "NORA Overall": NORA_CATEGORIES,
                "Meta-analysis": (META_ANALYSIS,),
                "Review": (REVIEW,),
                "GSS": (GSS,),
            },
        ),
        (54, {"Non-citable items": (NON_CITABLE,)}),
    )
    for number, groups in concordance_specs:
        output = output_dir / f"supplemental_figure_{number:02d}.pdf"
        figure, _ = plot_citation_concordance(
            concordance,
            config,
            output,
            category_groups=groups,
        )
        plt.close(figure)
        outputs.append(output)
    return outputs
