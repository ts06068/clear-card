"""Public plotting API for the CLEAR-CARD manuscript."""

# Re-exporting the supported plotters keeps notebooks independent of internal layout.
from clear_card.plotting.composition import (
    plot_composition_trends,
    summarize_composition,
)
from clear_card.plotting.correlations import (
    plot_correlation_panels,
    plot_dual_correlation_panels,
)
from clear_card.plotting.distributions import plot_histogram_collection
from clear_card.plotting.journal_trends import (
    plot_journal_trend,
    summarize_journal_trends,
)
from clear_card.plotting.style import FigureConfig, manuscript_style, save_figure
from clear_card.plotting.validation import (
    plot_citation_concordance,
    plot_validation_count_bars,
    prepare_citation_concordance,
    prepare_validation_counts,
)

__all__ = [
    "FigureConfig",
    "manuscript_style",
    "plot_citation_concordance",
    "plot_composition_trends",
    "plot_correlation_panels",
    "plot_dual_correlation_panels",
    "plot_histogram_collection",
    "plot_journal_trend",
    "plot_validation_count_bars",
    "prepare_citation_concordance",
    "prepare_validation_counts",
    "save_figure",
    "summarize_composition",
    "summarize_journal_trends",
]
