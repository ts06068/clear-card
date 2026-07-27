"""Public statistical API for CLEAR-CARD analyses."""

from clear_card.correlation import (
    CorrelationDifference,
    PairwiseCorrelationComparison,
    bootstrap_absolute_correlation_difference,
    build_correlation_dataset,
    dual_correlation,
    format_p_value,
    pairwise_correlation_comparison,
    safe_correlation,
)
from clear_card.summaries import (
    GROUP_COLUMNS,
    SummaryFormat,
    composition_by_category,
    format_composition_table,
    format_mean_sd,
    format_median_iqr,
    format_sum_percentage,
    require_columns,
    summarize_by_group,
    summarize_categories,
)

__all__ = [
    "CorrelationDifference",
    "GROUP_COLUMNS",
    "PairwiseCorrelationComparison",
    "SummaryFormat",
    "bootstrap_absolute_correlation_difference",
    "build_correlation_dataset",
    "composition_by_category",
    "dual_correlation",
    "format_composition_table",
    "format_mean_sd",
    "format_median_iqr",
    "format_p_value",
    "format_sum_percentage",
    "pairwise_correlation_comparison",
    "require_columns",
    "safe_correlation",
    "summarize_by_group",
    "summarize_categories",
]
