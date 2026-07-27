from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from clear_card.experiments.common import AnalysisInputs, load_inputs
from clear_card.experiments.fitness_validation import (
    generate_fitness_validation_figures,
)
from clear_card.experiments.main_figures import generate_main_figures
from clear_card.experiments.supplemental_figures import (
    generate_supplemental_analysis_figures,
)
from clear_card.experiments.validation_figures import generate_validation_figures
from clear_card.figures import FigureConfig
from clear_card.settings import RepositoryPaths
from clear_card.tables import build_supplemental_tables, write_supplemental_tables
from clear_card.taxonomy import build_supplemental_category_counts


PipelineTarget = Literal["all", "main", "supplemental", "tables"]


@dataclass(frozen=True)
class ReproductionResult:
    figures: tuple[Path, ...]
    tables: tuple[Path, ...]


def reproduce(
    *,
    target: PipelineTarget = "all",
    paths: RepositoryPaths | None = None,
) -> ReproductionResult:
    """Run one explicit portion of the manuscript reproduction pipeline."""

    repository_paths = paths or RepositoryPaths()
    repository_paths.create_outputs()
    inputs = load_inputs(repository_paths)
    figure_config = FigureConfig()
    figures: list[Path] = []
    tables: list[Path] = []

    if target in ("all", "tables"):
        tables.extend(_write_tables(inputs, repository_paths))
    if target in ("all", "main"):
        figures.extend(
            generate_main_figures(
                inputs.analysis,
                inputs.journal_metrics,
                repository_paths.figures,
                config=figure_config,
            )
        )
    if target in ("all", "supplemental"):
        figures.extend(
            generate_validation_figures(
                inputs.analysis,
                inputs.scopus_validation,
                inputs.icite_validation,
                inputs.wos_validation,
                inputs.wos_publication_counts,
                repository_paths.figures,
                config=figure_config,
            )
        )
        figures.extend(
            generate_fitness_validation_figures(
                inputs.analysis,
                repository_paths.figures,
                config=figure_config,
            )
        )
        figures.extend(
            generate_supplemental_analysis_figures(
                inputs.analysis,
                inputs.journal_metrics,
                repository_paths.figures,
                config=figure_config,
            )
        )
    return ReproductionResult(tuple(figures), tuple(tables))


def _write_tables(
    inputs: AnalysisInputs,
    paths: RepositoryPaths,
) -> list[Path]:
    tables = build_supplemental_tables(
        inputs.analysis,
        inputs.journal_metrics,
        inputs.impact_factor_simulation,
    )
    outputs = write_supplemental_tables(tables, paths.tables)
    supplemental_data = build_supplemental_category_counts(
        inputs.analysis, inputs.classifications
    )
    supplemental_data_path = paths.tables / "supplemental_data.xlsx"
    supplemental_data.to_excel(supplemental_data_path, index=False)
    return [*outputs, supplemental_data_path]
