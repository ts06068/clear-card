from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.figure import Figure

from clear_card.config import ALL_JOURNALS, JOURNAL_LABELS, PRIMARY_JOURNALS, STUDY


def _journal_labels() -> dict[str, str]:
    return dict(JOURNAL_LABELS)


def _journal_colors() -> dict[str, str]:
    return {
        "CIRCULATION": "#FF3333",
        "EUROPEAN HEART JOURNAL": "#33CC33",
        "JOURNAL OF THE AMERICAN COLLEGE OF CARDIOLOGY": "#3333FF",
        "JAMA CARDIOLOGY": "#B24DFF",
    }


def _journal_offsets() -> dict[str, float]:
    return {
        "CIRCULATION": -0.12,
        "EUROPEAN HEART JOURNAL": -0.06,
        "JOURNAL OF THE AMERICAN COLLEGE OF CARDIOLOGY": 0.06,
        "JAMA CARDIOLOGY": 0.12,
    }


def _histogram_shades() -> dict[str, tuple[str, ...]]:
    return {
        "CIRCULATION": ("#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#CC0000"),
        "EUROPEAN HEART JOURNAL": (
            "#CCFFCC",
            "#99FF99",
            "#66FF66",
            "#33CC33",
            "#009900",
        ),
        "JOURNAL OF THE AMERICAN COLLEGE OF CARDIOLOGY": (
            "#CCCCFF",
            "#9999FF",
            "#6666FF",
            "#3333FF",
            "#0000CC",
        ),
        "JAMA CARDIOLOGY": ("#F0CCFF", "#E099FF", "#CC66FF", "#B24DFF", "#7F00CC"),
    }


@dataclass(frozen=True)
class FigureConfig:
    """Visual constants shared by manuscript and supplemental figures."""

    journals: tuple[str, ...] = ALL_JOURNALS
    journal_labels: Mapping[str, str] = field(default_factory=_journal_labels)
    journal_colors: Mapping[str, str] = field(default_factory=_journal_colors)
    journal_offsets: Mapping[str, float] = field(default_factory=_journal_offsets)
    histogram_shades: Mapping[str, tuple[str, ...]] = field(
        default_factory=_histogram_shades
    )
    font_family: str = "Arial"
    figure_dpi: int = 300
    save_dpi: int = 600
    impact_factor_lag: int = STUDY.journal_metric_lag_years

    @property
    def primary_journals(self) -> tuple[str, ...]:
        """Return the three journals included in the primary analyses."""

        return tuple(
            journal for journal in PRIMARY_JOURNALS if journal in self.journals
        )

    def rc_params(self, *, large: bool = False) -> dict[str, object]:
        """Return isolated Matplotlib settings matching the manuscript style."""

        if large:
            font_size, title_size, label_size, legend_size, tick_size = (
                20,
                30,
                24,
                18,
                18,
            )
        else:
            font_size, title_size, label_size, legend_size, tick_size = 9, 11, 9, 8, 8
        return {
            "figure.dpi": self.figure_dpi,
            "savefig.dpi": self.save_dpi,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": self.font_family,
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "legend.fontsize": legend_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.4,
            "lines.markersize": 5.5,
            "legend.frameon": False,
        }


def manuscript_style(
    config: FigureConfig, *, large: bool = False
) -> AbstractContextManager[None]:
    """Return a context that applies manuscript styling without changing global rc state."""

    return mpl.rc_context(rc=config.rc_params(large=large))


def save_figure(figure: Figure, output_path: Path, config: FigureConfig) -> Path:
    """Create the destination directory and save a publication-quality figure."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.save_dpi, bbox_inches="tight", facecolor="white")
    return path


def _require_columns(frame: pd.DataFrame, columns: Collection[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _selected_journals(
    config: FigureConfig,
    journals: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = tuple(journals) if journals is not None else config.journals
    unknown = [journal for journal in selected if journal not in config.journal_labels]
    if unknown:
        raise ValueError(f"FigureConfig has no style for: {', '.join(unknown)}")
    return selected


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
