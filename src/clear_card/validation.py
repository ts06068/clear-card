from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from clear_card.config import (
    CITABLE_CATEGORIES,
    GSS,
    JAMA_CARDIOLOGY,
    META_ANALYSIS,
    NORA_CATEGORIES,
    ORA,
    REVIEW,
    STUDY,
)
from clear_card.data import load_analysis, load_classifications
from clear_card.settings import PROJECT_ROOT, RepositoryPaths
from clear_card.taxonomy import build_supplemental_category_counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Stream frozen datasets so checksum validation has bounded memory use.
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path = PROJECT_ROOT) -> list[str]:
    manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.exists():
            errors.append(f"missing: {item['path']}")
            continue
        if sha256(path) != item["sha256"]:
            errors.append(f"checksum mismatch: {item['path']}")
    return errors


def compute_benchmarks(paths: RepositoryPaths | None = None) -> dict[str, int]:
    frame = load_analysis(paths)
    primary = frame.loc[
        frame["pub_year"].between(STUDY.study_start_year, STUDY.study_end_year)
        & frame["target_journal"].isin(STUDY.primary_journals)
    ]
    citable = primary[primary["manuscript_category"].isin(CITABLE_CATEGORIES)]
    jama = frame.loc[frame["target_journal"].eq(JAMA_CARDIOLOGY)]
    return {
        "primary_citable": len(citable),
        "primary_original": int((primary["manuscript_category"] == ORA).sum()),
        "primary_meta": int((primary["manuscript_category"] == META_ANALYSIS).sum()),
        "primary_review": int((primary["manuscript_category"] == REVIEW).sum()),
        "primary_gss": int((primary["manuscript_category"] == GSS).sum()),
        "circulation_citable": int((citable["target_journal"] == "CIRCULATION").sum()),
        "ehj_citable": int(
            (citable["target_journal"] == "EUROPEAN HEART JOURNAL").sum()
        ),
        "jacc_citable": int(
            (
                citable["target_journal"]
                == "JOURNAL OF THE AMERICAN COLLEGE OF CARDIOLOGY"
            ).sum()
        ),
        "jama_citable": int(jama["manuscript_category"].isin(CITABLE_CATEGORIES).sum()),
        "jama_original": int((jama["manuscript_category"] == ORA).sum()),
        "jama_nora": int(jama["manuscript_category"].isin(NORA_CATEGORIES).sum()),
    }


def verify_benchmarks(root: Path = PROJECT_ROOT) -> list[str]:
    expected = json.loads(
        (root / "data" / "expected" / "manuscript_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    actual = compute_benchmarks(RepositoryPaths(root))
    return [
        f"{key}: expected {value}, got {actual.get(key)}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]


def verify_supplemental_data(
    paths: RepositoryPaths | None = None,
) -> list[str]:
    """Compare regenerated taxonomy counts with the frozen manuscript table."""

    paths = paths or RepositoryPaths()
    expected = pd.read_csv(paths.processed_data / "supplemental_category_counts.csv")
    actual = build_supplemental_category_counts(
        load_analysis(paths), load_classifications(paths)
    )
    if actual.columns.tolist() != expected.columns.tolist():
        return ["supplemental data columns do not match the frozen table"]

    sort_columns = ["journal", "category_1", "category_2", "category_3"]
    actual = actual.sort_values(sort_columns, na_position="first").reset_index(
        drop=True
    )
    expected = expected.sort_values(sort_columns, na_position="first").reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except AssertionError:
        return ["supplemental data values do not match the frozen table"]
    return []
