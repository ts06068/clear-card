from clear_card.validation import (
    verify_benchmarks,
    verify_manifest,
    verify_supplemental_data,
)

# These checks validate frozen inputs without running the figure pipeline.


def test_data_manifest() -> None:
    assert verify_manifest() == []


def test_manuscript_benchmarks() -> None:
    assert verify_benchmarks() == []


def test_supplemental_data() -> None:
    assert verify_supplemental_data() == []
