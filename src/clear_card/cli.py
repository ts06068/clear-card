from __future__ import annotations

import argparse

from clear_card.database import check_connection
from clear_card.pipeline import reproduce
from clear_card.validation import (
    verify_benchmarks,
    verify_manifest,
    verify_supplemental_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clear-card")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "validate", help="Validate checksums and manuscript benchmark counts."
    )
    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Generate manuscript figures and tables."
    )
    reproduce_parser.add_argument(
        "--target",
        choices=["all", "main", "supplemental", "tables"],
        default="all",
    )
    db_parser = subparsers.add_parser(
        "db-check", help="Test the configured SQL Server connection."
    )
    db_parser.add_argument("--database", choices=["scopus", "icite"], default="scopus")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Thin dispatch keeps command logic in its owning module.
    if args.command == "db-check":
        print(f"Connected to {check_connection(args.database)}")
        return 0

    if args.command == "reproduce":
        result = reproduce(target=args.target)
        print(f"Wrote {len(result.figures)} figures and {len(result.tables)} tables.")
        return 0

    errors = verify_manifest() + verify_benchmarks() + verify_supplemental_data()
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: checksums and manuscript benchmark counts match.")
    return 0
