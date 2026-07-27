from __future__ import annotations

import argparse
from pathlib import Path

from clear_card.database import extract_citation_edges


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export citation edges from SQL Server to an ignored local file."
    )
    parser.add_argument("--database", choices=["scopus", "icite"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Resolve early so creation and reporting use the same absolute path.
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = extract_citation_edges(args.database)
    frame.to_csv(
        output, index=False, compression="gzip" if output.suffix == ".gz" else None
    )
    print(f"Wrote {len(frame):,} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
