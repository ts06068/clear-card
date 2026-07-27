from __future__ import annotations

import sys

from clear_card.cli import main


if __name__ == "__main__":
    # Keep this convenience script thin; the package CLI owns all options.
    raise SystemExit(main(["reproduce", *sys.argv[1:]]))
