from __future__ import annotations

import sys

from clear_card.cli import main

if __name__ == "__main__":
    # Delegate parsing to the package CLI; this script is only a shortcut.
    raise SystemExit(main(["reproduce", *sys.argv[1:]]))
