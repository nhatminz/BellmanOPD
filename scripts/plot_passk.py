#!/usr/bin/env python3
"""CLI entry point for publication-quality Pass@K figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b200_experiment.passk_plotting import main


if __name__ == "__main__":
    raise SystemExit(main())
