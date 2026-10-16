"""Command-line entry point for per-chain analysis; never starts a simulation."""

import sys
from pathlib import Path

# Resolve the local package independently of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from phi4_analysis.cli import main

if __name__ == "__main__":
    main()
