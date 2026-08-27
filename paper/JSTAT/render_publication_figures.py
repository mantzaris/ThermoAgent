#!/usr/bin/env python3
"""Compatibility entry point for the canonical V15 figure generator.

Earlier V15 revisions carried a second presentation-only renderer whose
layouts could diverge from the source-data catalog.  Publication figures now
have one implementation: :mod:`thermoagent.statmech_llm_v15.figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thermoagent.statmech_llm_v15.figures import generate_figure1, generate_figures


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("all", "1"),
        default="all",
        help="regenerate all figures (default) or only the architecture figure",
    )
    arguments = parser.parse_args()
    generator = generate_figure1 if arguments.figure == "1" else generate_figures
    generator(ROOT)
