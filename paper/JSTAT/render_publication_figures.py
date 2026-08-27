#!/usr/bin/env python3
"""Compatibility entry point for the canonical publication figure generator.

Earlier V15 revisions carried a second presentation-only renderer whose
layouts could diverge from the source-data catalog.  Publication figures now
have one implementation: :mod:`thermoagent.statmech_llm_v15.figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thermoagent.statmech_llm_v15.figures import (
    generate_figures,
    generate_selected_figures,
)


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        action="append",
        choices=("all", "1", "2", "4"),
        help=(
            "regenerate all figures (default) or repeat this option to select "
            "canonical figures 1, 2, and/or 4"
        ),
    )
    arguments = parser.parse_args()
    selections = arguments.figure or ["all"]
    if "all" in selections:
        if len(selections) != 1:
            parser.error("--figure all cannot be combined with targeted figures")
        generate_figures(ROOT)
    else:
        generate_selected_figures(ROOT, [int(value) for value in selections])
