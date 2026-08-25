#!/usr/bin/env python3
"""Compatibility entry point for the canonical V15 figure generator.

Earlier V15 revisions carried a second presentation-only renderer whose
layouts could diverge from the source-data catalog.  Publication figures now
have one implementation: :mod:`thermoagent.statmech_llm_v15.figures`.
"""

from __future__ import annotations

from pathlib import Path

from thermoagent.statmech_llm_v15.figures import generate_figures


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    generate_figures(ROOT)
