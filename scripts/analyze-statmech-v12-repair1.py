#!/usr/bin/env python3
"""Run the frozen V12 analysis with a documented bookkeeping-only repair.

The first post-formal analysis attempt failed before writing result tables because
``_panel_statistics`` omitted the prospectively generated ``replicate`` field
from its summary row, while ``_factor_effects`` grouped on that field.  The raw
formal trajectories, estimands, grouping rule, bootstrap, and thresholds are
unchanged.  Keeping this wrapper outside the frozen execution-source checksum
preserves the exact formal source while making the repair explicit and
reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from thermoagent.statmech_llm_v12 import analysis
from thermoagent.statmech_llm_v12.workflow import repository_root


def install_replicate_bookkeeping_repair() -> Callable[..., object]:
    """Copy the design's replicate identifier into each panel summary row."""

    original = analysis._panel_statistics

    def repaired(frame, protocol, panel_definition):
        row, currents = original(frame, protocol, panel_definition)
        row["replicate"] = int(panel_definition["replicate"])
        return row, currents

    analysis._panel_statistics = repaired
    return original


def self_test() -> None:
    """Exercise the repair without reading any formal result."""

    actual = analysis._panel_statistics

    def sentinel(frame, protocol, panel_definition):
        return {"panel_id": "sentinel"}, []

    try:
        analysis._panel_statistics = sentinel
        install_replicate_bookkeeping_repair()
        row, currents = analysis._panel_statistics(None, None, {"replicate": 7})
        assert row == {"panel_id": "sentinel", "replicate": 7}
        assert currents == []
    finally:
        analysis._panel_statistics = actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        print("V12 analysis repair self-test passed")
        return
    install_replicate_bookkeeping_repair()
    result = analysis.analyze_formal(Path(repository_root()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
