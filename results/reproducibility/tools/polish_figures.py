#!/usr/bin/env python3
"""Apply post-freeze, presentation-only layout fixes to selected figures.

The experimental protocol and frozen figure/data logic remain untouched.  This
wrapper intercepts the final save call to adjust labels and axes after the
frozen plotting functions have constructed each figure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from thermoagent import figures


def polish(results_root: Path) -> dict[str, Any]:
    results_root = results_root.resolve()
    original_save = figures._save
    regenerated: list[str] = []

    def polished_save(fig: Any, name: str, root: Path) -> tuple[Path, Path]:
        if name == "communication_performance_pareto":
            fig.set_size_inches(7.2, 3.8, forward=True)
            for axis in fig.axes[:2]:
                axis.set_xlabel("Messages per episode\n(operational + monitor)")
            if fig.legends:
                old_legend = fig.legends[0]
                handles = getattr(old_legend, "legend_handles", None)
                if handles is None:
                    handles = old_legend.legendHandles
                labels = [text.get_text() for text in old_legend.get_texts()]
                old_legend.remove()
                fig.legend(
                    handles=handles,
                    labels=labels,
                    loc="lower center",
                    ncol=3,
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.015),
                )
            fig.tight_layout(rect=(0, 0.30, 1, 0.94), w_pad=3.6)
        elif name.startswith("network_snapshots_"):
            fig.set_size_inches(7.2, 6.5, forward=True)
            for axis in fig.axes[:5]:
                axis.set_xlim(-1.55, 1.55)
                axis.set_ylim(-1.55, 1.55)
                axis.set_aspect("equal", adjustable="box")
            fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.94), h_pad=1.6, w_pad=1.2)
        regenerated.append(name + ".pdf")
        return original_save(fig, name, root)

    figures._save = polished_save
    try:
        episodes = pd.read_csv(results_root / "processed" / "episodes.csv")
        figures.communication_pareto(episodes, results_root)
        figures.network_snapshots(results_root, "commercial")
        figures.network_snapshots(results_root, "humanitarian")
    finally:
        figures._save = original_save

    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "presentation-only post-freeze visual QA correction",
        "frozen_protocol_modified": False,
        "regenerated": regenerated,
        "changes": [
            "wrapped Pareto x-axis labels and increased inter-panel spacing",
            "moved the Pareto legend fully inside a taller three-column layout",
            "expanded network-panel limits so coalition outlines are not clipped",
        ],
        "command": "./results/reproducibility/tools/polish-figures.sh",
    }
    output = results_root / "reproducibility" / "postfreeze_figure_polish.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(polish(args.results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
