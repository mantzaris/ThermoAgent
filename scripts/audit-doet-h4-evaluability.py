#!/usr/bin/env python3
"""Audit and freeze a development-only, evaluable H4 collapse definition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from thermoagent.doet_analysis import (
    SEVERE_SERVICE_LOSS_PERSISTENCE,
    SEVERE_SERVICE_LOSS_THRESHOLD,
    _severe_service_collapse_step,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _old_collapse(series: pd.DataFrame, disruption_step: int) -> Optional[int]:
    pre = series[series["step"] < disruption_step]["service_loss"]
    if pre.empty:
        return None
    threshold = float(pre.mean() + 0.10)
    candidates = series[
        (series["step"] >= disruption_step)
        & (series["service_loss"] > threshold)
    ]
    return int(candidates.iloc[0]["step"]) if len(candidates) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=Path("results/entropy_triggered_v2"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.results
    output = arguments.output or root / "protocol" / "h4_evaluability_audit.json"
    rows: List[Dict[str, Any]] = []
    for stage in ("preflight_v2", "profile_v2"):
        raw = root / "raw" / stage
        for episode_path in sorted(raw.glob("*/episode.json")):
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            series = pd.DataFrame(episode["time_series"])
            disruption_steps = series[
                series["disruption_active"].map(bool)
            ]["step"]
            if disruption_steps.empty:
                continue
            disruption_step = int(disruption_steps.iloc[0])
            old = _old_collapse(series, disruption_step)
            severe = _severe_service_collapse_step(series, disruption_step)
            rows.append({
                "stage": stage,
                "source_episode": str(episode_path),
                "source_sha256": _sha256(episode_path),
                "run_id": episode["run_id"],
                "application": episode["application"],
                "method": episode["method"],
                "scenario": episode["scenario"],
                "disruption_step": disruption_step,
                "old_collapse_step": old,
                "old_strict_lead_window_periods": (
                    old - disruption_step if old is not None else None
                ),
                "revised_collapse_step": severe,
                "revised_strict_lead_window_periods": (
                    severe - disruption_step if severe is not None else None
                ),
                "trigger_activations": int(
                    episode["metrics"].get("trigger_activations", 0)
                ),
            })
    if not rows:
        raise RuntimeError("no non-nominal engineering episodes found")
    old_windows = [row["old_strict_lead_window_periods"] for row in rows]
    revised_windows = [
        row["revised_strict_lead_window_periods"] for row in rows
    ]
    primary_doet = [row for row in rows if row["method"] == "doet_rule"]
    report = {
        "status": "corrected and frozen before validation outcome inspection",
        "scope": (
            "deterministic mock preflight and real-Qwen throughput-profile "
            "episodes only; no validation or holdout output was read"
        ),
        "problem": (
            "The earlier pre-disruption-mean-plus-0.10 definition classified "
            "the disruption period itself as collapse in every inspected "
            "engineering episode because cumulative service loss rises during "
            "ordinary logistics lead-time warm-up. Strict detection lead time "
            "was therefore structurally impossible."
        ),
        "old_rule": (
            "first post-disruption service_loss > pre-disruption episode mean + 0.10"
        ),
        "revised_rule": {
            "threshold": SEVERE_SERVICE_LOSS_THRESHOLD,
            "interpretation": "at most 10% cumulative fulfillment",
            "required_consecutive_post_disruption_periods": (
                SEVERE_SERVICE_LOSS_PERSISTENCE
            ),
            "collapse_timestamp": "the final period of the first qualifying run",
            "timely_activation": (
                "first activation at or after disruption and strictly before "
                "the confirmed collapse timestamp"
            ),
            "no_collapse_rule": (
                "an episode without confirmed severe collapse receives no "
                "before-collapse success credit"
            ),
        },
        "selection_rationale": (
            "0.90 is a domain-readable severe normalized loss threshold; a "
            "three-period persistence rule rejects transients and restores a "
            "nonzero evaluation window. The inspected primary DOET engineering "
            "episodes had zero trigger activations, so this correction could "
            "not have been selected to convert a development failure into a "
            "success."
        ),
        "episodes_audited": len(rows),
        "old_zero_window_episodes": sum(value == 0 for value in old_windows),
        "revised_positive_window_episodes": sum(
            value is not None and value > 0 for value in revised_windows
        ),
        "primary_doet_engineering_episodes": len(primary_doet),
        "primary_doet_total_trigger_activations": sum(
            row["trigger_activations"] for row in primary_doet
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "episodes_audited": report["episodes_audited"],
        "old_zero_window_episodes": report["old_zero_window_episodes"],
        "revised_positive_window_episodes": report[
            "revised_positive_window_episodes"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
