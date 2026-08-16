"""Prospective V7 pilot design and guarded stage workflow."""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from .v5_experiments import atomic_json, write_csv
from .v7_analysis import analyze_pilot
from .v7_experiments import aggregate_stage, run_episode
from .v7_policies import V7SelectiveController


def _configuration(repository: Path, filename: str = "v7_pilot.yaml") -> Dict[str, Any]:
    return dict(yaml.safe_load(
        (repository / "configs" / filename).read_text(encoding="utf-8")
    ))


def run_pilots(
    repository: Path, results_root: Path, resume: bool = True,
    configuration_filename: str = "v7_pilot.yaml", stage: str = "pilots",
) -> Dict[str, Any]:
    lock_path = results_root / "logs" / (stage + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stage_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another V7 pilot writer holds the exclusive stage lock")
    stage_lock.seek(0)
    stage_lock.truncate()
    stage_lock.write("pid=%d\n" % os.getpid())
    stage_lock.flush()
    configuration = _configuration(repository, configuration_filename)
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row in configuration["panels"]:
        methods = row.get("controllers", ["always_act"])
        sketches = row.get("sketch_policies", ["event_triggered"])
        for method in methods:
            for sketch in sketches:
                controller = V7SelectiveController(
                    str(method),
                    float(row.get("autonomous_coverage", 1.0 if method == "always_act" else 0.60)),
                    int(row.get("operator_slots_per_epoch", 1)),
                )
                try:
                    output = run_episode(
                        str(row["application"]), str(row["complexity"]),
                        str(row["coupling"]), str(row["fragmentation"]),
                        str(row["network_disruption"]), str(row["topology_family"]),
                        int(row["environment_seed"]), controller,
                        str(row.get("information_condition", "private_fragmented")),
                        str(sketch), results_root, stage,
                        int(row.get("counterfactual_limit_per_epoch", 2)),
                        operational_communication_policy=str(
                            row.get(
                                "operational_communication_policy",
                                "agent_event_triggered",
                            )
                        ),
                    )
                    completed.append(dict(output["summary"]))
                except Exception as error:
                    failures.append({
                        **dict(row), "controller": method, "sketch_policy": sketch,
                        "failure_type": type(error).__name__, "failure_reason": str(error),
                    })
    if failures:
        write_csv(results_root / "negative_results" / "pilot_failures.csv", failures)
    execution = aggregate_stage(results_root, stage)
    analysis = analyze_pilot(results_root, stage)
    report = {
        "stage": stage, "completed_runs": len(completed),
        "failed_runs": len(failures), "execution": execution,
        "analysis": analysis,
    }
    atomic_json(results_root / stage / "pilot_run_summary.json", report)
    return report
