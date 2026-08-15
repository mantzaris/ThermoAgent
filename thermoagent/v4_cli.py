"""Command-line entry points for restartable ThermoHITL v4 workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .v4_analysis import analyze_v4_development
from .v4_experiments import initialize_v4_results, run_matrix
from .v4_qwen import run_real_qwen_qualification
from .v4_replay import replay_v4_results


APPLICATIONS = ("commercial", "humanitarian", "utility_restoration")
DISRUPTED_REGIMES = (
    "isolated_physical", "telemetry_integrity", "partition", "correlated", "compound",
)
ALL_DEVELOPMENT_REGIMES = ("nominal", *DISRUPTED_REGIMES)
DEVELOPMENT_SEEDS = tuple(range(24001, 24013))


def _repository(value: str) -> Path:
    return Path(value).resolve()


def run_development(repository: Path) -> dict:
    root = repository / "results" / "human_operator_v4"
    initialize_v4_results(root)
    reports = []
    reports.append(run_matrix(
        repository, root, "development_gate_coordination",
        applications=APPLICATIONS,
        regimes=DISRUPTED_REGIMES,
        information_conditions=("private_fragmented",),
        methods=("no_communication", "fixed_communication"),
        environment_seeds=DEVELOPMENT_SEEDS,
        counterfactual_probes=False,
        dense_candidates=False,
        operator_budget=1,
        resume=True,
    ))
    reports.append(run_matrix(
        repository, root, "development_gate_human",
        applications=APPLICATIONS,
        regimes=DISRUPTED_REGIMES,
        information_conditions=("private_fragmented",),
        methods=("autonomy_no_operator", "thermohitl_v4_rule"),
        environment_seeds=DEVELOPMENT_SEEDS,
        counterfactual_probes=True,
        dense_candidates=False,
        operator_budget=1,
        resume=True,
    ))
    reports.append(run_matrix(
        repository, root, "development_gate_monitoring",
        applications=APPLICATIONS,
        regimes=ALL_DEVELOPMENT_REGIMES,
        information_conditions=("private_fragmented", "globally_public"),
        methods=("autonomy_no_operator",),
        environment_seeds=DEVELOPMENT_SEEDS,
        counterfactual_probes=False,
        dense_candidates=True,
        operator_budget=1,
        resume=True,
    ))
    reports.append(run_matrix(
        repository, root, "development_gate_trigger",
        applications=APPLICATIONS,
        regimes=ALL_DEVELOPMENT_REGIMES,
        information_conditions=("private_fragmented", "globally_public"),
        methods=("thermohitl_v4_rule",),
        environment_seeds=DEVELOPMENT_SEEDS,
        counterfactual_probes=False,
        dense_candidates=False,
        operator_budget=1,
        resume=True,
    ))
    report = {
        "workflow": "formal_v4_development",
        "reports": reports,
        "episodes": sum(int(value["episodes"]) for value in reports),
        "failures": sum(int(value["failures"]) for value in reports),
        "resume_safe": True,
    }
    destination = root / "development" / "formal_development_run_report.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("development")
    subparsers.add_parser("replay")
    subparsers.add_parser("analyze")
    subparsers.add_parser("real-qwen")
    args = parser.parse_args(argv)
    repository = _repository(args.root)
    results_root = repository / "results" / "human_operator_v4"
    if args.command == "development":
        value = run_development(repository)
    elif args.command == "replay":
        value = replay_v4_results(results_root)
    elif args.command == "analyze":
        value = analyze_v4_development(repository)
    else:
        value = run_real_qwen_qualification(repository, results_root)
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
