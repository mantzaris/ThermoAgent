"""Command-line interface for the isolated V7 result namespace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def _paths(repository: str, results_root: Optional[str]) -> tuple:
    root = Path(repository).resolve()
    results = (
        Path(results_root).resolve()
        if results_root else root / "results" / "complexity_entropic_coordination_v7"
    )
    return root, results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ThermoAgent V7 workflow")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--results-root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pilots")
    subparsers.add_parser("pilots-iteration2")
    subparsers.add_parser("pilots-iteration3")
    subparsers.add_parser("analyze-pilots")
    subparsers.add_parser("replay")
    subparsers.add_parser("evaluate-feasibility-gates")
    subparsers.add_parser("freeze-protocol")
    subparsers.add_parser("index-artifacts")
    subparsers.add_parser("run-development-reference")
    subparsers.add_parser("run-development-dynamic")
    subparsers.add_parser("run-development-communication")
    subparsers.add_parser("evaluate-formal-development")
    subparsers.add_parser("train-multiseed")
    subparsers.add_parser("run-real-qwen")
    subparsers.add_parser("generate-figures")
    subparsers.add_parser("build-report")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--application", choices=("humanitarian", "utility_restoration"), default="humanitarian")
    args = parser.parse_args(argv)
    repository, results_root = _paths(args.repository, args.results_root)
    if args.command == "pilots":
        from .v7_workflow import run_pilots
        result = run_pilots(repository, results_root)
    elif args.command == "pilots-iteration2":
        from .v7_workflow import run_pilots
        result = run_pilots(
            repository, results_root,
            configuration_filename="v7_pilot_iteration2.yaml",
            stage="pilots_iteration2",
        )
    elif args.command == "pilots-iteration3":
        from .v7_workflow import run_pilots
        result = run_pilots(
            repository, results_root,
            configuration_filename="v7_pilot_iteration3.yaml",
            stage="pilots_iteration3",
        )
    elif args.command == "analyze-pilots":
        from .v7_analysis import analyze_pilot
        result = analyze_pilot(results_root, "pilots")
    elif args.command == "replay":
        from .v7_replay import replay_all
        result = replay_all(results_root)
    elif args.command == "evaluate-feasibility-gates":
        from .v7_gates import evaluate_feasibility_gates
        result = evaluate_feasibility_gates(repository, results_root)
    elif args.command == "freeze-protocol":
        from .v7_protocol import freeze_protocol
        result = freeze_protocol(repository, results_root)
    elif args.command == "index-artifacts":
        from .v7_artifacts import build_index, crlf_audit, verify_index
        build_index(results_root)
        verification = verify_index(results_root)
        text = crlf_audit(repository, results_root)
        result = {"index": verification, "text": text}
    elif args.command == "run-development-reference":
        from .v7_formal_workflow import run_reference_development
        result = run_reference_development(repository, results_root)
    elif args.command == "run-development-dynamic":
        from .v7_formal_workflow import run_crossfit_dynamic_development
        result = run_crossfit_dynamic_development(repository, results_root)
    elif args.command == "run-development-communication":
        from .v7_formal_workflow import run_communication_development
        result = run_communication_development(repository, results_root)
    elif args.command == "evaluate-formal-development":
        from .v7_formal_workflow import evaluate_formal_development_gates
        result = evaluate_formal_development_gates(results_root)
    elif args.command == "train-multiseed":
        from .v7_protocol import assert_stage_unlocked
        from .v7_training import train_multiseed
        assert_stage_unlocked(results_root, "RL_training")
        result = train_multiseed(
            repository, results_root,
            seeds=(78731, 78732, 78733, 78734, 78735),
            training_episodes=80, evaluation_episodes=24,
        )
    elif args.command == "run-real-qwen":
        from .v7_protocol import assert_stage_unlocked
        from .v7_qwen import run_qwen_qualification
        assert_stage_unlocked(results_root, "Qwen_qualification")
        result = run_qwen_qualification(repository, results_root)
    elif args.command == "generate-figures":
        from .v7_figures import generate_v7_figures
        result = generate_v7_figures(results_root)
    elif args.command == "build-report":
        from .v7_reporting import build_v7_report
        result = build_v7_report(repository, results_root)
    elif args.command == "smoke":
        from .v7_experiments import run_episode
        from .v7_policies import V7SelectiveController
        topology = "small_world" if args.application == "humanitarian" else "grid"
        result = run_episode(
            args.application, "small", "medium", "medium", "medium",
            topology, 770001, V7SelectiveController("kpi_confidence", 0.60),
            results_root=results_root, stage="smoke",
            counterfactual_limit_per_epoch=1,
        )["summary"]
    else:
        raise RuntimeError("unsupported V7 command")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
