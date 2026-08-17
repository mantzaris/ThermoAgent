"""Command-line entry points for the V8 staged workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .v8_analysis import (
    analyze_v8_development, analyze_v8_final_development,
    analyze_v8_hysteresis_repair, analyze_v8_pilot,
    analyze_v8_pilot_no_go, analyze_v8_primary_stage,
    analyze_v8_seed_stability, combine_v8_development_gates,
    analyze_v8_estimator_calibration,
)
from .v8_training import train_v8_multiseed
from .v8_workflow import run_configured_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("results/entropy_triggered_belief_monitoring_v8"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilots = subparsers.add_parser("pilots")
    pilots.add_argument("--config", default="v8_pilot.yaml")
    pilots.add_argument("--stage", default="pilots")
    pilots.add_argument("--encoding-checks", action="store_true")
    analysis = subparsers.add_parser("analyze-pilots")
    analysis.add_argument("--stage", default="pilots")
    development = subparsers.add_parser("analyze-development")
    development.add_argument("--stage", default="development")
    development.add_argument("--bootstrap-replicates", type=int, default=10000)
    final_development = subparsers.add_parser("analyze-final-development")
    final_development.add_argument("--stage", default="development_final")
    final_development.add_argument("--bootstrap-replicates", type=int, default=10000)
    hysteresis_repair = subparsers.add_parser("analyze-hysteresis-repair")
    hysteresis_repair.add_argument("--stage", default="hysteresis_repair_pilot")
    no_go = subparsers.add_parser("analyze-no-go")
    no_go.add_argument("--stage", default="hysteresis_repair_pilot_v3")
    no_go.add_argument("--bootstrap-replicates", type=int, default=10000)
    training = subparsers.add_parser("train-multiseed")
    training.add_argument("--seeds", default="88201,88202,88203,88204,88205")
    training.add_argument("--episodes", type=int, default=18)
    training.add_argument("--update-epochs", type=int, default=4)
    primary = subparsers.add_parser("analyze-primary")
    primary.add_argument("--stage", required=True)
    stability = subparsers.add_parser("analyze-seed-stability")
    stability.add_argument("--stage", default="seed_stability")
    subparsers.add_parser("combine-development-gates")
    calibration = subparsers.add_parser("analyze-calibration")
    calibration.add_argument("--stage", required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    results_root = arguments.results_root
    if not results_root.is_absolute():
        results_root = repository / results_root
    if arguments.command == "pilots":
        result = run_configured_stage(
            repository, results_root,
            configuration_filename=arguments.config,
            stage=arguments.stage,
            include_encoding_checks=arguments.encoding_checks,
        )
    elif arguments.command == "analyze-pilots":
        result = analyze_v8_pilot(results_root, arguments.stage)
    elif arguments.command == "analyze-development":
        result = analyze_v8_development(
            results_root, arguments.stage,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
    elif arguments.command == "analyze-final-development":
        result = analyze_v8_final_development(
            results_root, arguments.stage,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
    elif arguments.command == "analyze-hysteresis-repair":
        result = analyze_v8_hysteresis_repair(results_root, arguments.stage)
    elif arguments.command == "analyze-no-go":
        result = analyze_v8_pilot_no_go(
            results_root, arguments.stage,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
    elif arguments.command == "train-multiseed":
        seeds = tuple(int(value) for value in arguments.seeds.split(",") if value.strip())
        result = train_v8_multiseed(
            repository=repository, results_root=results_root, seeds=seeds,
            training_episodes=arguments.episodes,
            update_epochs=arguments.update_epochs,
        )
    elif arguments.command == "analyze-primary":
        result = analyze_v8_primary_stage(results_root, arguments.stage)
    elif arguments.command == "analyze-seed-stability":
        result = analyze_v8_seed_stability(results_root, arguments.stage)
    elif arguments.command == "combine-development-gates":
        result = combine_v8_development_gates(results_root)
    else:
        result = analyze_v8_estimator_calibration(results_root, arguments.stage)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
