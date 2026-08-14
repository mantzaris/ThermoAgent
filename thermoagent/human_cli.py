"""One-command, fail-closed workflows for the ThermoHITL v3 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .human_experiments import (
    V3_RESULTS,
    calibrate_human_thermodynamics,
    diagnose_v2_actionability,
    run_human_matrix,
    train_human_multiseed,
    utc_now,
    _atomic_json,
)
from .experiments import load_yaml
from .human_gates import assert_holdout_unlocked, evaluate_development_gates
from .human_monitoring import analyze_dense_causal_value
from .human_operator import EscalationConfig, HumanMethod


FINAL_DEVELOPMENT_TRIGGER = EscalationConfig(
    tau_on=1.5,
    tau_off=0.6,
    actionable_tau_on=1.1,
    minimum_dwell=2,
    cooldown=3,
)


def _development(root: Path, results: Path) -> Dict[str, Any]:
    """Complete or resume the prospectively declared CPU development stages."""

    config = load_yaml(root / "configs" / "human_operator_development.yaml")
    calibration = results / "calibration" / "thermodynamic_calibration_n10.json"
    if not calibration.is_file():
        calibrate_human_thermodynamics(
            results,
            seeds=config["calibration"]["seeds"],
            horizon=int(config["horizon"]),
            n_agents=int(config["n_agents"]),
            topology=str(config["topology"]),
        )
    stages = []
    matrices = config["matrices"]
    trigger = EscalationConfig(**config["trigger"])
    for matrix in matrices:
        stage = str(matrix["stage"])
        summary = results / stage / "episode_summary.csv"
        if summary.is_file():
            stages.append({"stage": stage, "status": "already_complete"})
            continue
        rows = run_human_matrix(
            root=root,
            results_root=results,
            stage=stage,
            methods=matrix["methods"],
            applications=config["applications"],
            regimes=matrix["regimes"],
            communications=config["communication"],
            seeds=matrix["seeds"],
            horizon=int(config["horizon"]),
            n_agents=int(config["n_agents"]),
            topology=str(config["topology"]),
            planner_backend="mock",
            calibration_path=calibration,
            escalation_config=trigger,
            counterfactual_probes=bool(matrix.get("counterfactual_probes")),
            dense_counterfactual_probes=bool(matrix.get("dense_counterfactual_probes")),
        )
        stages.append({"stage": stage, "status": "complete", "episodes": len(rows)})
    monitoring = analyze_dense_causal_value(results, "dense_causal_development_n10_v3")
    gates = evaluate_development_gates(root, results)
    return {"stages": stages, "monitoring": monitoring, "gates": gates}


def _real_profile(results: Path, episodes: int = 4) -> Dict[str, Any]:
    # Frozen antecedent throughput measurement: 8 v2 episodes, 714.10 seconds
    # including model load. The v3 qualification is deliberately bounded and
    # uses a conservative additive load reserve rather than optimistic scaling.
    antecedent_seconds_per_episode = 714.10 / 8.0
    projected_seconds = 180.0 + episodes * antecedent_seconds_per_episode
    buffered = projected_seconds * 1.15
    record = {
        "created_at": utc_now(),
        "stage": "v3 real-LLM actionability qualification projection",
        "source_measurement": "v2 real-Qwen profile: 8 episodes / 714.10 seconds including load",
        "planned_episodes": episodes,
        "model_load_reserve_seconds": 180.0,
        "projected_seconds_before_safety": projected_seconds,
        "safety_reserve_fraction": 0.15,
        "projected_single_gpu_hours": buffered / 3600.0,
        "projected_cost_usd_at_0_34_per_hour": buffered / 3600.0 * 0.34,
        "projected_llm_calls": episodes * 70,
        "projected_prompt_tokens": episodes * 140_000,
        "projected_generated_tokens": episodes * 9_000,
        "projected_disk_mb": 20,
        "cap_single_gpu_hours": 40.0,
        "training_validation_holdout_hours": 0.0,
        "reason": "Gate 5 already blocks all expensive stages; only Gate 2 model qualification remains",
    }
    _atomic_json(results / "reproducibility" / "v3_real_llm_prelaunch_projection.json", record)
    return record


def _real_actionability(
    root: Path,
    results: Path,
    seeds: Sequence[int],
    stage: str = "development_real_llm_actionability",
) -> Dict[str, Any]:
    _real_profile(results, episodes=2 * len(seeds))
    config = load_yaml(root / "configs" / "human_operator_real_llm_actionability.yaml")
    development = load_yaml(root / "configs" / "human_operator_development.yaml")
    calibration = results / "calibration" / "thermodynamic_calibration_n10.json"
    rows = run_human_matrix(
        root=root,
        results_root=results,
        stage=stage,
        methods=config["methods"],
        applications=config["applications"],
        regimes=config["regimes"],
        communications=config["communications"],
        seeds=seeds,
        horizon=int(config["horizon"]),
        n_agents=int(config["n_agents"]),
        topology=str(config["topology"]),
        planner_backend=str(config["planner_backend"]),
        llm_seed=int(config["llm_seed"]),
        calibration_path=calibration,
        escalation_config=EscalationConfig(**development["trigger"]),
        operator_profile=str(config["operator_profile"]),
        counterfactual_probes=bool(config["counterfactual_probes"]),
    )
    gates = evaluate_development_gates(root, results)
    return {"episodes": len(rows), "gate_2": gates["gates"][1], "gate_decision": gates["decision"]}


def _guarded_stage(results: Path, name: str) -> Dict[str, Any]:
    assert_holdout_unlocked(results)
    raise NotImplementedError(
        "%s was not implemented because the prospective no-go gate blocked it; "
        "a future study must version a new protocol rather than bypass this guard" % name
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermoagent-human")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--results", type=Path, default=V3_RESULTS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diagnostics")
    sub.add_parser("calibrate")
    sub.add_parser("development")
    sub.add_parser("monitoring")
    sub.add_parser("gates")
    sub.add_parser("profile-real-llm")
    real = sub.add_parser("real-llm-actionability")
    real.add_argument("--seeds", default="13101,13102")
    real.add_argument("--stage", default="development_real_llm_actionability")
    sub.add_parser("train")
    sub.add_parser("validation")
    sub.add_parser("design-holdout")
    sub.add_parser("freeze-holdout")
    sub.add_parser("holdout")
    sub.add_parser("counterfactuals")
    sub.add_parser("ablations")
    sub.add_parser("analyze")
    sub.add_parser("figures")
    replay = sub.add_parser("replay")
    replay.add_argument("--report-name", default="development_replay_report.json")
    sub.add_parser("validate-pdfs")
    sub.add_parser("index")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    results = args.results
    if args.command == "diagnostics":
        value: Any = diagnose_v2_actionability(root, results)
    elif args.command == "calibrate":
        value = calibrate_human_thermodynamics(results, horizon=24, n_agents=10)
    elif args.command == "development":
        value = _development(root, results)
    elif args.command == "monitoring":
        value = analyze_dense_causal_value(results)
    elif args.command == "gates":
        value = evaluate_development_gates(root, results)
    elif args.command == "profile-real-llm":
        value = _real_profile(results)
    elif args.command == "real-llm-actionability":
        seeds = [int(item) for item in args.seeds.split(",") if item]
        value = _real_actionability(root, results, seeds, args.stage)
    elif args.command == "train":
        assert_holdout_unlocked(results)
        value = train_human_multiseed(results)
    elif args.command in ("validation", "design-holdout", "freeze-holdout", "holdout"):
        value = _guarded_stage(results, args.command)
    elif args.command in ("counterfactuals", "ablations", "analyze"):
        from .human_analysis import run

        value = run(root, results)
    elif args.command == "figures":
        from .human_figures import run

        value = run(results)
    elif args.command == "replay":
        from .human_replay import replay_human_results

        stages = sorted(path.name for path in (results / "raw").iterdir() if path.is_dir())
        value = replay_human_results(results, stages, args.report_name)
    elif args.command == "validate-pdfs":
        from .figures import validate_pdfs

        value = validate_pdfs(results)
    elif args.command == "index":
        from .human_analysis import build_index

        value = {"index": str(build_index(results))}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
