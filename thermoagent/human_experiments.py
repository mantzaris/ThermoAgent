"""Restartable v3 calibration, development, training, and evaluation workflows."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import EventLedger, sha256_file
from .experiments import git_provenance, source_checksum
from .human_environment import HumanOversightEnvironment, HumanScenarioConfig
from .human_operator import (
    DistributedThermodynamicMonitor,
    EscalationConfig,
    HumanMethod,
    ThermodynamicCalibration,
)
from .human_policy import HumanAttentionPolicy, train_contextual_bandit
from .human_runner import HumanOperatorEpisodeRunner, write_human_episode
from .planners import PLANNER_PROMPT_REVISION, MockPlanner, TransformersPlanner


PRIMARY_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PRIMARY_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
V3_RESULTS = Path("results/human_operator_v3")


V3_DIRECTORIES = (
    "protocol", "diagnostics", "development", "monitoring",
    "operator_models", "dashboard", "validation", "training",
    "checkpoints", "holdout_locked", "counterfactuals", "raw",
    "processed", "statistics", "tables", "figures/pdf",
    "figures/previews", "logs/setup", "logs/diagnostics",
    "logs/training", "logs/validation", "logs/holdout",
    "logs/analysis", "manifests", "reproducibility",
    "reproducibility/pdf_qa",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_v3_tree(results_root: Path = V3_RESULTS) -> None:
    for relative in V3_DIRECTORIES:
        (results_root / relative).mkdir(parents=True, exist_ok=True)


def _dependency_versions() -> Dict[str, str]:
    packages = (
        "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
        "torch", "transformers", "bitsandbytes", "accelerate", "PyMuPDF",
    )
    output = {"python": platform.python_version()}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


def _hardware_summary() -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        value.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as error:  # pragma: no cover - environment diagnostic
        value["torch_error"] = type(error).__name__
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _calibration_rows(
    application: str,
    seeds: Sequence[int],
    horizon: int,
    n_agents: int,
    topology: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []
    disabled_escalation = EscalationConfig(tau_on=999.0, tau_off=998.0)
    for seed in seeds:
        config = HumanScenarioConfig(
            application=application,
            seed=int(seed),
            horizon=int(horizon),
            n_agents=int(n_agents),
            topology=topology,
            disruption="nominal",
            communication="reliable",
            decision_interval=2,
            communication_budget=100,
            operator_seed=int(seed) + 20_000,
        )
        runner = HumanOperatorEpisodeRunner(
            config,
            HumanMethod.THERMOHITL_RULE.value,
            escalation_config=disabled_escalation,
            enable_counterfactual_probes=False,
        )
        result = runner.run("calibration-%s-s%d" % (application, seed))
        for event in runner.env.ledger.events:
            if event.kind != "thermodynamic_state":
                continue
            payload = event.payload
            rows.append({
                "application": application,
                "seed": int(seed),
                "step": int(event.step),
                "agent_id": event.actor,
                "role": payload["role"],
                "energy": float(payload["distributed_energy"]),
                "distributed_entropy": float(payload["distributed_entropy"]),
                "flow_entropy": float(payload["flow_entropy"]),
                "belief_entropy": float(payload["belief_entropy"]),
                "free_energy": float(payload["free_energy"]),
            })
        episode_summaries.append({
            "application": application,
            "seed": int(seed),
            "primary_outcome": result.metrics["primary_outcome"],
            "material_actions_reached_demand": result.metrics["material_actions_reached_demand"],
            "conservation_error": result.metrics["conservation_error"],
        })
    return rows, {"episodes": episode_summaries}


def calibrate_human_thermodynamics(
    results_root: Path = V3_RESULTS,
    seeds: Sequence[int] = (10401, 10402, 10403, 10404, 10405, 10406),
    horizon: int = 24,
    n_agents: int = 10,
    topology: str = "human_v3_development",
    artifact_stem: str = "thermodynamic_calibration_n10",
) -> Dict[str, Any]:
    """Fit nominal statistics before any v3 disruption outcome comparison."""

    ensure_v3_tree(results_root)
    all_rows: List[Dict[str, Any]] = []
    applications: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian"):
        rows, summary = _calibration_rows(
            application, seeds, horizon, n_agents, topology
        )
        calibration = ThermodynamicCalibration.fit(rows)
        applications[application] = asdict(calibration)
        diagnostics[application] = summary
        all_rows.extend(rows)
    record = {
        "stage": "development_nominal_calibration",
        "created_at": utc_now(),
        "data_boundary": "nominal development seeds only",
        "seeds": [int(seed) for seed in seeds],
        "horizon": int(horizon),
        "n_agents": int(n_agents),
        "topology": topology,
        "applications": applications,
        "energy_weights": {
            "backlog": 0.24, "unmet": 0.22, "congestion": 0.16,
            "lateness": 0.14, "commitment": 0.12, "safety": 0.12,
        },
        "diagnostics": diagnostics,
    }
    calibration_path = results_root / "calibration" / (artifact_stem + ".json")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(calibration_path, record)
    csv_path = results_root / "calibration" / (artifact_stem + "_agent_periods.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    record["artifacts"] = {
        "calibration": str(calibration_path),
        "rows": str(csv_path),
        "calibration_sha256": sha256_file(calibration_path),
        "rows_sha256": sha256_file(csv_path),
    }
    return record


def load_human_calibrations(path: Path) -> Dict[str, ThermodynamicCalibration]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        application: ThermodynamicCalibration.from_mapping(parameters)
        for application, parameters in value["applications"].items()
    }


def _run_id(
    stage: str,
    application: str,
    method: str,
    regime: str,
    communication: str,
    seed: int,
    operator_seed: int,
    rl_seed: Optional[int],
) -> str:
    return "%s-%s-%s-%s-%s-e%05d-o%05d-r%s" % (
        stage,
        application,
        method,
        regime,
        communication,
        int(seed),
        int(operator_seed),
        "none" if rl_seed is None else "%05d" % int(rl_seed),
    )


def _topology_checksum(environment: HumanOversightEnvironment) -> str:
    topology = next(
        event.payload for event in environment.ledger.events
        if event.kind == "topology_snapshot"
    )
    return hashlib.sha256(
        json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest(
    root: Path,
    results_root: Path,
    stage: str,
    result: Any,
    runner: HumanOperatorEpisodeRunner,
    output_checksums: Mapping[str, str],
    started_at: str,
    ended_at: str,
    planner_backend: str,
    trigger_config: EscalationConfig,
    protocol_checksum: Optional[str],
    checkpoint: Optional[Path],
) -> Dict[str, Any]:
    uses_llm = planner_backend == "transformers"
    return {
        "run_id": result.run_id,
        "study": "thermohitl_v3",
        "stage": stage,
        "source": {
            **git_provenance(root),
            "checksum": source_checksum(root),
        },
        "protocol_checksum": protocol_checksum,
        "configuration": asdict(runner.config),
        "application": result.application,
        "method": result.method,
        "model_identifier": PRIMARY_MODEL_ID if uses_llm else "none",
        "model_revision": PRIMARY_MODEL_REVISION if uses_llm else "none",
        "precision": "bitsandbytes NF4; BF16 compute" if uses_llm else "none",
        "serving_library": "transformers 4.55.4" if uses_llm else "none",
        "planner_revision": result.planner_metrics["planner_revision"],
        "prompt_revision": PLANNER_PROMPT_REVISION,
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
        "trigger_type": result.method,
        "trigger_parameters": asdict(trigger_config),
        "communication_mode": _underlying_communication(result.method),
        "operator_profile": result.operator_profile,
        "operator_view": result.operator_view,
        "environment_seed": result.environment_seed,
        "llm_seed": result.llm_seed,
        "rl_training_seed": result.rl_seed,
        "operator_seed": result.operator_seed,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None,
        "topology_identifier": runner.config.topology,
        "topology_checksum": _topology_checksum(runner.env),
        "dependencies": _dependency_versions(),
        "hardware": _hardware_summary(),
        "start_timestamp": started_at,
        "end_timestamp": ended_at,
        "wall_clock_seconds": result.wall_clock_seconds,
        "single_gpu_hours": result.wall_clock_seconds / 3600.0 if uses_llm else 0.0,
        "estimated_gpu_cost_usd": result.wall_clock_seconds / 3600.0 * 0.34 if uses_llm else 0.0,
        "environment_steps": runner.config.horizon,
        "llm_calls": result.planner_metrics["llm_calls"],
        "prompt_tokens": result.planner_metrics["prompt_tokens"],
        "generated_tokens": result.planner_metrics["generated_tokens"],
        "tool_calls": runner.env.tool_calls,
        "agent_messages": result.metrics["agent_messages"],
        "entropy_energy_sketch_messages": result.metrics["thermodynamic_sketch_messages"],
        "operator_messages": result.metrics["operator_messages"],
        "structured_bytes": result.metrics["total_communication_bytes"],
        "completion_status": result.completion_status,
        "failure_reason": None,
        "output_checksums": dict(output_checksums),
    }


def _underlying_communication(method: str) -> str:
    if method == HumanMethod.NO_COMMUNICATION.value:
        return "none"
    if method == HumanMethod.FIXED_COMMUNICATION_NO_HUMAN.value:
        return "fixed_always_on"
    return "periodic_independent_agents_plus_operator_if_applicable"


def _existing_complete(
    manifest_path: Path,
    episode_dir: Path,
    expected_source: str,
) -> Optional[Dict[str, Any]]:
    if not manifest_path.is_file() or not (episode_dir / "episode.json").is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completion_status") != "complete":
        return None
    if manifest.get("source", {}).get("checksum") != expected_source:
        raise RuntimeError("run-ID collision with a different source checksum")
    for name, checksum in manifest.get("output_checksums", {}).items():
        path = episode_dir / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise RuntimeError("published v3 episode checksum mismatch: %s" % path)
    return json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))


def run_human_matrix(
    root: Path,
    results_root: Path,
    stage: str,
    methods: Sequence[str],
    applications: Sequence[str],
    regimes: Sequence[str],
    communications: Sequence[str],
    seeds: Sequence[int],
    operator_seeds: Optional[Sequence[int]] = None,
    horizon: int = 24,
    n_agents: int = 8,
    topology: str = "human_v3_development",
    planner_backend: str = "mock",
    llm_seed: int = 0,
    calibration_path: Optional[Path] = None,
    escalation_config: Optional[EscalationConfig] = None,
    checkpoint_map: Optional[Mapping[Tuple[str, int], Path]] = None,
    rl_seeds: Sequence[Optional[int]] = (None,),
    operator_profile: str = "high_accuracy_bounded",
    protocol_checksum: Optional[str] = None,
    counterfactual_probes: bool = True,
    dense_counterfactual_probes: bool = False,
) -> List[Dict[str, Any]]:
    ensure_v3_tree(results_root)
    if planner_backend not in ("mock", "transformers"):
        raise ValueError("planner backend must be mock or transformers")
    calibrations = (
        load_human_calibrations(calibration_path)
        if calibration_path else {
            application: ThermodynamicCalibration() for application in applications
        }
    )
    trigger = escalation_config or EscalationConfig()
    planner: Any = MockPlanner()
    if planner_backend == "transformers":
        planner = TransformersPlanner(
            PRIMARY_MODEL_ID,
            PRIMARY_MODEL_REVISION,
            max_new_tokens=128,
            max_input_tokens=2560,
            load_in_4bit=True,
            seed=llm_seed,
        )
    expected_source = source_checksum(root)
    summaries: List[Dict[str, Any]] = []
    operator_seeds = list(operator_seeds or [int(seed) + 30_000 for seed in seeds])
    if len(operator_seeds) not in (1, len(seeds)):
        raise ValueError("operator seeds must contain one value or match environment seeds")
    for application in applications:
        for regime in regimes:
            for communication in communications:
                for seed_index, seed in enumerate(seeds):
                    operator_seed = operator_seeds[0] if len(operator_seeds) == 1 else operator_seeds[seed_index]
                    for method_value in methods:
                        learned = method_value in (
                            HumanMethod.LEARNED_NO_THERMODYNAMICS.value,
                            HumanMethod.THERMOHITL_RL.value,
                        )
                        selected_rl_seeds = rl_seeds if learned else (None,)
                        for rl_seed in selected_rl_seeds:
                            checkpoint: Optional[Path] = None
                            scorer: Optional[Any] = None
                            if learned:
                                if rl_seed is None:
                                    raise ValueError("learned methods require an RL seed")
                                checkpoint = (checkpoint_map or {}).get((method_value, int(rl_seed)))
                                if checkpoint is None or not checkpoint.is_file():
                                    raise FileNotFoundError("missing learned checkpoint for %s seed %s" % (method_value, rl_seed))
                                scorer = HumanAttentionPolicy.load(checkpoint)
                            run_id = _run_id(
                                stage, application, method_value, regime,
                                communication, seed, operator_seed, rl_seed,
                            )
                            episode_dir = results_root / "raw" / stage / run_id
                            manifest_path = results_root / "manifests" / (run_id + ".json")
                            existing = _existing_complete(manifest_path, episode_dir, expected_source)
                            if existing is not None:
                                summaries.append(_summary_row(existing, manifest_path, episode_dir))
                                continue
                            config = HumanScenarioConfig(
                                application=application,
                                seed=int(seed),
                                horizon=int(horizon),
                                n_agents=int(n_agents),
                                topology=topology,
                                disruption=regime,
                                communication=communication,
                                decision_interval=2,
                                communication_budget=max(100, horizon * 5),
                                operator_seed=int(operator_seed),
                                operator_profile=operator_profile,
                                intervention_budget=6,
                            )
                            started_at = utc_now()
                            try:
                                runner = HumanOperatorEpisodeRunner(
                                    config,
                                    method_value,
                                    planner=planner,
                                    thermodynamic_calibration=calibrations[application],
                                    escalation_config=trigger,
                                    learned_score=scorer,
                                    rl_seed=rl_seed,
                                    llm_seed=llm_seed,
                                    enable_counterfactual_probes=counterfactual_probes,
                                    dense_counterfactual_probes=dense_counterfactual_probes,
                                )
                                result = runner.run(run_id)
                                checksums = write_human_episode(result, runner.env.ledger, episode_dir)
                                ended_at = utc_now()
                                manifest = _manifest(
                                    root, results_root, stage, result, runner,
                                    checksums, started_at, ended_at, planner_backend,
                                    trigger, protocol_checksum, checkpoint,
                                )
                                _atomic_json(manifest_path, manifest)
                                summaries.append(_summary_row(asdict(result), manifest_path, episode_dir))
                            except Exception as error:
                                ended_at = utc_now()
                                failure = {
                                    "run_id": run_id,
                                    "study": "thermohitl_v3",
                                    "stage": stage,
                                    "source": {**git_provenance(root), "checksum": expected_source},
                                    "configuration": asdict(config),
                                    "application": application,
                                    "method": method_value,
                                    "environment_seed": seed,
                                    "operator_seed": operator_seed,
                                    "rl_training_seed": rl_seed,
                                    "start_timestamp": started_at,
                                    "end_timestamp": ended_at,
                                    "completion_status": "failed",
                                    "failure_reason": "%s: %s" % (type(error).__name__, error),
                                    "traceback": traceback.format_exc(),
                                    "retry_policy": "retain; infrastructure retry only under the frozen policy",
                                }
                                _atomic_json(manifest_path, failure)
                                raise
    summary_path = results_root / stage / "episode_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dict_csv(summary_path, summaries)
    sweep = {
        "stage": stage,
        "created_at": utc_now(),
        "planner_backend": planner_backend,
        "episodes": len(summaries),
        "complete": sum(row["completion_status"] == "complete" for row in summaries),
        "failed": sum(row["completion_status"] != "complete" for row in summaries),
        "source_checksum": expected_source,
        "summary_sha256": sha256_file(summary_path),
    }
    _atomic_json(results_root / "manifests" / (stage + "_sweep.json"), sweep)
    return summaries


def _summary_row(result: Mapping[str, Any], manifest_path: Path, episode_dir: Path) -> Dict[str, Any]:
    metrics = result.get("metrics", {})
    operator = result.get("operator_metrics", {})
    actionability = result.get("actionability", {})
    return {
        "run_id": result.get("run_id"),
        "application": result.get("application"),
        "method": result.get("method"),
        "scenario": result.get("scenario"),
        "environment_seed": result.get("environment_seed"),
        "llm_seed": result.get("llm_seed"),
        "rl_seed": result.get("rl_seed"),
        "operator_seed": result.get("operator_seed"),
        "operator_profile": result.get("operator_profile"),
        "completion_status": result.get("completion_status"),
        "primary_outcome": metrics.get("primary_outcome"),
        "service_loss_auc": metrics.get("service_loss_auc"),
        "cumulative_unmet_weighted_need": metrics.get("cumulative_unmet_weighted_need"),
        "operator_requests": metrics.get("operator_requests"),
        "operator_interventions": metrics.get("operator_interventions"),
        "operator_minutes": operator.get("operator_minutes"),
        "trigger_activations": operator.get("trigger_activations"),
        "pre_disruption_false_activation": operator.get("pre_disruption_false_activation"),
        "nominal_false_activation": operator.get("nominal_false_activation"),
        "timely_activation": operator.get("timely_activation"),
        "missed_activation": operator.get("missed_activation"),
        "material_actions_accepted": metrics.get("material_actions_accepted"),
        "material_actions_reached_demand": metrics.get("material_actions_reached_demand"),
        "first_pass_valid_rate": actionability.get("first_pass_valid_rate"),
        "valid_after_one_repair_rate": actionability.get("valid_after_one_repair_rate"),
        "conservation_error": metrics.get("conservation_error"),
        "total_communication_messages": metrics.get("total_communication_messages"),
        "llm_calls": metrics.get("llm_calls"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "generated_tokens": metrics.get("generated_tokens"),
        "llm_latency_seconds": metrics.get("llm_latency_seconds"),
        "manifest": str(manifest_path),
        "episode_directory": str(episode_dir),
    }


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def collect_bandit_training_rows(
    results_root: Path,
    stages: Sequence[str] = ("development",),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage in stages:
        for episode_path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            series = episode["time_series"]
            event_path = episode_path.with_name("events.jsonl.gz")
            states: Dict[Tuple[int, str], Dict[str, Any]] = {}
            with gzip.open(event_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    event = json.loads(line)
                    if event["kind"] == "thermodynamic_state":
                        states[(int(event["step"]), str(event["actor"]))] = event["payload"]
            disruption_step = max(2, int(series[-1]["step"] + 1) // 3)
            for (step, agent_id), state in states.items():
                current = series[step]
                future = series[step + 1 : min(len(series), step + 5)]
                if not future:
                    continue
                if episode["application"] == "commercial":
                    future_risk = float(np.mean([row["service_loss"] for row in future]))
                else:
                    future_risk = float(np.mean([
                        row["weighted_backlog"] / max(row["cumulative_demand"], 1.0)
                        for row in future
                    ]))
                    future_risk = min(1.5, future_risk)
                features = {
                    "local_kpi_risk": float(state["local_kpi_risk"]),
                    "local_disruption_risk": float(state["local_disruption_risk"]),
                    "actionability_evidence": float(state.get("actionability_evidence", 0.0)),
                    "consensus_confidence": float(state["consensus_confidence"]),
                    "local_energy_residual": float(state.get("local_energy_residual", 0.0)),
                    "energy_residual": float(state["energy_residual"]),
                    "entropy_residual": float(state["entropy_residual"]),
                    "entropy_slope": float(state["entropy_slope"]),
                    "disagreement": float(state["disagreement"]),
                }
                active = episode["scenario"].find("nominal") < 0 and step >= disruption_step
                if not active and future_risk < 0.55:
                    best_action = 0
                elif features["disagreement"] >= 0.16:
                    best_action = 5
                elif features["local_disruption_risk"] >= 0.55 and features["local_kpi_risk"] >= 0.45:
                    best_action = 6
                elif features["energy_residual"] >= 1.25:
                    best_action = 4
                elif features["entropy_residual"] >= 1.25:
                    best_action = 2
                else:
                    best_action = 3
                operator_cost = 0.10
                rewards = [
                    -1.5 * future_risk if active else 0.0,
                    -0.65 * future_risk - 0.03 if active else -0.03,
                ]
                for action in range(2, 7):
                    if not active:
                        reward = -operator_cost
                    elif action == best_action:
                        reward = future_risk + 0.35 * features["local_kpi_risk"] - operator_cost
                    else:
                        reward = 0.20 * future_risk - 1.5 * operator_cost
                    rewards.append(float(reward))
                rows.append({
                    "application": episode["application"],
                    "scenario": episode["scenario"],
                    "environment_seed": episode["environment_seed"],
                    "step": step,
                    "agent_id": agent_id,
                    "features": features,
                    "action_rewards": rewards,
                    "training_only_best_action": best_action,
                    "future_risk": future_risk,
                })
    if len(rows) < 64:
        raise ValueError("insufficient development contexts for bandit training")
    return rows


def train_human_multiseed(
    results_root: Path,
    seeds: Sequence[int] = (11301, 11302, 11303, 11304, 11305),
) -> Dict[str, Any]:
    ensure_v3_tree(results_root)
    rows = collect_bandit_training_rows(results_root)
    dataset_path = results_root / "training" / "contextual_bandit_dataset.json"
    _atomic_json(dataset_path, rows)
    learning_rows: List[Dict[str, Any]] = []
    seed_manifest: List[Dict[str, Any]] = []
    for variant in (
        HumanMethod.LEARNED_NO_THERMODYNAMICS.value,
        HumanMethod.THERMOHITL_RL.value,
    ):
        for seed in seeds:
            checkpoint = results_root / "checkpoints" / "%s-seed-%d.pt" % (variant, seed)
            history = train_contextual_bandit(rows, checkpoint, int(seed), variant)
            learning_rows.extend(history)
            seed_manifest.append({
                "variant": variant,
                "rl_seed": int(seed),
                "training_rows": len(rows),
                "fixed_epochs": len(history),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "selection_rule": "final fixed epoch; no outcome-based seed removal",
                "status": "complete",
                "final_loss": history[-1]["loss"],
                "final_best_action_accuracy": history[-1]["best_action_accuracy"],
            })
    _write_dict_csv(results_root / "training" / "learning_curves.csv", learning_rows)
    _write_dict_csv(results_root / "training" / "seed_manifest.csv", seed_manifest)
    _write_dict_csv(results_root / "training" / "checkpoint_selection.csv", seed_manifest)
    return {
        "training_rows": len(rows),
        "seeds": [int(seed) for seed in seeds],
        "checkpoints": len(seed_manifest),
        "failed": sum(row["status"] != "complete" for row in seed_manifest),
        "dataset_sha256": sha256_file(dataset_path),
    }


def checkpoint_map(results_root: Path) -> Dict[Tuple[str, int], Path]:
    output: Dict[Tuple[str, int], Path] = {}
    for path in (results_root / "checkpoints").glob("*-seed-*.pt"):
        prefix, seed_text = path.stem.rsplit("-seed-", 1)
        output[(prefix, int(seed_text))] = path
    return output


def diagnose_v2_actionability(
    repository_root: Path,
    results_root: Path = V3_RESULTS,
) -> Dict[str, Any]:
    """Derive v3 diagnostics from immutable v2 ledgers without rewriting them."""

    ensure_v3_tree(results_root)
    source_root = repository_root / "results" / "entropy_triggered_v2"
    ledger_paths = sorted((source_root / "raw").glob("*/*/events.jsonl.gz"))
    counts: Dict[Tuple[str, str], int] = {}
    material_proposals = 0
    accepted_material = 0
    arrivals = 0
    for path in ledger_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event["kind"] == "tool_result":
                    tool = str(event["payload"].get("tool", "unknown"))
                    code = str(event["payload"].get("code", "unknown"))
                    counts[(tool, code)] = counts.get((tool, code), 0) + 1
                    if tool in ("schedule_shipment", "transfer_resource"):
                        material_proposals += 1
                        accepted_material += int(bool(event["payload"].get("ok")))
                elif (
                    event["kind"] == "environment_transition"
                    and event["payload"].get("transition") == "shipment_arrival"
                ):
                    arrivals += 1
    rows = [
        {"tool": tool, "result_code": code, "count": count}
        for (tool, code), count in sorted(counts.items())
    ]
    output = results_root / "diagnostics" / "v2_tool_result_codes.csv"
    _write_dict_csv(output, rows)
    source_checksums = {
        str(path.relative_to(repository_root)): sha256_file(path)
        for path in ledger_paths
    }
    summary = {
        "created_at": utc_now(),
        "source_namespace": "results/entropy_triggered_v2/raw",
        "source_ledgers": len(ledger_paths),
        "source_checksum_digest": hashlib.sha256(
            json.dumps(source_checksums, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "material_proposals": material_proposals,
        "accepted_material_actions": accepted_material,
        "shipment_arrivals": arrivals,
        "acceptance_rate": accepted_material / max(material_proposals, 1),
        "arrival_per_accepted_action": arrivals / max(accepted_material, 1),
        "frozen_inputs_modified": False,
    }
    _atomic_json(results_root / "diagnostics" / "v2_actionability_summary.json", summary)
    return summary
