"""Generate the unseen balanced v2 holdout before protocol freeze."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .events import sha256_file
from .experiments import expand_matrix


RL_SEEDS = (7301, 7302, 7303, 7304, 7305)
NON_NOMINAL_SEEDS = tuple(range(8101, 8117))
NOMINAL_SEEDS = tuple(range(8201, 8209))
CORE_METHODS = (
    "autonomous_no_comm",
    "fixed_always_on",
    "periodic_communication",
    "random_budget_matched",
    "learned_no_entropy",
    "thermoagent",
    "doet_rule",
    "doet_rl",
    "kpi_cusum_trigger",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_maps(results_root: Path) -> Dict[str, Dict[str, str]]:
    training_path = results_root / "training" / "seed_manifest.csv"
    if not training_path.exists():
        raise FileNotFoundError(training_path)
    with training_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: Dict[str, Dict[str, str]] = {
        "no_entropy": {}, "thermo": {}, "doet_rl": {},
    }
    for variant in output:
        variant_rows = [row for row in rows if row["variant"] == variant]
        by_seed = {int(row["rl_training_seed"]): row for row in variant_rows}
        if set(by_seed) != set(RL_SEEDS):
            raise ValueError(
                "%s requires exactly RL seeds %s; observed %s"
                % (variant, list(RL_SEEDS), sorted(by_seed))
            )
        for seed, row in sorted(by_seed.items()):
            if row["status"] != "complete":
                raise ValueError(
                    "%s seed %d is not complete: %s"
                    % (variant, seed, row.get("failure_reason", ""))
                )
            relative = Path(row["checkpoint"])
            checkpoint = results_root / relative
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            observed = sha256_file(checkpoint)
            if observed != row["checkpoint_sha256"]:
                raise ValueError("checkpoint checksum changed: %s" % checkpoint)
            output[variant][str(seed)] = str(
                Path("results/entropy_triggered_v2") / relative
            )
    return output


def _precision_analysis(
    validation_pairs: pd.DataFrame,
    validation_sweep: Mapping[str, Any],
    episode_count: int,
) -> Dict[str, Any]:
    non_nominal = validation_pairs[
        validation_pairs["scenario_name"] != "nominal"
    ]
    applications: Dict[str, Any] = {}
    normal = NormalDist()
    for application, values in non_nominal.groupby("application"):
        degradation = values["relative_degradation"].to_numpy(dtype=float)
        standard_deviation = float(np.std(degradation, ddof=1))
        standard_error = standard_deviation / math.sqrt(16)
        expected_mean = float(np.mean(degradation))
        upper_bound = expected_mean + 1.645 * standard_error
        if standard_error > 0:
            power = normal.cdf(
                (0.02 - expected_mean) / standard_error - 1.645
            )
        else:
            power = float(expected_mean < 0.02)
        applications[str(application)] = {
            "validation_mean_relative_degradation": expected_mean,
            "validation_pair_standard_deviation": standard_deviation,
            "planned_pairs_per_regime": 16,
            "approximate_standard_error_per_regime": standard_error,
            "expected_one_sided_95_upper_bound": upper_bound,
            "approximate_probability_of_noninferiority_if_validation_effect_repeats": float(power),
        }
    validation_wall = float(
        validation_sweep["wall_clock_seconds_including_model_load"]
    )
    validation_episodes = int(validation_sweep["episodes_complete"])
    seconds_per_episode = validation_wall / max(validation_episodes, 1)
    projected_hours = seconds_per_episode * episode_count / 3600.0 * 1.15
    return {
        "method": (
            "normal approximation using validation paired degradation variance; "
            "confirmatory inference will use hierarchical paired bootstrap"
        ),
        "applications": applications,
        "validation_seconds_per_episode": seconds_per_episode,
        "holdout_episode_count": episode_count,
        "projected_holdout_single_gpu_hours_with_15_percent_buffer": projected_hours,
        "validation_single_gpu_hours": validation_wall / 3600.0,
        "projected_additional_gpu_hours_validation_plus_holdout": (
            validation_wall / 3600.0 + projected_hours
        ),
        "budget_limit_single_gpu_hours": 35.0,
    }


def run(results_root: Path, config_path: Path) -> Dict[str, Any]:
    selection_path = results_root / "validation" / "trigger_selection.json"
    controls_path = results_root / "validation" / "budget_matched_controls.json"
    pairs_path = results_root / "validation" / "selected_trigger_pairs.csv"
    validation_sweep_path = results_root / "manifests" / "validation_sweep.json"
    for path in (
        selection_path, controls_path, pairs_path, validation_sweep_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    validation_sweep = json.loads(
        validation_sweep_path.read_text(encoding="utf-8")
    )
    if validation_sweep.get("planner_backend") != "transformers":
        raise ValueError("holdout design requires real-LLM validation throughput")
    if int(validation_sweep.get("episodes_failed", 1)) != 0:
        raise ValueError("validation contains failed episodes")
    selected_trigger = dict(selection["selected_trigger"])
    trigger_parameters = dict(selected_trigger["parameters"])
    checkpoints = _checkpoint_maps(results_root)
    config: Dict[str, Any] = {
        "stage": "holdout_locked",
        "protocol_freeze_path": "results/entropy_triggered_v2/protocol/holdout_freeze.json",
        "prompt_template_revision": "planner-json-v7-route-affordances",
        "agentic_metric_revision": "agentic-metrics-v2-two-party-joined-coalition",
        "llm_seed": 9101,
        "rl_seeds": list(RL_SEEDS),
        "balanced_rl_assignment": True,
        "decision_interval": 4,
        "communication_budget": 300,
        "fixed_broadcast_fanout": 3,
        "periodic_interval": int(controls["periodic_interval"]),
        "random_gate_probability": float(controls["random_gate_probability"]),
        "calibration": "results/reproducibility/macrostate_calibration.json",
        "checkpoints": checkpoints,
        "model": {
            "identifier": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "precision": "bitsandbytes NF4, bfloat16 compute, double quantization",
            "load_in_4bit": True,
            "max_input_tokens": 2560,
            "max_new_tokens": 160,
            "decoding": {
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        },
        "trigger": {
            "normalizers_path": "results/entropy_triggered_v2/calibration/trigger_nominal_calibration.json",
            "normalizers_key": "normalizers",
            "parameters": trigger_parameters,
        },
        "applications": {
            "commercial": {"n_agents": 10},
            "humanitarian": {"n_agents": 10},
        },
        "methods": list(CORE_METHODS),
        "seeds": list(NON_NOMINAL_SEEDS),
        "scenarios": {
            "nominal": {
                "seeds": list(NOMINAL_SEEDS),
                "horizon": 24,
                "private_information": 0.8,
                "objective_misalignment": 0.8,
                "communication": "reliable",
                "disruption": "nominal",
                "topology": "tri_region_bridge_v2",
            },
            "isolated": {
                "horizon": 24,
                "private_information": 0.8,
                "objective_misalignment": 0.8,
                "communication": "reliable",
                "disruption": "moderate",
                "topology": "tri_region_bridge_v2",
            },
            "communication_partition": {
                "horizon": 24,
                "private_information": 0.8,
                "objective_misalignment": 0.8,
                "communication": "partition",
                "disruption": "moderate",
                "topology": "tri_region_bridge_v2",
            },
            "correlated": {
                "horizon": 24,
                "private_information": 1.0,
                "objective_misalignment": 1.0,
                "communication": "intermittent",
                "disruption": "correlated",
                "topology": "tri_region_bridge_v2",
            },
            "compound_ood": {
                "horizon": 24,
                "private_information": 1.0,
                "objective_misalignment": 1.0,
                "communication": "partition",
                "disruption": "compound",
                "topology": "tri_region_bridge_v2",
            },
        },
    }
    matrix = expand_matrix(config)
    episode_count = len(matrix)
    expected_count = 1296
    if episode_count != expected_count:
        raise ValueError(
            "locked design expected %d episodes, got %d"
            % (expected_count, episode_count)
        )
    # Verify exact balance: 144 panels/method and either 28 or 29 panels per
    # training seed for every learned method.
    learned_counts: Dict[str, Dict[int, int]] = {}
    for _, _, _, method, scenario in matrix:
        if method not in ("learned_no_entropy", "thermoagent", "doet_rl"):
            continue
        seed = int(scenario["_rl_seed"])
        learned_counts.setdefault(method, {}).setdefault(seed, 0)
        learned_counts[method][seed] += 1
    if any(
        set(counts) != set(RL_SEEDS)
        or max(counts.values()) - min(counts.values()) > 1
        for counts in learned_counts.values()
    ):
        raise ValueError("balanced RL assignment failed: %s" % learned_counts)
    precision = _precision_analysis(
        pd.read_csv(pairs_path), validation_sweep, episode_count
    )
    if precision["projected_additional_gpu_hours_validation_plus_holdout"] > 35.0:
        raise RuntimeError(
            "projected validation + holdout compute %.2f GPU-hours exceeds 35; "
            "do not freeze or launch without a reduced design or user approval"
            % precision["projected_additional_gpu_hours_validation_plus_holdout"]
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    protocol_dir = results_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    design_rows: List[Dict[str, Any]] = []
    for application, n_agents, environment_seed, method, scenario in matrix:
        design_rows.append({
            "application": application,
            "n_agents": n_agents,
            "scenario": scenario["name"],
            "environment_seed": environment_seed,
            "method": method,
            "rl_training_seed": scenario["_rl_seed"],
            "topology": scenario["topology"],
            "communication": scenario["communication"],
            "disruption": scenario["disruption"],
        })
    design_path = protocol_dir / "holdout_design.csv"
    pd.DataFrame(design_rows).to_csv(design_path, index=False)
    precision_path = protocol_dir / "power_precision_analysis.json"
    precision_path.write_text(
        json.dumps(precision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "designed but not yet frozen or run",
        "generated_at": _utc_now(),
        "new_environment_seeds": {
            "non_nominal": list(NON_NOMINAL_SEEDS),
            "nominal": list(NOMINAL_SEEDS),
        },
        "new_llm_seed": 9101,
        "topology": "tri_region_bridge_v2",
        "base_scenario_panels": 144,
        "non_nominal_base_panels": 128,
        "nominal_base_panels": 16,
        "episode_count": episode_count,
        "methods": list(CORE_METHODS),
        "rl_training_seeds": list(RL_SEEDS),
        "learned_assignment_counts": learned_counts,
        "selected_trigger_variant": selection["selected_method_variant"],
        "selected_trigger_checksum": sha256_file(selection_path),
        "validation_pairs_checksum": sha256_file(pairs_path),
        "config_path": str(config_path),
        "config_checksum": sha256_file(config_path),
        "design_checksum": sha256_file(design_path),
        "precision_checksum": sha256_file(precision_path),
    }
    manifest_path = protocol_dir / "holdout_design_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
