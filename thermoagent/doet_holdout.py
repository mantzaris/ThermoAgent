"""Generate the unseen balanced v2 holdout before protocol freeze."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .events import sha256_file
from .experiments import expand_matrix


RL_SEEDS = (7301, 7302, 7303, 7304, 7305)
NON_NOMINAL_SEEDS = tuple(range(8101, 8117))
NOMINAL_SEEDS = tuple(range(8201, 8209))
SECONDARY_SUBSET_SEEDS = (8101, 8106, 8111)
# Prospectively ordered, nested compute fallbacks. Selection uses measured Pod
# time only, never validation outcome values: retain every priority-method panel
# and every RL seed, then reduce the shared secondary-comparator panel subset.
SECONDARY_SEED_LADDER = (
    SECONDARY_SUBSET_SEEDS,
    (8101, 8111),
    (8106,),
)
PRIMARY_FULL_METHODS = (
    "fixed_always_on",
    "learned_no_entropy",
    "doet_rule",
    "doet_rl",
)
SECONDARY_SUBSET_METHODS = (
    "autonomous_no_comm",
    "periodic_communication",
    "random_budget_matched",
    "thermoagent",
    "kpi_cusum_trigger",
)
CORE_METHODS = (
    *PRIMARY_FULL_METHODS,
    *SECONDARY_SUBSET_METHODS,
)


def _method_seed_map(
    primary_seeds: Sequence[int],
    secondary_seeds: Sequence[int],
) -> Dict[str, List[int]]:
    return {
        **{
            method: [int(seed) for seed in primary_seeds]
            for method in PRIMARY_FULL_METHODS
        },
        **{
            method: [int(seed) for seed in secondary_seeds]
            for method in SECONDARY_SUBSET_METHODS
        },
    }


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
    training_manifest: Mapping[str, Any],
    profile_sweep: Mapping[str, Any],
    model_smoke: Mapping[str, Any],
    episode_count: int,
) -> Dict[str, Any]:
    non_nominal = validation_pairs[
        validation_pairs["scenario_name"] != "nominal"
    ]
    applications: Dict[str, Any] = {}
    simulation_replicates = 20_000
    simulation_seed = 20260814
    # The validation set has no standalone partition cell. Its prospectively
    # defined compound-partition cell is the conservative variance proxy for
    # both partitioned holdout families.
    holdout_to_validation = {
        "isolated": "isolated",
        "communication_partition": "compound_partition",
        "correlated": "correlated",
        "compound_ood": "compound_partition",
    }
    for application, values in non_nominal.groupby("application"):
        rng = np.random.RandomState(simulation_seed + (
            0 if str(application) == "commercial" else 1
        ))
        draws = []
        source_counts: Dict[str, int] = {}
        for planned_regime, source_regime in holdout_to_validation.items():
            degradation = values[
                values["scenario_name"] == source_regime
            ]["relative_degradation"].to_numpy(dtype=float)
            if not len(degradation):
                raise ValueError(
                    "precision simulation missing %s validation values for %s"
                    % (source_regime, application)
                )
            source_counts[planned_regime] = int(len(degradation))
            draws.append(degradation[rng.randint(
                0, len(degradation), size=(simulation_replicates, 16)
            )])
        simulated_means = np.concatenate(draws, axis=1).mean(axis=1)
        expected_mean = float(np.mean(simulated_means))
        standard_error = float(np.std(simulated_means, ddof=1))
        centered_upper = float(np.quantile(
            simulated_means - expected_mean, 0.95
        ))
        upper_bound = expected_mean + centered_upper
        probability_noninferior = float(np.mean(
            simulated_means + centered_upper < 0.02
        ))
        applications[str(application)] = {
            "validation_mean_relative_degradation": expected_mean,
            "planned_pairs_per_regime": 16,
            "planned_non_nominal_pairs": 64,
            "validation_proxy_mapping": holdout_to_validation,
            "validation_rows_available_by_planned_regime": source_counts,
            "simulated_standard_error_of_application_mean": standard_error,
            "simulated_mean_quantile_2_5": float(np.quantile(simulated_means, 0.025)),
            "simulated_mean_quantile_97_5": float(np.quantile(simulated_means, 0.975)),
            "expected_one_sided_95_upper_bound": upper_bound,
            "simulated_probability_of_noninferiority_if_validation_effect_repeats": probability_noninferior,
        }
    validation_wall = float(
        validation_sweep["wall_clock_seconds_including_model_load"]
    )
    validation_gpu_hours = max(
        validation_wall / 3600.0,
        float(validation_sweep.get(
            "cumulative_episode_single_gpu_hours", 0.0
        )),
    )
    validation_episodes = int(validation_sweep["episodes_complete"])
    seconds_per_episode = (
        validation_gpu_hours * 3600.0 / max(validation_episodes, 1)
    )
    projected_hours = seconds_per_episode * episode_count / 3600.0 * 1.15
    training_hours = float(training_manifest.get(
        "single_gpu_hours_reserved",
        float(training_manifest.get("wall_clock_seconds", 0.0) or 0.0)
        / 3600.0,
    ))
    profile_hours = max(
        float(profile_sweep.get(
            "wall_clock_seconds_including_model_load", 0.0
        )) / 3600.0,
        float(profile_sweep.get(
            "cumulative_episode_single_gpu_hours", 0.0
        )),
    )
    model_smoke_hours = (
        float(model_smoke.get("load_seconds", 0.0) or 0.0)
        + float(model_smoke.get("batched_inference_seconds", 0.0) or 0.0)
    ) / 3600.0
    unmeasured_setup_reserve_hours = 0.1
    return {
        "method": (
            "20,000-replicate stratified Monte Carlo resampling of validation "
            "paired degradation, drawing 16 panels for each of four planned "
            "non-nominal regimes; confirmatory inference uses hierarchical "
            "paired bootstrap"
        ),
        "simulation_replicates": simulation_replicates,
        "simulation_seed": simulation_seed,
        "applications": applications,
        "validation_seconds_per_episode": seconds_per_episode,
        "holdout_episode_count": episode_count,
        "projected_holdout_single_gpu_hours_with_15_percent_buffer": projected_hours,
        "validation_single_gpu_hours": validation_gpu_hours,
        "training_single_gpu_hours_reserved": training_hours,
        "profile_single_gpu_hours": profile_hours,
        "model_smoke_single_gpu_hours": model_smoke_hours,
        "unmeasured_setup_gpu_hour_reserve": unmeasured_setup_reserve_hours,
        "projected_additional_gpu_hours_all_required_stages": (
            validation_gpu_hours + training_hours + projected_hours
            + profile_hours + model_smoke_hours
            + unmeasured_setup_reserve_hours
        ),
        "budget_limit_single_gpu_hours": 35.0,
    }


def run(results_root: Path, config_path: Path) -> Dict[str, Any]:
    selection_path = results_root / "validation" / "trigger_selection.json"
    controls_path = results_root / "validation" / "budget_matched_controls.json"
    pairs_path = results_root / "validation" / "selected_trigger_pairs.csv"
    validation_sweep_path = results_root / "manifests" / "validation_sweep.json"
    training_manifest_path = results_root / "training" / "training_manifest.json"
    profile_sweep_path = results_root / "manifests" / "profile_v2_sweep.json"
    model_smoke_path = results_root / "logs" / "setup" / "model_smoke.json"
    for path in (
        selection_path, controls_path, pairs_path, validation_sweep_path,
        training_manifest_path, profile_sweep_path, model_smoke_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if "kpi_trigger" not in controls:
        raise ValueError(
            "validation controls lack the frozen budget-matched KPI trigger"
        )
    validation_sweep = json.loads(
        validation_sweep_path.read_text(encoding="utf-8")
    )
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    profile_sweep = json.loads(
        profile_sweep_path.read_text(encoding="utf-8")
    )
    model_smoke = json.loads(model_smoke_path.read_text(encoding="utf-8"))
    if profile_sweep.get("planner_backend") != "transformers":
        raise ValueError("holdout design requires a real-LLM throughput profile")
    if int(profile_sweep.get("episodes_failed", 1)) != 0:
        raise ValueError("throughput profile contains failed episodes")
    if model_smoke.get("status") != "complete":
        raise ValueError("CUDA/model smoke did not complete")
    if validation_sweep.get("planner_backend") != "transformers":
        raise ValueError("holdout design requires real-LLM validation throughput")
    if int(validation_sweep.get("episodes_failed", 1)) != 0:
        raise ValueError("validation contains failed episodes")
    selected_trigger = dict(selection["selected_trigger"])
    trigger_parameters = dict(selected_trigger["parameters"])
    checkpoints = _checkpoint_maps(results_root)

    # Select the largest prospectively defined common secondary subset whose
    # conservative projection fits the hard 35-hour cap. Four priority methods
    # always retain all 144 panels (576 episodes total), and all five independent
    # RL seeds remain represented. Each secondary seed adds one panel in every
    # application/regime cell for all five secondary methods: 40 episodes.
    budget_candidates: List[Dict[str, Any]] = []
    selected_secondary_seeds: Sequence[int] | None = None
    precision: Dict[str, Any] | None = None
    for secondary_seeds in SECONDARY_SEED_LADDER:
        candidate_episode_count = (
            len(PRIMARY_FULL_METHODS) * 144
            + len(SECONDARY_SUBSET_METHODS) * 8 * len(secondary_seeds)
        )
        candidate_precision = _precision_analysis(
            pd.read_csv(pairs_path), validation_sweep, training_manifest,
            profile_sweep, model_smoke, candidate_episode_count
        )
        candidate_total = float(candidate_precision[
            "projected_additional_gpu_hours_all_required_stages"
        ])
        budget_candidates.append({
            "secondary_environment_seeds": list(secondary_seeds),
            "episode_count": candidate_episode_count,
            "projected_additional_single_gpu_hours": candidate_total,
            "within_35_hour_cap": candidate_total <= 35.0,
        })
        if candidate_total <= 35.0:
            selected_secondary_seeds = secondary_seeds
            precision = candidate_precision
            break
    if selected_secondary_seeds is None or precision is None:
        minimum = budget_candidates[-1]
        raise RuntimeError(
            "even the preregistered minimum secondary subset projects %.2f "
            "single-GPU hours; do not freeze or launch without user approval "
            "or a newly documented design"
            % minimum["projected_additional_single_gpu_hours"]
        )
    precision["secondary_subset_budget_candidates"] = budget_candidates
    precision["secondary_subset_selection_rule"] = (
        "choose the largest preregistered common secondary seed subset that "
        "keeps validation + five-seed training + buffered holdout + measured "
        "model setup/profile below 35 single-GPU hours; selection uses runtime "
        "only and never validation outcomes"
    )
    precision["selected_secondary_environment_seeds"] = list(
        selected_secondary_seeds
    )
    config: Dict[str, Any] = {
        "stage": "holdout_locked",
        "source_provenance_path": "results/entropy_triggered_v2/reproducibility/execution_source.json",
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
        "budget_match_calibration": controls,
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
        "method_variants": {
            "kpi_cusum_trigger": [{
                "name": "budget_matched",
                "trigger": controls["kpi_trigger"],
            }],
        },
        "seeds": list(NON_NOMINAL_SEEDS),
        "scenarios": {
            "nominal": {
                "seeds": list(NOMINAL_SEEDS),
                "method_seeds": _method_seed_map(NOMINAL_SEEDS, ()),
                "horizon": 16,
                "private_information": 0.8,
                "objective_misalignment": 0.8,
                "communication": "reliable",
                "disruption": "nominal",
                "topology": "tri_region_bridge_v2",
            },
            "isolated": {
                "method_seeds": _method_seed_map(
                    NON_NOMINAL_SEEDS, selected_secondary_seeds
                ),
                "horizon": 16,
                "private_information": 0.8,
                "objective_misalignment": 0.8,
                "communication": "reliable",
                "disruption": "moderate",
                "topology": "tri_region_bridge_v2",
            },
            "communication_partition": {
                "method_seeds": _method_seed_map(
                    NON_NOMINAL_SEEDS, selected_secondary_seeds
                ),
                "horizon": 16,
                "private_information": 0.8,
                "objective_misalignment": 0.8,
                "communication": "partition",
                "disruption": "moderate",
                "topology": "tri_region_bridge_v2",
            },
            "correlated": {
                "method_seeds": _method_seed_map(
                    NON_NOMINAL_SEEDS, selected_secondary_seeds
                ),
                "horizon": 16,
                "private_information": 1.0,
                "objective_misalignment": 1.0,
                "communication": "intermittent",
                "disruption": "correlated",
                "topology": "tri_region_bridge_v2",
            },
            "compound_ood": {
                "method_seeds": _method_seed_map(
                    NON_NOMINAL_SEEDS, selected_secondary_seeds
                ),
                "horizon": 16,
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
    expected_count = (
        len(PRIMARY_FULL_METHODS) * 144
        + len(SECONDARY_SUBSET_METHODS) * 8
        * len(selected_secondary_seeds)
    )
    if episode_count != expected_count:
        raise ValueError(
            "locked design expected %d episodes, got %d"
            % (expected_count, episode_count)
        )
    # The compute-capped design keeps the four preregistered priority methods
    # on all 144 panels and evaluates every secondary comparator on the same
    # 24 stratified non-nominal panels. Learned checkpoints remain balanced.
    method_evaluation_counts: Dict[str, int] = {}
    for _, _, _, method, _ in matrix:
        method_evaluation_counts[method] = (
            method_evaluation_counts.get(method, 0) + 1
        )
    expected_method_counts = {
        **{method: 144 for method in PRIMARY_FULL_METHODS},
        **{
            method: 8 * len(selected_secondary_seeds)
            for method in SECONDARY_SUBSET_METHODS
        },
    }
    if method_evaluation_counts != expected_method_counts:
        raise ValueError(
            "compute-capped method balance failed: %s"
            % method_evaluation_counts
        )
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
    projected_total = precision[
        "projected_additional_gpu_hours_all_required_stages"
    ]
    if projected_total > 35.0:
        raise RuntimeError(
            "projected validation + training + holdout resource use %.2f "
            "single-GPU hours exceeds 35; "
            "do not freeze or launch without a reduced design or user approval"
            % projected_total
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
        "primary_full_panel_methods": list(PRIMARY_FULL_METHODS),
        "secondary_subset_methods": list(SECONDARY_SUBSET_METHODS),
        "secondary_subset_environment_seeds": list(selected_secondary_seeds),
        "secondary_subset_selection_rule": precision[
            "secondary_subset_selection_rule"
        ],
        "secondary_subset_budget_candidates": budget_candidates,
        "method_evaluation_counts": method_evaluation_counts,
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
