"""Budget-gated extended DOET signal/oracle ablation design."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from .events import sha256_file
from .experiments import expand_matrix


ABLATION_SEEDS = (8301, 8302, 8303)
EPISODE_COUNT = 96


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(results_root: Path, config_path: Path) -> Dict[str, Any]:
    selection_path = results_root / "validation" / "trigger_selection.json"
    controls_path = results_root / "validation" / "budget_matched_controls.json"
    validation_sweep_path = results_root / "manifests" / "validation_sweep.json"
    profile_sweep_path = results_root / "manifests" / "profile_v2_sweep.json"
    model_smoke_path = results_root / "logs" / "setup" / "model_smoke.json"
    holdout_sweep_path = results_root / "manifests" / "holdout_locked_sweep.json"
    training_manifest_path = results_root / "training" / "training_manifest.json"
    provenance_path = results_root / "reproducibility" / "execution_source.json"
    for path in (
        selection_path, controls_path, validation_sweep_path, profile_sweep_path,
        model_smoke_path, holdout_sweep_path, training_manifest_path,
        provenance_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if "kpi_trigger" not in controls:
        raise ValueError("validation controls lack budget-matched KPI trigger")
    validation_sweep = json.loads(
        validation_sweep_path.read_text(encoding="utf-8")
    )
    profile_sweep = json.loads(
        profile_sweep_path.read_text(encoding="utf-8")
    )
    model_smoke = json.loads(model_smoke_path.read_text(encoding="utf-8"))
    holdout_sweep = json.loads(
        holdout_sweep_path.read_text(encoding="utf-8")
    )
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    parameters = dict(selection["selected_trigger"]["parameters"])
    tau_on = float(parameters["tau_on"])
    parameters_without_hysteresis = {
        **parameters,
        "tau_off": max(0.0, tau_on - 1e-6),
        "minimum_dwell": 1,
        "cooldown": 0,
    }
    config: Dict[str, Any] = {
        "stage": "ablations",
        "protocol_freeze_path": (
            "results/entropy_triggered_v2/protocol/ablation_freeze.json"
        ),
        "source_provenance_path": (
            "results/entropy_triggered_v2/reproducibility/execution_source.json"
        ),
        "prompt_template_revision": "planner-json-v7-route-affordances",
        "agentic_metric_revision": (
            "agentic-metrics-v2-two-party-joined-coalition"
        ),
        "llm_seed": 9301,
        "decision_interval": 4,
        "communication_budget": 300,
        "calibration": "results/reproducibility/macrostate_calibration.json",
        "model": {
            "identifier": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "precision": (
                "bitsandbytes NF4, bfloat16 compute, double quantization"
            ),
            "load_in_4bit": True,
            "max_input_tokens": 2560,
            "max_new_tokens": 160,
            "decoding": {
                "do_sample": False, "temperature": 0.0, "top_p": 1.0,
            },
        },
        "trigger": {
            "normalizers_path": (
                "results/entropy_triggered_v2/calibration/"
                "trigger_nominal_calibration.json"
            ),
            "normalizers_key": "normalizers",
            "parameters": parameters,
        },
        "applications": {
            "commercial": {"n_agents": 10},
            "humanitarian": {"n_agents": 10},
        },
        "methods": [
            "doet_rule", "global_entropy_trigger_oracle",
            "disruption_label_oracle", "kpi_cusum_trigger",
        ],
        "seeds": list(ABLATION_SEEDS),
        "method_variants": {
            "doet_rule": [
                {"name": "selected"},
                {
                    "name": "without_hysteresis",
                    "trigger": {"parameters": parameters_without_hysteresis},
                },
                {
                    "name": "without_local_surprisal",
                    "trigger": {
                        "parameters": {"crisis_surprisal": 1e9}
                    },
                },
                {
                    "name": "without_distributed_gossip",
                    "trigger": {"parameters": {"disable_gossip": True}},
                },
                {
                    "name": "partition_signal_noise",
                    "trigger": {"parameters": {"signal_noise_std": 0.02}},
                },
            ],
            "kpi_cusum_trigger": [{
                "name": "private_local_kpi",
                "trigger": controls["kpi_trigger"],
            }],
        },
        "scenarios": {
            "correlated": {
                "horizon": 24,
                "private_information": 1.0,
                "objective_misalignment": 1.0,
                "communication": "intermittent",
                "disruption": "correlated",
                "topology": "tri_region_bridge_v2",
            },
            "compound_partition": {
                "horizon": 24,
                "private_information": 1.0,
                "objective_misalignment": 1.0,
                "communication": "partition",
                "disruption": "compound",
                "topology": "tri_region_bridge_v2",
            },
        },
    }
    observed_count = len(expand_matrix(config))
    if observed_count != EPISODE_COUNT:
        raise ValueError(
            "extended ablation expected %d episodes, got %d"
            % (EPISODE_COUNT, observed_count)
        )
    validation_hours = max(
        float(validation_sweep["wall_clock_seconds_including_model_load"])
        / 3600.0,
        float(validation_sweep.get(
            "cumulative_episode_single_gpu_hours", 0.0
        )),
    )
    holdout_hours = max(
        float(holdout_sweep["wall_clock_seconds_including_model_load"])
        / 3600.0,
        float(holdout_sweep.get(
            "cumulative_episode_single_gpu_hours", 0.0
        )),
    )
    validation_seconds = validation_hours * 3600.0
    holdout_seconds = holdout_hours * 3600.0
    seconds_per_episode = validation_seconds / max(
        int(validation_sweep["episodes_complete"]), 1
    )
    projected_seconds = seconds_per_episode * EPISODE_COUNT * 1.15
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
    projected_total_hours = (
        (validation_seconds + holdout_seconds + projected_seconds) / 3600.0
        + training_hours
        + profile_hours
        + model_smoke_hours
        + unmeasured_setup_reserve_hours
    )
    authorized = projected_total_hours <= 35.0
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    record = {
        "status": (
            "authorized within measured 35-hour cap"
            if authorized else
            "extended design retained but not authorized by compute cap"
        ),
        "authorized": authorized,
        "generated_at": _utc_now(),
        "episode_count": EPISODE_COUNT,
        "environment_seeds": list(ABLATION_SEEDS),
        "llm_seed": 9301,
        "signal_noise_std": 0.02,
        "signal_noise_rule": (
            "fixed prospectively at the calibration scale floor; exploratory"
        ),
        "validation_hours": validation_hours,
        "training_single_gpu_hours_reserved": training_hours,
        "profile_single_gpu_hours": profile_hours,
        "model_smoke_single_gpu_hours": model_smoke_hours,
        "unmeasured_setup_gpu_hour_reserve": unmeasured_setup_reserve_hours,
        "holdout_hours": holdout_hours,
        "projected_ablation_hours_with_15_percent_buffer": (
            projected_seconds / 3600.0
        ),
        "projected_total_additional_gpu_hours": projected_total_hours,
        "budget_limit_gpu_hours": 35.0,
        "config_path": str(config_path),
        "config_checksum": sha256_file(config_path),
        "selected_trigger_checksum": sha256_file(selection_path),
    }
    output = results_root / "protocol" / "extended_ablation_design.json"
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record
