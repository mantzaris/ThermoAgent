"""Pilot-based compute, token, disk, and cost projection before protocol freeze."""

from __future__ import annotations

import json
import gzip
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .analysis import _analysis_valid_mask, collect_results
from .experiments import expand_matrix


PILOT_SCENARIOS = {"paired_nominal_v8", "paired_compound_v8"}
CORE_METHODS = {
    "centralized_lookahead",
    "centralized_llm",
    "scripted_independent",
    "autonomous_no_comm",
    "autonomous_fixed_comm",
    "learned_no_entropy",
    "thermoagent",
}
METHOD_ANALOGUE = {
    "entropy_llm_only": "autonomous_fixed_comm",
    "no_episodic_memory": "thermoagent",
    "random_gate": "thermoagent",
    "global_entropy_oracle": "thermoagent",
    "shuffled_entropy": "thermoagent",
}
PROJECTED_METRICS = (
    "wall_clock_seconds",
    "llm_calls",
    "prompt_tokens",
    "generated_tokens",
)


def _quantile_summary(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for method, group in frame.groupby("method"):
        row: Dict[str, float] = {"episodes": int(len(group))}
        for metric in PROJECTED_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row[metric + "_mean"] = float(np.mean(values)) if values.size else 0.0
            row[metric + "_p90"] = float(np.quantile(values, 0.90)) if values.size else 0.0
        summaries[str(method)] = row
    return summaries


def _scale_factor(method: str, n_agents: int, horizon: int, decision_interval: int) -> float:
    """Scale the 8-agent, horizon-20 pilot workload to a configured cell."""

    epoch_ratio = math.ceil(horizon / decision_interval) / math.ceil(20 / 4)
    if method == "centralized_llm":
        # The final legal-coordinator baseline receives up to one dispatch slot
        # for each of three demand organizations; paired-v8 used one demand at
        # its eight-agent pilot size. Private cells often use only one no-op,
        # so three is a conservative upper workload factor.
        return 3.0 * epoch_ratio
    if method == "centralized_lookahead":
        # The frozen oracle controller replans every period; paired-v8 used the
        # ordinary five decision epochs before this pre-freeze strengthening.
        return 4.0 * (n_agents / 8.0) * epoch_ratio
    return (n_agents / 8.0) * epoch_ratio


def project_matrix(
    config: Mapping[str, Any],
    method_summaries: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """Project one configured matrix from conservative paired-pilot summaries."""

    totals = {
        "expected_wall_clock_seconds": 0.0,
        "upper_wall_clock_seconds": 0.0,
        "expected_llm_calls": 0.0,
        "upper_llm_calls": 0.0,
        "expected_prompt_tokens": 0.0,
        "upper_prompt_tokens": 0.0,
        "expected_generated_tokens": 0.0,
        "upper_generated_tokens": 0.0,
    }
    matrix = expand_matrix(config)
    analogue_counts: Dict[str, int] = {}
    for _, n_agents, _, method, scenario in matrix:
        analogue = METHOD_ANALOGUE.get(method, method)
        if analogue not in method_summaries:
            raise ValueError("pilot has no profiling analogue for %s" % method)
        analogue_counts[analogue] = analogue_counts.get(analogue, 0) + 1
        scale = _scale_factor(
            method,
            n_agents,
            int(scenario["horizon"]),
            int(config.get("decision_interval", 4)),
        )
        summary = method_summaries[analogue]
        for metric in PROJECTED_METRICS:
            totals["expected_" + metric] += float(summary[metric + "_mean"]) * scale
            totals["upper_" + metric] += float(summary[metric + "_p90"]) * scale
    return {
        "stage": str(config["stage"]),
        "episodes": len(matrix),
        "profiling_analogue_episode_counts": analogue_counts,
        **totals,
    }


def _raw_episode_sizes(results_root: Path, run_ids: set[str]) -> np.ndarray:
    sizes = []
    for episode_path in (results_root / "raw" / "pilot").glob("*/episode.json"):
        value = json.loads(episode_path.read_text(encoding="utf-8"))
        if value.get("run_id") not in run_ids:
            continue
        sizes.append(sum(path.stat().st_size for path in episode_path.parent.rglob("*") if path.is_file()))
    return np.asarray(sizes, dtype=float)


def _ordinary_message_activation_summary(
    results_root: Path,
    run_ids: Sequence[str],
) -> Dict[str, float]:
    """Measure decision epochs that produced at least one ordinary message.

    A coalition proposal can fan out to several recipients, so raw messages
    divided by decisions is not a Bernoulli probability and can exceed one.
    The random-gate control instead matches the fraction of independent agent
    decision epochs with any successfully validated, policy-originated message.
    Mandatory monitor sketches are a separate accounting channel and automatic
    breach/late-delivery notices are not coordination-policy decisions.
    """

    decision_keys: set[tuple[str, str, int]] = set()
    message_keys: set[tuple[str, str, int]] = set()
    response_events = 0
    ordinary_messages = 0
    automatic_kinds = {"commitment_breach", "late_delivery"}
    for run_id in sorted(str(value) for value in run_ids):
        path = results_root / "raw" / "pilot" / run_id / "events.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError("missing paired-v8 event ledger: %s" % path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                key = (run_id, str(event.get("actor")), int(event.get("step", -1)))
                if event.get("kind") == "llm_structured_response":
                    response_events += 1
                    if key in decision_keys:
                        raise RuntimeError(
                            "multiple planner responses share one agent/step in %s" % run_id
                        )
                    decision_keys.add(key)
                elif event.get("kind") == "message":
                    message_kind = event.get("payload", {}).get("kind")
                    if message_kind not in automatic_kinds:
                        ordinary_messages += 1
                        message_keys.add(key)
    if response_events != len(decision_keys) or not decision_keys:
        raise RuntimeError("cannot derive a unique paired-v8 communication activation rate")
    active = decision_keys & message_keys
    return {
        "decision_epochs": int(len(decision_keys)),
        "message_active_epochs": int(len(active)),
        "ordinary_validated_messages": int(ordinary_messages),
        "probability": float(len(active) / len(decision_keys)),
    }


def profile_budget(
    results_root: Path,
    config_paths: Sequence[Path],
    output: Path,
    hourly_rates: Sequence[float] = (0.34, 0.69),
) -> Dict[str, Any]:
    """Validate paired-v8 completeness and write a conservative launch budget."""

    episodes, _, _ = collect_results(results_root)
    pilot = episodes[
        (episodes["stage"] == "pilot")
        & (episodes["scenario_name"].isin(PILOT_SCENARIOS))
        & _analysis_valid_mask(episodes)
    ].copy()
    expected = 2 * 3 * 2 * len(CORE_METHODS)
    completed = pilot[pilot["completion_status"] == "complete"].copy()
    if len(pilot) != expected or len(completed) != expected:
        raise RuntimeError(
            "paired-v8 pilot is incomplete: expected %d, found %d rows (%d complete)"
            % (expected, len(pilot), len(completed))
        )
    observed_methods = set(completed["method"])
    if observed_methods != CORE_METHODS:
        raise RuntimeError("paired-v8 method set mismatch: %s" % sorted(observed_methods))
    if float(completed["conservation_error"].abs().max()) > 1e-8:
        raise RuntimeError("paired-v8 contains a resource-conservation failure")

    summaries = _quantile_summary(completed)
    thermo = completed[completed["method"] == "thermoagent"]
    activation = _ordinary_message_activation_summary(
        results_root, thermo["run_id"].astype(str).tolist()
    )
    expected_calls = int(pd.to_numeric(thermo["llm_calls"], errors="coerce").fillna(0).sum())
    if activation["decision_epochs"] != expected_calls:
        raise RuntimeError(
            "paired-v8 ledger/planner-call mismatch: %d decision epochs versus %d calls"
            % (activation["decision_epochs"], expected_calls)
        )

    projects = []
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        projects.append(project_matrix(config, summaries))
    load_seconds = 0.0
    model_smoke = results_root / "smoke" / "model_smoke.json"
    if model_smoke.exists():
        load_seconds = float(json.loads(model_smoke.read_text(encoding="utf-8")).get("load_seconds", 0.0))
    expected_seconds = sum(row["expected_wall_clock_seconds"] for row in projects) + load_seconds * len(projects)
    upper_seconds = sum(row["upper_wall_clock_seconds"] for row in projects) + load_seconds * len(projects)
    expected_gpu_hours = expected_seconds / 3600.0
    upper_gpu_hours = upper_seconds / 3600.0
    raw_sizes = _raw_episode_sizes(results_root, set(completed["run_id"].astype(str)))
    total_projected_episodes = sum(row["episodes"] for row in projects)
    expected_disk = float(np.mean(raw_sizes) * total_projected_episodes) if raw_sizes.size else 0.0
    upper_disk = float(np.quantile(raw_sizes, 0.90) * total_projected_episodes) if raw_sizes.size else 0.0
    rates = sorted(float(rate) for rate in hourly_rates)
    record: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_gate": {
            "expected_episodes": expected,
            "complete_episodes": len(completed),
            "failed_episodes": int((pilot["completion_status"] != "complete").sum()),
            "max_absolute_conservation_error": float(completed["conservation_error"].abs().max()),
            "methods": sorted(observed_methods),
            "scenarios": sorted(set(completed["scenario_name"])),
        },
        "pilot_method_summaries": summaries,
        "thermoagent_communication_active_probability": activation["probability"],
        "communication_probability_calibration": activation,
        "communication_probability_definition": (
            "paired-v8 ThermoAgent independent agent decision epochs with at least one "
            "validated ordinary policy-originated message / all ThermoAgent decision epochs; "
            "monitor sketches and automatic notices excluded"
        ),
        "projection_basis": {
            "centralized_llm": "pilot p90 scaled by decision epochs",
            "other_methods": "pilot p90 scaled by agent count and decision epochs",
            "upper_definition": "sum of method-specific empirical 90th-percentile episode workloads",
            "model_load_overhead_seconds_per_sweep": load_seconds,
        },
        "stage_projections": projects,
        "total_projected_episodes": total_projected_episodes,
        "expected_wall_clock_seconds": expected_seconds,
        "upper_wall_clock_seconds": upper_seconds,
        "expected_gpu_hours": expected_gpu_hours,
        "upper_gpu_hours": upper_gpu_hours,
        "within_24_gpu_hour_limit": upper_gpu_hours <= 24.0,
        "expected_prompt_tokens": sum(row["expected_prompt_tokens"] for row in projects),
        "upper_prompt_tokens": sum(row["upper_prompt_tokens"] for row in projects),
        "expected_generated_tokens": sum(row["expected_generated_tokens"] for row in projects),
        "upper_generated_tokens": sum(row["upper_generated_tokens"] for row in projects),
        "expected_llm_calls": sum(row["expected_llm_calls"] for row in projects),
        "upper_llm_calls": sum(row["upper_llm_calls"] for row in projects),
        "expected_raw_disk_bytes": expected_disk,
        "upper_raw_disk_bytes": upper_disk,
        "hourly_rate_assumptions_usd": rates,
        "expected_cost_range_usd": [expected_gpu_hours * rate for rate in rates],
        "upper_cost_range_usd": [upper_gpu_hours * rate for rate in rates],
        "pricing_note": "Advertised RTX 4090 starting-rate assumptions checked 2026-08-12; actual Pod console rate controls.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
