"""Fail-closed prospective development-gate evaluation for ThermoHITL v3."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .events import sha256_file
from .human_experiments import _atomic_json, _write_dict_csv, source_checksum, utc_now


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _episode_documents(results_root: Path, stage: str) -> Iterable[Dict[str, Any]]:
    for path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def _paired_reduction(
    frame: pd.DataFrame,
    reference: str,
    treatment: str,
) -> Tuple[float, Dict[str, float], int]:
    pivot = frame.pivot_table(
        index=["application", "environment_seed", "scenario"],
        columns="method",
        values="primary_outcome",
        aggfunc="first",
    ).dropna(subset=[reference, treatment])
    reduction = (pivot[reference] - pivot[treatment]) / np.maximum(
        np.abs(pivot[reference]), 1e-9
    )
    by_application = {
        str(application): float(values.mean())
        for application, values in reduction.groupby(level="application")
    }
    return float(reduction.mean()), by_application, len(reduction)


def _gate_engineering(results_root: Path) -> Dict[str, Any]:
    tests = _read_json(results_root / "reproducibility" / "test_summary.json")
    replay = _read_json(results_root / "reproducibility" / "development_replay_report.json")
    passed = bool(
        tests.get("passed")
        and replay.get("episodes_checked", 0) > 0
        and replay.get("mismatches") == 0
        and replay.get("maximum_absolute_conservation_residual", 1.0) < 1e-8
        and replay.get("operator_view_privacy_failures", 0) == 0
        and replay.get("nonfinite_values", 0) == 0
    )
    return {
        "gate": 1,
        "name": "engineering",
        "status": "passed" if passed else "pending_or_failed",
        "passed": passed,
        "tests": tests.get("tests_collected", 0),
        "replays": replay.get("episodes_checked", 0),
        "replay_mismatches": replay.get("mismatches"),
        "maximum_conservation_residual": replay.get(
            "maximum_absolute_conservation_residual"
        ),
        "evidence": "reproducibility/test_summary.json; reproducibility/development_replay_report.json",
    }


def _gate_actionability(results_root: Path) -> Dict[str, Any]:
    stage = "development_trigger_candidate_n10_v4"
    episodes = list(_episode_documents(results_root, stage))
    attempts = sum(row["actionability"]["structured_attempts"] for row in episodes)
    first = sum(row["actionability"]["first_pass_valid"] for row in episodes)
    repaired = sum(row["actionability"]["valid_after_one_repair"] for row in episodes)
    accepted = sum(row["actionability"]["material_actions_accepted"] for row in episodes)
    next_stage = sum(row["actionability"]["material_actions_next_stage"] for row in episodes)
    demand = sum(row["actionability"]["material_actions_reached_demand"] for row in episodes)
    causal_by_application = {
        application: sum(
            probe.get("primary_outcome_changed", False)
            for episode in episodes if episode["application"] == application
            for probe in episode.get("counterfactuals", [])
        )
        for application in ("commercial", "humanitarian")
    }
    first_rate = first / max(attempts, 1)
    repaired_rate = repaired / max(attempts, 1)
    next_rate = next_stage / max(accepted, 1)
    demand_rate = demand / max(accepted, 1)
    mechanics_passed = bool(
        first_rate >= 0.90
        and repaired_rate >= 0.98
        and next_rate >= 0.70
        and demand_rate >= 0.30
        and all(value > 0 for value in causal_by_application.values())
    )
    real_attempt_stages = [
        "development_real_llm_actionability_retry1",
        "development_real_llm_actionability",
    ]
    retained_real_stages = [
        candidate for candidate in real_attempt_stages
        if any(_episode_documents(results_root, candidate))
    ]
    real_stage = (
        retained_real_stages[0]
        if retained_real_stages
        else real_attempt_stages[-1]
    )
    real_episodes = list(_episode_documents(results_root, real_stage))
    real_attempts = sum(
        row["actionability"]["structured_attempts"] for row in real_episodes
    )
    real_first = sum(
        row["actionability"]["first_pass_valid"] for row in real_episodes
    )
    real_final = sum(
        row["actionability"]["valid_after_one_repair"] for row in real_episodes
    )
    real_accepted = sum(
        row["actionability"]["material_actions_accepted"] for row in real_episodes
    )
    real_next = sum(
        row["actionability"]["material_actions_next_stage"] for row in real_episodes
    )
    real_demand = sum(
        row["actionability"]["material_actions_reached_demand"]
        for row in real_episodes
    )
    real_first_rate = real_first / max(real_attempts, 1)
    real_final_rate = real_final / max(real_attempts, 1)
    real_next_rate = real_next / max(real_accepted, 1)
    real_demand_rate = real_demand / max(real_accepted, 1)
    real_applications = sorted({row["application"] for row in real_episodes})
    real_manifests = [
        _read_json(path)
        for path in (results_root / "manifests").glob(
            real_stage + "-*.json"
        )
    ]
    real_qualification_passed = bool(
        set(real_applications) == {"commercial", "humanitarian"}
        and real_attempts > 0
        and real_first_rate >= 0.90
        and real_final_rate >= 0.98
        and real_accepted > 0
        and real_next_rate >= 0.70
        and real_demand_rate >= 0.30
        and len(real_manifests) == len(real_episodes)
        and all(
            manifest.get("model_identifier") != "none"
            and manifest.get("completion_status") == "complete"
            for manifest in real_manifests
        )
    )
    # Deterministic episodes establish causal mechanics and conservation. The
    # independently generated Qwen episodes must separately clear the declared
    # structured-output and material-progression thresholds; mere presence of
    # a model manifest is not sufficient.
    passed = bool(mechanics_passed and real_qualification_passed)
    return {
        "gate": 2,
        "name": "agent_actionability",
        "status": "passed" if passed else (
            "mechanics_passed_real_llm_not_run_or_failed"
            if mechanics_passed else "failed"
        ),
        "passed": passed,
        "mechanics_passed": mechanics_passed,
        "real_llm_qualification_episodes": len(real_episodes),
        "real_llm_selected_stage": real_stage,
        "real_llm_retained_attempt_stages": retained_real_stages,
        "real_llm_applications": real_applications,
        "real_llm_qualification_passed": real_qualification_passed,
        "real_llm_structured_attempts": real_attempts,
        "real_llm_first_pass_validity": real_first_rate,
        "real_llm_valid_after_one_repair": real_final_rate,
        "real_llm_material_actions_accepted": real_accepted,
        "real_llm_accepted_to_next_stage": real_next_rate,
        "real_llm_accepted_to_demand": real_demand_rate,
        "structured_attempts": attempts,
        "first_pass_validity": first_rate,
        "valid_after_one_repair": repaired_rate,
        "accepted_to_next_stage": next_rate,
        "accepted_to_demand": demand_rate,
        "causal_outcome_changes": causal_by_application,
        "evidence": "raw/%s; raw/%s; retained prior real-LLM attempts; manifests" % (stage, real_stage),
    }


def _gate_coordination(results_root: Path) -> Dict[str, Any]:
    stage = "development_gate_coordination_n10"
    path = results_root / stage / "episode_summary.csv"
    if not path.is_file():
        return {"gate": 3, "name": "coordination_necessity", "status": "pending", "passed": False}
    frame = pd.read_csv(path)
    aggregate, applications, panels = _paired_reduction(
        frame, "no_communication", "fixed_communication_no_human"
    )
    regime_rows: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian"):
        selected = frame[frame.application == application]
        for regime in ("moderate", "correlated", "compound"):
            regime_frame = selected[selected.scenario.str.contains("-" + regime + "-")]
            value, _, count = _paired_reduction(
                regime_frame, "no_communication", "fixed_communication_no_human"
            )
            regime_rows.append({
                "application": application,
                "regime": regime,
                "relative_loss_reduction": value,
                "paired_panels": count,
            })
    improved_regimes = {
        application: sum(
            row["relative_loss_reduction"] > 0.0
            for row in regime_rows if row["application"] == application
        )
        for application in ("commercial", "humanitarian")
    }
    passed = bool(
        all(applications.get(application, -1.0) >= 0.05 for application in applications)
        and all(value >= 2 for value in improved_regimes.values())
    )
    _write_dict_csv(results_root / "development" / "coordination_gate_by_regime.csv", regime_rows)
    return {
        "gate": 3,
        "name": "coordination_necessity",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "aggregate_relative_loss_reduction": aggregate,
        "application_relative_loss_reduction": applications,
        "improved_regimes": improved_regimes,
        "paired_panels": panels,
        "evidence": "%s/episode_summary.csv" % stage,
    }


def _gate_human_usefulness(results_root: Path) -> Dict[str, Any]:
    stage = "development_gate_preliminary_v3_n10"
    path = results_root / stage / "episode_summary.csv"
    if not path.is_file():
        return {"gate": 4, "name": "human_causal_usefulness", "status": "pending", "passed": False}
    frame = pd.read_csv(path)
    aggregate, applications, panels = _paired_reduction(
        frame, "autonomous_no_human", "local_kpi_trigger"
    )
    regime_improvements: Dict[str, int] = {}
    for application in ("commercial", "humanitarian"):
        count = 0
        for regime in ("moderate", "correlated", "compound"):
            selected = frame[
                (frame.application == application)
                & frame.scenario.str.contains("-" + regime + "-")
            ]
            reduction, _, _ = _paired_reduction(
                selected, "autonomous_no_human", "local_kpi_trigger"
            )
            count += int(reduction > 0.0)
        regime_improvements[application] = count
    causal = {"commercial": 0, "humanitarian": 0}
    full_chain = {"commercial": 0, "humanitarian": 0}
    for episode in _episode_documents(results_root, stage):
        if episode["method"] != "local_kpi_trigger":
            continue
        for probe in episode.get("counterfactuals", []):
            application = episode["application"]
            causal[application] += int(probe.get("intervention_effect", 0.0) > 1e-9)
            full_chain[application] += int(
                probe.get("agent_accepted")
                and probe.get("material_action_accepted")
                and probe.get("material_reached_demand")
                and probe.get("primary_outcome_changed")
            )
    passed = bool(
        all(applications.get(application, 0.0) > 0.0 for application in applications)
        and all(value >= 2 for value in regime_improvements.values())
        and all(value > 0 for value in full_chain.values())
    )
    return {
        "gate": 4,
        "name": "human_causal_usefulness",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "aggregate_relative_loss_reduction": aggregate,
        "application_relative_loss_reduction": applications,
        "improved_regimes": regime_improvements,
        "beneficial_counterfactuals": causal,
        "complete_causal_chains": full_chain,
        "paired_panels": panels,
        "evidence": "%s/episode_summary.csv; raw/%s" % (stage, stage),
    }


def _gate_information_value(results_root: Path) -> Dict[str, Any]:
    summary_path = results_root / "monitoring" / "causal_value_summary.json"
    summary = _read_json(summary_path)
    passed = bool(summary.get("gate_5_causal_passed_both_applications"))
    return {
        "gate": 5,
        "name": "thermodynamic_information_value",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "application_results": summary.get("incremental", []),
        "evidence": "monitoring/causal_value_summary.json; monitoring/causal_incremental_value.csv",
    }


def _gate_trigger(results_root: Path) -> Dict[str, Any]:
    stage = "development_trigger_candidate_n10_v4"
    path = results_root / stage / "episode_summary.csv"
    if not path.is_file():
        return {"gate": 6, "name": "trigger_feasibility", "status": "pending", "passed": False}
    frame = pd.read_csv(path)
    disrupted = frame[~frame.scenario.str.contains("-nominal-")]
    nominal = frame[frame.scenario.str.contains("-nominal-")]
    regimes: List[Dict[str, Any]] = []
    for (application, scenario), rows in disrupted.groupby(["application", "scenario"], sort=True):
        regimes.append({
            "application": application,
            "scenario": scenario,
            "episodes": len(rows),
            "mean_requests": float(rows.operator_requests.mean()),
            "timely_rate": float(rows.timely_activation.astype(bool).mean()),
            "missed_rate": float(rows.missed_activation.astype(bool).mean()),
            "pre_disruption_false_rate": float(rows.pre_disruption_false_activation.astype(bool).mean()),
        })
    causal = [
        probe
        for episode in _episode_documents(results_root, stage)
        for probe in episode.get("counterfactuals", [])
        if probe.get("intervention_effect", 0.0) > 1e-9
        and probe.get("material_reached_demand")
        and probe.get("primary_outcome_changed")
    ]
    passed = bool(
        regimes
        and all(row["mean_requests"] > 0.0 for row in regimes)
        and all(row["timely_rate"] >= 0.75 for row in regimes)
        and all(row["pre_disruption_false_rate"] <= 0.10 for row in regimes)
        and float(nominal.nominal_false_activation.astype(bool).mean()) <= 0.10
        and float(disrupted.operator_requests.mean()) < 10 * 24 / 2
        and causal
    )
    _write_dict_csv(results_root / "development" / "trigger_gate_by_regime.csv", regimes)
    return {
        "gate": 6,
        "name": "trigger_feasibility",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "nominal_false_activation_rate": float(
            nominal.nominal_false_activation.astype(bool).mean()
        ),
        "regimes": regimes,
        "beneficial_complete_causal_probes": len(causal),
        "frozen_candidate_parameters": {
            "tau_on": 1.5,
            "tau_off": 0.6,
            "actionable_tau_on": 1.1,
            "minimum_dwell": 2,
            "cooldown": 3,
        },
        "evidence": "%s/episode_summary.csv; raw/%s" % (stage, stage),
    }


def evaluate_development_gates(
    repository_root: Path,
    results_root: Path,
) -> Dict[str, Any]:
    gates = [
        _gate_engineering(results_root),
        _gate_actionability(results_root),
        _gate_coordination(results_root),
        _gate_human_usefulness(results_root),
        _gate_information_value(results_root),
        _gate_trigger(results_root),
    ]
    eligible = all(bool(gate["passed"]) for gate in gates)
    record = {
        "created_at": utc_now(),
        "study": "thermohitl_v3",
        "source_checksum": source_checksum(repository_root),
        "prospective_rule": "all six gates must pass before validation or holdout",
        "gates": gates,
        "validation_unlocked": eligible,
        "holdout_unlocked": eligible,
        "decision": (
            "eligible_to_continue" if eligible
            else "fail_closed_stop_before_validation_and_holdout"
        ),
        "failed_or_incomplete_gates": [
            gate["gate"] for gate in gates if not gate["passed"]
        ],
    }
    output = results_root / "development" / "gate_status.json"
    _atomic_json(output, record)
    _write_dict_csv(
        results_root / "tables" / "development_gates.csv",
        [{
            "gate": gate["gate"],
            "name": gate["name"],
            "status": gate["status"],
            "passed": gate["passed"],
            "evidence": gate.get("evidence", ""),
        } for gate in gates],
    )
    record["gate_status_sha256"] = sha256_file(output)
    return record


def assert_holdout_unlocked(results_root: Path) -> Dict[str, Any]:
    record = _read_json(results_root / "development" / "gate_status.json")
    if not record.get("holdout_unlocked"):
        raise RuntimeError(
            "ThermoHITL holdout is locked: every prospective development gate must pass"
        )
    return record
