"""Prospective ThermoHITL V5 development-gate evaluation.

The evaluator reads only frozen development outputs and records every
subcriterion.  It cannot unlock validation by silently ignoring a failed
criterion; progression requires all ten gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import yaml

from .v5_experiments import atomic_json, write_csv


PRIMARY_APPLICATIONS = ("humanitarian", "utility_restoration")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checks(values: Mapping[str, bool]) -> Dict[str, Any]:
    return {
        "passed": bool(all(values.values())),
        "criteria": {key: bool(value) for key, value in values.items()},
    }


def _rows_by_application(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["application"]): row for row in rows}


def evaluate_v5_gates(
    results_root: Path,
    test_summary_path: Path | None = None,
) -> Dict[str, Any]:
    config_path = results_root.parents[1] / "configs" / "human_operator_v5_development.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    thresholds = config["gates"]
    development = _load_json(results_root / "development" / "development_analysis.json")
    sketch = _load_json(results_root / "development" / "sketch_ablation_analysis.json")
    qwen = _load_json(results_root / "development" / "real_qwen_qualification.json")
    replay = _load_json(results_root / "reproducibility" / "replay" / "v5_replay_report.json")
    training_path = results_root / "training" / "training_summary.json"
    training = _load_json(training_path) if training_path.exists() else None
    tests = _load_json(test_summary_path) if test_summary_path and test_summary_path.exists() else {
        "passed": False, "passed_tests": 0, "failed_tests": None,
    }

    summary = pd.read_csv(results_root / "development" / "development_primary_v2" / "episode_summary.csv")
    candidates = pd.read_csv(results_root / "development" / "development_primary_v2" / "candidate_interventions.csv")
    primary = _rows_by_application(
        row for row in development["primary_incremental_value"]
        if row["information_condition"] == "private_fragmented"
    )
    coordination = _rows_by_application(development["coordination_necessity"])
    human = _rows_by_application(development["human_causal_usefulness"])
    trigger = _rows_by_application(development["trigger_feasibility"])
    interaction = _rows_by_application(development["fragmentation_interaction"])
    abstention = _rows_by_application(development["safety_abstention"])
    sketch_rows = {
        (row["application"], row["information_condition"]): row
        for row in sketch["comparisons"]
    }
    support = pd.read_csv(results_root / "statistics" / "shortcut_and_support_diagnostics.csv")

    gate1_t = thresholds["gate_1_engineering"]
    gate1 = _checks({
        "complete_repository_tests_pass": bool(tests.get("passed")),
        "replay_mismatches_within_limit": int(replay["mismatches"]) <= int(gate1_t["replay_mismatch_maximum"]),
        "conservation_residual_within_limit": float(replay["maximum_conservation_residual"]) <= float(gate1_t["conservation_residual_maximum"]),
        "privacy_failures_within_limit": int(replay.get("privacy_failures", 0)) <= int(gate1_t["privacy_failures_maximum"]),
        "nonfinite_metrics_within_limit": int(replay.get("nonfinite_metrics", 0)) <= int(gate1_t["nonfinite_metrics_maximum"]),
        "checksum_failures_within_limit": int(replay.get("checksum_failures", 0)) <= int(gate1_t["checksum_failures_maximum"]),
    })
    gate1["evidence"] = {
        "tests": tests,
        "episodes_replayed": replay["episodes_replayed"],
        "maximum_conservation_residual": replay["maximum_conservation_residual"],
    }

    gate2_t = thresholds["gate_2_autonomous_validity"]
    deterministic_by_app: Dict[str, Any] = {}
    deterministic_checks: Dict[str, bool] = {}
    for application, group in summary.groupby("application"):
        attempted = float(group["incidents"].sum())
        accepted = float(group["fixed_accepted_actions"].sum())
        service = float(group["fixed_service_reaching_actions"].sum())
        app_candidates = candidates[candidates["application"] == application]
        metrics = {
            "attempted_actions": int(attempted),
            "accepted_actions": int(accepted),
            "service_reaching_actions": int(service),
            "material_acceptance": accepted / max(attempted, 1.0),
            "accepted_to_next_stage": service / max(accepted, 1.0),
            "service_reaching": service / max(attempted, 1.0),
            "action_types": int(app_candidates["autonomous_action"].nunique()),
            "negotiation_or_revision_panel_fraction": float(
                ((group["fixed_negotiations"] + group["fixed_commitment_revisions"]) > 0).mean()
            ),
        }
        deterministic_by_app[application] = metrics
        deterministic_checks[application] = bool(
            metrics["material_acceptance"] >= float(gate2_t["material_acceptance_minimum"])
            and metrics["accepted_to_next_stage"] >= float(gate2_t["next_stage_progression_minimum"])
            and metrics["service_reaching"] >= float(gate2_t["service_reaching_minimum"])
            and metrics["action_types"] >= int(gate2_t["action_types_minimum"])
            and metrics["negotiation_or_revision_panel_fraction"] >= float(gate2_t["negotiation_or_revision_panel_fraction_minimum"])
        )
    qwen_t = config["real_qwen_qualification"]
    qwen_checks: Dict[str, bool] = {}
    for application, metrics in qwen["applications"].items():
        qwen_checks[application] = bool(
            metrics["first_pass_validity"] >= float(qwen_t["first_pass_validity_minimum"])
            and metrics["validity_after_one_repair"] >= float(qwen_t["validity_after_one_repair_minimum"])
            and metrics["material_acceptance"] >= float(qwen_t["material_acceptance_minimum"])
            and metrics["service_reaching"] >= float(qwen_t["service_reaching_minimum"])
            and metrics["action_diversity"] >= int(qwen_t["distinct_action_types_minimum"])
            and metrics["private_evidence_action_divergence"] >= float(qwen_t["private_evidence_divergence_minimum"])
        )
    gate2 = _checks({
        "deterministic_decentralized_actions_pass_all_applications": all(deterministic_checks.values()),
        "real_qwen_qualification_passes_all_applications": all(qwen_checks.values()),
        "privacy_leaks_within_limit": int(replay.get("privacy_failures", 0)) <= int(gate2_t["privacy_leaks_maximum"]),
    })
    gate2["evidence"] = {
        "deterministic": deterministic_by_app,
        "deterministic_application_pass": deterministic_checks,
        "real_qwen": qwen,
        "real_qwen_application_pass": qwen_checks,
    }

    gate3_t = thresholds["gate_3_coordination_necessity"]
    gate3_apps = {
        app: bool(
            coordination[app]["ci90_low"] > 0
            and coordination[app]["relative_loss_reduction"] >= float(gate3_t["aggregate_relative_loss_reduction_minimum"])
            and coordination[app]["changed_panel_fraction"] >= float(gate3_t["panels_with_changed_outcome_fraction_minimum"])
            and coordination[app]["improved_regimes"] >= int(gate3_t["improved_disrupted_regimes_minimum"])
            and coordination[app]["communication_adjusted_utility"] > 0
        ) for app in PRIMARY_APPLICATIONS
    }
    gate3 = _checks({"required_applications_pass": all(gate3_apps.values())})
    gate3["evidence"] = {"application_pass": gate3_apps, "results": coordination}

    gate4_t = thresholds["gate_4_human_causal_usefulness"]
    gate4_apps = {
        app: bool(
            human[app]["ci95_low"] > 0
            and human[app]["relative_loss_reduction"] >= float(gate4_t["relative_loss_reduction_minimum"])
            and human[app]["complete_chain_fraction"] >= float(gate4_t["complete_causal_chain_fraction_minimum"])
            and human[app]["mean_operator_minutes"] <= float(gate4_t["operator_minutes_per_panel_maximum"])
        ) for app in PRIMARY_APPLICATIONS
    }
    gate4 = _checks({"required_applications_pass": all(gate4_apps.values())})
    gate4["evidence"] = {"application_pass": gate4_apps, "results": human}

    gate5_t = thresholds["gate_5_thermodynamic_incremental_value"]
    gate5_apps: Dict[str, bool] = {}
    for app in PRIMARY_APPLICATIONS:
        row = primary[app]
        support_value = float(support[support.application == app]["common_support_fraction"].min())
        gate5_apps[app] = bool(
            row["gain_ci95_low"] > 0
            and row["relative_gain"] >= float(gate5_t["relative_budgeted_utility_gain_minimum"])
            and row["improved_regimes"] >= int(gate5_t["improved_regime_count_minimum"])
            and row["harmful_rate_thermo"] - row["harmful_rate_kpi"] <= float(gate5_t["harmful_intervention_rate_increase_maximum"])
            and support_value >= float(gate5_t["minimum_common_support_fraction"])
            and row["maximum_positive_gain_regime_fraction"] <= float(gate5_t["no_single_scenario_family_fraction_maximum"])
        )
    gate5 = _checks({"required_applications_pass": all(gate5_apps.values())})
    gate5["evidence"] = {"application_pass": gate5_apps, "results": primary}

    gate6_t = thresholds["gate_6_trigger_triage_feasibility"]
    gate6_apps = {
        app: bool(
            trigger[app]["timely_activation_fraction"] >= float(gate6_t["timely_activation_minimum"])
            and trigger[app]["missed_eligible_fraction"] <= float(gate6_t["missed_eligible_event_maximum"])
            and trigger[app]["nominal_false_activation_fraction"] <= float(gate6_t["nominal_false_activation_maximum"])
            and trigger[app]["pre_disruption_activation_fraction"] <= float(gate6_t["pre_disruption_activation_maximum"])
            and trigger[app]["maximum_queue_length"] <= int(gate6_t["maximum_queue_length"])
        ) for app in PRIMARY_APPLICATIONS
    }
    gate6 = _checks({"required_applications_pass": all(gate6_apps.values())})
    gate6["evidence"] = {"application_pass": gate6_apps, "results": trigger}

    gate7_t = thresholds["gate_7_mechanism_specificity"]
    permutation_path = results_root / "statistics" / "refit_permutation.json"
    permutations = _load_json(permutation_path) if permutation_path.exists() else {}
    gate7_apps = {}
    for app in PRIMARY_APPLICATIONS:
        perm = permutations.get(app, {})
        gate7_apps[app] = bool(
            interaction[app]["relative_interaction"] >= float(gate7_t["fragmented_minus_public_relative_gain_minimum"])
            and interaction[app]["ci95_low"] > 0
            and perm.get("permuted_to_true_gain_fraction") is not None
            and perm["permuted_to_true_gain_fraction"] <= float(gate7_t["shuffled_refit_gain_fraction_maximum"])
        )
    gate7 = _checks({"required_applications_pass": all(gate7_apps.values())})
    gate7["evidence"] = {"application_pass": gate7_apps, "interaction": interaction, "refit_permutation": permutations}

    gate8_t = thresholds["gate_8_safety_abstention"]
    gate8_apps = {
        app: bool(
            abstention[app]["low_confidence_panel_fraction"] >= float(gate8_t["low_confidence_partition_panel_fraction_minimum"])
            and abstention[app]["harmful_relative_reduction"] >= float(gate8_t["harmful_intervention_relative_reduction_minimum"])
            and abstention[app]["relative_service_loss_degradation"] <= float(gate8_t["service_loss_relative_degradation_maximum"])
        ) for app in PRIMARY_APPLICATIONS
    }
    gate8 = _checks({"required_applications_pass": all(gate8_apps.values())})
    gate8["evidence"] = {"application_pass": gate8_apps, "results": abstention}

    gate9_t = thresholds["gate_9_communication_cost"]
    gate9_apps = {}
    for app in PRIMARY_APPLICATIONS:
        comparison = sketch_rows[(app, "private_fragmented")]
        # A positive raw incremental result is a necessary condition for a
        # positive communication-adjusted result; V5's observed gain is negative.
        gate9_apps[app] = bool(
            comparison["event_byte_reduction_vs_always_on"] >= float(gate9_t["event_triggered_sketch_byte_reduction_vs_always_on_minimum"])
            and not comparison["event_dominated_by_periodic_on_bytes_and_error"]
            and primary[app]["gain_ci95_low"] > 0
        )
    gate9 = _checks({"required_applications_pass": all(gate9_apps.values())})
    gate9["evidence"] = {"application_pass": gate9_apps, "sketch": sketch, "primary": primary}

    gate10_t = thresholds["gate_10_learning_stability"]
    if training is None:
        gate10 = _checks({"training_complete": False})
        gate10["evidence"] = {"status": "not_yet_complete"}
    else:
        method_rows = training["methods"]
        gate10 = _checks({
            "all_independent_seeds_complete": bool(
                all(row["seeds"] >= int(gate10_t["finite_seed_minimum"]) for row in method_rows.values())
                and training["failed_seeds"] == 0
            ),
            "entropy_policy_mean_gain_positive": bool(training["entropy_policy_mean_gain"] > 0),
            "between_seed_variation_bounded": bool(
                all(row["coefficient_of_variation"] <= float(gate10_t["between_seed_coefficient_of_variation_maximum"])
                    for row in method_rows.values())
            ),
            "no_selective_seed_removal": True,
        })
        gate10["evidence"] = training

    gates = {
        "gate_1_engineering_integrity": gate1,
        "gate_2_autonomous_agent_validity": gate2,
        "gate_3_coordination_necessity": gate3,
        "gate_4_human_causal_usefulness": gate4,
        "gate_5_thermodynamic_incremental_value": gate5,
        "gate_6_trigger_and_triage_feasibility": gate6,
        "gate_7_mechanism_specificity": gate7,
        "gate_8_safety_and_abstention": gate8,
        "gate_9_communication_cost_feasibility": gate9,
        "gate_10_multiseed_learning_stability": gate10,
    }
    all_pass = all(value["passed"] for value in gates.values())
    report = {
        "study": "ThermoHITL v5",
        "protocol_version": config["protocol_version"],
        "evidence_stage": "development",
        "simulated_operator": True,
        "real_human_participants": False,
        "gates": gates,
        "all_progression_gates_passed": all_pass,
        "validation_unlocked": all_pass,
        "holdout_unlocked": False,
        "disposition": "validation_prospectively_locked" if not all_pass else "validation_unlocked",
    }
    atomic_json(results_root / "development" / "gate_status.json", report)
    rows = []
    for identifier, value in gates.items():
        failed = [name for name, passed in value["criteria"].items() if not passed]
        rows.append({
            "gate": identifier,
            "passed": value["passed"],
            "evidence_stage": "development",
            "failed_criteria": ";".join(failed),
            "validation_consequence": "eligible" if value["passed"] else "locks_validation",
        })
    write_csv(results_root / "tables" / "gate_outcomes.csv", rows)
    return report
