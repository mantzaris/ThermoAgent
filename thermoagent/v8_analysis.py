"""Panel-level V8 pilot selection and prospective primary analysis helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .v5_experiments import atomic_json, write_csv
from .v8_io import read_csv_gzip


NON_ENTROPIC = {
    "periodic", "matched_random", "kpi_change",
    "predictive_uncertainty_change", "l1_belief_drift",
    "age_of_information",
}


def _load(results_root: Path, stage: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(results_root / stage / "episode_summary.csv")
    registry = pd.read_csv(results_root / stage / "candidate_registry.csv")
    frame = frame.merge(
        registry[["candidate_name", "configuration_digest", "method", "encoding"]],
        left_on=["trigger_configuration_digest", "encoding"],
        right_on=["configuration_digest", "encoding"],
        how="left",
        suffixes=("", "_registry"),
    )
    if frame.candidate_name.isna().any():
        missing = frame.loc[frame.candidate_name.isna(), "trigger_configuration_digest"].unique()
        raise RuntimeError("candidate registry is incomplete: %s" % list(missing))
    frame["panel_id"] = frame.application.astype(str) + ":" + frame.environment_seed.astype(str)
    return frame, registry


def _paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    encoding_counts = frame.groupby("encoding").candidate_name.nunique()
    primary_encoding = str(encoding_counts.sort_values(ascending=False).index[0])
    always = frame[
        (frame.scheduler == "always_on") & (frame.encoding == primary_encoding)
    ].copy()
    reference_columns = [
        "panel_id", "sketch_on_wire_bytes", "fully_counted_bytes",
        "transmitted_sketch_messages", "fully_counted_messages",
        "normalized_time_integrated_estimation_error",
        "primary_distributed_state_error",
        "primary_distributed_state_error_p95",
        "pointwise_estimation_mae_p95",
        "disagreement_time_integrated_error", "mean_detection_delay_steps",
        "service_loss", "autonomous_harmful_actions", "normalized_autonomous_reward",
    ]
    always = always[reference_columns].rename(columns={
        key: key + "_always" for key in reference_columns if key != "panel_id"
    })
    paired = frame[frame.encoding == primary_encoding].merge(always, on="panel_id", how="inner")
    paired["sketch_byte_reduction"] = 1.0 - (
        paired.sketch_on_wire_bytes / paired.sketch_on_wire_bytes_always.replace(0, np.nan)
    )
    paired["fully_counted_byte_reduction"] = 1.0 - (
        paired.fully_counted_bytes / paired.fully_counted_bytes_always.replace(0, np.nan)
    )
    paired["message_reduction"] = 1.0 - (
        paired.transmitted_sketch_messages
        / paired.transmitted_sketch_messages_always.replace(0, np.nan)
    )
    paired["log_sketch_message_ratio"] = np.log(
        (paired.transmitted_sketch_messages + 1.0)
        / (paired.transmitted_sketch_messages_always + 1.0)
    )
    paired["log_wire_byte_ratio"] = np.log(
        (paired.sketch_on_wire_bytes + 1.0)
        / (paired.sketch_on_wire_bytes_always + 1.0)
    )
    paired["estimation_error_increase"] = (
        paired.normalized_time_integrated_estimation_error
        - paired.normalized_time_integrated_estimation_error_always
    )
    paired["primary_estimation_error_increase"] = (
        paired.primary_distributed_state_error
        - paired.primary_distributed_state_error_always
    )
    paired["primary_p95_increase"] = (
        paired.primary_distributed_state_error_p95
        - paired.primary_distributed_state_error_p95_always
    )
    paired["pointwise_p95_increase"] = (
        paired.pointwise_estimation_mae_p95
        - paired.pointwise_estimation_mae_p95_always
    )
    paired["disagreement_error_increase"] = (
        paired.disagreement_time_integrated_error
        - paired.disagreement_time_integrated_error_always
    )
    paired["detection_delay_increase"] = (
        paired.mean_detection_delay_steps - paired.mean_detection_delay_steps_always
    )
    paired["service_loss_increase"] = paired.service_loss - paired.service_loss_always
    paired["harmful_action_increase"] = (
        paired.autonomous_harmful_actions - paired.autonomous_harmful_actions_always
    )
    paired["reward_degradation"] = (
        paired.normalized_autonomous_reward_always - paired.normalized_autonomous_reward
    )
    return paired


def analyze_v8_pilot(results_root: Path, stage: str = "pilots") -> Dict[str, Any]:
    frame, registry = _load(results_root, stage)
    paired = _paired_comparisons(frame)
    write_csv(results_root / stage / "paired_scheduler_results.csv", paired.to_dict("records"))
    aggregate_rows: List[Dict[str, Any]] = []
    for (candidate, application), values in paired.groupby(["candidate_name", "application"]):
        aggregate_rows.append({
            "candidate_name": candidate,
            "application": application,
            "method": values.scheduler.iloc[0],
            "panels": int(values.panel_id.nunique()),
            "mean_sketch_bytes": float(values.sketch_on_wire_bytes.mean()),
            "mean_fully_counted_bytes": float(values.fully_counted_bytes.mean()),
            "mean_sketch_byte_reduction": float(values.sketch_byte_reduction.mean()),
            "mean_fully_counted_byte_reduction": float(values.fully_counted_byte_reduction.mean()),
            "mean_message_reduction": float(values.message_reduction.mean()),
            "mean_estimation_error": float(values.normalized_time_integrated_estimation_error.mean()),
            "mean_estimation_error_increase": float(values.estimation_error_increase.mean()),
            "mean_primary_estimation_error": float(values.primary_distributed_state_error.mean()),
            "mean_primary_estimation_error_increase": float(values.primary_estimation_error_increase.mean()),
            "maximum_primary_p95": float(values.primary_distributed_state_error_p95.max()),
            "maximum_primary_p95_increase": float(values.primary_p95_increase.max()),
            "mean_disagreement_error": float(values.disagreement_time_integrated_error.mean()),
            "mean_disagreement_error_increase": float(values.disagreement_error_increase.mean()),
            "mean_detection_delay_increase": float(values.detection_delay_increase.mean()),
            "maximum_pointwise_p95": float(values.pointwise_estimation_mae_p95.max()),
            "maximum_pointwise_p95_increase": float(values.pointwise_p95_increase.max()),
            "mean_service_loss_increase": float(values.service_loss_increase.mean()),
            "mean_harmful_action_increase": float(values.harmful_action_increase.mean()),
            "mean_reward_degradation": float(values.reward_degradation.mean()),
            "mean_activation_rate": float(values.trigger_activation_rate.mean()),
        })
    aggregates = pd.DataFrame(aggregate_rows)
    write_csv(results_root / stage / "scheduler_pilot_summary.csv", aggregate_rows)

    encoding_frame = frame[frame.scheduler == "always_on"].copy()
    encoding_counts = encoding_frame.groupby("panel_id").encoding.nunique()
    matched_encoding_panels = set(encoding_counts[encoding_counts >= 3].index)
    if matched_encoding_panels:
        encoding_frame = encoding_frame[encoding_frame.panel_id.isin(matched_encoding_panels)]
    elif encoding_frame.encoding.nunique() != 1:
        raise RuntimeError("encoding comparison lacks matched panels")
    encoding_rows = []
    for encoding, values in encoding_frame.groupby("encoding"):
        encoding_rows.append({
            "encoding": encoding,
            "episodes": len(values),
            "mean_sketch_bytes": float(values.sketch_on_wire_bytes.mean()),
            "mean_quantization_l1_error": float(values.mean_quantization_l1_error.mean()),
            "mean_estimation_error": float(values.normalized_time_integrated_estimation_error.mean()),
        })
    write_csv(results_root / stage / "encoding_pilot_summary.csv", encoding_rows)
    encodings = {row["encoding"]: row for row in encoding_rows}
    selected_encoding = str(encoding_rows[0]["encoding"]) if len(encoding_rows) == 1 else "fp32"
    if (
        "fp16" in encodings
        and encodings["fp16"]["mean_quantization_l1_error"] <= 0.001
        and encodings["fp16"]["mean_sketch_bytes"] < encodings.get("fp32", {"mean_sketch_bytes": math.inf})["mean_sketch_bytes"]
    ):
        selected_encoding = "fp16"
    if (
        "uint8_simplex" in encodings
        and encodings["uint8_simplex"]["mean_quantization_l1_error"] <= 0.005
        and encodings["uint8_simplex"]["mean_estimation_error"]
        <= encodings[selected_encoding]["mean_estimation_error"] + 0.002
        and encodings["uint8_simplex"]["mean_sketch_bytes"]
        < encodings[selected_encoding]["mean_sketch_bytes"]
    ):
        selected_encoding = "uint8_simplex"

    generalized = aggregates[aggregates.method == "generalized_information"]
    eligible_candidates = []
    for candidate, values in generalized.groupby("candidate_name"):
        applications = set(values.application)
        eligible = (
            applications == {"humanitarian", "utility_restoration"}
            and bool((values.mean_sketch_byte_reduction >= 0.25).all())
            and bool((values.mean_primary_estimation_error_increase <= 0.02).all())
            and bool((values.maximum_pointwise_p95_increase <= 0.01).all())
            and bool((values.mean_detection_delay_increase <= 5.0).all())
        )
        if eligible:
            eligible_candidates.append({
                "candidate_name": candidate,
                "worst_estimation_increase": float(values.mean_primary_estimation_error_increase.max()),
                "worst_delay_increase": float(values.mean_detection_delay_increase.max()),
                "mean_byte_reduction": float(values.mean_sketch_byte_reduction.mean()),
            })
    eligible_candidates.sort(key=lambda value: (
        value["worst_estimation_increase"], value["worst_delay_increase"],
        -value["mean_byte_reduction"], value["candidate_name"],
    ))
    selected_trigger = eligible_candidates[0]["candidate_name"] if eligible_candidates else None

    selected_comparator = None
    comparator_ranking: List[Dict[str, Any]] = []
    if selected_trigger is not None:
        selected_rows = paired[paired.candidate_name == selected_trigger][
            ["panel_id", "sketch_on_wire_bytes"]
        ].rename(columns={"sketch_on_wire_bytes": "selected_bytes"})
        for candidate, values in paired[paired.scheduler.isin(NON_ENTROPIC)].groupby("candidate_name"):
            matched = values.merge(selected_rows, on="panel_id", how="inner")
            closeness = float(np.mean(np.abs(np.log(
                (matched.sketch_on_wire_bytes + 1.0) / (matched.selected_bytes + 1.0)
            ))))
            comparator_ranking.append({
                "candidate_name": candidate,
                "budget_log_distance": closeness,
                "mean_disagreement_error": float(matched.disagreement_time_integrated_error.mean()),
                "mean_detection_delay": float(matched.mean_detection_delay_steps.mean()),
                "mean_service_loss": float(matched.service_loss.mean()),
            })
        comparator_ranking.sort(key=lambda value: (
            value["budget_log_distance"], value["mean_disagreement_error"],
            value["mean_detection_delay"], value["mean_service_loss"],
            value["candidate_name"],
        ))
        selected_comparator = comparator_ranking[0]["candidate_name"] if comparator_ranking else None
    write_csv(results_root / stage / "nonentropic_comparator_ranking.csv", comparator_ranking)

    primary_encoding = str(paired.encoding.iloc[0]) if len(paired) else "fp16"
    always = frame[(frame.scheduler == "always_on") & (frame.encoding == primary_encoding)]
    none = frame[(frame.scheduler == "none") & (frame.encoding == primary_encoding)]
    base = always[["panel_id", "primary_distributed_state_error"]].merge(
        none[["panel_id", "primary_distributed_state_error"]],
        on="panel_id", suffixes=("_always", "_none"),
    )
    by_application = []
    for application in sorted(frame.application.unique()):
        panel_ids = set(frame.loc[frame.application == application, "panel_id"])
        subset = base[base.panel_id.isin(panel_ids)]
        by_application.append(bool(len(subset) and (
            subset.primary_distributed_state_error_none
            - subset.primary_distributed_state_error_always
        ).mean() > 0.0))
    always_better = bool(by_application and all(by_application))
    selected_frame = frame[frame.candidate_name == selected_trigger] if selected_trigger else frame.iloc[0:0]
    feasibility = {
        "selected_trigger_exists": selected_trigger is not None,
        "activation_nonzero": bool(len(selected_frame) and (selected_frame.trigger_activations > 0).all()),
        "not_always_on": bool(len(selected_frame) and (selected_frame.trigger_activation_rate < 0.95).all()),
        "always_on_improves_estimation_over_none": always_better,
        "delivered_sketches_update_private_beliefs": bool(
            len(selected_frame) and (selected_frame.belief_updates_from_delivered_sketches > 0).all()
        ),
        "beneficial_action_opportunities_present": bool((frame.autonomous_beneficial_actions > 0).any()),
        "harmful_action_opportunities_present": bool((frame.autonomous_harmful_actions > 0).any()),
        "equal_entropy_mode_switch_unit_test_required": True,
        "wire_roundtrip_unit_test_required": True,
    }
    feasibility["pilot_feasible"] = bool(all(feasibility.values()))
    selection = {
        "stage": "development_pilot_only",
        "selected_encoding": selected_encoding,
        "selected_generalized_trigger": selected_trigger,
        "selected_strongest_nonentropic_comparator": selected_comparator,
        "eligible_generalized_candidates": eligible_candidates,
        "selection_rule_frozen_before_pilot": True,
        "feasibility": feasibility,
    }
    atomic_json(results_root / stage / "pilot_selection.json", selection)
    atomic_json(results_root / stage / "pilot_feasibility.json", feasibility)
    return selection


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, replicates: int = 10000,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap input must be a finite nonempty vector")
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, len(array), size=(int(replicates), len(array)))
    distribution = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "bootstrap_replicates": int(replicates),
    }


def analyze_v8_development(
    results_root: Path, stage: str = "development",
    *, selected_generalized: str = "generalized_0125_u8",
    bootstrap_replicates: int = 10000,
) -> Dict[str, Any]:
    """Select the comparator and quantify development precision by panel."""
    frame, _ = _load(results_root, stage)
    paired = _paired_comparisons(frame)
    write_csv(
        results_root / stage / "paired_scheduler_results.csv",
        paired.to_dict("records"),
    )
    if selected_generalized not in set(paired.candidate_name):
        raise RuntimeError("preregistered generalized candidate is absent")
    selected = paired[paired.candidate_name == selected_generalized].copy()
    selected_budget = selected[["panel_id", "application", "sketch_on_wire_bytes"]].rename(
        columns={"sketch_on_wire_bytes": "selected_bytes"},
    )
    ranking: List[Dict[str, Any]] = []
    for candidate, values in paired[paired.scheduler.isin(NON_ENTROPIC)].groupby(
        "candidate_name",
    ):
        matched = values.merge(
            selected_budget, on=["panel_id", "application"], how="inner",
        )
        if matched.panel_id.nunique() != selected.panel_id.nunique():
            continue
        per_application_distance = matched.groupby("application").apply(
            lambda group: float(np.mean(np.abs(np.log(
                (group.sketch_on_wire_bytes + 1.0) / (group.selected_bytes + 1.0)
            ))))
        )
        ranking.append({
            "candidate_name": candidate,
            "worst_application_budget_log_distance": float(per_application_distance.max()),
            "mean_budget_log_distance": float(per_application_distance.mean()),
            "mean_primary_distributed_state_error": float(
                matched.primary_distributed_state_error.mean()
            ),
            "mean_disagreement_error": float(
                matched.disagreement_time_integrated_error.mean()
            ),
            "mean_detection_delay": float(matched.mean_detection_delay_steps.mean()),
            "mean_service_loss": float(matched.service_loss.mean()),
        })
    ranking.sort(key=lambda value: (
        value["worst_application_budget_log_distance"],
        value["mean_primary_distributed_state_error"],
        value["mean_disagreement_error"],
        value["mean_detection_delay"],
        value["mean_service_loss"],
        value["candidate_name"],
    ))
    if not ranking:
        raise RuntimeError("no complete non-entropic comparator was available")
    comparator = str(ranking[0]["candidate_name"])
    write_csv(results_root / stage / "nonentropic_comparator_ranking.csv", ranking)

    interval_rows: List[Dict[str, Any]] = []
    primary_metrics = (
        "sketch_byte_reduction", "message_reduction",
        "fully_counted_byte_reduction", "primary_estimation_error_increase",
        "primary_p95_increase", "detection_delay_increase",
        "service_loss_increase", "harmful_action_increase", "reward_degradation",
    )
    seed_counter = 0
    for application, values in selected.groupby("application"):
        for metric in primary_metrics:
            interval = _bootstrap_mean_interval(
                values[metric].to_numpy(), seed=881800 + seed_counter,
                replicates=bootstrap_replicates,
            )
            seed_counter += 1
            interval_rows.append({
                "comparison": "%s_vs_always_on" % selected_generalized,
                "application": application, "metric": metric,
                "independent_panels": int(values.panel_id.nunique()), **interval,
            })
    generalized_direct = frame[frame.candidate_name == selected_generalized].copy()
    comparator_direct = frame[frame.candidate_name == comparator].copy()
    direct = generalized_direct.merge(
        comparator_direct,
        on=["panel_id", "application"], suffixes=("_generalized", "_comparator"),
    )
    h2_metrics = {
        "primary_state_error_advantage": (
            direct.primary_distributed_state_error_comparator
            - direct.primary_distributed_state_error_generalized
        ),
        "disagreement_error_advantage": (
            direct.disagreement_time_integrated_error_comparator
            - direct.disagreement_time_integrated_error_generalized
        ),
        "detection_delay_advantage": (
            direct.mean_detection_delay_steps_comparator
            - direct.mean_detection_delay_steps_generalized
        ),
        "consensus_recovery_advantage": (
            direct.consensus_recovery_steps_comparator
            - direct.consensus_recovery_steps_generalized
        ),
        "stale_row_advantage": (
            direct.stale_belief_rows_comparator
            - direct.stale_belief_rows_generalized
        ),
    }
    for application in sorted(direct.application.unique()):
        mask = direct.application.eq(application).to_numpy()
        for metric, values in h2_metrics.items():
            interval = _bootstrap_mean_interval(
                np.asarray(values)[mask], seed=881800 + seed_counter,
                replicates=bootstrap_replicates,
            )
            seed_counter += 1
            interval_rows.append({
                "comparison": "%s_vs_%s" % (selected_generalized, comparator),
                "application": application, "metric": metric,
                "independent_panels": int(mask.sum()), **interval,
            })
    write_csv(results_root / stage / "panel_bootstrap_intervals.csv", interval_rows)

    def interval(application: str, metric: str) -> Mapping[str, Any]:
        matches = [
            value for value in interval_rows
            if value["comparison"] == "%s_vs_always_on" % selected_generalized
            and value["application"] == application and value["metric"] == metric
        ]
        if len(matches) != 1:
            raise AssertionError("development interval lookup is not unique")
        return matches[0]

    applications = ("humanitarian", "utility_restoration")
    feasibility = {
        "minimum_24_independent_panels_per_application": all(
            selected[selected.application.eq(value)].panel_id.nunique() >= 24
            for value in applications
        ),
        "message_reduction_point_at_least_0_25": all(
            interval(value, "message_reduction")["mean"] >= 0.25
            for value in applications
        ),
        "byte_reduction_point_at_least_0_25": all(
            interval(value, "sketch_byte_reduction")["mean"] >= 0.25
            for value in applications
        ),
        "primary_error_upper_interval_at_most_0_02": all(
            interval(value, "primary_estimation_error_increase")["ci_high"] <= 0.02
            for value in applications
        ),
        "p95_error_increase_upper_interval_at_most_0_01": all(
            interval(value, "primary_p95_increase")["ci_high"] <= 0.01
            for value in applications
        ),
        "detection_delay_upper_interval_at_most_one_epoch": all(
            interval(value, "detection_delay_increase")["ci_high"] <= 5.0
            for value in applications
        ),
        "nondegenerate_activation": bool(
            (generalized_direct.trigger_activation_rate > 0.0).all()
            and (generalized_direct.trigger_activation_rate < 0.95).all()
        ),
        "privacy_and_conservation": bool(
            generalized_direct.privacy_boundary_pass.all()
            and generalized_direct.conservation_feasible.all()
        ),
        "delivered_information_changed_private_beliefs": bool(
            (generalized_direct.belief_updates_from_delivered_sketches > 0).all()
        ),
        "downstream_outcomes_can_change_across_schedulers": bool(
            (selected.service_loss_increase.abs() > 1e-12).any()
            or (selected.harmful_action_increase.abs() > 1e-12).any()
        ),
    }
    feasibility["development_progression_feasible"] = bool(all(feasibility.values()))
    h2_rows = [
        value for value in interval_rows
        if value["comparison"] == "%s_vs_%s" % (selected_generalized, comparator)
        and value["metric"] == "primary_state_error_advantage"
    ]
    entropy_specific_supported_in_development = bool(
        len(h2_rows) == 2 and all(value["ci_low"] > 0.0 for value in h2_rows)
    )
    report = {
        "stage": "formal_development",
        "selected_generalized_trigger": selected_generalized,
        "selected_strongest_nonentropic_comparator": comparator,
        "encoding": "uint8_simplex",
        "independent_panels_per_application": {
            value: int(selected[selected.application.eq(value)].panel_id.nunique())
            for value in applications
        },
        "bootstrap_replicates": int(bootstrap_replicates),
        "development_feasibility": feasibility,
        "entropy_specific_h2_supported_in_development": entropy_specific_supported_in_development,
        "h2_is_extension_not_progression_gate": True,
        "validation_or_holdout_evidence": False,
    }
    atomic_json(results_root / stage / "development_selection.json", report)
    atomic_json(results_root / stage / "development_gates.json", feasibility)
    return report


def analyze_v8_final_development(
    results_root: Path, stage: str = "development_final",
    *, bootstrap_replicates: int = 10000,
) -> Dict[str, Any]:
    """Apply the prewritten final two-candidate development selection rule."""
    frame, _ = _load(results_root, stage)
    paired = _paired_comparisons(frame)
    pre_rows: List[Dict[str, Any]] = []
    for value in frame.to_dict("records"):
        run_dir = results_root / "raw" / stage / str(value["run_id"])
        trigger_path = run_dir / "triggers.csv.gz"
        if not trigger_path.exists():
            continue
        onset = max(
            4,
            int(round(
                (0.18 if value["application"] == "humanitarian" else 0.16)
                * int(value["horizon"])
            )),
        )
        trigger_rows = read_csv_gzip(trigger_path)
        eligible_rows = [
            row for row in trigger_rows
            if int(row["step"]) < onset and row["reason"] != "initial_reference"
        ]
        transmissions = sum(
            str(row["transmit"]).lower() == "true" for row in eligible_rows
        )
        transmitted_reasons = [
            str(row["reason"]) for row in trigger_rows
            if str(row["transmit"]).lower() == "true"
        ]
        information_transmissions = sum(
            reason.startswith("generalized_information_")
            for reason in transmitted_reasons
        )
        noninitial_nonrecovery = sum(
            reason not in ("initial_reference", "partition_recovery")
            for reason in transmitted_reasons
        )
        pre_rows.append({
            "run_id": value["run_id"], "candidate_name": value["candidate_name"],
            "application": value["application"], "environment_seed": value["environment_seed"],
            "disruption_step": onset,
            "pre_disruption_noninitial_evaluations": len(eligible_rows),
            "pre_disruption_noninitial_transmissions": transmissions,
            "pre_disruption_noninitial_transmission_rate": float(
                transmissions / max(len(eligible_rows), 1)
            ),
            "information_score_transmissions": information_transmissions,
            "noninitial_nonrecovery_transmissions": noninitial_nonrecovery,
        })
    write_csv(results_root / stage / "pre_disruption_transmission_rates.csv", pre_rows)
    pre_frame = pd.DataFrame(pre_rows)
    candidate_rows: List[Dict[str, Any]] = []
    eligible: List[Dict[str, Any]] = []
    counter = 0
    generalized_candidates = sorted(
        str(value) for value in frame.loc[
            frame.scheduler.eq("generalized_information"), "candidate_name"
        ].unique()
    )
    for candidate in generalized_candidates:
        values = paired[paired.candidate_name.eq(candidate)]
        candidate_eligible = True
        ranking_errors = []
        ranking_reductions = []
        for application in ("humanitarian", "utility_restoration"):
            subset = values[values.application.eq(application)]
            intervals = {}
            for metric in (
                "message_reduction", "sketch_byte_reduction",
                "primary_estimation_error_increase", "primary_p95_increase",
                "detection_delay_increase",
            ):
                intervals[metric] = _bootstrap_mean_interval(
                    subset[metric].to_numpy(), seed=881950 + counter,
                    replicates=bootstrap_replicates,
                )
                counter += 1
            integration = bool(
                (frame[
                    frame.candidate_name.eq(candidate)
                    & frame.application.eq(application)
                ].belief_updates_from_delivered_sketches > 0).all()
            )
            pre_subset = pre_frame[
                pre_frame.candidate_name.eq(candidate)
                & pre_frame.application.eq(application)
            ]
            mean_pre_disruption_rate = float(
                pre_subset.pre_disruption_noninitial_transmission_rate.mean()
            ) if len(pre_subset) else math.inf
            information_transmissions = int(
                pre_subset.information_score_transmissions.sum()
            ) if len(pre_subset) else 0
            noninitial_nonrecovery = int(
                pre_subset.noninitial_nonrecovery_transmissions.sum()
            ) if len(pre_subset) else 0
            information_fraction = float(
                information_transmissions / max(noninitial_nonrecovery, 1)
            )
            passed = bool(
                intervals["message_reduction"]["ci_low"] >= 0.25
                and intervals["sketch_byte_reduction"]["ci_low"] >= 0.25
                and intervals["primary_estimation_error_increase"]["ci_high"] <= 0.02
                and intervals["primary_p95_increase"]["ci_high"] <= 0.01
                and intervals["detection_delay_increase"]["ci_high"] <= 5.0
                and integration
                and mean_pre_disruption_rate <= 0.10
                and information_transmissions > 0
                and information_fraction >= 0.05
            )
            candidate_eligible = candidate_eligible and passed
            ranking_errors.append(float(
                frame[
                    frame.candidate_name.eq(candidate)
                    & frame.application.eq(application)
                ].primary_distributed_state_error.mean()
            ))
            ranking_reductions.append(intervals["sketch_byte_reduction"]["mean"])
            candidate_rows.append({
                "candidate_name": candidate, "application": application,
                "independent_panels": int(subset.panel_id.nunique()),
                "all_panels_integrated_delivered_beliefs": integration,
                "mean_pre_disruption_noninitial_transmission_rate": mean_pre_disruption_rate,
                "pre_disruption_rate_limit": 0.10,
                "information_score_transmissions": information_transmissions,
                "noninitial_nonrecovery_transmissions": noninitial_nonrecovery,
                "information_score_transmission_fraction": information_fraction,
                "minimum_information_score_fraction": 0.05,
                "eligible_in_application": passed,
                **{
                    "%s_%s" % (metric, field): result[field]
                    for metric, result in intervals.items()
                    for field in ("mean", "ci_low", "ci_high")
                },
            })
        if candidate_eligible:
            eligible.append({
                "candidate_name": candidate,
                "worst_application_primary_error": max(ranking_errors),
                "mean_byte_reduction": float(np.mean(ranking_reductions)),
            })
    write_csv(results_root / stage / "final_trigger_candidate_intervals.csv", candidate_rows)
    eligible.sort(key=lambda value: (
        value["worst_application_primary_error"],
        -value["mean_byte_reduction"], value["candidate_name"],
    ))
    selection = eligible[0]["candidate_name"] if eligible else None
    if selection is None:
        report = {
            "stage": stage, "selected_generalized_trigger": None,
            "eligible_candidates": eligible,
            "development_progression_feasible": False,
            "reason": "no candidate met the frozen final development rule",
        }
        atomic_json(results_root / stage / "development_selection.json", report)
        return report
    report = analyze_v8_development(
        results_root, stage, selected_generalized=selection,
        bootstrap_replicates=bootstrap_replicates,
    )
    report["eligible_final_candidates"] = eligible
    report["selection_rule"] = (
        "all H1 constraints and all-panel integration; then minimum worst-application "
        "primary error, byte reduction, stable name"
    )
    atomic_json(results_root / stage / "development_selection.json", report)
    return report


def analyze_v8_hysteresis_repair(
    results_root: Path, stage: str = "hysteresis_repair_pilot",
) -> Dict[str, Any]:
    """Apply the mechanism-only rule written before the repair pilot."""
    frame, _ = _load(results_root, stage)
    paired = _paired_comparisons(frame)
    candidates: List[Dict[str, Any]] = []
    generalized_candidates = sorted(
        str(value) for value in frame.loc[
            frame.scheduler.eq("generalized_information"), "candidate_name"
        ].unique()
    )
    for candidate in generalized_candidates:
        candidate_frame = frame[frame.candidate_name.eq(candidate)]
        by_application: List[Dict[str, Any]] = []
        eligible = True
        for application in ("humanitarian", "utility_restoration"):
            subset = candidate_frame[candidate_frame.application.eq(application)]
            reason_counts: Dict[str, int] = {}
            predisruption_evaluations = 0
            predisruption_transmissions = 0
            for value in subset.to_dict("records"):
                path = results_root / "raw" / stage / str(value["run_id"]) / "triggers.csv.gz"
                onset = max(
                    4,
                    int(round(
                        (0.18 if application == "humanitarian" else 0.16)
                        * int(value["horizon"])
                    )),
                )
                for row in read_csv_gzip(path):
                    if int(row["step"]) < onset and row["reason"] != "initial_reference":
                        predisruption_evaluations += 1
                        predisruption_transmissions += int(
                            str(row["transmit"]).lower() == "true"
                        )
                    if str(row["transmit"]).lower() == "true":
                        reason = str(row["reason"])
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
            information = int(sum(
                count for reason, count in reason_counts.items()
                if reason.startswith("generalized_information_")
            ))
            denominator = sum(
                count for reason, count in reason_counts.items()
                if reason not in ("initial_reference", "partition_recovery")
            )
            fraction = float(information / max(denominator, 1))
            predisruption_rate = float(
                predisruption_transmissions / max(predisruption_evaluations, 1)
            )
            application_eligible = bool(
                information > 0
                and fraction >= 0.05
                and (subset.trigger_activation_rate > 0.0).all()
                and (subset.trigger_activation_rate < 0.95).all()
                and (subset.belief_updates_from_delivered_sketches > 0).all()
                and predisruption_rate <= 0.10
            )
            eligible = eligible and application_eligible
            by_application.append({
                "application": application,
                "independent_panels": int(subset.panel_id.nunique()),
                "transmission_reasons": reason_counts,
                "information_score_transmissions": information,
                "noninitial_nonrecovery_transmissions": denominator,
                "information_score_fraction": fraction,
                "pre_disruption_noninitial_transmission_rate": predisruption_rate,
                "activation_rate_mean": float(subset.trigger_activation_rate.mean()),
                "all_panels_integrated_peer_beliefs": bool(
                    (subset.belief_updates_from_delivered_sketches > 0).all()
                ),
                "eligible": application_eligible,
            })
        values = paired[paired.candidate_name.eq(candidate)]
        candidates.append({
            "candidate_name": candidate,
            "eligible": bool(eligible),
            "worst_application_primary_error": float(
                candidate_frame.groupby("application").primary_distributed_state_error.mean().max()
            ),
            "mean_sketch_byte_reduction": float(values.sketch_byte_reduction.mean()),
            "applications": by_application,
        })
    eligible_candidates = [value for value in candidates if value["eligible"]]
    eligible_candidates.sort(key=lambda value: (
        value["worst_application_primary_error"],
        -value["mean_sketch_byte_reduction"], value["candidate_name"],
    ))
    selected = (
        str(eligible_candidates[0]["candidate_name"])
        if eligible_candidates else None
    )
    report = {
        "stage": stage,
        "mechanism_rule_recorded_before_execution": True,
        "candidates": candidates,
        "selected_generalized_candidate": selected,
        "mechanism_feasible": selected is not None,
        "validation_or_holdout_evidence": False,
    }
    atomic_json(results_root / stage / "hysteresis_repair_selection.json", report)
    return report


def analyze_v8_pilot_no_go(
    results_root: Path, stage: str = "hysteresis_repair_pilot_v3",
    *, bootstrap_replicates: int = 10000,
) -> Dict[str, Any]:
    """Quantify the retained pilot evidence after the prospective stop."""
    frame, _ = _load(results_root, stage)
    paired = _paired_comparisons(frame)
    diagnostic_candidate = "generalized_0115_u8"
    comparator_candidate = "kpi_012_u8"
    diagnostic = paired[paired.candidate_name.eq(diagnostic_candidate)].copy()
    if diagnostic.panel_id.nunique() != 12:
        raise RuntimeError("V8 no-go analysis requires the complete 12-panel pilot")
    interval_rows: List[Dict[str, Any]] = []
    metrics = (
        "message_reduction", "sketch_byte_reduction",
        "fully_counted_byte_reduction", "primary_estimation_error_increase",
        "primary_p95_increase", "detection_delay_increase",
        "service_loss_increase", "harmful_action_increase", "reward_degradation",
    )
    counter = 0
    for application, values in diagnostic.groupby("application"):
        for metric in metrics:
            interval_rows.append({
                "stage": stage,
                "status": "development_pilot_no_go",
                "diagnostic_candidate": diagnostic_candidate,
                "application": application,
                "metric": metric,
                "independent_panels": int(values.panel_id.nunique()),
                **_bootstrap_mean_interval(
                    values[metric].to_numpy(), seed=888100 + counter,
                    replicates=bootstrap_replicates,
                ),
            })
            counter += 1
    write_csv(results_root / "statistics" / "v8_pilot_no_go_intervals.csv", interval_rows)

    feasibility = json.loads(
        (results_root / stage / "hysteresis_repair_selection.json").read_text(
            encoding="utf-8"
        )
    )
    feasibility_rows: List[Dict[str, Any]] = []
    for candidate in feasibility["candidates"]:
        for application in candidate["applications"]:
            feasibility_rows.append({
                "candidate_name": candidate["candidate_name"],
                "application": application["application"],
                "independent_panels": application["independent_panels"],
                "activation_rate_mean": application["activation_rate_mean"],
                "information_score_transmissions": application["information_score_transmissions"],
                "noninitial_nonrecovery_transmissions": application["noninitial_nonrecovery_transmissions"],
                "information_score_fraction": application["information_score_fraction"],
                "pre_disruption_noninitial_transmission_rate": application[
                    "pre_disruption_noninitial_transmission_rate"
                ],
                "information_fraction_gate": 0.05,
                "nominal_transmission_rate_gate": 0.10,
                "eligible": application["eligible"],
            })
    write_csv(results_root / "tables" / "trigger_feasibility.csv", feasibility_rows)

    direct = frame[frame.candidate_name.eq(diagnostic_candidate)].merge(
        frame[frame.candidate_name.eq(comparator_candidate)],
        on=["panel_id", "application"], suffixes=("_generalized", "_kpi"),
    )
    diagnostic_rows: List[Dict[str, Any]] = []
    for application in ("humanitarian", "utility_restoration"):
        values = diagnostic[diagnostic.application.eq(application)]
        comparison = direct[direct.application.eq(application)]
        diagnostic_rows.append({
            "application": application,
            "independent_panels": int(values.panel_id.nunique()),
            "mean_generalized_transmitted_sketch_messages": float(
                values.transmitted_sketch_messages.mean()
            ),
            "mean_always_transmitted_sketch_messages": float(
                values.transmitted_sketch_messages_always.mean()
            ),
            "mean_generalized_actual_wire_bytes": float(values.sketch_on_wire_bytes.mean()),
            "mean_always_actual_wire_bytes": float(values.sketch_on_wire_bytes_always.mean()),
            "mean_message_reduction": float(values.message_reduction.mean()),
            "mean_actual_wire_byte_reduction": float(values.sketch_byte_reduction.mean()),
            "mean_fully_counted_byte_reduction": float(
                values.fully_counted_byte_reduction.mean()
            ),
            "mean_primary_error_increase": float(
                values.primary_estimation_error_increase.mean()
            ),
            "mean_primary_error_advantage_vs_kpi": float(
                comparison.primary_distributed_state_error_kpi.mean()
                - comparison.primary_distributed_state_error_generalized.mean()
            ),
            "mean_service_loss_difference_rule_policy": float(
                values.service_loss_increase.mean()
            ),
            "mean_harmful_action_count_difference_rule_policy": float(
                values.harmful_action_increase.mean()
            ),
            "mean_reward_degradation_rule_policy": float(values.reward_degradation.mean()),
        })
    write_csv(results_root / "tables" / "pilot_no_go_application_summary.csv", diagnostic_rows)
    report = {
        "stage": stage,
        "evidence_status": "development_pilot_only",
        "diagnostic_candidate_not_frozen_primary": diagnostic_candidate,
        "nonentropic_reference_not_frozen_comparator": comparator_candidate,
        "independent_panels": int(diagnostic.panel_id.nunique()),
        "independent_panels_per_application": 6,
        "bootstrap_replicates": int(bootstrap_replicates),
        "trigger_mechanism_feasible": False,
        "formal_development_unlocked": False,
        "multi_seed_training_unlocked": False,
        "validation_unlocked": False,
        "holdout_unlocked": False,
        "stop_reason": (
            "no generalized candidate met the frozen nominal pre-disruption "
            "transmission-rate limit in both applications"
        ),
        "confirmatory_claims_supported": False,
        "application_summaries": diagnostic_rows,
    }
    atomic_json(results_root / "statistics" / "v8_pilot_no_go_summary.json", report)
    atomic_json(results_root / "negative_results" / "v8_stop_decision.json", report)
    return report


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return [float(value) for value in adjusted]


def _write_primary_heterogeneity(
    results_root: Path, stage: str, paired: pd.DataFrame,
    *, bootstrap_replicates: int,
) -> None:
    """Write secondary panel-level scale, topology, and regime intervals."""
    rows: List[Dict[str, Any]] = []
    factors = (
        "complexity_generalized", "topology_family_generalized",
        "fragmentation_generalized", "network_disruption_generalized",
    )
    metrics = (
        "wire_byte_reduction", "primary_error_increase",
        "relative_service_degradation", "reward_degradation",
    )
    counter = 0
    for factor in factors:
        if factor not in paired:
            continue
        for (application, level), values in paired.groupby(["application", factor]):
            for metric in metrics:
                interval = _bootstrap_mean_interval(
                    values[metric].to_numpy(), seed=887000 + counter,
                    replicates=bootstrap_replicates,
                )
                counter += 1
                rows.append({
                    "stage": stage, "application": application,
                    "factor": factor.replace("_generalized", ""),
                    "level": level, "metric": metric,
                    "independent_panels": int(values.panel_id.nunique()),
                    **interval,
                })
    write_csv(results_root / stage / "secondary_heterogeneity_intervals.csv", rows)


def analyze_v8_primary_stage(
    results_root: Path, stage: str, *, bootstrap_replicates: int = 10000,
) -> Dict[str, Any]:
    """Frozen H1--H3 panel analysis for development, validation, or holdout."""
    protocol = json.loads(
        (results_root / "protocol" / "v8_frozen_protocol.json").read_text(encoding="utf-8")
    )
    generalized_name = str(protocol["primary_trigger"])
    comparator_name = str(protocol["strongest_nonentropic_comparator"])
    margins = dict(protocol["primary_margins"])
    frame, _ = _load(results_root, stage)
    if frame.action_policy_id.nunique() != 1:
        raise RuntimeError("primary stage must use one frozen five-seed ensemble")
    generalized = frame[frame.candidate_name.eq(generalized_name)].copy()
    always = frame[frame.candidate_name.eq("always_on_u8")].copy()
    comparator = frame[frame.candidate_name.eq(comparator_name)].copy()
    keys = ["panel_id", "application"]
    paired = generalized.merge(always, on=keys, suffixes=("_generalized", "_always"))
    paired = paired.merge(
        comparator, on=keys, how="inner", suffixes=("", "_comparator"),
    )
    # Comparator columns without a collision retain their original names;
    # explicitly rename the primary fields to make the estimand auditable.
    comparator_lookup = comparator.set_index("panel_id")
    paired["message_reduction"] = 1.0 - (
        paired.transmitted_sketch_messages_generalized
        / paired.transmitted_sketch_messages_always.replace(0, np.nan)
    )
    paired["wire_byte_reduction"] = 1.0 - (
        paired.sketch_on_wire_bytes_generalized
        / paired.sketch_on_wire_bytes_always.replace(0, np.nan)
    )
    paired["fully_counted_byte_reduction"] = 1.0 - (
        paired.fully_counted_bytes_generalized
        / paired.fully_counted_bytes_always.replace(0, np.nan)
    )
    paired["fully_counted_message_reduction"] = 1.0 - (
        paired.fully_counted_messages_generalized
        / paired.fully_counted_messages_always.replace(0, np.nan)
    )
    paired["log_sketch_message_ratio"] = np.log(
        (paired.transmitted_sketch_messages_generalized + 1.0)
        / (paired.transmitted_sketch_messages_always + 1.0)
    )
    paired["log_wire_byte_ratio"] = np.log(
        (paired.sketch_on_wire_bytes_generalized + 1.0)
        / (paired.sketch_on_wire_bytes_always + 1.0)
    )
    paired["primary_error_increase"] = (
        paired.primary_distributed_state_error_generalized
        - paired.primary_distributed_state_error_always
    )
    paired["primary_pointwise_p95_increase"] = (
        paired.primary_distributed_state_error_p95_generalized
        - paired.primary_distributed_state_error_p95_always
    )
    paired["detection_delay_increase"] = (
        paired.mean_detection_delay_steps_generalized
        - paired.mean_detection_delay_steps_always
    )
    paired["relative_service_degradation"] = (
        paired.service_loss_generalized - paired.service_loss_always
    ) / paired.service_loss_always.abs().clip(lower=1e-9)
    paired["absolute_service_loss_difference"] = (
        paired.service_loss_generalized - paired.service_loss_always
    )
    paired["causal_utility_degradation"] = (
        paired.net_causal_utility_always - paired.net_causal_utility_generalized
    )
    generalized_actions = (
        paired.autonomous_beneficial_actions_generalized
        + paired.autonomous_harmful_actions_generalized
        + paired.autonomous_neutral_actions_generalized
    ).clip(lower=1)
    always_actions = (
        paired.autonomous_beneficial_actions_always
        + paired.autonomous_harmful_actions_always
        + paired.autonomous_neutral_actions_always
    ).clip(lower=1)
    paired["harmful_action_rate_degradation"] = (
        paired.autonomous_harmful_actions_generalized / generalized_actions
        - paired.autonomous_harmful_actions_always / always_actions
    )
    paired["harmful_action_count_difference"] = (
        paired.autonomous_harmful_actions_generalized
        - paired.autonomous_harmful_actions_always
    )
    paired["reward_degradation"] = (
        paired.normalized_autonomous_reward_always
        - paired.normalized_autonomous_reward_generalized
    )
    paired["comparator_primary_error"] = [
        float(comparator_lookup.loc[panel, "primary_distributed_state_error"])
        for panel in paired.panel_id
    ]
    paired["comparator_disagreement_error"] = [
        float(comparator_lookup.loc[panel, "disagreement_time_integrated_error"])
        for panel in paired.panel_id
    ]
    paired["comparator_detection_delay"] = [
        float(comparator_lookup.loc[panel, "mean_detection_delay_steps"])
        for panel in paired.panel_id
    ]
    paired["comparator_recovery"] = [
        float(comparator_lookup.loc[panel, "consensus_recovery_steps"])
        for panel in paired.panel_id
    ]
    paired["primary_error_advantage_vs_comparator"] = (
        paired.comparator_primary_error
        - paired.primary_distributed_state_error_generalized
    )
    paired["disagreement_error_advantage_vs_comparator"] = (
        paired.comparator_disagreement_error
        - paired.disagreement_time_integrated_error_generalized
    )
    paired["detection_delay_advantage_vs_comparator"] = (
        paired.comparator_detection_delay
        - paired.mean_detection_delay_steps_generalized
    )
    paired["recovery_advantage_vs_comparator"] = (
        paired.comparator_recovery - paired.consensus_recovery_steps_generalized
    )
    write_csv(results_root / stage / "primary_paired_panels.csv", paired.to_dict("records"))

    interval_rows: List[Dict[str, Any]] = []
    metric_names = (
        "message_reduction", "wire_byte_reduction", "fully_counted_byte_reduction",
        "fully_counted_message_reduction",
        "log_sketch_message_ratio", "log_wire_byte_ratio",
        "primary_error_increase", "detection_delay_increase",
        "primary_pointwise_p95_increase",
        "relative_service_degradation", "absolute_service_loss_difference",
        "causal_utility_degradation", "harmful_action_rate_degradation",
        "harmful_action_count_difference", "reward_degradation",
        "primary_error_advantage_vs_comparator",
        "disagreement_error_advantage_vs_comparator",
        "detection_delay_advantage_vs_comparator",
        "recovery_advantage_vs_comparator",
    )
    seed_counter = 0
    for application, values in paired.groupby("application"):
        for metric in metric_names:
            interval = _bootstrap_mean_interval(
                values[metric].to_numpy(), seed=885000 + seed_counter,
                replicates=bootstrap_replicates,
            )
            seed_counter += 1
            # Paired sign-randomization Monte Carlo p-value around zero.
            rng = np.random.RandomState(886000 + seed_counter)
            array = values[metric].to_numpy(dtype=float)
            signs = rng.choice((-1.0, 1.0), size=(bootstrap_replicates, len(array)))
            null = (signs * array).mean(axis=1)
            p_value = float((np.sum(np.abs(null) >= abs(array.mean())) + 1) / (
                bootstrap_replicates + 1
            ))
            interval_rows.append({
                "stage": stage, "application": application, "metric": metric,
                "independent_panels": int(values.panel_id.nunique()),
                "paired_randomization_p": p_value, **interval,
            })
    adjusted = _holm_adjust([value["paired_randomization_p"] for value in interval_rows])
    for row, value in zip(interval_rows, adjusted):
        row["holm_adjusted_p"] = value
    write_csv(results_root / stage / "primary_bootstrap_intervals.csv", interval_rows)
    _write_primary_heterogeneity(
        results_root, stage, paired,
        bootstrap_replicates=bootstrap_replicates,
    )

    lookup = {
        (value["application"], value["metric"]): value for value in interval_rows
    }
    application_gates: Dict[str, Dict[str, bool]] = {}
    for application in ("humanitarian", "utility_restoration"):
        values = paired[paired.application.eq(application)]
        get = lambda metric: lookup[(application, metric)]
        application_gates[application] = {
            "H1_message_reduction": get("message_reduction")["ci_low"]
            >= margins["H1_message_reduction_lower_95"],
            "H1_wire_byte_reduction": get("wire_byte_reduction")["ci_low"]
            >= margins["H1_wire_byte_reduction_lower_95"],
            "H1_estimation_noninferiority": get("primary_error_increase")["ci_high"]
            <= margins["H1_primary_error_increase_upper_95"],
            "H1_pointwise_p95_noninferiority": get(
                "primary_pointwise_p95_increase"
            )["ci_high"] <= margins["H1_primary_pointwise_p95_increase_upper_95"],
            "H1_detection_delay_noninferiority": get("detection_delay_increase")["ci_high"]
            <= margins["H1_detection_delay_increase_steps_upper_95"],
            "H2_entropy_specific_superiority": (
                get("primary_error_advantage_vs_comparator")["ci_low"] >
                margins["H2_primary_error_advantage_lower_95"]
                and get("primary_error_advantage_vs_comparator")["mean"] >=
                margins["H2_practical_primary_error_advantage"]
            ),
            "H3_service_noninferiority": get("relative_service_degradation")["ci_high"]
            <= margins["H3_relative_service_degradation_upper_95"],
            "H3_harm_noninferiority": get("harmful_action_rate_degradation")["ci_high"]
            <= margins["H3_harmful_action_rate_degradation_upper_95"],
            "H3_reward_noninferiority": get("reward_degradation")["ci_high"]
            <= margins["H3_reward_degradation_upper_95"],
        }
    h1 = all(all(
        value[key] for key in value if key.startswith("H1_")
    ) for value in application_gates.values())
    h2 = all(value["H2_entropy_specific_superiority"] for value in application_gates.values())
    h3 = all(all(
        value[key] for key in value if key.startswith("H3_")
    ) for value in application_gates.values())
    report = {
        "stage": stage,
        "action_policy_id": str(frame.action_policy_id.iloc[0]),
        "independent_panels": int(paired.panel_id.nunique()),
        "independent_panels_per_application": {
            key: int(paired[paired.application.eq(key)].panel_id.nunique())
            for key in ("humanitarian", "utility_restoration")
        },
        "application_gates": application_gates,
        "H1_communication_efficient_estimation_pass": h1,
        "H2_entropy_specific_extension_pass": h2,
        "H3_downstream_policy_retention_pass": h3,
        "progression_pass": bool(h1 and h3),
        "H2_is_not_a_progression_gate": True,
        "bootstrap_replicates": int(bootstrap_replicates),
    }
    atomic_json(results_root / stage / "primary_gate_results.json", report)
    return report


def analyze_v8_seed_stability(
    results_root: Path, stage: str = "seed_stability",
) -> Dict[str, Any]:
    protocol = json.loads(
        (results_root / "protocol" / "v8_frozen_protocol.json").read_text(encoding="utf-8")
    )
    frame, _ = _load(results_root, stage)
    generalized_name = str(protocol["primary_trigger"])
    seeds = sorted(frame.action_policy_id.unique())
    rows: List[Dict[str, Any]] = []
    for policy_id in seeds:
        values = frame[frame.action_policy_id.eq(policy_id)]
        generalized = values[values.candidate_name.eq(generalized_name)]
        always = values[values.candidate_name.eq("always_on_u8")]
        paired = generalized.merge(
            always, on=["panel_id", "application"], suffixes=("_generalized", "_always"),
        )
        relative_service = (
            paired.service_loss_generalized - paired.service_loss_always
        ) / paired.service_loss_always.abs().clip(lower=1e-9)
        reward_degradation = (
            paired.normalized_autonomous_reward_always
            - paired.normalized_autonomous_reward_generalized
        )
        delegation_columns = [
            "policy_delegation_execute_autonomously", "policy_delegation_defer",
            "policy_delegation_abstain", "policy_delegation_escalate_operator",
        ]
        rows.append({
            "action_policy_id": policy_id,
            "completed_arms": len(values),
            "independent_panels": int(values.panel_id.nunique()),
            "scheduler_count": int(values.candidate_name.nunique()),
            "minimum_episode_delegation_diversity": int(
                values.policy_delegation_diversity.min()
            ),
            "aggregate_delegation_diversity": int(sum(
                float(values[column].sum()) > 0.0 for column in delegation_columns
            )),
            "accepted_physical_actions": int(values.accepted_physical_actions_v8.sum()),
            "service_reaching_actions": int(values.service_reaching_actions.sum()),
            "mean_relative_service_degradation": float(relative_service.mean()),
            "mean_reward_degradation": float(reward_degradation.mean()),
            "mean_generalized_harmful_actions": float(
                paired.autonomous_harmful_actions_generalized.mean()
            ),
        })
    write_csv(results_root / stage / "seed_level_stability.csv", rows)
    training = json.loads(
        (results_root / "training" / "training_summary.json").read_text(encoding="utf-8")
    )
    rewards = np.asarray([value["mean_reward_degradation"] for value in rows])
    gates = {
        "all_five_training_seeds_completed": bool(
            training.get("completed_seeds") == 5 and training.get("failed_seeds") == 0
            and len(rows) == 5
        ),
        "all_evaluation_arms_accounted_for": bool(
            rows and all(value["completed_arms"] == 48 for value in rows)
        ),
        "at_least_two_delegation_actions_per_seed": bool(
            rows and all(value["aggregate_delegation_diversity"] >= 2 for value in rows)
        ),
        "accepted_physical_actions_each_seed": bool(
            rows and all(value["accepted_physical_actions"] > 0 for value in rows)
        ),
        "service_reaching_actions_each_seed": bool(
            rows and all(value["service_reaching_actions"] > 0 for value in rows)
        ),
        "bounded_between_seed_reward_effect_sd": bool(
            len(rewards) == 5 and float(rewards.std(ddof=1)) <= 0.03
        ),
        "bounded_absolute_seed_mean_reward_degradation": bool(
            len(rewards) == 5 and bool((np.abs(rewards) <= 0.05).all())
        ),
    }
    gates["multi_seed_stability_pass"] = bool(all(gates.values()))
    report = {
        "stage": stage, "seed_count": len(rows), "seed_rows": rows,
        "gates": gates,
    }
    atomic_json(results_root / stage / "seed_stability_gates.json", report)
    return report


def combine_v8_development_gates(results_root: Path) -> Dict[str, Any]:
    monitoring = json.loads(
        (results_root / "development_final" / "development_selection.json").read_text(encoding="utf-8")
    )
    downstream = json.loads(
        (results_root / "development_agent" / "primary_gate_results.json").read_text(encoding="utf-8")
    )
    stability = json.loads(
        (results_root / "seed_stability" / "seed_stability_gates.json").read_text(encoding="utf-8")
    )
    gates = {
        "monitoring_feasibility": bool(
            monitoring["development_feasibility"]["development_progression_feasible"]
        ),
        "H1_development": bool(downstream["H1_communication_efficient_estimation_pass"]),
        "H3_development": bool(downstream["H3_downstream_policy_retention_pass"]),
        "multi_seed_stability": bool(stability["gates"]["multi_seed_stability_pass"]),
        "H2_entropy_specific_extension": bool(downstream["H2_entropy_specific_extension_pass"]),
    }
    gates["validation_unlocked"] = bool(
        gates["monitoring_feasibility"] and gates["H1_development"]
        and gates["H3_development"] and gates["multi_seed_stability"]
    )
    gates["H2_not_required_for_progression"] = True
    report = {
        "stage": "development_gate_evaluation",
        "gates": gates,
        "validation_unlocked": gates["validation_unlocked"],
    }
    atomic_json(results_root / "development_final" / "combined_progression_gates.json", report)
    return report


def analyze_v8_estimator_calibration(
    results_root: Path, stage: str,
) -> Dict[str, Any]:
    """Descriptive calibration using rows nested within independent panels."""
    frame, _ = _load(results_root, stage)
    run_lookup = frame.set_index("run_id")[["application", "candidate_name", "environment_seed"]]
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted((results_root / "raw" / stage).glob("*")):
        if not run_dir.is_dir() or run_dir.name not in run_lookup.index:
            continue
        path = run_dir / "estimation.csv.gz"
        if not path.exists():
            continue
        metadata = run_lookup.loc[run_dir.name]
        for value in read_csv_gzip(path):
            rows.append({
                "application": metadata["application"],
                "candidate_name": metadata["candidate_name"],
                "environment_seed": int(metadata["environment_seed"]),
                "prediction": float(value["distributed_disrupted_probability"]),
                "outcome": float(str(value["evaluator_disrupted"]).lower() == "true"),
            })
    calibration_rows: List[Dict[str, Any]] = []
    if rows:
        values = pd.DataFrame(rows)
        for (application, candidate), group in values.groupby(["application", "candidate_name"]):
            prediction = group.prediction.to_numpy(dtype=float)
            outcome = group.outcome.to_numpy(dtype=float)
            bins = np.minimum((prediction * 10).astype(int), 9)
            ece = 0.0
            for index in range(10):
                mask = bins == index
                if mask.any():
                    ece += float(mask.mean()) * abs(float(prediction[mask].mean() - outcome[mask].mean()))
            calibration_rows.append({
                "stage": stage, "application": application,
                "candidate_name": candidate,
                "independent_panels": int(group.environment_seed.nunique()),
                "nested_estimation_rows": len(group),
                "brier_score_descriptive": float(np.mean((prediction - outcome) ** 2)),
                "expected_calibration_error_10_bins_descriptive": float(ece),
                "mean_prediction": float(prediction.mean()),
                "observed_disruption_prevalence": float(outcome.mean()),
            })
    write_csv(results_root / stage / "estimator_calibration.csv", calibration_rows)
    report = {
        "stage": stage, "scheduler_application_rows": len(calibration_rows),
        "nested_rows_are_not_independent_units": True,
        "status": "complete" if calibration_rows else "no_unpacked_rows",
    }
    atomic_json(results_root / stage / "estimator_calibration_summary.json", report)
    return report
