"""Same-information monitoring and intervention-value analysis for v3."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .human_experiments import _write_dict_csv, utc_now


LOCAL_KPI_FEATURES = (
    "component_backlog",
    "component_unmet",
    "component_congestion",
    "component_lateness",
    "component_commitment",
    "component_safety",
    "local_kpi_risk",
    "local_disruption_risk",
    "actionability_evidence",
)

THERMODYNAMIC_FEATURES = (
    "local_energy_residual",
    "distributed_energy",
    "energy_residual",
    "distributed_entropy",
    "entropy_residual",
    "entropy_slope",
    "entropy_acceleration",
    "flow_entropy",
    "belief_entropy",
    "disagreement",
    "consensus_confidence",
    "free_energy",
    "free_energy_residual",
)


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if labels.sum() else float("nan")


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def _event_data(event_path: Path) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], Dict[str, Any]]:
    states: Dict[Tuple[int, str], Dict[str, Any]] = {}
    disruption: Dict[str, Any] = {}
    topology: Dict[str, Any] = {}
    with gzip.open(event_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event["kind"] == "thermodynamic_state":
                states[(int(event["step"]), str(event["actor"]))] = event["payload"]
            elif event["kind"] == "topology_snapshot":
                topology = event["payload"]
            elif event["kind"] == "disruption" and event["actor"] == "v3_actionability_mechanism":
                disruption = dict(event["payload"])
                # The event ledger is the authoritative onset clock.  Keeping
                # it with the evaluator-only label avoids the period-zero
                # direction artifact that invalidated several v2 trigger
                # interpretations.
                disruption["event_step"] = int(event["step"])
    return states, {"disruption": disruption, "topology": topology}


def build_monitoring_frame(results_root: Path, stage: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode_path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        states, context = _event_data(episode_path.with_name("events.jsonl.gz"))
        disruption = context["disruption"]
        topology = context["topology"]
        roles = {
            agent_id: values["role"]
            for agent_id, values in topology.get("agents", {}).items()
        }
        affected_demands = set(disruption.get("affected_demand_nodes", []))
        disruption_step = int(
            disruption.get("event_step", max(2, len(episode["time_series"]) // 3))
        )
        for (step, agent_id), state in states.items():
            current = episode["time_series"][step]
            future = episode["time_series"][step + 1 : min(step + 5, len(episode["time_series"]))]
            if not future:
                continue
            future_loss = (
                float(np.mean([row["service_loss"] for row in future]))
                if episode["application"] == "commercial"
                else float(np.mean([
                    row["weighted_backlog"] / max(row["cumulative_demand"], 1.0)
                    for row in future
                ]))
            )
            demand_role = roles.get(agent_id) in ("retailer", "clinic", "community")
            eligible = bool(
                step >= disruption_step
                and agent_id in affected_demands
                and demand_role
                and future_loss >= (0.55 if episode["application"] == "commercial" else 0.75)
            )
            components = state["components"]
            row = {
                "run_id": episode["run_id"],
                "application": episode["application"],
                "scenario": episode["scenario"],
                "method": episode["method"],
                "environment_seed": int(episode["environment_seed"]),
                "operator_seed": int(episode["operator_seed"]),
                "step": int(step),
                "agent_id": agent_id,
                "role": roles.get(agent_id, state.get("role", "unknown")),
                "disruption_step": disruption_step,
                "affected_demand": int(agent_id in affected_demands),
                "eligible_intervention": int(eligible),
                "future_loss": future_loss,
                "component_backlog": float(components["backlog"]),
                "component_unmet": float(components["unmet"]),
                "component_congestion": float(components["congestion"]),
                "component_lateness": float(components["lateness"]),
                "component_commitment": float(components["commitment"]),
                "component_safety": float(components["safety"]),
            }
            for name in (
                "local_kpi_risk", "local_disruption_risk", "actionability_evidence",
                *THERMODYNAMIC_FEATURES,
            ):
                # The explicit actionability field was introduced after the
                # first immutable development diagnostic; retain that panel
                # as a zero-valued historical feature rather than rewriting
                # its ledgers.
                row[name] = float(state.get(name, 0.0))
            rows.append(row)
    if not rows:
        raise ValueError("no v3 monitoring episodes found for stage %s" % stage)
    return pd.DataFrame(rows)


def _seed_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = sorted(frame["environment_seed"].unique())
    if len(seeds) < 3:
        raise ValueError("monitoring analysis requires at least three environment seeds")
    test_count = max(1, len(seeds) // 3)
    test_seeds = set(seeds[-test_count:])
    return frame[~frame.environment_seed.isin(test_seeds)].copy(), frame[frame.environment_seed.isin(test_seeds)].copy()


def _fit_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    label_column: str = "eligible_intervention",
) -> np.ndarray:
    labels = train[label_column].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("training split has only one intervention label")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=31991,
        ),
    )
    model.fit(train[list(features)].to_numpy(dtype=float), labels)
    return model.predict_proba(test[list(features)].to_numpy(dtype=float))[:, 1]


def _metrics(
    application: str,
    detector: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    prevalence = float(labels.mean())
    return {
        "application": application,
        "detector": detector,
        "rows": len(labels),
        "positive_rows": int(labels.sum()),
        "prevalence": prevalence,
        "average_precision": _safe_ap(labels, scores),
        "ap_lift_over_prevalence": _safe_ap(labels, scores) / prevalence if prevalence > 0 else None,
        "roc_auc": _safe_auc(labels, scores),
        "brier_score": float(brier_score_loss(labels, np.clip(scores, 0.0, 1.0))),
    }


def _threshold_at_false_alarm(
    train: pd.DataFrame,
    scores: np.ndarray,
    maximum_false_alarm: float = 0.10,
) -> float:
    negative = scores[train["eligible_intervention"].to_numpy(dtype=int) == 0]
    if not len(negative):
        return 1.0
    return float(np.quantile(negative, 1.0 - maximum_false_alarm, method="higher"))


def _timing_rows(
    test: pd.DataFrame,
    score_column: str,
    threshold: float,
    detector: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_id, episode in test.groupby("run_id", sort=True):
        disruption_step = int(episode.disruption_step.iloc[0])
        active = episode[episode[score_column] >= threshold]
        pre = active[active.step < disruption_step]
        post = active[active.step >= disruption_step]
        first_post = int(post.step.min()) if not post.empty else None
        rows.append({
            "run_id": run_id,
            "application": episode.application.iloc[0],
            "scenario": episode.scenario.iloc[0],
            "detector": detector,
            "threshold": threshold,
            "first_post_disruption_activation": first_post,
            "activation_delay": first_post - disruption_step if first_post is not None else None,
            "pre_disruption_false_activation": not pre.empty,
            "detected": first_post is not None,
        })
    return rows


def _localization_rows(test: pd.DataFrame, score_column: str, detector: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    post = test[test.step >= test.disruption_step]
    for (run_id, step), group in post.groupby(["run_id", "step"], sort=True):
        positives = set(group.loc[group.affected_demand == 1, "agent_id"])
        if not positives:
            continue
        ranked = group.sort_values([score_column, "agent_id"], ascending=[False, True]).agent_id.tolist()
        rows.append({
            "run_id": run_id,
            "step": int(step),
            "application": group.application.iloc[0],
            "detector": detector,
            "top1_hit": int(bool(set(ranked[:1]) & positives)),
            "top3_hit": int(bool(set(ranked[:3]) & positives)),
            "positive_agents": ";".join(sorted(positives)),
            "ranked_agents": ";".join(ranked),
        })
    return rows


def analyze_monitoring(
    results_root: Path,
    stage: str = "monitoring_development",
) -> Dict[str, Any]:
    frame = build_monitoring_frame(results_root, stage)
    processed = results_root / "monitoring" / "agent_period_features.csv"
    processed.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed, index=False)
    metric_rows: List[Dict[str, Any]] = []
    incremental_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    localization_rows: List[Dict[str, Any]] = []
    scored_frames: List[pd.DataFrame] = []
    for application, application_frame in frame.groupby("application", sort=True):
        train, test = _seed_split(application_frame)
        train = train.copy()
        test = test.copy()
        kpi_train_scores = _fit_scores(train, train, LOCAL_KPI_FEATURES)
        thermo_train_scores = _fit_scores(
            train, train, LOCAL_KPI_FEATURES + THERMODYNAMIC_FEATURES
        )
        test["local_kpi_logistic"] = _fit_scores(train, test, LOCAL_KPI_FEATURES)
        test["kpi_plus_thermodynamic_logistic"] = _fit_scores(
            train, test, LOCAL_KPI_FEATURES + THERMODYNAMIC_FEATURES
        )
        # Direct interpretable signals are min-max mapped only for Brier
        # comparability; ranking metrics are invariant to this transform.
        direct = {
            "local_kpi_risk": np.clip(test.local_kpi_risk.to_numpy(float), 0.0, 1.0),
            "entropy_anomaly": 1.0 - np.exp(-np.clip(test.entropy_residual.to_numpy(float), 0.0, None)),
            "energy_severity": 1.0 / (1.0 + np.exp(-test.energy_residual.to_numpy(float))),
            "disagreement": np.clip(test.disagreement.to_numpy(float) * 4.0, 0.0, 1.0),
            "free_energy_diagnostic": 1.0 - np.exp(-np.clip(test.free_energy_residual.to_numpy(float), 0.0, None)),
        }
        labels = test.eligible_intervention.to_numpy(dtype=int)
        for detector, scores in {
            **direct,
            "local_kpi_logistic": test.local_kpi_logistic.to_numpy(float),
            "kpi_plus_thermodynamic_logistic": test.kpi_plus_thermodynamic_logistic.to_numpy(float),
        }.items():
            test["score_" + detector] = scores
            metric_rows.append(_metrics(application, detector, labels, scores))
        kpi = next(row for row in metric_rows if row["application"] == application and row["detector"] == "local_kpi_logistic")
        thermo = next(row for row in metric_rows if row["application"] == application and row["detector"] == "kpi_plus_thermodynamic_logistic")
        incremental_rows.append({
            "application": application,
            "comparison": "same_information_local_KPI_plus_thermodynamics_minus_local_KPI",
            "delta_average_precision": thermo["average_precision"] - kpi["average_precision"],
            "delta_roc_auc": thermo["roc_auc"] - kpi["roc_auc"],
            "delta_brier_score": thermo["brier_score"] - kpi["brier_score"],
            "gate_threshold": 0.05,
            "rank_gate_passed": bool(
                thermo["average_precision"] - kpi["average_precision"] >= 0.05
                or thermo["roc_auc"] - kpi["roc_auc"] >= 0.05
            ),
        })
        for detector, training_scores in (
            ("local_kpi_logistic", kpi_train_scores),
            ("kpi_plus_thermodynamic_logistic", thermo_train_scores),
        ):
            threshold = _threshold_at_false_alarm(train, training_scores)
            timing_rows.extend(_timing_rows(
                test,
                "score_" + detector,
                threshold,
                detector,
            ))
            localization_rows.extend(_localization_rows(
                test,
                "score_" + detector,
                detector,
            ))
        test["split"] = "held_out_development_seeds"
        scored_frames.append(test)
    _write_dict_csv(results_root / "monitoring" / "monitoring_baselines.csv", metric_rows)
    _write_dict_csv(results_root / "monitoring" / "incremental_value.csv", incremental_rows)
    _write_dict_csv(results_root / "monitoring" / "detection_lead_time.csv", timing_rows)
    _write_dict_csv(results_root / "monitoring" / "localization.csv", localization_rows)
    pd.concat(scored_frames, ignore_index=True).to_csv(
        results_root / "monitoring" / "held_out_development_predictions.csv",
        index=False,
    )
    summary = {
        "created_at": utc_now(),
        "stage": stage,
        "rows": len(frame),
        "applications": sorted(frame.application.unique()),
        "incremental": incremental_rows,
        "gate_5_rank_passed_both_applications": all(
            row["rank_gate_passed"] for row in incremental_rows
        ),
        "interpretation": (
            "Thermodynamic features are tested only against ordinary KPIs from the same private-local boundary. "
            "The intervention label is evaluator-only and used for development training/evaluation, never actor execution."
        ),
    }
    (results_root / "monitoring" / "README.md").write_text(
        "# V3 monitoring and intervention-value analysis\n\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return summary


CAUSAL_LOCAL_FEATURES = (
    "component_backlog", "component_unmet", "component_congestion",
    "component_lateness", "component_commitment", "component_safety",
    "local_kpi_risk", "local_disruption_risk", "actionability_evidence",
    "role_is_demand", "candidate_is_route", "candidate_is_information",
    "candidate_is_priority",
)


def build_dense_causal_frame(results_root: Path, stage: str) -> pd.DataFrame:
    """Load actor-visible predictors and evaluator-only branch effects."""

    rows: List[Dict[str, Any]] = []
    calibration_path = results_root / "calibration" / "thermodynamic_calibration_n10.json"
    calibration_record = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.is_file() else {"applications": {}}
    )
    for episode_path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        for probe in episode.get("counterfactuals", []):
            if probe.get("probe_type") != "dense_development_candidate":
                continue
            state = probe["features"]
            components = state["components"]
            role = str(probe["role"])
            tool = str(probe["candidate_tool"])
            row: Dict[str, Any] = {
                "run_id": episode["run_id"],
                "application": episode["application"],
                "scenario": episode["scenario"],
                "environment_seed": int(episode["environment_seed"]),
                "step": int(probe["step"]),
                "agent_id": probe["agent_id"],
                "role": role,
                "candidate_tool": tool,
                "candidate_arguments_sha256": probe["candidate_arguments_sha256"],
                "candidate_publicly_redundant": int(probe["candidate_publicly_redundant"]),
                "beneficial_intervention": int(probe["beneficial_intervention"]),
                "intervention_effect": float(probe["intervention_effect"]),
                "role_is_demand": int(role in ("retailer", "clinic", "community")),
                "candidate_is_route": int(tool in ("authorize_emergency_route", "temporary_emergency_override")),
                "candidate_is_information": int(tool == "authorize_information_sharing"),
                "candidate_is_priority": int(tool == "adjust_priorities"),
            }
            for name, value in components.items():
                row["component_" + name] = float(value)
            for name in (
                "local_kpi_risk", "local_disruption_risk",
                "actionability_evidence", *THERMODYNAMIC_FEATURES,
            ):
                row[name] = float(state.get(name, 0.0))
            if "local_energy_residual" not in state:
                application_calibration = calibration_record.get(
                    "applications", {}
                ).get(episode["application"], {})
                role_calibration = application_calibration.get(
                    "by_role", {}
                ).get(role, {})
                center = float(role_calibration.get(
                    "energy_center",
                    application_calibration.get("energy_center", 0.0),
                ))
                scale = max(1e-6, float(role_calibration.get(
                    "energy_scale",
                    application_calibration.get("energy_scale", 1.0),
                )))
                row["local_energy_residual"] = (
                    float(state["energy"]) - center
                ) / scale
            rows.append(row)
    if not rows:
        raise ValueError("no dense causal probes found for stage %s" % stage)
    return pd.DataFrame(rows)


def _budgeted_allocation_rows(
    frame: pd.DataFrame,
    score_column: str,
    detector: str,
    budget: int = 6,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_id, episode in frame.groupby("run_id", sort=True):
        # A repeated request for the same bounded route/scope is one incident.
        # Retain the highest-scored decision epoch without using its outcome.
        unique = (
            episode.sort_values([score_column, "step"], ascending=[False, True])
            .drop_duplicates(
                ["agent_id", "candidate_tool", "candidate_arguments_sha256"],
                keep="first",
            )
            .head(int(budget))
        )
        rows.append({
            "run_id": run_id,
            "application": episode.application.iloc[0],
            "scenario": episode.scenario.iloc[0],
            "detector": detector,
            "fixed_intervention_budget": int(budget),
            "selected_interventions": len(unique),
            "beneficial_selected": int(unique.beneficial_intervention.sum()),
            "realized_counterfactual_utility": float(unique.intervention_effect.sum()),
            "mean_selected_effect": float(unique.intervention_effect.mean()) if len(unique) else 0.0,
        })
    return rows


def analyze_dense_causal_value(
    results_root: Path,
    stage: str = "dense_causal_development_n10_v3",
) -> Dict[str, Any]:
    """Test incremental value against actual paired intervention effects."""

    frame = build_dense_causal_frame(results_root, stage)
    output_frame = results_root / "monitoring" / "dense_causal_features.csv"
    frame.to_csv(output_frame, index=False)
    metric_rows: List[Dict[str, Any]] = []
    incremental_rows: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []
    for application, application_frame in frame.groupby("application", sort=True):
        train, test = _seed_split(application_frame)
        test = test.copy()
        test["score_causal_local_kpi_logistic"] = _fit_scores(
            train, test, CAUSAL_LOCAL_FEATURES,
            label_column="beneficial_intervention",
        )
        test["score_causal_kpi_plus_thermodynamic_logistic"] = _fit_scores(
            train, test, CAUSAL_LOCAL_FEATURES + THERMODYNAMIC_FEATURES,
            label_column="beneficial_intervention",
        )
        labels = test.beneficial_intervention.to_numpy(dtype=int)
        for detector in (
            "causal_local_kpi_logistic",
            "causal_kpi_plus_thermodynamic_logistic",
        ):
            scores = test["score_" + detector].to_numpy(dtype=float)
            metric_rows.append(_metrics(application, detector, labels, scores))
            utility_rows.extend(_budgeted_allocation_rows(
                test, "score_" + detector, detector, budget=6
            ))
        kpi = next(
            row for row in metric_rows
            if row["application"] == application
            and row["detector"] == "causal_local_kpi_logistic"
        )
        thermo = next(
            row for row in metric_rows
            if row["application"] == application
            and row["detector"] == "causal_kpi_plus_thermodynamic_logistic"
        )
        kpi_utility = np.mean([
            row["realized_counterfactual_utility"] for row in utility_rows
            if row["application"] == application
            and row["detector"] == "causal_local_kpi_logistic"
        ])
        thermo_utility = np.mean([
            row["realized_counterfactual_utility"] for row in utility_rows
            if row["application"] == application
            and row["detector"] == "causal_kpi_plus_thermodynamic_logistic"
        ])
        utility_gain = (
            float((thermo_utility - kpi_utility) / max(abs(kpi_utility), 1e-9))
        )
        incremental_rows.append({
            "application": application,
            "comparison": "paired_causal_effect_KPI_plus_thermodynamics_minus_same_information_KPI",
            "test_environment_seeds": ";".join(
                str(value) for value in sorted(test.environment_seed.unique())
            ),
            "positive_prevalence": float(labels.mean()),
            "delta_average_precision": thermo["average_precision"] - kpi["average_precision"],
            "delta_roc_auc": thermo["roc_auc"] - kpi["roc_auc"],
            "delta_brier_score": thermo["brier_score"] - kpi["brier_score"],
            "kpi_mean_budgeted_utility": float(kpi_utility),
            "thermodynamic_mean_budgeted_utility": float(thermo_utility),
            "relative_budgeted_utility_gain": utility_gain,
            "rank_threshold": 0.05,
            "utility_threshold": 0.05,
            "gate_passed": bool(
                thermo["average_precision"] - kpi["average_precision"] >= 0.05
                or thermo["roc_auc"] - kpi["roc_auc"] >= 0.05
                or utility_gain >= 0.05
            ),
        })
        test["split"] = "held_out_development_seeds"
        predictions.append(test)
    _write_dict_csv(results_root / "monitoring" / "causal_monitoring_baselines.csv", metric_rows)
    _write_dict_csv(results_root / "monitoring" / "causal_incremental_value.csv", incremental_rows)
    _write_dict_csv(results_root / "monitoring" / "causal_allocation_utility.csv", utility_rows)
    pd.concat(predictions, ignore_index=True).to_csv(
        results_root / "monitoring" / "dense_causal_predictions.csv", index=False
    )
    summary = {
        "created_at": utc_now(),
        "stage": stage,
        "rows": len(frame),
        "paired_branch_label": "positive six-step loss_without_minus_loss_with",
        "normal_operator_redundancy_rule_applied": True,
        "common_randomness_verified": True,
        "incremental": incremental_rows,
        "gate_5_causal_passed_both_applications": all(
            row["gate_passed"] for row in incremental_rows
        ),
        "allocation_caveat": (
            "Budgeted allocation is a retrospective fixed-budget ranking diagnostic; "
            "episode-level execution remains the primary decision evaluation."
        ),
    }
    (results_root / "monitoring" / "causal_value_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
