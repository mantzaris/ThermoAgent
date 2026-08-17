"""Prospective V7 feasibility gates; no validation or holdout unlocking here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .v5_experiments import atomic_json, write_csv
from .v7_io import read_csv_artifact


def _check(
    rows: List[Dict[str, Any]], gate: str, criterion: str,
    observed: Any, threshold: Any, passed: bool, evidence: str,
) -> None:
    rows.append({
        "gate": gate, "criterion": criterion, "observed": observed,
        "threshold": threshold, "pass": bool(passed), "evidence": evidence,
    })


def evaluate_feasibility_gates(
    repository: Path,
    results_root: Path,
    stage: str = "pilots_iteration3",
) -> Dict[str, Any]:
    config = yaml.safe_load(
        (repository / "configs" / "v7_progression_gates_draft.yaml").read_text(
            encoding="utf-8",
        )
    )
    if not bool(config.get("created_before_iteration3_outcomes")):
        raise RuntimeError("V7 feasibility thresholds are not prospective")
    analysis_path = results_root / stage / "analysis" / "pilot_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError("pilot analysis is incomplete")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    candidates = read_csv_artifact(results_root / stage / "candidate_decisions.csv")
    episodes = read_csv_artifact(results_root / stage / "episode_summary.csv")
    accepted = candidates[
        candidates.counterfactual_evaluated.astype(bool)
        & candidates.counterfactual_action_accepted.astype(bool)
    ].copy()
    accepted["harmful_label"] = accepted.counterfactual_harmful.astype(int)
    replay_path = results_root / "reproducibility" / "replay" / "replay_summary.json"
    replay = (
        json.loads(replay_path.read_text(encoding="utf-8"))
        if replay_path.exists() else {}
    )
    rows: List[Dict[str, Any]] = []
    engineering = config["gate_a_engineering"]
    _check(
        rows, "A", "replay mismatches",
        replay.get("replay_mismatches", "not_run"),
        engineering["maximum_replay_mismatches"],
        replay.get("replay_mismatches") == engineering["maximum_replay_mismatches"],
        "reproducibility/replay/replay_summary.json",
    )
    _check(
        rows, "A", "maximum reconstructed conservation residual",
        replay.get("maximum_conservation_residual", "not_run"),
        engineering["maximum_conservation_residual"],
        float(replay.get("maximum_conservation_residual", np.inf))
        <= float(engineering["maximum_conservation_residual"]),
        "reproducibility/replay/replay_summary.json",
    )
    _check(
        rows, "A", "privacy failures",
        replay.get("privacy_failures", "not_run"),
        engineering["maximum_privacy_failures"],
        replay.get("privacy_failures") == engineering["maximum_privacy_failures"],
        "reproducibility/replay/replay_summary.json",
    )

    validity = config["gate_b_environment_agent_validity"]
    mechanism = config["gate_c_primary_mechanism_feasibility"]
    variation = {
        str(value["application"]): value for value in analysis["variation"]
    }
    for application in ("humanitarian", "utility_restoration"):
        subset = accepted[accepted.application == application]
        episode_subset = episodes[episodes.application == application]
        values = variation.get(application, {})
        prefix = "%s: " % application
        probes = len(subset)
        beneficial = int(subset.counterfactual_beneficial.astype(bool).sum())
        harmful = int(subset.counterfactual_harmful.astype(bool).sum())
        beneficial_fraction = beneficial / max(probes, 1)
        action_types = subset.proposed_operational_action.nunique()
        kpi_auc = float("nan")
        if subset.harmful_label.nunique() > 1:
            kpi_auc = float(roc_auc_score(
                subset.harmful_label, subset.risk_kpi_confidence,
            ))
        entropy_auc = float("nan")
        if subset.harmful_label.nunique() > 1:
            entropy_auc = float(roc_auc_score(
                subset.harmful_label,
                subset.shannon_local + subset.js_disagreement,
            ))
        checks = (
            ("independent panels", int(values.get("independent_panels", 0)), validity["minimum_independent_panels_per_application"], lambda a, b: a >= b),
            ("accepted counterfactual probes", probes, validity["minimum_accepted_counterfactual_probes_per_application"], lambda a, b: a >= b),
            ("beneficial probes", beneficial, validity["minimum_beneficial_probes_per_application"], lambda a, b: a >= b),
            ("harmful probes", harmful, validity["minimum_harmful_probes_per_application"], lambda a, b: a >= b),
            ("beneficial fraction lower bound", beneficial_fraction, validity["minimum_beneficial_fraction"], lambda a, b: a >= b),
            ("beneficial fraction upper bound", beneficial_fraction, validity["maximum_beneficial_fraction"], lambda a, b: a <= b),
            ("physical action types", int(action_types), validity["minimum_physical_action_types_per_application"], lambda a, b: a >= b),
            ("causal chain depth", int(episode_subset.maximum_cascade_depth.max()), validity["minimum_causal_chain_depth"], lambda a, b: a >= b),
            ("cross-community messages", int(episode_subset.cross_community_messages.sum()), validity["minimum_cross_community_messages"], lambda a, b: a >= b),
            ("KPI harm AUC not near-perfect", kpi_auc, validity["maximum_kpi_harm_auc"], lambda a, b: np.isfinite(a) and a <= b),
        )
        for criterion, observed, threshold, comparison in checks:
            _check(
                rows, "B", prefix + criterion, observed, threshold,
                comparison(observed, threshold),
                "%s/candidate_decisions.csv; %s/episode_summary.csv" % (stage, stage),
            )
        mechanism_checks = (
            ("competitive panels", int(values.get("competitive_panels", 0)), mechanism["minimum_competitive_panels_per_application"], lambda a, b: a >= b),
            ("Shannon standard deviation", float(values.get("shannon_std", 0.0)), mechanism["minimum_shannon_standard_deviation"], lambda a, b: a >= b),
            ("JS standard deviation", float(values.get("js_disagreement_std", 0.0)), mechanism["minimum_js_standard_deviation"], lambda a, b: a >= b),
            ("graph disagreement standard deviation", float(values.get("graph_disagreement_std", 0.0)), mechanism["minimum_graph_disagreement_standard_deviation"], lambda a, b: a >= b),
            ("mean distributed contributors", float(values.get("distributed_contributors_mean", 0.0)), mechanism["minimum_mean_distributed_contributors"], lambda a, b: a >= b),
            ("single entropy score AUC not a disguised label", entropy_auc, mechanism["maximum_single_entropy_harm_auc"], lambda a, b: np.isfinite(a) and a <= b),
        )
        for criterion, observed, threshold, comparison in mechanism_checks:
            _check(
                rows, "C", prefix + criterion, observed, threshold,
                comparison(observed, threshold),
                "%s/analysis/mechanism_variation.csv" % stage,
            )
    gate_status = {
        gate: bool(all(value["pass"] for value in rows if value["gate"] == gate))
        for gate in ("A", "B", "C")
    }
    progression = bool(all(gate_status.values()))
    report = {
        "stage": stage,
        "threshold_version": config["version"],
        "gate_status": gate_status,
        "formal_development_unlocked": progression,
        "validation_unlocked": False,
        "holdout_unlocked": False,
        "interpretation": (
            "feasibility gates pass; formal protocol may now be frozen"
            if progression else
            "prospective feasibility gate failed; stop before formal development"
        ),
    }
    destination = results_root / "development" / "gate_feasibility"
    write_csv(destination / "gate_checks.csv", rows)
    atomic_json(destination / "gate_summary.json", report)
    return report
