"""Development-only learnability ceilings and nontrivial-policy checks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .v5_experiments import atomic_json, write_csv
from .v6_environment import PHYSICAL_ACTIONS, V6PanelEnvironment
from .v6_types import INCIDENT_MODES, OPERATIONAL_ACTIONS, V6ActionProposal, V6DecisionContext


class ActionValueCollector:
    def __init__(self, environment: V6PanelEnvironment) -> None:
        self.environment = environment
        self.rows: List[Dict[str, Any]] = []

    def __call__(
        self, contexts: Sequence[V6DecisionContext], step: int,
    ) -> Mapping[str, str]:
        for context in contexts:
            agent = self.environment.agents[context.proposal.agent_id]
            belief = agent.private_beliefs[context.proposal.incident_id]
            allowed = [
                action for action in self.environment.registry.allowed_actions(context.proposal.role)
                if action in PHYSICAL_ACTIONS
            ]
            allowed.append("no_action")
            for action in allowed:
                proposal = V6ActionProposal(**{
                    **asdict(context.proposal),
                    "action": action,
                    "reason_code": "development_action_value_candidate",
                })
                effect = 0.0 if action == "no_action" else self.environment.preview_direct_effect(proposal, step)
                self.rows.append({
                    "cluster_id": self.environment.cluster_id,
                    "application": self.environment.application,
                    "regime": self.environment.regime,
                    "information_condition": self.environment.information_condition,
                    "environment_seed": self.environment.seed,
                    "split_family": self.environment.split_family,
                    "topology_family": self.environment.topology_family,
                    "scenario_family": self.environment.scenario_family,
                    "decision_id": "%s|%d|%s|%s" % (
                        self.environment.cluster_id, step,
                        context.proposal.incident_id, context.proposal.agent_id,
                    ),
                    "step": step,
                    "incident_id": context.proposal.incident_id,
                    "agent_id": context.proposal.agent_id,
                    "role": context.proposal.role,
                    "candidate_action": action,
                    **context.local_kpis,
                    "action_probability": context.proposal.action_probability,
                    "action_value": context.proposal.action_value,
                    "value_margin": context.proposal.value_margin,
                    **{"belief_%d" % index: float(value) for index, value in enumerate(belief)},
                    "causal_effect": float(effect),
                    "harmful": bool(effect < -1e-9),
                    "beneficial": bool(effect > 1e-9),
                })
        return {context.proposal.incident_id: "abstain" for context in contexts}


def generate_action_value_dataset(
    seeds: Sequence[int],
    applications: Sequence[str] = (
        "commercial", "humanitarian", "utility_restoration",
    ),
    regimes: Sequence[str] = (
        "isolated_physical", "telemetry_integrity", "partition",
        "correlated", "compound", "ood",
    ),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for application in applications:
        for regime in regimes:
            for seed in seeds:
                environment = V6PanelEnvironment(
                    application, regime, "private_fragmented", int(seed),
                    "event_triggered",
                )
                collector = ActionValueCollector(environment)
                environment.run(collector, "action_value_collector")
                rows.extend(collector.rows)
    return pd.DataFrame(rows)


def _encoded_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "visible_severity", "visible_backlog", "visible_delay",
        "resource_scarcity", "safety_risk", "commitment_strain",
        "action_probability", "action_value", "value_margin",
        *["belief_%d" % value for value in range(len(INCIDENT_MODES))],
    ]
    categories = pd.get_dummies(
        frame[["candidate_action", "role", "regime"]], dtype=float,
    )
    return pd.concat([
        frame[numeric].reset_index(drop=True), categories.reset_index(drop=True),
    ], axis=1)


def evaluate_action_value_learnability(frame: pd.DataFrame) -> Dict[str, Any]:
    groups = frame["split_family"].to_numpy()
    if len(np.unique(groups)) < 5:
        raise ValueError("learnability evaluation requires five split families")
    features = _encoded_features(frame)
    target = frame["causal_effect"].to_numpy(dtype=float)
    predictions = np.zeros(len(frame), dtype=float)
    splitter = GroupKFold(n_splits=5)
    fold_rows: List[Dict[str, Any]] = []
    for fold, (train, test) in enumerate(splitter.split(features, target, groups), start=1):
        model = RandomForestRegressor(
            n_estimators=160, min_samples_leaf=8, max_features=0.75,
            random_state=66900 + fold, n_jobs=-1,
        )
        model.fit(features.iloc[train], target[train])
        predictions[test] = model.predict(features.iloc[test])
        train_frame = frame.iloc[train]
        test_frame = frame.iloc[test]
        fold_rows.append({
            "fold": fold,
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "environment_seed_disjoint": set(train_frame.environment_seed).isdisjoint(set(test_frame.environment_seed)),
            "topology_family_disjoint": set(train_frame.topology_family).isdisjoint(set(test_frame.topology_family)),
            "scenario_family_disjoint": set(train_frame.scenario_family).isdisjoint(set(test_frame.scenario_family)),
        })
    scored = frame.copy()
    scored["predicted_effect"] = predictions
    selected_rows: List[Dict[str, Any]] = []
    for decision_id, group in scored.groupby("decision_id", sort=True):
        predicted = group.sort_values(
            ["predicted_effect", "candidate_action"],
            ascending=[False, True], kind="mergesort",
        ).iloc[0]
        oracle = group.sort_values(
            ["causal_effect", "candidate_action"],
            ascending=[False, True], kind="mergesort",
        ).iloc[0]
        selected_rows.append({
            "decision_id": decision_id,
            "cluster_id": predicted.cluster_id,
            "application": predicted.application,
            "regime": predicted.regime,
            "environment_seed": int(predicted.environment_seed),
            "predicted_action": predicted.candidate_action,
            "predicted_action_effect": float(predicted.causal_effect),
            "oracle_action": oracle.candidate_action,
            "oracle_effect": float(oracle.causal_effect),
            "regret": float(oracle.causal_effect - predicted.causal_effect),
        })
    selections = pd.DataFrame(selected_rows)
    application_rows: List[Dict[str, Any]] = []
    for application, subset in selections.groupby("application", sort=True):
        application_rows.append({
            "application": application,
            "decisions": int(len(subset)),
            "supervised_mean_action_utility": float(subset.predicted_action_effect.mean()),
            "authorized_oracle_mean_action_utility": float(subset.oracle_effect.mean()),
            "gain_over_always_no_action": float(subset.predicted_action_effect.mean()),
            "mean_regret": float(subset.regret.mean()),
            "selected_action_diversity": int(subset.predicted_action.nunique()),
        })
    return {
        "rows": int(len(frame)),
        "independent_panels": int(frame.cluster_id.nunique()),
        "folds": fold_rows,
        "applications": application_rows,
        "scored": scored,
        "selections": selections,
    }


def run_learnability_diagnostics(
    results_root: Path,
    seeds: Sequence[int] = tuple(range(66101, 66131)),
) -> Dict[str, Any]:
    output = results_root / "development" / "learnability"
    frame = generate_action_value_dataset(seeds)
    result = evaluate_action_value_learnability(frame)
    write_csv(output / "action_value_candidates.csv", frame.to_dict("records"))
    write_csv(output / "crossfit_action_value_predictions.csv", result.pop("scored").to_dict("records"))
    write_csv(output / "crossfit_action_selections.csv", result.pop("selections").to_dict("records"))
    write_csv(output / "grouped_fold_audit.csv", result["folds"])
    write_csv(output / "application_summary.csv", result["applications"])
    atomic_json(output / "learnability_summary.json", result)
    return result
