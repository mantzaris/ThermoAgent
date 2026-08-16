"""Deployable selective-autonomy controllers for matched risk evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .v6_types import V6DecisionContext


CONTROLLER_METHODS = (
    "always_act",
    "random_abstention",
    "fixed_severity",
    "kpi_confidence",
    "action_value_margin",
    "calibrated_max_probability",
    "predictive_action_entropy",
    "ensemble_variance_proxy",
    "conformal_margin_proxy",
    "local_shannon",
    "pooled_shannon",
    "jensen_shannon",
    "gini_simpson",
    "tsallis_q_0_5",
    "tsallis_q_1_5",
    "tsallis_q_2",
    "tsallis_q_3",
    "jensen_tsallis_q_0_5",
    "jensen_tsallis_q_1_5",
    "jensen_tsallis_q_2",
    "jensen_tsallis_q_3",
    "graph_disagreement",
    "combined_generalized_entropic",
    "oracle_risk",
)


def binary_entropy(probability: float) -> float:
    value = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    return float(-(value * np.log(value) + (1.0 - value) * np.log(1.0 - value)) / np.log(2.0))


def risk_score(method: str, context: V6DecisionContext) -> float:
    """Return larger values for less reliable autonomous proposals.

    Except for the explicitly labeled oracle, every input is contained in the
    deployable context. The oracle is supplied separately by experiments and
    is rejected here to prevent accidental evaluator leakage.
    """
    if method == "fixed_severity":
        return float(1.0 - context.local_kpis["visible_severity"])
    if method == "kpi_confidence":
        return float(
            0.45 * (1.0 - context.proposal.action_probability)
            + 0.25 * context.local_kpis["safety_risk"]
            + 0.20 * context.local_kpis["resource_scarcity"]
            + 0.10 * context.local_kpis["visible_delay"]
        )
    if method in ("action_value_margin", "conformal_margin_proxy"):
        return float(1.0 - context.proposal.value_margin)
    if method == "calibrated_max_probability":
        return float(1.0 - context.proposal.action_probability)
    if method == "predictive_action_entropy":
        return binary_entropy(context.proposal.action_probability)
    if method == "ensemble_variance_proxy":
        return float(
            abs(context.local_kpis["visible_severity"] - context.proposal.action_probability)
            + 0.35 * context.local_kpis["visible_delay"]
        )
    if method == "local_shannon":
        return float(context.shannon_local)
    if method == "pooled_shannon":
        return float(context.pooled_uncertainty)
    if method == "jensen_shannon":
        return float(context.js_disagreement)
    if method == "gini_simpson":
        return float(context.gini_simpson_local)
    if method == "tsallis_q_0_5":
        return float(context.tsallis_0_5_local)
    if method == "tsallis_q_1_5":
        return float(context.tsallis_1_5_local)
    if method == "tsallis_q_2":
        return float(context.tsallis_2_local)
    if method == "tsallis_q_3":
        return float(context.tsallis_3_local)
    if method == "jensen_tsallis_q_0_5":
        return float(context.jt_disagreement_0_5)
    if method == "jensen_tsallis_q_1_5":
        return float(context.jt_disagreement_1_5)
    if method == "jensen_tsallis_q_2":
        return float(context.jt_disagreement_2)
    if method == "jensen_tsallis_q_3":
        return float(context.jt_disagreement_3)
    if method == "graph_disagreement":
        return float(context.graph_disagreement + 0.35 * context.consensus_residual)
    if method == "combined_generalized_entropic":
        # Prespecified spectrum summary: tail-sensitive disagreement, ordinary
        # JS, graph disagreement, staleness, and action margin. No true state or
        # counterfactual outcome enters this score.
        scaled_js = min(context.js_disagreement / 0.22, 1.0)
        scaled_tail = min(context.jt_disagreement_0_5 / 0.16, 1.0)
        scaled_graph = min(context.graph_disagreement / 0.22, 1.0)
        return float(
            0.26 * (1.0 - context.proposal.value_margin)
            + 0.10 * context.shannon_local
            + 0.25 * scaled_js
            + 0.18 * scaled_tail
            + 0.10 * scaled_graph
            + 0.11 * min(context.consensus_residual, 1.0)
        )
    if method == "random_abstention":
        digest = hashlib.sha256(
            (context.proposal.incident_id + "|" + str(context.step)).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") / float(2 ** 64 - 1)
    if method == "always_act":
        return -1.0
    if method == "oracle_risk":
        raise ValueError("oracle risk requires evaluator-only injection")
    raise ValueError("unknown V6 risk controller: %s" % method)


@dataclass
class SelectiveController:
    method: str
    autonomous_coverage: float
    escalation_slots_per_epoch: int = 1
    escalation_risk_threshold: float = 0.80
    request_evidence_when_unserved: bool = False

    def __post_init__(self) -> None:
        if self.method not in CONTROLLER_METHODS or self.method == "oracle_risk":
            raise ValueError("unknown or evaluator-only controller")
        if not 0.0 <= float(self.autonomous_coverage) <= 1.0:
            raise ValueError("coverage must be in [0, 1]")

    def __call__(
        self, contexts: Sequence[V6DecisionContext], step: int,
    ) -> Mapping[str, str]:
        if self.method == "always_act":
            return {value.proposal.incident_id: "execute_autonomously" for value in contexts}
        decisions: Dict[str, str] = {
            value.proposal.incident_id: "abstain"
            for value in contexts if value.proposal.action == "no_action"
        }
        actionable = [
            value for value in contexts if value.proposal.action != "no_action"
        ]
        scored = sorted(
            ((risk_score(self.method, value), value) for value in actionable),
            key=lambda item: (item[0], item[1].proposal.incident_id),
        )
        autonomous_count = int(round(float(self.autonomous_coverage) * len(scored)))
        autonomous_count = max(0, min(len(scored), autonomous_count))
        for _, context in scored[:autonomous_count]:
            decisions[context.proposal.incident_id] = "execute_autonomously"
        unserved = list(reversed(scored[autonomous_count:]))
        escalated = 0
        for risk, context in unserved:
            if (
                escalated < max(0, int(self.escalation_slots_per_epoch))
                and float(risk) >= float(self.escalation_risk_threshold)
            ):
                decisions[context.proposal.incident_id] = "escalate_operator"
                escalated += 1
                continue
            decisions[context.proposal.incident_id] = (
                "request_evidence" if self.request_evidence_when_unserved else "abstain"
            )
        return decisions


class NeverActController:
    def __call__(self, contexts: Sequence[V6DecisionContext], step: int) -> Mapping[str, str]:
        return {value.proposal.incident_id: "abstain" for value in contexts}
