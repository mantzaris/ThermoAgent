"""Same-capacity Level-2 risk and communication controllers for V7."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .v7_types import V7RiskContext


METHODS = (
    "always_act",
    "random_abstention",
    "fixed_severity",
    "kpi_confidence",
    "predictive_uncertainty",
    "calibrated_max_probability",
    "action_value_margin",
    "conformal_risk_proxy",
    "shannon_js",
    "generalized_tsallis_gini",
    "graph_disagreement",
    "combined_generalized_entropic",
)


def decision_key(context: V7RiskContext) -> str:
    return "%s|%s" % (
        context.proposal.agent_id, context.proposal.target_asset_or_location,
    )


def risk_score(method: str, context: V7RiskContext) -> float:
    """Higher is riskier; all inputs are deployable except explicit oracles."""
    severity = float(context.local_kpis.get("severity", 0.0))
    safety = float(context.local_kpis.get("safety_risk", 0.0))
    scarcity = float(context.local_kpis.get("resource_scarcity", 0.0))
    delay = min(float(context.local_kpis.get("delay", 0.0)) / 5.0, 1.0)
    if method == "always_act":
        return -1.0
    if method == "random_abstention":
        payload = "%s|%s|%s" % (
            context.proposal.agent_id,
            context.proposal.target_asset_or_location,
            context.step,
        )
        raw = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(raw[:8], "big") / float(2 ** 64 - 1)
    if method == "fixed_severity":
        return float(1.0 - severity)
    if method == "kpi_confidence":
        return float(
            0.34 * context.predictive_uncertainty
            + 0.24 * safety + 0.22 * scarcity + 0.12 * delay
            + 0.08 * (1.0 - context.communication_reliability)
        )
    if method in ("predictive_uncertainty", "calibrated_max_probability"):
        return float(context.predictive_uncertainty)
    if method in ("action_value_margin", "conformal_risk_proxy"):
        return float(1.0 - context.action_value_margin)
    if method == "shannon_js":
        return float(
            0.40 * context.predictive_uncertainty
            + 0.25 * context.shannon_local
            + 0.35 * min(context.js_disagreement / 0.25, 1.0)
        )
    if method == "generalized_tsallis_gini":
        tail = min(context.jt_disagreement_0_5 / 0.20, 1.0)
        concentration = 0.5 * context.tsallis_2_local + 0.5 * context.gini_simpson_local
        return float(
            0.40 * context.predictive_uncertainty
            + 0.30 * tail + 0.30 * concentration
        )
    if method == "graph_disagreement":
        return float(
            0.42 * context.predictive_uncertainty
            + 0.38 * min(context.graph_disagreement / 0.25, 1.0)
            + 0.20 * min(context.consensus_residual, 1.0)
        )
    if method == "combined_generalized_entropic":
        # Same scalar capacity as the KPI baseline: a frozen linear risk score,
        # not a larger nonlinear model. Pilot-only coefficients must be frozen
        # before formal development.
        spectrum_disagreement = np.mean([
            context.js_disagreement,
            context.jt_disagreement_0_5,
            context.jt_disagreement_1_5,
            context.jt_disagreement_2,
        ])
        return float(
            0.28 * context.predictive_uncertainty
            + 0.12 * safety
            + 0.23 * min(float(spectrum_disagreement) / 0.22, 1.0)
            + 0.16 * min(context.graph_disagreement / 0.22, 1.0)
            + 0.13 * min(context.consensus_residual, 1.0)
            + 0.08 * min(max(context.disagreement_slope, 0.0) / 0.10, 1.0)
        )
    raise ValueError("unknown V7 Level-2 controller: %s" % method)


@dataclass
class V7SelectiveController:
    method: str
    autonomous_coverage: float = 0.60
    operator_slots_per_epoch: int = 1
    escalation_threshold: float = 0.82
    communicate_threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError("unknown V7 controller")
        if not 0.0 <= float(self.autonomous_coverage) <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if int(self.operator_slots_per_epoch) < 0:
            raise ValueError("operator slots cannot be negative")

    def __call__(
        self, contexts: Sequence[V7RiskContext], step: int,
    ) -> Mapping[str, str]:
        actionable = [
            context for context in contexts if context.proposal.is_physical
        ]
        decisions: Dict[str, str] = {
            decision_key(context): "execute_autonomously"
            for context in contexts if not context.proposal.is_physical
        }
        scored = sorted(
            ((risk_score(self.method, context), context) for context in actionable),
            key=lambda value: (value[0], decision_key(value[1])),
        )
        execute_count = int(round(self.autonomous_coverage * len(scored)))
        execute_count = min(max(execute_count, 0), len(scored))
        for _, context in scored[:execute_count]:
            decisions[decision_key(context)] = "execute_autonomously"
        escalated = 0
        for score, context in reversed(scored[execute_count:]):
            key = decision_key(context)
            if score >= self.escalation_threshold and escalated < self.operator_slots_per_epoch:
                decisions[key] = "escalate_operator"
                escalated += 1
            elif score >= self.communicate_threshold:
                decisions[key] = "defer"
            else:
                decisions[key] = "abstain"
        return decisions


class V7NeverActController:
    method = "never_act"

    def __call__(
        self, contexts: Sequence[V7RiskContext], step: int,
    ) -> Mapping[str, str]:
        return {
            decision_key(context): (
                "execute_autonomously" if not context.proposal.is_physical else "abstain"
            )
            for context in contexts
        }
