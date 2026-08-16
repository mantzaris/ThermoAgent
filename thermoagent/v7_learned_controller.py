"""Cross-fitted same-capacity Level-2 risk controllers for V7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .v7_formal_analysis import CATEGORICAL
from .v7_policies import decision_key
from .v7_types import V7RiskContext


def context_row(context: V7RiskContext, topology_family: str) -> Dict[str, Any]:
    """Convert only deployable context fields to the frozen analysis schema."""
    return {
        "severity": float(context.local_kpis.get("severity", 0.0)),
        "safety_risk": float(context.local_kpis.get("safety_risk", 0.0)),
        "resource_scarcity": float(context.local_kpis.get("resource_scarcity", 0.0)),
        "delay": float(context.local_kpis.get("delay", 0.0)),
        "predictive_uncertainty": context.predictive_uncertainty,
        "action_probability": context.proposal.action_probability,
        "action_value": context.proposal.action_value,
        "action_value_margin": context.proposal.value_margin,
        "communication_reliability": context.communication_reliability,
        "coupling_numeric": context.coupling_strength,
        "fragmentation_numeric": context.fragmentation,
        "size_normalized": context.size_normalized,
        "shannon_local": context.shannon_local,
        "pooled_uncertainty": context.pooled_uncertainty,
        "js_disagreement": context.js_disagreement,
        "jt_disagreement_0_5": context.jt_disagreement_0_5,
        "jt_disagreement_1_5": context.jt_disagreement_1_5,
        "jt_disagreement_2": context.jt_disagreement_2,
        "jt_disagreement_3": context.jt_disagreement_3,
        "graph_disagreement": context.graph_disagreement,
        "consensus": context.consensus,
        "consensus_residual": context.consensus_residual,
        "entropy_slope": context.entropy_slope,
        "disagreement_slope": context.disagreement_slope,
        "proposed_operational_action": context.proposal.proposed_operational_action,
        "topology_family": str(topology_family),
    }


@dataclass
class FittedRiskController:
    model: Any
    method: str
    topology_family: str
    autonomous_coverage: float = 0.60
    operator_slots_per_epoch: int = 1
    escalation_threshold: float = 0.82
    communicate_threshold: float = 0.65

    def __call__(
        self, contexts: Sequence[V7RiskContext], step: int,
    ) -> Mapping[str, str]:
        actionable = [value for value in contexts if value.proposal.is_physical]
        output: Dict[str, str] = {
            decision_key(value): "execute_autonomously"
            for value in contexts if not value.proposal.is_physical
        }
        if not actionable:
            return output
        frame = pd.DataFrame([
            context_row(value, self.topology_family) for value in actionable
        ])
        probabilities = np.asarray(self.model.predict_proba(frame))[:, 1]
        ordered = sorted(
            zip(probabilities, actionable),
            key=lambda value: (float(value[0]), decision_key(value[1])),
        )
        execute_count = int(round(float(self.autonomous_coverage) * len(ordered)))
        execute_count = min(max(execute_count, 0), len(ordered))
        for _, context in ordered[:execute_count]:
            output[decision_key(context)] = "execute_autonomously"
        escalations = 0
        for score, context in reversed(ordered[execute_count:]):
            key = decision_key(context)
            if float(score) >= self.escalation_threshold and escalations < self.operator_slots_per_epoch:
                output[key] = "escalate_operator"
                escalations += 1
            elif float(score) >= self.communicate_threshold:
                output[key] = "defer"
            else:
                output[key] = "abstain"
        return output
