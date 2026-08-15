"""Typed state for the prospective ThermoHITL v5 study.

The thermodynamic quantities are operational summaries of probability and
service state. They are not literal physical thermodynamics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


APPLICATIONS = ("commercial", "humanitarian", "utility_restoration")
INFORMATION_CONDITIONS = ("private_fragmented", "public_shared")
REGIMES = (
    "nominal", "isolated_physical", "telemetry_integrity", "partition",
    "correlated", "compound", "ood",
)
SKETCH_POLICIES = ("none", "periodic", "event_triggered", "always_on")

INCIDENT_MODES = (
    "evidence_conflict",
    "resource_shortage",
    "route_or_service_failure",
    "unsafe_or_compromised_component",
    "commitment_deadlock",
)

OPERATOR_ACTIONS = (
    "verify",
    "request_peer_evidence",
    "authorize_emergency_resource",
    "reroute_or_reconfigure",
    "deploy_repair_capacity",
    "isolate_or_quarantine",
    "revise_commitment",
    "defer",
    "no_action",
)

PRIMARY_ACTION_FOR_MODE = {
    "evidence_conflict": "verify",
    "resource_shortage": "authorize_emergency_resource",
    "route_or_service_failure": "reroute_or_reconfigure",
    "unsafe_or_compromised_component": "isolate_or_quarantine",
    "commitment_deadlock": "revise_commitment",
}

SECONDARY_ACTION_FOR_MODE = {
    "evidence_conflict": "request_peer_evidence",
    "resource_shortage": "deploy_repair_capacity",
    "route_or_service_failure": "deploy_repair_capacity",
    "unsafe_or_compromised_component": "verify",
    "commitment_deadlock": "request_peer_evidence",
}


def normalized_entropy(values: List[float]) -> float:
    probabilities = np.asarray(values, dtype=float)
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(probabilities)))


def jensen_shannon(values: List[List[float]]) -> float:
    matrix = np.asarray(values, dtype=float)
    matrix = np.maximum(matrix, 1e-12)
    matrix /= matrix.sum(axis=1, keepdims=True)
    center = matrix.mean(axis=0)
    divergences = np.sum(matrix * np.log(matrix / center), axis=1)
    return float(np.mean(divergences) / math.log(2.0))


@dataclass(frozen=True)
class V5Identity:
    agent_id: str
    application: str
    role: str
    incident_scope: Tuple[str, ...]
    authority: Tuple[str, ...]


@dataclass(frozen=True)
class V5PrivateObservation:
    step: int
    incident_id: str
    visible_severity: float
    visible_backlog: float
    visible_delay: float
    resource_scarcity: float
    safety_risk: float
    commitment_strain: float
    telemetry_confidence: float
    communication_reliability: float
    private_evidence: Tuple[float, ...]
    private_inventory: float
    private_cost: float
    private_priority: float

    def kpis(self) -> Dict[str, float]:
        return {
            "visible_severity": self.visible_severity,
            "visible_backlog": self.visible_backlog,
            "visible_delay": self.visible_delay,
            "resource_scarcity": self.resource_scarcity,
            "safety_risk": self.safety_risk,
            "commitment_strain": self.commitment_strain,
        }


@dataclass
class V5Commitment:
    commitment_id: str
    proposer: str
    recipient: str
    incident_id: str
    action: str
    quantity: float
    status: str = "proposed"
    revision: int = 0


@dataclass(frozen=True)
class V5Incident:
    incident_id: str
    scenario_family: str
    topology_family: str
    true_mode: str
    correct_action: str
    secondary_action: str
    severity: float
    priority: float
    fragmentation: float
    telemetry_integrity: float
    base_loss: float
    disruption_step: int
    location: Tuple[float, float]


@dataclass(frozen=True)
class V5ThermodynamicState:
    operational_energy: float
    mean_belief_entropy: float
    entropy_dispersion: float
    js_disagreement: float
    distributed_entropy: float
    entropy_slope: float
    consensus_residual: float
    consensus_confidence: float
    effective_temperature: float
    free_energy: float
    sketch_messages: int
    sketch_bytes: int
    sketch_latency: float
    contributors: Tuple[str, ...]

    def deployable_features(self) -> Dict[str, float]:
        value = asdict(self)
        for key in ("sketch_messages", "sketch_bytes", "sketch_latency", "contributors"):
            value.pop(key, None)
        return {str(key): float(item) for key, item in value.items()}


@dataclass(frozen=True)
class V5ActionEffect:
    incident_id: str
    action: str
    loss_without: float
    loss_with: float
    causal_effect: float
    intervention_cost: float
    operator_minutes: float
    delay_steps: int
    beneficial: bool
    harmful: bool
    changed_commitment: bool
    accepted_action: bool
    reached_next_stage: bool
    reached_service: bool
    stochastic_tape_digest: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
