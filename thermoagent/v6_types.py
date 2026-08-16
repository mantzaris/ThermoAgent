"""Typed V6 state for generalized-entropic selective autonomy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


APPLICATIONS = ("commercial", "humanitarian", "utility_restoration")
PRIMARY_APPLICATIONS = ("humanitarian", "utility_restoration")
INFORMATION_CONDITIONS = ("private_fragmented", "public_shared")
REGIMES = (
    "nominal", "isolated_physical", "telemetry_integrity", "partition",
    "correlated", "compound", "ood",
)
SKETCH_POLICIES = ("none", "periodic", "event_triggered", "always_on")

INCIDENT_MODES = (
    "observation_ambiguity",
    "resource_shortage",
    "route_or_service_failure",
    "physical_equipment_failure",
    "unsafe_or_compromised_component",
    "commitment_deadlock",
    # A genuine no-incident state is required for prospective false-alert and
    # pre-disruption timing tests.  It is appended so the semantics of the six
    # V5-informed incident indices remain stable.
    "nominal",
)

OPERATIONAL_ACTIONS = (
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
    "observation_ambiguity": "verify",
    "resource_shortage": "authorize_emergency_resource",
    "route_or_service_failure": "reroute_or_reconfigure",
    "physical_equipment_failure": "deploy_repair_capacity",
    "unsafe_or_compromised_component": "isolate_or_quarantine",
    "commitment_deadlock": "revise_commitment",
    "nominal": "no_action",
}

SECONDARY_ACTION_FOR_MODE = {
    "observation_ambiguity": "request_peer_evidence",
    "resource_shortage": "deploy_repair_capacity",
    "route_or_service_failure": "deploy_repair_capacity",
    "physical_equipment_failure": "authorize_emergency_resource",
    "unsafe_or_compromised_component": "verify",
    "commitment_deadlock": "request_peer_evidence",
    "nominal": "defer",
}

DELEGATION_ACTIONS = (
    "execute_autonomously", "communicate", "request_evidence",
    "defer", "abstain", "escalate_operator",
)


@dataclass(frozen=True)
class V6Identity:
    agent_id: str
    application: str
    role: str
    incident_scope: Tuple[str, ...]
    authority: Tuple[str, ...]


@dataclass(frozen=True)
class V6Utility:
    service_weight: float
    safety_weight: float
    cost_weight: float
    delay_weight: float
    disclosure_cost: float
    risk_tolerance: float


@dataclass(frozen=True)
class V6PrivateObservation:
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
class V6Commitment:
    commitment_id: str
    proposer: str
    recipient: str
    incident_id: str
    action: str
    quantity: float
    status: str = "proposed"
    revision: int = 0


@dataclass
class V6Incident:
    incident_id: str
    application: str
    regime: str
    scenario_family: str
    topology_family: str
    true_mode: str
    correct_action: str
    secondary_action: str
    severity: float
    priority: float
    fragmentation: float
    telemetry_integrity: float
    disruption_step: int
    service_deficit: float = 0.0
    backlog: float = 0.0
    cumulative_loss: float = 0.0
    resolved_fraction: float = 0.0
    last_effect: float = 0.0


@dataclass(frozen=True)
class V6InformationState:
    q: float
    average_local_uncertainty: float
    pooled_uncertainty: float
    generalized_disagreement: float
    consensus: float
    graph_disagreement: float
    consensus_residual: float
    entropy_slope: float
    disagreement_slope: float
    consensus_slope: float
    contributors: Tuple[str, ...]
    missing_agents: Tuple[str, ...]
    sketch_messages: int
    sketch_bytes: int
    sketch_latency: float

    def deployable(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V6ActionProposal:
    agent_id: str
    role: str
    incident_id: str
    action: str
    quantity: float
    action_probability: float
    action_value: float
    value_margin: float
    reason_code: str


@dataclass(frozen=True)
class V6DecisionContext:
    step: int
    proposal: V6ActionProposal
    local_kpis: Dict[str, float]
    operational_energy: float
    effective_temperature: float
    free_energy_diagnostic: float
    shannon_local: float
    tsallis_0_5_local: float
    tsallis_1_5_local: float
    tsallis_2_local: float
    tsallis_3_local: float
    gini_simpson_local: float
    average_local_uncertainty: float
    pooled_uncertainty: float
    js_disagreement: float
    jt_disagreement_0_5: float
    jt_disagreement_1_5: float
    jt_disagreement_2: float
    jt_disagreement_3: float
    graph_disagreement: float
    consensus: float
    consensus_residual: float
    entropy_slope: float
    entropy_acceleration: float
    entropy_ewma: float
    entropy_time_above: int
    disagreement_slope: float
    disagreement_acceleration: float
    disagreement_ewma: float
    disagreement_time_above: int
    consensus_slope: float
    consensus_ewma: float
    communication_reliability: float
    contributors: Tuple[str, ...]
    missing_agents: Tuple[str, ...]

    def deployable(self) -> Dict[str, Any]:
        value = asdict(self)
        value["proposal"] = asdict(self.proposal)
        return value


@dataclass(frozen=True)
class V6ActionResult:
    accepted_typed_action: bool
    accepted_physical_action: bool
    action: str
    incident_id: str
    scheduled_step: Optional[int]
    completed_step: Optional[int]
    causal_effect: float
    harmful: bool
    beneficial: bool
    changed_commitment: bool
    reached_next_stage: bool
    reached_service: bool
    validation_code: str
    resource_key: Optional[str]
    resource_quantity: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceAccount:
    initial: float
    remaining: float
    consumed: float = 0.0
    transferred: float = 0.0
    losses: float = 0.0

    def residual(self) -> float:
        return float(self.initial - self.remaining - self.consumed - self.transferred - self.losses)


@dataclass(frozen=True)
class V6ToolCall:
    action: str
    incident_id: str
    quantity: float = 1.0
    target_agent: Optional[str] = None
    reason_code: str = "local_authorized_evidence"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
