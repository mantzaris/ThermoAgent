"""Typed state for the V7 coupled-network study.

The action schema deliberately separates physical operations, information
gathering, communication, and oversight delegation. Information requests are
never counted as physical service actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


PRIMARY_APPLICATIONS = ("humanitarian", "utility_restoration")
COMPLEXITY_LEVELS = ("small", "medium", "large")
INFORMATION_CONDITIONS = ("private_fragmented", "public_shared")
COUPLING_LEVELS = ("low", "medium", "high")
FRAGMENTATION_LEVELS = ("low", "medium", "high")
NETWORK_DISRUPTION_LEVELS = ("low", "medium", "high")
SKETCH_POLICIES = ("none", "periodic", "event_triggered", "always_on")
OPERATIONAL_COMMUNICATION_POLICIES = (
    "none", "periodic", "always_on", "kpi_event_triggered",
    "agent_event_triggered",
)

INFORMATION_ACTIONS = (
    "no_information_action",
    "request_peer_evidence",
    "verify_observation",
)
COMMUNICATION_ACTIONS = (
    "no_communication_action",
    "send_targeted_summary",
    "broadcast_alert",
    "negotiate_commitment",
)
DELEGATION_ACTIONS = (
    "execute_autonomously",
    "defer",
    "abstain",
    "escalate_operator",
)

HUMANITARIAN_ACTIONS = (
    "allocate_shipment",
    "redirect_vehicle",
    "release_emergency_reserve",
    "revise_delivery_priority",
    "cancel_risky_dispatch",
    "no_operational_action",
)
UTILITY_ACTIONS = (
    "dispatch_repair_crew",
    "allocate_spare_component",
    "reconfigure_service_edge",
    "isolate_component",
    "deploy_mobile_generation",
    "restore_communication_relay",
    "no_operational_action",
)


@dataclass(frozen=True)
class ComplexitySpec:
    level: str
    agent_count: int
    operational_nodes: int
    horizon: int
    decision_interval: int
    concurrent_disruptions: int

    @property
    def decision_steps(self) -> Tuple[int, ...]:
        return tuple(range(0, self.horizon, self.decision_interval))


DEFAULT_COMPLEXITY: Dict[str, ComplexitySpec] = {
    "small": ComplexitySpec("small", 12, 8, 30, 3, 2),
    "medium": ComplexitySpec("medium", 28, 16, 60, 4, 4),
    "large": ComplexitySpec("large", 52, 30, 100, 5, 7),
}


@dataclass(frozen=True)
class V7Identity:
    agent_id: str
    application: str
    role: str
    asset_scope: Tuple[str, ...]
    location_scope: Tuple[str, ...]
    physical_authority: Tuple[str, ...]


@dataclass(frozen=True)
class V7Utility:
    service_weight: float
    safety_weight: float
    equity_weight: float
    cost_weight: float
    disclosure_cost: float
    commitment_weight: float
    risk_tolerance: float


@dataclass(frozen=True)
class V7PrivateObservation:
    step: int
    focal_asset: str
    local_kpis: Mapping[str, float]
    belief_distribution: Tuple[float, ...]
    telemetry_confidence: float
    message_age: float
    communication_reliability: float
    available_resources: Mapping[str, float]
    feasible_physical_actions: Tuple[str, ...]
    active_commitments: Tuple[str, ...]


@dataclass(frozen=True)
class V7OperationalProposal:
    agent_id: str
    application: str
    role: str
    proposed_operational_action: str
    target_asset_or_location: Optional[str]
    source_asset_or_location: Optional[str]
    commodity_or_resource: Optional[str]
    quantity_or_capacity: float
    expected_delay: int
    action_probability: float
    action_value: float
    value_margin: float
    reason_code: str

    @property
    def is_physical(self) -> bool:
        return self.proposed_operational_action not in (
            "no_operational_action", "",
        )


@dataclass(frozen=True)
class V7StructuredDecision:
    proposal: V7OperationalProposal
    information_action: str
    communication_action: str
    delegation_action: str
    confidence: float
    compact_plan_summary: str

    def as_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["proposal"] = asdict(self.proposal)
        return value


@dataclass(frozen=True)
class V7DistributedState:
    q: float
    local_entropy: float
    average_local_uncertainty: float
    pooled_uncertainty: float
    generalized_disagreement: float
    graph_disagreement: float
    consensus: float
    consensus_residual: float
    entropy_slope: float
    disagreement_slope: float
    contributors: Tuple[str, ...]
    missing_agents: Tuple[str, ...]
    sketch_messages: int
    sketch_bytes: int
    dropped_sketch_messages: int
    maximum_message_age: float


@dataclass(frozen=True)
class V7RiskContext:
    step: int
    proposal: V7OperationalProposal
    local_kpis: Mapping[str, float]
    predictive_uncertainty: float
    action_value_margin: float
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
    disagreement_slope: float
    communication_reliability: float
    coupling_strength: float
    fragmentation: float
    size_normalized: float
    contributors: Tuple[str, ...]
    missing_agents: Tuple[str, ...]

    def deployable(self) -> Dict[str, Any]:
        value = asdict(self)
        value["proposal"] = asdict(self.proposal)
        return value


@dataclass
class V7Commitment:
    commitment_id: str
    proposer: str
    recipient: str
    action: str
    resource: str
    quantity: float
    due_step: int
    status: str = "proposed"
    revision: int = 0


@dataclass(frozen=True)
class V7ActionResult:
    accepted_typed_action: bool
    accepted_physical_action: bool
    action: str
    actor: str
    target: Optional[str]
    scheduled_step: Optional[int]
    completed_step: Optional[int]
    validation_code: str
    causal_effect: float = 0.0
    beneficial: bool = False
    harmful: bool = False
    reached_next_stage: bool = False
    reached_service: bool = False
    causal_chain_id: Optional[str] = None


@dataclass
class V7ResourceAccount:
    initial: float
    remaining: float
    consumed: float = 0.0
    in_transit: float = 0.0
    delivered: float = 0.0
    losses: float = 0.0

    def residual(self) -> float:
        return float(
            self.initial - self.remaining - self.consumed - self.in_transit
            - self.delivered - self.losses
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "initial": float(self.initial),
            "remaining": float(self.remaining),
            "consumed": float(self.consumed),
            "in_transit": float(self.in_transit),
            "delivered": float(self.delivered),
            "losses": float(self.losses),
            "residual": self.residual(),
        }


@dataclass(frozen=True)
class V7TopologyDiagnostics:
    family: str
    node_count: int
    edge_count: int
    density: float
    mean_degree: float
    degree_variance: float
    connected_components: int
    giant_component_fraction: float
    average_shortest_path: Optional[float]
    clustering_coefficient: float
    modularity: Optional[float]
    algebraic_connectivity: Optional[float]
    diameter: Optional[int]
    assortativity: Optional[float]
    edge_reliability_mean: float
    graph6_sha256: str


@dataclass
class V7EpisodeSummary:
    run_id: str
    application: str
    complexity: str
    coupling: str
    fragmentation: str
    network_disruption: str
    topology_family: str
    environment_seed: int
    controller: str
    sketch_policy: str
    service_loss: float = 0.0
    harmful_actions: int = 0
    beneficial_actions: int = 0
    neutral_actions: int = 0
    physical_actions: int = 0
    actionable_opportunities: int = 0
    service_reaching_actions: int = 0
    information_actions: int = 0
    operational_messages: int = 0
    sketch_messages: int = 0
    dropped_messages: int = 0
    total_bytes: int = 0
    causal_utility: float = 0.0
    maximum_cascade_depth: int = 0
    maximum_conservation_residual: float = 0.0
    replay_status: str = "pending"
    extra: Dict[str, Any] = field(default_factory=dict)
