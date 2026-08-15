"""Typed v4 research objects and execution-time information boundaries.

The cyber-physical fields in this module are abstract simulator quantities.
They cannot address, inspect, or control a real utility or operational system.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class V4Application(str, Enum):
    COMMERCIAL = "commercial"
    HUMANITARIAN = "humanitarian"
    UTILITY = "utility_restoration"


class InformationCondition(str, Enum):
    PRIVATE_FRAGMENTED = "private_fragmented"
    GLOBALLY_PUBLIC = "globally_public"


class V4Method(str, Enum):
    NO_COMMUNICATION = "no_communication"
    AUTONOMY_NO_OPERATOR = "autonomy_no_operator"
    FIXED_COMMUNICATION = "fixed_communication"
    PERIODIC_REVIEW = "periodic_operator_review"
    RANDOM_REVIEW = "random_budget_matched_review"
    LOCAL_KPI_TRIGGER = "local_kpi_trigger"
    KPI_CAUSAL_TRIAGE = "kpi_only_causal_value"
    ENERGY_TRIAGE = "energy_only_triage"
    ENTROPY_DISAGREEMENT_TRIAGE = "entropy_disagreement_only_triage"
    KPI_ENERGY_TRIAGE = "kpi_plus_energy"
    KPI_THERMO_TRIAGE = "kpi_plus_entropy_disagreement"
    COMPLETE_THERMO_TRIAGE = "complete_thermodynamic_view"
    THERMOHITL_RULE = "thermohitl_v4_rule"
    LEARNED_NON_THERMO = "learned_no_thermodynamics"
    THERMOHITL_RL = "thermohitl_v4_rl"
    CENTRALIZED_FULL_INFORMATION = "centralized_full_information"
    BOUNDED_ORACLE = "bounded_operator_oracle"
    FULL_ORACLE = "full_information_oracle"
    SHUFFLED_THERMO = "shuffled_thermodynamics"
    KPI_STRATUM_PERMUTED = "kpi_stratum_permuted_thermodynamics"


class OperatorViewCondition(str, Enum):
    KPI_ONLY = "local_kpi_only"
    ENERGY_ONLY = "energy_only"
    ENTROPY_DISAGREEMENT = "entropy_disagreement_only"
    KPI_ENERGY = "kpi_plus_energy"
    KPI_THERMO = "kpi_plus_entropy_disagreement"
    COMPLETE_THERMO = "complete_thermodynamic"
    ORACLE = "evaluator_global_oracle"


@dataclass(frozen=True)
class V4PrivateObservation:
    step: int
    incident_id: str
    local_service_deficit: float
    local_backlog: float
    local_lateness: float
    local_resource_scarcity: float
    local_commitment_strain: float
    local_safety_stress: float
    local_disruption_risk: float
    local_actionability_flag: float
    belief_operational: float
    belief_telemetry_unreliable: float
    telemetry_confidence: float
    observed_resource_available: float
    communication_reliability: float
    private_cost: float
    private_priority: float
    authorized_actions: Tuple[str, ...]

    def local_kpis(self) -> Dict[str, float]:
        return {
            "service_deficit": float(self.local_service_deficit),
            "backlog": float(self.local_backlog),
            "lateness": float(self.local_lateness),
            "resource_scarcity": float(self.local_resource_scarcity),
            "commitment_strain": float(self.local_commitment_strain),
            "safety_stress": float(self.local_safety_stress),
            "disruption_risk": float(self.local_disruption_risk),
            "actionability_flag": float(self.local_actionability_flag),
        }


@dataclass(frozen=True)
class V4Identity:
    agent_id: str
    role: str
    application: str
    organization: str
    location: Tuple[float, float]
    incident_scope: Tuple[str, ...]


@dataclass
class V4Commitment:
    commitment_id: str
    proposer: str
    recipient: str
    incident_id: str
    resource: str
    quantity: float
    due_step: int
    status: str = "proposed"
    parent_commitment_id: Optional[str] = None
    negotiation_round: int = 0


@dataclass(frozen=True)
class ThermodynamicFeaturesV4:
    raw_service_deficit: float
    raw_backlog: float
    raw_lateness: float
    raw_safety_stress: float
    raw_commitment_strain: float
    raw_resource_scarcity: float
    operational_energy: float
    standardized_energy: float
    belief_entropy: float
    alternative_entropy: float
    commitment_entropy: float
    distributed_entropy: float
    entropy_residual: float
    entropy_anomaly: float
    entropy_slope: float
    entropy_acceleration: float
    belief_disagreement: float
    consensus_confidence: float
    consensus_error: float
    temperature_diagnostic: float
    free_energy_diagnostic: float
    free_energy_residual: float
    sketch_contributors: int
    sketch_messages: int
    sketch_bytes: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionRequestV4:
    request_id: str
    incident_id: str
    requesting_agent: str
    step: int
    reason_code: str
    requested_action: str
    severity: float
    predicted_benefit: float
    uncertainty: float
    estimated_operator_minutes: float
    priority_score: float
    consensus_confidence: float
    predicted_steps_until_collapse: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorInterventionV4:
    intervention_id: str
    incident_id: str
    step: int
    action: str
    target_agent: str
    arguments: Dict[str, Any]
    mandatory: bool
    service_steps: int
    estimated_minutes: float
    reason_code: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


OPERATOR_VIEW_FIELDS: Dict[OperatorViewCondition, Tuple[str, ...]] = {
    OperatorViewCondition.KPI_ONLY: (
        "service_deficit", "backlog", "lateness", "resource_scarcity",
        "commitment_strain", "safety_stress", "disruption_risk",
        "actionability_flag",
    ),
    OperatorViewCondition.ENERGY_ONLY: ("operational_energy", "standardized_energy"),
    OperatorViewCondition.ENTROPY_DISAGREEMENT: (
        "distributed_entropy", "entropy_anomaly", "entropy_slope",
        "belief_disagreement", "consensus_confidence",
    ),
    OperatorViewCondition.KPI_ENERGY: (
        "service_deficit", "backlog", "lateness", "resource_scarcity",
        "commitment_strain", "safety_stress", "disruption_risk",
        "actionability_flag", "operational_energy", "standardized_energy",
    ),
    OperatorViewCondition.KPI_THERMO: (
        "service_deficit", "backlog", "lateness", "resource_scarcity",
        "commitment_strain", "safety_stress", "disruption_risk",
        "actionability_flag", "distributed_entropy", "entropy_anomaly",
        "entropy_slope", "belief_disagreement", "consensus_confidence",
    ),
    OperatorViewCondition.COMPLETE_THERMO: (
        "service_deficit", "backlog", "lateness", "resource_scarcity",
        "commitment_strain", "safety_stress", "disruption_risk",
        "actionability_flag", "operational_energy", "standardized_energy",
        "belief_entropy", "alternative_entropy", "commitment_entropy",
        "distributed_entropy", "entropy_residual", "entropy_anomaly",
        "entropy_slope", "entropy_acceleration", "belief_disagreement",
        "consensus_confidence", "temperature_diagnostic",
        "free_energy_diagnostic", "free_energy_residual",
    ),
    OperatorViewCondition.ORACLE: (
        "service_deficit", "backlog", "lateness", "resource_scarcity",
        "commitment_strain", "safety_stress", "disruption_risk",
        "actionability_flag", "operational_energy", "standardized_energy",
        "belief_entropy", "alternative_entropy", "commitment_entropy",
        "distributed_entropy", "entropy_residual", "entropy_anomaly",
        "entropy_slope", "entropy_acceleration", "belief_disagreement",
        "consensus_confidence", "temperature_diagnostic",
        "free_energy_diagnostic", "free_energy_residual", "oracle_state",
    ),
}


FORBIDDEN_DEPLOYABLE_VIEW_KEYS = {
    "true_disruption",
    "true_telemetry_corrupted",
    "future_disruptions",
    "future_loss",
    "counterfactual_loss",
    "rng_state",
    "evaluator_global_state",
    "oracle_state",
    "private_agent_state",
}


@dataclass(frozen=True)
class OperatorViewV4:
    condition: str
    step: int
    application: str
    incident_id: str
    features: Dict[str, Any]
    alert: Dict[str, Any]
    public_network: Dict[str, Any]
    workload: Dict[str, Any]
    provenance: Dict[str, Any]
    oracle: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        blob = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_operator_view_v4(view: OperatorViewV4) -> None:
    condition = OperatorViewCondition(view.condition)
    if bool(view.oracle) != (condition == OperatorViewCondition.ORACLE):
        raise ValueError("oracle flag and view condition disagree")
    allowed = set(OPERATOR_VIEW_FIELDS[condition])
    extras = set(view.features) - allowed
    if extras:
        raise ValueError("operator view exposes fields outside condition: %s" % sorted(extras))
    if condition != OperatorViewCondition.ORACLE:
        blob = json.dumps(view.as_dict(), sort_keys=True)
        leaked = sorted(key for key in FORBIDDEN_DEPLOYABLE_VIEW_KEYS if key in blob)
        if leaked:
            raise ValueError("deployable operator view leaks evaluator/private fields: %s" % leaked)
    if not view.provenance.get("information_boundary"):
        raise ValueError("operator view lacks information-boundary provenance")


@dataclass(frozen=True)
class CausalChainV4:
    request_entered_queue: bool
    allocator_selected: bool
    operator_received_authorized_view: bool
    operator_acted: bool
    agent_commitment_changed: bool
    accepted_action_changed: bool
    material_or_service_flow_changed: bool
    reached_demand_or_critical_service: bool
    primary_outcome_changed: bool
    intervention_effect: float
    harmful: bool
    common_randomness_verified: bool

    @property
    def complete(self) -> bool:
        return all((
            self.request_entered_queue,
            self.allocator_selected,
            self.operator_received_authorized_view,
            self.operator_acted,
            self.agent_commitment_changed or self.accepted_action_changed,
            self.material_or_service_flow_changed,
            self.reached_demand_or_critical_service,
            self.primary_outcome_changed,
        ))

    def as_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "complete": self.complete}


def bounded_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    import math

    values = [max(0.0, float(value)) for value in probabilities]
    total = sum(values)
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    distribution = [value / total for value in values]
    entropy = -sum(value * math.log(value) for value in distribution if value > 0.0)
    return float(entropy / math.log(len(distribution)))


def jensen_shannon_disagreement(distributions: Sequence[Sequence[float]]) -> float:
    """Bounded generalized Jensen-Shannon divergence in [0, 1]."""

    if not distributions:
        return 0.0
    width = len(distributions[0])
    if width <= 1 or any(len(row) != width for row in distributions):
        raise ValueError("belief distributions must have one common width greater than one")
    normalized: List[List[float]] = []
    for row in distributions:
        values = [max(float(value), 1e-12) for value in row]
        total = sum(values)
        normalized.append([value / total for value in values])
    mean = [sum(row[index] for row in normalized) / len(normalized) for index in range(width)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        import math

        return sum(a * math.log(a / b) for a, b in zip(left, right))

    import math

    js = sum(kl(row, mean) for row in normalized) / len(normalized)
    return min(1.0, max(0.0, float(js / math.log(width))))
