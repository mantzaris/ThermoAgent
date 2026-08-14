"""ThermoHITL v3 thermodynamic triage and bounded simulated operators.

The quantities in this module are statistical-mechanics-inspired operational
constructs. They are not literal physical energy, entropy, or temperature.
Normal execution policies consume only :class:`OperatorView` payloads built
from authorized local/distributed fields. Evaluator-global diagnostics are
returned separately and never inserted into those payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .consensus import (
    consensus_rmse,
    gossip_distributions_with_trace,
    local_consensus_residuals,
    metropolis_matrix,
    one_hot_sketch,
)
from .mechanics import normalized_entropy


class HumanMethod(str, Enum):
    """V3 method identifiers; v1/v2 :class:`types.Method` remains unchanged."""

    AUTONOMOUS_NO_HUMAN = "autonomous_no_human"
    FIXED_COMMUNICATION_NO_HUMAN = "fixed_communication_no_human"
    ALWAYS_ON_HUMAN_REVIEW = "always_on_human_review"
    PERIODIC_HUMAN_REVIEW = "periodic_human_review"
    RANDOM_BUDGET_MATCHED_HUMAN = "random_budget_matched_human"
    LOCAL_KPI_TRIGGER = "local_kpi_trigger"
    ENTROPY_ONLY_TRIGGER = "entropy_only_trigger"
    ENERGY_ONLY_TRIGGER = "energy_only_trigger"
    FREE_ENERGY_TRIGGER = "free_energy_trigger"
    DISAGREEMENT_TRIGGER = "disagreement_trigger"
    THERMOHITL_RULE = "thermohitl_rule"
    LEARNED_NO_THERMODYNAMICS = "learned_no_thermodynamics"
    THERMOHITL_RL = "thermohitl_rl"
    NO_COMMUNICATION = "no_communication"
    CENTRALIZED_FULL_INFORMATION = "centralized_full_information"
    BOUNDED_HUMAN_ORACLE = "bounded_human_oracle"
    FULL_INFORMATION_ORACLE = "full_information_oracle"


class OperatorViewCondition(str, Enum):
    LOCAL_KPI = "local_kpi"
    ENTROPY_ONLY = "entropy_only"
    ENERGY_ONLY = "energy_only"
    THERMODYNAMIC = "thermodynamic"
    THERMODYNAMIC_DISAGREEMENT = "thermodynamic_plus_disagreement"
    EVALUATOR_ORACLE = "evaluator_global_oracle"


class AutonomyLevel(IntEnum):
    QUIET_DECENTRALIZED = 0
    TARGETED_COORDINATION = 1
    HUMAN_INFORMATION = 2
    HUMAN_RECOMMENDATION = 3
    HUMAN_APPROVAL = 4
    HUMAN_CONFLICT_RESOLUTION = 5
    EMERGENCY_OVERRIDE = 6


class AssistanceKind(str, Enum):
    INFORMATION = "human_information"
    RECOMMENDATION = "human_recommendation"
    APPROVAL = "human_approval"
    CONFLICT_RESOLUTION = "conflict_resolution"
    EMERGENCY_OVERRIDE = "emergency_override"


class OperatorAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_MORE_INFORMATION = "request_more_information"
    AUTHORIZE_DATA_SHARING = "authorize_data_sharing"
    RESOLVE_CONFLICT = "resolve_conflict"
    APPROVE_EMERGENCY_RESOURCE = "approve_emergency_resource"
    ADJUST_PRIORITIES = "adjust_priorities"
    INITIATE_OVERRIDE = "initiate_temporary_override"
    RETURN_CONTROL = "return_control"


@dataclass(frozen=True)
class EnergyWeights:
    backlog: float = 0.24
    unmet: float = 0.22
    congestion: float = 0.16
    lateness: float = 0.14
    commitment: float = 0.12
    safety: float = 0.12

    def normalized(self) -> np.ndarray:
        values = np.asarray([
            self.backlog,
            self.unmet,
            self.congestion,
            self.lateness,
            self.commitment,
            self.safety,
        ], dtype=float)
        if np.any(values < 0.0) or values.sum() <= 0.0:
            raise ValueError("energy weights must be nonnegative with positive sum")
        return values / values.sum()


@dataclass
class ThermodynamicCalibration:
    """Nominal-only centers and scales used by execution-time actors."""

    energy_center: float = 0.24
    energy_scale: float = 0.10
    entropy_center: float = 0.45
    entropy_scale: float = 0.10
    flow_entropy_center: float = 0.15
    flow_entropy_scale: float = 0.10
    belief_entropy_center: float = 0.35
    belief_entropy_scale: float = 0.12
    free_energy_center: float = 0.08
    free_energy_scale: float = 0.08
    temperature_center: float = 0.25
    by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    revision: str = "thermohitl-nominal-calibration-dev-v1"

    def values_for_role(self, role: str) -> Dict[str, float]:
        values = {
            "energy_center": self.energy_center,
            "energy_scale": self.energy_scale,
            "entropy_center": self.entropy_center,
            "entropy_scale": self.entropy_scale,
            "flow_entropy_center": self.flow_entropy_center,
            "flow_entropy_scale": self.flow_entropy_scale,
            "belief_entropy_center": self.belief_entropy_center,
            "belief_entropy_scale": self.belief_entropy_scale,
            "free_energy_center": self.free_energy_center,
            "free_energy_scale": self.free_energy_scale,
        }
        values.update(self.by_role.get(str(role), {}))
        for key in list(values):
            if key.endswith("_scale"):
                values[key] = max(float(values[key]), 1e-6)
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ThermodynamicCalibration":
        allowed = {
            "energy_center", "energy_scale", "entropy_center", "entropy_scale",
            "flow_entropy_center", "flow_entropy_scale",
            "belief_entropy_center", "belief_entropy_scale",
            "free_energy_center", "free_energy_scale", "temperature_center",
            "by_role", "revision",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("unknown calibration fields: %s" % sorted(unknown))
        return cls(**dict(values))

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]]) -> "ThermodynamicCalibration":
        if len(rows) < 20:
            raise ValueError("nominal calibration requires at least 20 agent-period rows")

        def center_scale(name: str, subset: Sequence[Mapping[str, Any]]) -> Tuple[float, float]:
            values = np.asarray([float(row[name]) for row in subset], dtype=float)
            center = float(np.median(values))
            # Robust nominal scale; a floor prevents unstable standardized residuals.
            mad = float(np.median(np.abs(values - center))) * 1.4826
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            return center, max(mad, 0.25 * std, 0.025)

        fields = ("energy", "distributed_entropy", "flow_entropy", "belief_entropy", "free_energy")
        calibration_name = {
            "distributed_entropy": "entropy",
        }
        fitted: Dict[str, float] = {}
        for field_name in fields:
            center, scale = center_scale(field_name, rows)
            output_name = calibration_name.get(field_name, field_name)
            fitted[output_name + "_center"] = center
            fitted[output_name + "_scale"] = scale
        roles = sorted({str(row["role"]) for row in rows})
        by_role: Dict[str, Dict[str, float]] = {}
        for role in roles:
            subset = [row for row in rows if str(row["role"]) == role]
            if len(subset) < 8:
                continue
            by_role[role] = {}
            for field_name in fields:
                center, scale = center_scale(field_name, subset)
                output_name = calibration_name.get(field_name, field_name)
                by_role[role][output_name + "_center"] = center
                by_role[role][output_name + "_scale"] = scale
        return cls(by_role=by_role, **fitted)


@dataclass
class LocalThermodynamicState:
    agent_id: str
    role: str
    step: int
    energy: float
    local_energy_residual: float
    distributed_energy: float
    energy_residual: float
    flow_entropy: float
    belief_entropy: float
    distributed_entropy: float
    entropy_residual: float
    entropy_slope: float
    entropy_acceleration: float
    disagreement: float
    consensus_confidence: float
    local_disruption_risk: float
    local_kpi_risk: float
    actionability_evidence: float
    temperature: float
    free_energy: float
    free_energy_residual: float
    components: Dict[str, float]
    macrostate: int
    sketch_contributors: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluatorThermodynamicState:
    step: int
    exact_entropy: float
    exact_energy: float
    exact_flow_entropy: float
    exact_belief_entropy: float
    exact_disagreement: float
    exact_free_energy: float
    entropy_rmse: float
    energy_rmse: float
    sketch_messages: int
    sketch_bytes: int


@dataclass
class ThermodynamicUpdate:
    local: Dict[str, LocalThermodynamicState]
    evaluator: EvaluatorThermodynamicState


def _clip(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _standardized(value: float, center: float, scale: float) -> float:
    return float((value - center) / max(scale, 1e-6))


def _entropy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = np.clip(array, 0.0, None)
    if array.sum() <= 0.0:
        return 0.0
    array /= array.sum()
    positive = array[array > 0.0]
    if positive.size <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / math.log(array.size))


def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    """Bounded base-2 Jensen-Shannon divergence in [0, 1]."""

    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    if p.shape != q.shape or p.ndim != 1 or np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("Jensen-Shannon inputs must be matching nonnegative vectors")
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    midpoint = 0.5 * (p + q)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0.0
        return float(np.sum(first[mask] * np.log2(first[mask] / second[mask])))

    return float(max(0.0, min(1.0, 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint))))


class DistributedThermodynamicMonitor:
    """Link-respecting local/gossip estimator with evaluator-only diagnostics."""

    def __init__(
        self,
        agent_ids: Sequence[str],
        calibration: Optional[ThermodynamicCalibration] = None,
        energy_weights: Optional[EnergyWeights] = None,
        gossip_rounds: int = 3,
        alpha: float = 0.1,
    ) -> None:
        if gossip_rounds < 0:
            raise ValueError("gossip_rounds cannot be negative")
        self.agent_ids = sorted(str(agent_id) for agent_id in agent_ids)
        self.calibration = calibration or ThermodynamicCalibration()
        self.energy_weights = energy_weights or EnergyWeights()
        self.gossip_rounds = int(gossip_rounds)
        self.alpha = float(alpha)
        self.previous_entropy = {agent_id: self.calibration.entropy_center for agent_id in self.agent_ids}
        self.previous_slope = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.cumulative_sketch_messages = 0
        self.cumulative_sketch_bytes = 0

    @staticmethod
    def _belief_distribution(observation: Any) -> np.ndarray:
        impairment = _clip(float(observation.impairment))
        delay = _clip(float(observation.delay) / 2.0)
        shortage = _clip(max(
            float(observation.service_shortfall),
            float(observation.backlog) / max(float(observation.local_forecast), 1.0),
        ))
        comm = _clip(1.0 - float(observation.communication_reliability))
        local = 0.50 * impairment + 0.20 * delay + 0.20 * shortage + 0.10 * comm
        systemic = 0.20 * impairment + 0.30 * delay + 0.25 * shortage + 0.25 * comm
        nominal = max(0.01, 1.0 - max(local, systemic))
        values = np.asarray([nominal, max(0.01, local), max(0.01, systemic)], dtype=float)
        return values / values.sum()

    @staticmethod
    def _flow_entropy(env: Any, agent_id: str) -> float:
        recent: Dict[str, float] = {}
        lower_step = max(0, int(env.step_index) - 6)
        for shipment in list(env.shipments.values()) + list(env.completed_shipments.values()):
            if shipment.sender == agent_id and shipment.sent_step >= lower_step:
                recent[shipment.recipient] = recent.get(shipment.recipient, 0.0) + float(shipment.quantity)
        return _entropy(list(recent.values()))

    def _components(self, env: Any, agent_id: str) -> Tuple[Dict[str, float], np.ndarray, float]:
        agent = env.agents[agent_id]
        observation = agent.vault.observation(agent_id)
        backlog = _clip(float(observation.backlog) / max(float(observation.local_forecast), 1.0))
        unmet = _clip(float(observation.service_shortfall))
        congestion = _clip(
            max(float(observation.impairment), 1.0 - float(observation.capacity) / max(
                float(observation.capacity) + float(observation.backlog), 1.0
            ))
        )
        lateness = _clip(float(observation.delay) / 2.0)
        commitment = _clip(float(observation.commitment_strain))
        known_failed_route = any(
            memory.kind == "failure" and "no_route" in memory.summary
            for memory in agent.vault.retrieve(agent_id, 8)
        )
        safety = _clip(max(
            0.65 * float(observation.impairment) + 0.20 * lateness,
            0.85 if known_failed_route else 0.0,
        ))
        components = {
            "backlog": backlog,
            "unmet": unmet,
            "congestion": congestion,
            "lateness": lateness,
            "commitment": commitment,
            "safety": safety,
        }
        weights = self.energy_weights.normalized()
        energy = float(np.dot(weights, np.asarray(list(components.values()), dtype=float)))
        belief = self._belief_distribution(observation)
        return components, belief, energy

    @staticmethod
    def _macrostate(energy: float, belief_entropy: float, communication_risk: float) -> int:
        return (
            4 * int(energy >= 0.45)
            + 2 * int(belief_entropy >= 0.55)
            + int(communication_risk >= 0.30)
        )

    @staticmethod
    def _average_consensus(
        values: Mapping[str, Sequence[float]],
        edges_by_round: Sequence[Iterable[Tuple[str, str]]],
    ) -> Dict[str, np.ndarray]:
        ids = sorted(values)
        matrix_values = np.asarray([values[agent_id] for agent_id in ids], dtype=float)
        for edges in edges_by_round:
            matrix_values = metropolis_matrix(ids, edges).dot(matrix_values)
        return {agent_id: matrix_values[index].copy() for index, agent_id in enumerate(ids)}

    def update(self, env: Any, edges_by_round: Optional[Sequence[Iterable[Tuple[str, str]]]] = None) -> ThermodynamicUpdate:
        local_components: Dict[str, Dict[str, float]] = {}
        beliefs: Dict[str, np.ndarray] = {}
        energies: Dict[str, float] = {}
        flow_entropies: Dict[str, float] = {}
        belief_entropies: Dict[str, float] = {}
        macrostates: Dict[str, int] = {}
        for agent_id in self.agent_ids:
            components, belief, energy = self._components(env, agent_id)
            observation = env.agents[agent_id].vault.observation(agent_id)
            local_components[agent_id] = components
            beliefs[agent_id] = belief
            energies[agent_id] = energy
            flow_entropies[agent_id] = self._flow_entropy(env, agent_id)
            belief_entropies[agent_id] = _entropy(belief)
            macrostates[agent_id] = self._macrostate(
                energy,
                belief_entropies[agent_id],
                1.0 - float(observation.communication_reliability),
            )

        if edges_by_round is None:
            edges = set(env.active_communication_edges())
            edges_by_round = [edges for _ in range(self.gossip_rounds)]
        edges_by_round = [set(edges) for edges in edges_by_round]
        sketches = {
            agent_id: one_hot_sketch(
                macrostates[agent_id], k=8, alpha=self.alpha,
                population_size=len(self.agent_ids),
            )
            for agent_id in self.agent_ids
        }
        if edges_by_round:
            estimates, trace = gossip_distributions_with_trace(sketches, edges_by_round)
        else:
            estimates = {agent_id: value.copy() for agent_id, value in sketches.items()}
            trace = []
        coarse_values = {
            agent_id: [
                round(energies[agent_id], 1),
                round(flow_entropies[agent_id], 1),
                round(belief_entropies[agent_id], 1),
                *np.round(beliefs[agent_id], 1).tolist(),
            ]
            for agent_id in self.agent_ids
        }
        averaged = self._average_consensus(coarse_values, edges_by_round)
        final_edges = edges_by_round[-1] if edges_by_round else set()
        residuals = local_consensus_residuals(estimates, final_edges)

        step_messages = 0
        step_bytes = 0
        for round_index, (round_edges, round_estimates) in enumerate(zip(edges_by_round, trace), start=1):
            neighbors: Dict[str, List[str]] = {agent_id: [] for agent_id in self.agent_ids}
            for left, right in sorted(round_edges):
                neighbors[left].append(right)
                neighbors[right].append(left)
                for sender, recipient in ((left, right), (right, left)):
                    payload = {
                        "protocol": "thermohitl-sketch-v1",
                        "sender": sender,
                        "recipient": recipient,
                        "step": int(env.step_index),
                        "round": round_index,
                        "macro_distribution": np.round(round_estimates[sender], 6).tolist(),
                        "coarse_statistics": coarse_values[sender],
                    }
                    step_messages += 1
                    step_bytes += len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            for agent_id in self.agent_ids:
                env.ledger.append(
                    env.step_index,
                    "thermodynamic_sketch",
                    agent_id,
                    {
                        "round": round_index,
                        "neighbors": sorted(neighbors[agent_id]),
                        "macro_distribution": np.round(round_estimates[agent_id], 6).tolist(),
                        "privacy": "coarse_local_summary",
                    },
                    private_to=agent_id,
                )
        self.cumulative_sketch_messages += step_messages
        self.cumulative_sketch_bytes += step_bytes

        exact_distribution = np.bincount(
            np.asarray(list(macrostates.values()), dtype=int), minlength=8
        ).astype(float)
        exact_distribution = (exact_distribution + self.alpha) / (
            exact_distribution.sum() + self.alpha * 8
        )
        exact_entropy = normalized_entropy(exact_distribution)
        exact_energy = float(np.mean(list(energies.values())))
        exact_flow = float(np.mean(list(flow_entropies.values())))
        exact_belief = float(np.mean(list(belief_entropies.values())))
        exact_belief_distribution = np.mean(np.asarray(list(beliefs.values())), axis=0)
        exact_disagreement = float(np.mean([
            jensen_shannon_divergence(belief, exact_belief_distribution)
            for belief in beliefs.values()
        ]))
        temperature = _clip(0.15 + 0.55 * float(np.std(list(energies.values()))) + 0.30 * exact_disagreement)
        exact_free = exact_energy - temperature * exact_entropy

        local_states: Dict[str, LocalThermodynamicState] = {}
        entropy_errors: List[float] = []
        energy_errors: List[float] = []
        for agent_id in self.agent_ids:
            role = env.agents[agent_id].identity.role
            calibration = self.calibration.values_for_role(role)
            distributed_entropy = normalized_entropy(estimates[agent_id])
            distributed_energy = float(averaged[agent_id][0])
            distributed_belief = np.clip(averaged[agent_id][3:6], 1e-9, None)
            distributed_belief /= distributed_belief.sum()
            disagreement = jensen_shannon_divergence(beliefs[agent_id], distributed_belief)
            local_temperature = _clip(
                0.15 + 0.45 * abs(distributed_energy - energies[agent_id]) + 0.40 * disagreement
            )
            free_energy = distributed_energy - local_temperature * distributed_entropy
            entropy_slope = distributed_entropy - self.previous_entropy[agent_id]
            entropy_acceleration = entropy_slope - self.previous_slope[agent_id]
            observation = env.agents[agent_id].vault.observation(agent_id)
            disruption_risk = _clip(
                0.35 * float(observation.impairment)
                + 0.20 * min(float(observation.delay) / 2.0, 1.0)
                + 0.25 * float(observation.service_shortfall)
                + 0.20 * (1.0 - float(observation.communication_reliability))
            )
            working = env.agents[agent_id].vault.working_memory(agent_id)
            last_result = str(working.get("last_result", ""))
            recent_material_failure = last_result in (
                "no_route", "lead_time_infeasible", "handling_capacity_exceeded",
                "insufficient_inventory", "operator_authorization_expired",
            )
            pending_need = any(
                message.kind in ("need", "quote_request")
                for message in list(env.agents[agent_id].inbox)[-8:]
            )
            binding_commitment = any(
                commitment.status in ("accepted", "breached", "in_transit")
                for commitment in env.agents[agent_id].commitments.values()
            )
            actionability_evidence = _clip(max(
                1.0 if recent_material_failure else 0.0,
                0.35 if pending_need else 0.0,
                0.45 if binding_commitment else 0.0,
            ))
            local_kpi_risk = _clip(
                0.40 * local_components[agent_id]["backlog"]
                + 0.30 * local_components[agent_id]["unmet"]
                + 0.15 * local_components[agent_id]["congestion"]
                + 0.15 * local_components[agent_id]["lateness"]
            )
            state = LocalThermodynamicState(
                agent_id=agent_id,
                role=role,
                step=int(env.step_index),
                energy=energies[agent_id],
                local_energy_residual=_standardized(
                    energies[agent_id],
                    calibration["energy_center"],
                    calibration["energy_scale"],
                ),
                distributed_energy=distributed_energy,
                energy_residual=_standardized(
                    distributed_energy,
                    calibration["energy_center"],
                    calibration["energy_scale"],
                ),
                flow_entropy=flow_entropies[agent_id],
                belief_entropy=belief_entropies[agent_id],
                distributed_entropy=distributed_entropy,
                entropy_residual=abs(_standardized(
                    distributed_entropy,
                    calibration["entropy_center"],
                    calibration["entropy_scale"],
                )),
                entropy_slope=entropy_slope,
                entropy_acceleration=entropy_acceleration,
                disagreement=disagreement,
                consensus_confidence=_clip(1.0 - min(1.0, residuals[agent_id] * 4.0)),
                local_disruption_risk=disruption_risk,
                local_kpi_risk=local_kpi_risk,
                actionability_evidence=actionability_evidence,
                temperature=local_temperature,
                free_energy=free_energy,
                free_energy_residual=abs(_standardized(
                    free_energy,
                    calibration["free_energy_center"],
                    calibration["free_energy_scale"],
                )),
                components=local_components[agent_id],
                macrostate=macrostates[agent_id],
                sketch_contributors=max(1, 1 + sum(agent_id in edge for edge in final_edges)),
            )
            local_states[agent_id] = state
            self.previous_entropy[agent_id] = distributed_entropy
            self.previous_slope[agent_id] = entropy_slope
            entropy_errors.append(distributed_entropy - exact_entropy)
            energy_errors.append(distributed_energy - exact_energy)
            env.agents[agent_id].vault.update_working(
                agent_id,
                {"thermodynamic_state": state.as_dict()},
            )
            env.ledger.append(
                env.step_index,
                "thermodynamic_state",
                agent_id,
                {
                    **state.as_dict(),
                    "information_boundary": "private_local_plus_received_sketches",
                },
                private_to=agent_id,
            )

        return ThermodynamicUpdate(
            local=local_states,
            evaluator=EvaluatorThermodynamicState(
                step=int(env.step_index),
                exact_entropy=exact_entropy,
                exact_energy=exact_energy,
                exact_flow_entropy=exact_flow,
                exact_belief_entropy=exact_belief,
                exact_disagreement=exact_disagreement,
                exact_free_energy=exact_free,
                entropy_rmse=float(np.sqrt(np.mean(np.square(entropy_errors)))),
                energy_rmse=float(np.sqrt(np.mean(np.square(energy_errors)))),
                sketch_messages=step_messages,
                sketch_bytes=step_bytes,
            ),
        )


@dataclass(frozen=True)
class EscalationConfig:
    alpha: float = 0.30
    beta: float = 0.22
    gamma: float = 0.12
    delta: float = 0.14
    eta: float = 0.22
    workload_penalty: float = 0.25
    tau_on: float = 1.15
    tau_off: float = 0.45
    actionable_tau_on: Optional[float] = None
    minimum_dwell: int = 2
    cooldown: int = 3
    periodic_interval: int = 6
    random_probability: float = 0.20

    def __post_init__(self) -> None:
        if self.tau_off >= self.tau_on:
            raise ValueError("tau_off must be lower than tau_on")
        if (
            self.actionable_tau_on is not None
            and not self.tau_off < self.actionable_tau_on <= self.tau_on
        ):
            raise ValueError("actionable_tau_on must lie above tau_off and at or below tau_on")
        if self.minimum_dwell < 1 or self.cooldown < 0 or self.periodic_interval < 1:
            raise ValueError("invalid escalation timing configuration")
        if not 0.0 <= self.random_probability <= 1.0:
            raise ValueError("random probability must be in [0,1]")


@dataclass
class AgentEscalationState:
    active: bool = False
    activated_step: Optional[int] = None
    last_request_step: Optional[int] = None
    last_score: float = 0.0
    request_count: int = 0


@dataclass
class AssistanceRequest:
    incident_id: str
    requesting_agent: str
    application: str
    step: int
    assistance_kind: str
    reason: str
    severity: float
    entropy_anomaly: float
    disagreement: float
    consensus_confidence: float
    local_kpi_risk: float
    expected_loss_without: float
    expected_loss_with: float
    expected_benefit: float
    prediction_uncertainty: float
    estimated_operator_minutes: float
    priority_score: float
    predicted_steps_until_collapse: int
    suggested_intervention: str
    intervention_arguments: Dict[str, Any]
    requested_autonomy_level: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndependentEscalationController:
    """Per-agent stateful escalation; there is no shared central alarm state."""

    def __init__(self, agent_ids: Sequence[str], config: Optional[EscalationConfig] = None) -> None:
        self.config = config or EscalationConfig()
        self.states = {str(agent_id): AgentEscalationState() for agent_id in agent_ids}

    def score(
        self,
        method: HumanMethod,
        state: LocalThermodynamicState,
        workload: float,
        learned_score: Optional[float] = None,
    ) -> float:
        if method == HumanMethod.LOCAL_KPI_TRIGGER:
            # The conventional baseline receives the same private failure and
            # commitment evidence as ThermoHITL; only distributed
            # thermodynamic fields are withheld.
            return (
                2.25 * state.local_kpi_risk
                + 1.20 * state.actionability_evidence
                - self.config.workload_penalty * workload
            )
        if method == HumanMethod.ENTROPY_ONLY_TRIGGER:
            return state.entropy_residual + 0.35 * max(0.0, state.entropy_slope)
        if method == HumanMethod.ENERGY_ONLY_TRIGGER:
            return max(0.0, state.energy_residual, state.local_energy_residual)
        if method == HumanMethod.FREE_ENERGY_TRIGGER:
            return state.free_energy_residual
        if method == HumanMethod.DISAGREEMENT_TRIGGER:
            return 4.0 * state.disagreement
        if method in (HumanMethod.LEARNED_NO_THERMODYNAMICS, HumanMethod.THERMOHITL_RL):
            if learned_score is None:
                raise ValueError("learned escalation method requires a learned score")
            return float(learned_score)
        if method in (HumanMethod.BOUNDED_HUMAN_ORACLE, HumanMethod.FULL_INFORMATION_ORACLE):
            return max(0.0, state.energy_residual) + state.local_disruption_risk + state.entropy_residual
        if method == HumanMethod.THERMOHITL_RULE:
            energy_signal = max(0.0, state.energy_residual)
            if state.actionability_evidence >= 0.95:
                energy_signal = max(energy_signal, state.local_energy_residual)
            score = float(
                self.config.alpha * max(0.0, energy_signal)
                + self.config.beta * state.entropy_residual
                + self.config.gamma * max(0.0, state.entropy_slope / 0.05)
                + self.config.delta * (state.disagreement / 0.10)
                + self.config.eta * (state.local_disruption_risk / 0.25)
                - self.config.workload_penalty * workload
            )
            # A system-wide anomaly is not itself an actionable request. A
            # demand organization may legitimately request route authority
            # from distributed early warning alone; every other role must
            # also possess private evidence of a failed/blocked obligation.
            demand_roles = {"retailer", "clinic", "community"}
            if state.role not in demand_roles and state.actionability_evidence < 0.95:
                return 0.0
            return score
        return 0.0

    def should_request(
        self,
        agent_id: str,
        method: HumanMethod,
        state: LocalThermodynamicState,
        workload: float,
        rng: np.random.RandomState,
        learned_score: Optional[float] = None,
    ) -> Tuple[bool, float, bool, bool]:
        private = self.states[agent_id]
        score = self.score(method, state, workload, learned_score)
        activated = False
        deactivated = False
        if method in (
            HumanMethod.AUTONOMOUS_NO_HUMAN,
            HumanMethod.FIXED_COMMUNICATION_NO_HUMAN,
            HumanMethod.NO_COMMUNICATION,
            HumanMethod.CENTRALIZED_FULL_INFORMATION,
        ):
            private.last_score = score
            return False, score, activated, deactivated
        if method == HumanMethod.ALWAYS_ON_HUMAN_REVIEW:
            request = state.step % 2 == 0
        elif method == HumanMethod.PERIODIC_HUMAN_REVIEW:
            request = state.step % self.config.periodic_interval == 0
        elif method == HumanMethod.RANDOM_BUDGET_MATCHED_HUMAN:
            request = bool(rng.rand() < self.config.random_probability)
        else:
            activation_threshold = (
                self.config.actionable_tau_on
                if (
                    state.actionability_evidence >= 0.95
                    and self.config.actionable_tau_on is not None
                )
                else self.config.tau_on
            )
            if not private.active and score >= activation_threshold:
                private.active = True
                private.activated_step = state.step
                activated = True
            elif (
                private.active
                and private.activated_step is not None
                and state.step - private.activated_step >= self.config.minimum_dwell
                and score <= self.config.tau_off
            ):
                private.active = False
                private.activated_step = None
                deactivated = True
            # Stateful triggers emit one request on the rising edge. Persisting
            # risk keeps the autonomy level elevated but does not flood the
            # scarce operator queue every cooldown interval.
            request = activated
        if request and private.last_request_step is not None:
            request = state.step - private.last_request_step >= self.config.cooldown
        if request:
            private.last_request_step = state.step
            private.request_count += 1
        private.last_score = score
        return bool(request), float(score), activated, deactivated


FORBIDDEN_NORMAL_VIEW_KEYS = {
    "raw_private_state",
    "private_observation",
    "private_inventory",
    "private_cost",
    "rng_state",
    "future_disruption",
    "true_disruption_label",
    "counterfactual_outcome",
    "exact_global_entropy",
    "exact_global_energy",
    "evaluator_full_state",
}


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


@dataclass(frozen=True)
class OperatorView:
    schema_version: str
    condition: str
    step: int
    incident_id: str
    payload: Dict[str, Any]
    sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_operator_view(view: OperatorView) -> None:
    condition = OperatorViewCondition(view.condition)
    if condition != OperatorViewCondition.EVALUATOR_ORACLE:
        leaked = FORBIDDEN_NORMAL_VIEW_KEYS.intersection(set(_walk_keys(view.payload)))
        if leaked:
            raise PermissionError("operator view contains forbidden fields: %s" % sorted(leaked))
    if canonical_payload_sha256(view.payload) != view.sha256:
        raise ValueError("operator view payload hash mismatch")


def build_operator_view(
    request: AssistanceRequest,
    state: LocalThermodynamicState,
    condition: OperatorViewCondition,
    workload: Mapping[str, Any],
    public_network: Mapping[str, Any],
    oracle_payload: Optional[Mapping[str, Any]] = None,
) -> OperatorView:
    features: Dict[str, Any] = {
        "local_kpi_risk": state.local_kpi_risk,
        "actionability_evidence": state.actionability_evidence,
        "local_kpi_bands": {
            name: "high" if value >= 0.67 else "nominal" if value >= 0.33 else "low"
            for name, value in state.components.items()
        },
    }
    if condition == OperatorViewCondition.ENTROPY_ONLY:
        features = {
            "distributed_entropy": state.distributed_entropy,
            "entropy_anomaly": state.entropy_residual,
            "entropy_slope": state.entropy_slope,
            "consensus_confidence": state.consensus_confidence,
        }
    elif condition == OperatorViewCondition.ENERGY_ONLY:
        features = {
            "local_energy": state.energy,
            "local_energy_residual": state.local_energy_residual,
            "distributed_energy": state.distributed_energy,
            "energy_residual": state.energy_residual,
            "local_kpi_bands": features["local_kpi_bands"],
        }
    elif condition in (
        OperatorViewCondition.THERMODYNAMIC,
        OperatorViewCondition.THERMODYNAMIC_DISAGREEMENT,
    ):
        features = {
            "local_energy": state.energy,
            "local_energy_residual": state.local_energy_residual,
            "distributed_energy": state.distributed_energy,
            "energy_residual": state.energy_residual,
            "distributed_entropy": state.distributed_entropy,
            "flow_entropy": state.flow_entropy,
            "belief_entropy": state.belief_entropy,
            "entropy_anomaly": state.entropy_residual,
            "entropy_slope": state.entropy_slope,
            "free_energy_diagnostic": state.free_energy,
            "free_energy_residual": state.free_energy_residual,
            "temperature_diagnostic": state.temperature,
            "consensus_confidence": state.consensus_confidence,
            "local_kpi_risk": state.local_kpi_risk,
            "actionability_evidence": state.actionability_evidence,
        }
        if condition == OperatorViewCondition.THERMODYNAMIC_DISAGREEMENT:
            features["agent_disagreement"] = state.disagreement
            features["sketch_contributors"] = state.sketch_contributors
    elif condition == OperatorViewCondition.EVALUATOR_ORACLE:
        if oracle_payload is None:
            raise ValueError("oracle view requires explicit evaluator payload")
        features = dict(oracle_payload)
    payload = {
        "schema_version": "thermohitl-operator-view-v1",
        "condition": condition.value,
        "step": request.step,
        "incident": {
            "incident_id": request.incident_id,
            "requesting_agent": request.requesting_agent,
            "application": request.application,
            "assistance_kind": request.assistance_kind,
            "reason": request.reason,
            "severity": request.severity,
            "expected_loss_without": request.expected_loss_without,
            "expected_loss_with": request.expected_loss_with,
            "expected_benefit": request.expected_benefit,
            "prediction_uncertainty": request.prediction_uncertainty,
            "estimated_operator_minutes": request.estimated_operator_minutes,
            "priority_score": request.priority_score,
            "predicted_steps_until_collapse": request.predicted_steps_until_collapse,
            "suggested_intervention": request.suggested_intervention,
            "intervention_arguments": request.intervention_arguments,
        },
        "features": features,
        "operator_workload": dict(workload),
        "public_network": dict(public_network),
        "provenance": {
            "information_boundary": (
                "evaluator_global_oracle" if condition == OperatorViewCondition.EVALUATOR_ORACLE
                else "requesting_agent_local_plus_distributed_coarse_sketches"
            ),
            "timestamp_step": request.step,
        },
    }
    view = OperatorView(
        schema_version="thermohitl-operator-view-v1",
        condition=condition.value,
        step=request.step,
        incident_id=request.incident_id,
        payload=payload,
        sha256=canonical_payload_sha256(payload),
    )
    validate_operator_view(view)
    return view


@dataclass(frozen=True)
class OperatorProfile:
    name: str
    slots: int
    base_latency_steps: int
    service_minutes: float
    base_accuracy: float
    fatigue_sensitivity: float
    workload_recovery: float
    risk_aversion: float

    def __post_init__(self) -> None:
        if self.slots < 1 or self.base_latency_steps < 0 or self.service_minutes <= 0:
            raise ValueError("operator capacity and service time must be positive")
        if not 0.0 <= self.base_accuracy <= 1.0:
            raise ValueError("operator accuracy must be in [0,1]")
        if not 0.0 <= self.workload_recovery <= 1.0:
            raise ValueError("workload recovery must be in [0,1]")


OPERATOR_PROFILES: Dict[str, OperatorProfile] = {
    "high_accuracy_bounded": OperatorProfile("high_accuracy_bounded", 2, 1, 8.0, 0.93, 0.12, 0.15, 0.45),
    "fast_imperfect": OperatorProfile("fast_imperfect", 2, 0, 5.0, 0.78, 0.18, 0.20, 0.30),
    "slow_accurate": OperatorProfile("slow_accurate", 1, 2, 12.0, 0.96, 0.08, 0.12, 0.50),
    "fatigue_sensitive": OperatorProfile("fatigue_sensitive", 2, 1, 8.0, 0.92, 0.45, 0.08, 0.40),
    "risk_averse": OperatorProfile("risk_averse", 1, 1, 10.0, 0.90, 0.15, 0.12, 0.80),
    "oracle": OperatorProfile("oracle", 4, 0, 4.0, 1.00, 0.00, 0.30, 0.00),
}


@dataclass
class ScheduledOperatorDecision:
    request: AssistanceRequest
    view: OperatorView
    allocated_step: int
    completion_step: int
    operator_action: str


@dataclass
class OperatorIntervention:
    intervention_id: str
    incident_id: str
    requesting_agent: str
    step: int
    action: str
    bounded_tool: str
    arguments: Dict[str, Any]
    mandatory: bool
    view_sha256: str
    expected_benefit: float
    operator_minutes: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AttentionAllocator:
    """Ranks authorized views without reading the simulator."""

    def __init__(self, policy: str, seed: int = 0) -> None:
        allowed = {
            "fcfs", "highest_energy", "highest_entropy", "highest_disagreement",
            "random", "local_kpi", "learned_non_entropic",
            "thermodynamic_expected_benefit", "benefit_per_minute", "oracle",
        }
        if policy not in allowed:
            raise ValueError("unknown attention allocation policy: %s" % policy)
        self.policy = policy
        self.rng = np.random.RandomState(int(seed))

    @staticmethod
    def _feature(view: OperatorView, key: str) -> float:
        return float(view.payload.get("features", {}).get(key, 0.0))

    def rank(self, rows: Sequence[Tuple[AssistanceRequest, OperatorView]]) -> List[Tuple[AssistanceRequest, OperatorView]]:
        rows = list(rows)
        if self.policy == "random":
            order = self.rng.permutation(len(rows)).tolist()
            return [rows[index] for index in order]
        if self.policy == "fcfs":
            return sorted(rows, key=lambda row: (row[0].step, row[0].incident_id))

        def score(row: Tuple[AssistanceRequest, OperatorView]) -> float:
            request, view = row
            if self.policy == "highest_energy":
                return self._feature(view, "distributed_energy")
            if self.policy == "highest_entropy":
                return self._feature(view, "entropy_anomaly")
            if self.policy == "highest_disagreement":
                return self._feature(view, "agent_disagreement")
            if self.policy in ("local_kpi", "learned_non_entropic"):
                return self._feature(view, "local_kpi_risk")
            if self.policy in ("benefit_per_minute", "thermodynamic_expected_benefit"):
                return request.expected_benefit / max(request.estimated_operator_minutes, 1e-6)
            if self.policy == "oracle":
                return float(view.payload.get("features", {}).get("true_intervention_benefit", request.expected_benefit))
            return request.priority_score

        return sorted(rows, key=lambda row: (-score(row), row[0].step, row[0].incident_id))


class SimulatedOperator:
    """Finite-capacity, latency- and fatigue-aware simulated operator.

    This model supports engineering evaluation only. It makes no claim about
    actual human usability, trust, workload, or behavior.
    """

    def __init__(self, profile: OperatorProfile, allocator: AttentionAllocator, seed: int = 0) -> None:
        self.profile = profile
        self.allocator = allocator
        self.rng = np.random.RandomState(int(seed))
        self.queue: List[Tuple[AssistanceRequest, OperatorView]] = []
        self.active: List[ScheduledOperatorDecision] = []
        self.seen_incidents: set[str] = set()
        self.workload = 0.0
        self.fatigue = 0.0
        self.operator_minutes = 0.0
        self.intervention_count = 0
        self.rejected_count = 0
        self.maximum_queue = 0
        self.maximum_workload = 0.0
        self.queue_wait_steps: List[int] = []
        self._intervention_counter = 0

    def workload_snapshot(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.name,
            "workload": self.workload,
            "fatigue": self.fatigue,
            "active_interventions": len(self.active),
            "queue_length": len(self.queue),
            "available_attention_slots": max(0, self.profile.slots - len(self.active)),
            "operator_minutes": self.operator_minutes,
        }

    def enqueue(self, request: AssistanceRequest, view: OperatorView) -> bool:
        validate_operator_view(view)
        if request.incident_id in self.seen_incidents:
            return False
        self.seen_incidents.add(request.incident_id)
        self.queue.append((request, view))
        self.maximum_queue = max(self.maximum_queue, len(self.queue))
        return True

    def _choose_action(self, request: AssistanceRequest, view: OperatorView) -> str:
        effective_accuracy = _clip(
            self.profile.base_accuracy
            * (1.0 - self.profile.fatigue_sensitivity * self.fatigue)
        )
        useful_prediction = request.expected_benefit > (
            self.profile.risk_aversion * request.prediction_uncertainty
        )
        if request.suggested_intervention in (
            "authorize_emergency_route", "temporary_emergency_override"
        ):
            source = request.intervention_arguments.get("source")
            target = request.intervention_arguments.get("target")
            active_edges = {
                tuple(edge)
                for edge in view.payload.get("public_network", {}).get(
                    "physical_edges", []
                )
            }
            # A bounded operator should not spend authority on a route that is
            # already feasible in its authorized infrastructure view.
            if (source, target) in active_edges:
                useful_prediction = False
        correct = bool(self.rng.rand() <= effective_accuracy)
        if correct and useful_prediction:
            mapping = {
                "authorize_information_sharing": OperatorAction.AUTHORIZE_DATA_SHARING.value,
                "resolve_contract_conflict": OperatorAction.RESOLVE_CONFLICT.value,
                "approve_emergency_resource": OperatorAction.APPROVE_EMERGENCY_RESOURCE.value,
                "adjust_priorities": OperatorAction.ADJUST_PRIORITIES.value,
                "authorize_emergency_route": OperatorAction.APPROVE.value,
                "temporary_emergency_override": OperatorAction.INITIATE_OVERRIDE.value,
            }
            return mapping.get(request.suggested_intervention, OperatorAction.APPROVE.value)
        if correct:
            return OperatorAction.REJECT.value
        return (
            OperatorAction.REQUEST_MORE_INFORMATION.value
            if self.rng.rand() < 0.5 else OperatorAction.REJECT.value
        )

    def step(self, step: int) -> List[OperatorIntervention]:
        self.workload *= 1.0 - self.profile.workload_recovery
        self.fatigue *= 1.0 - 0.5 * self.profile.workload_recovery
        completed: List[OperatorIntervention] = []
        remaining_active: List[ScheduledOperatorDecision] = []
        for decision in self.active:
            if decision.completion_step > step:
                remaining_active.append(decision)
                continue
            request = decision.request
            self._intervention_counter += 1
            approved_actions = {
                OperatorAction.APPROVE.value,
                OperatorAction.AUTHORIZE_DATA_SHARING.value,
                OperatorAction.RESOLVE_CONFLICT.value,
                OperatorAction.APPROVE_EMERGENCY_RESOURCE.value,
                OperatorAction.ADJUST_PRIORITIES.value,
                OperatorAction.INITIATE_OVERRIDE.value,
            }
            if decision.operator_action in approved_actions:
                bounded_tool = request.suggested_intervention
                arguments = dict(request.intervention_arguments)
                self.intervention_count += 1
            elif decision.operator_action == OperatorAction.REQUEST_MORE_INFORMATION.value:
                bounded_tool = "request_more_information"
                arguments = {"agent_id": request.requesting_agent, "topic": request.reason}
            else:
                bounded_tool = "reject_request"
                arguments = {"reason": "bounded operator declined the request"}
                self.rejected_count += 1
            completed.append(OperatorIntervention(
                intervention_id="HI%06d" % self._intervention_counter,
                incident_id=request.incident_id,
                requesting_agent=request.requesting_agent,
                step=int(step),
                action=decision.operator_action,
                bounded_tool=bounded_tool,
                arguments=arguments,
                mandatory=bounded_tool == "temporary_emergency_override",
                view_sha256=decision.view.sha256,
                expected_benefit=request.expected_benefit,
                operator_minutes=request.estimated_operator_minutes,
            ))
        self.active = remaining_active

        slots = max(0, self.profile.slots - len(self.active))
        if slots and self.queue:
            ranked = self.allocator.rank(self.queue)
            selected = ranked[:slots]
            selected_ids = {request.incident_id for request, _ in selected}
            self.queue = [row for row in self.queue if row[0].incident_id not in selected_ids]
            for request, view in selected:
                service_steps = max(1, int(math.ceil(request.estimated_operator_minutes / 5.0)))
                completion = int(step + self.profile.base_latency_steps + service_steps)
                action = self._choose_action(request, view)
                self.active.append(ScheduledOperatorDecision(
                    request=request,
                    view=view,
                    allocated_step=int(step),
                    completion_step=completion,
                    operator_action=action,
                ))
                self.queue_wait_steps.append(int(step - request.step))
                self.operator_minutes += request.estimated_operator_minutes
                workload_increment = request.estimated_operator_minutes / 30.0
                self.workload += workload_increment
                self.fatigue = _clip(self.fatigue + 0.20 * workload_increment)
        self.maximum_workload = max(self.maximum_workload, self.workload)
        return completed


def request_assistance_kind(state: LocalThermodynamicState) -> AssistanceKind:
    if state.disagreement >= 0.18:
        return AssistanceKind.CONFLICT_RESOLUTION
    if state.energy_residual >= 2.0 and state.local_disruption_risk >= 0.65:
        return AssistanceKind.EMERGENCY_OVERRIDE
    if state.energy_residual >= 1.25:
        return AssistanceKind.APPROVAL
    if state.entropy_residual >= 1.5:
        return AssistanceKind.INFORMATION
    return AssistanceKind.RECOMMENDATION


def autonomy_level_for_request(kind: AssistanceKind) -> AutonomyLevel:
    return {
        AssistanceKind.INFORMATION: AutonomyLevel.HUMAN_INFORMATION,
        AssistanceKind.RECOMMENDATION: AutonomyLevel.HUMAN_RECOMMENDATION,
        AssistanceKind.APPROVAL: AutonomyLevel.HUMAN_APPROVAL,
        AssistanceKind.CONFLICT_RESOLUTION: AutonomyLevel.HUMAN_CONFLICT_RESOLUTION,
        AssistanceKind.EMERGENCY_OVERRIDE: AutonomyLevel.EMERGENCY_OVERRIDE,
    }[kind]
