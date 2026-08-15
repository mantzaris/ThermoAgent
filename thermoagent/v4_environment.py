"""Three-application v4 environment with abstract utility restoration.

The utility application is a defensive logistics and service-continuity
simulation. It contains no protocol, device, credential, exploit, or real-world
target interaction.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .consensus import consensus_rmse, gossip_distributions_with_trace, local_consensus_residuals
from .events import EventLedger
from .tools import ToolRegistry
from .types import Message, PlanOutput, ToolResult
from .v4_agents import IndependentV4Agent, V4AgentUtility
from .v4_types import (
    InformationCondition,
    OperatorInterventionV4,
    ThermodynamicFeaturesV4,
    V4Application,
    V4Commitment,
    V4Identity,
    V4PrivateObservation,
    bounded_probability,
    jensen_shannon_disagreement,
    normalized_entropy,
)


ENERGY_WEIGHTS = {
    "service_deficit": 0.34,
    "backlog": 0.18,
    "lateness": 0.10,
    "unsafe_stress": 0.12,
    "failed_commitments": 0.10,
    "congestion": 0.08,
    "resource_scarcity": 0.08,
}


@dataclass
class IncidentState:
    incident_id: str
    location: Tuple[float, float]
    criticality: float
    true_mode: str
    resource_required: str
    disruption_step: int
    active: bool = False
    ambiguous: bool = False
    service_deficit: float = 0.0
    backlog: float = 0.0
    lateness: float = 0.0
    safety_stress: float = 0.0
    commitment_strain: float = 0.0
    congestion: float = 0.0
    verified: bool = False
    public_information: bool = False
    isolated: bool = False
    restored_fraction: float = 0.0
    visible_collapse: bool = False
    priority_multiplier: float = 1.0
    # Evaluator-side scenario parameter.  It controls how inconsistent the
    # private evidence is while leaving visible KPI severity unchanged.  It is
    # never included in a deployable operator view.
    belief_fragmentation: float = 0.0


@dataclass
class TransitRecord:
    action_id: str
    actor: str
    incident_id: str
    resource: str
    quantity: float
    accepted_step: int
    arrival_step: int
    correct_resource: bool
    stage: str = "accepted"
    intervention_id: Optional[str] = None


@dataclass(frozen=True)
class ApplicationProfile:
    application: str
    roles: Tuple[str, ...]
    resource_names: Tuple[str, ...]
    demand_label: str
    service_label: str
    utility_layered: bool


PROFILES: Dict[str, ApplicationProfile] = {
    V4Application.COMMERCIAL.value: ApplicationProfile(
        application=V4Application.COMMERCIAL.value,
        roles=(
            "supplier", "supplier", "manufacturer", "carrier", "warehouse",
            "retailer", "retailer", "coordinator", "carrier", "warehouse",
        ),
        resource_names=("standard_goods", "expedited_goods", "alternate_input"),
        demand_label="retailer demand",
        service_label="fulfillment",
        utility_layered=False,
    ),
    V4Application.HUMANITARIAN.value: ApplicationProfile(
        application=V4Application.HUMANITARIAN.value,
        roles=(
            "ngo", "agency", "depot", "transport", "clinic",
            "community", "community", "coordinator", "ngo", "transport",
        ),
        resource_names=("medical_kit", "water_supply", "shelter_supply"),
        demand_label="weighted unmet need",
        service_label="relief delivery",
        utility_layered=False,
    ),
    V4Application.UTILITY.value: ApplicationProfile(
        application=V4Application.UTILITY.value,
        roles=(
            "distribution_zone", "substation", "microgrid", "crew_dispatch",
            "parts_depot", "mobile_generation", "critical_load", "critical_load",
            "critical_load", "incident_coordinator",
        ),
        resource_names=("switch_module", "telemetry_module", "transformer_module"),
        demand_label="critical unserved load",
        service_label="restored critical service",
        utility_layered=True,
    ),
}


ROLE_PREFIX = {
    "supplier": "supplier",
    "manufacturer": "manufacturer",
    "carrier": "carrier",
    "warehouse": "warehouse",
    "retailer": "retailer",
    "coordinator": "coordinator",
    "ngo": "ngo",
    "agency": "agency",
    "depot": "depot",
    "transport": "transport",
    "clinic": "clinic",
    "community": "community",
    "distribution_zone": "zone",
    "substation": "substation",
    "microgrid": "microgrid",
    "crew_dispatch": "crew_dispatch",
    "parts_depot": "parts_depot",
    "mobile_generation": "mobile_generation",
    "critical_load": "critical_load",
    "incident_coordinator": "incident_coordinator",
}


class FragmentedOversightEnvironment:
    """Event-sourced environment shared by all v4 applications."""

    def __init__(
        self,
        application: str,
        regime: str,
        information_condition: str,
        seed: int,
        horizon: int = 20,
        disruption_step: int = 6,
        communication_enabled: bool = True,
    ) -> None:
        if application not in PROFILES:
            raise ValueError("unknown v4 application")
        if regime not in {
            "nominal", "isolated_physical", "telemetry_integrity", "partition",
            "correlated", "compound", "ood",
        }:
            raise ValueError("unknown v4 disruption regime")
        self.application = application
        self.profile = PROFILES[application]
        self.regime = regime
        self.information_condition = InformationCondition(information_condition)
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.disruption_step = int(disruption_step)
        self.communication_enabled = bool(communication_enabled)
        self.step_index = 0
        self.rng = np.random.RandomState(self.seed)
        self.ledger = EventLedger()
        self.registry = ToolRegistry()
        self.incidents = self._build_incidents()
        self.agents = self._build_agents()
        self.service_edges, self.communication_edges, self.logistics_edges = self._build_layers()
        self.initial_communication_edges = tuple(self.communication_edges)
        self.resources = self._initial_resources()
        self.initial_resources = deepcopy(self.resources)
        self.emergency_additions = {key: 0.0 for key in self.resources}
        self.consumed = {key: 0.0 for key in self.resources}
        self.transit: List[TransitRecord] = []
        self.completed_transit: List[TransitRecord] = []
        self.crew_assignments: Dict[str, str] = {}
        self.generator_assignments: Dict[str, str] = {}
        self.authorized_edges: Dict[Tuple[str, str], int] = {}
        self.coordinated_incidents = set()
        self.message_counter = 0
        self.action_counter = 0
        self.commitment_counter = 0
        self.metric_counters: Dict[str, float] = {
            "structured_attempts": 0,
            "first_pass_valid": 0,
            "valid_after_repair": 0,
            "material_actions_accepted": 0,
            "material_actions_next_stage": 0,
            "material_actions_reached_service": 0,
            "messages": 0,
            "message_bytes": 0,
            "thermodynamic_sketch_messages": 0,
            "thermodynamic_sketch_bytes": 0,
            "commitment_changes": 0,
            "tool_calls": 0,
        }
        self.loss_history: List[float] = []
        self.thermodynamic_history: Dict[str, List[ThermodynamicFeaturesV4]] = {
            incident_id: [] for incident_id in self.incidents
        }
        self._last_transition_state = self.state_payload()
        self.ledger.append(
            0,
            "topology_snapshot",
            "simulator",
            {
                "application": self.application,
                "abstract_defensive_simulation": True,
                "agents": {agent_id: asdict(agent.identity) for agent_id, agent in self.agents.items()},
                "service_edges": [list(edge) for edge in self.service_edges],
                "communication_edges": [list(edge) for edge in self.communication_edges],
                "logistics_edges": [list(edge) for edge in self.logistics_edges],
                "initial_state": self._last_transition_state,
                "initial_state_digest": self.state_digest(),
            },
        )

    def _build_incidents(self) -> Dict[str, IncidentState]:
        names = {
            V4Application.COMMERCIAL.value: ("retailer_north", "retailer_east", "retailer_south"),
            V4Application.HUMANITARIAN.value: ("clinic_region", "water_region", "shelter_region"),
            V4Application.UTILITY.value: ("hospital_zone", "water_zone", "communications_zone"),
        }[self.application]
        locations = ((-0.75, 0.55), (0.70, 0.60), (0.05, -0.78))
        criticalities = (1.0, 0.86, 0.74)
        resources = self.profile.resource_names
        incidents: Dict[str, IncidentState] = {}
        for index, name in enumerate(names):
            incidents[name] = IncidentState(
                incident_id=name,
                location=locations[index],
                criticality=criticalities[index],
                true_mode="none",
                resource_required=resources[index],
                disruption_step=self.disruption_step + (1 if index == 2 else 0),
            )
        return incidents

    def _build_agents(self) -> Dict[str, IndependentV4Agent]:
        agents: Dict[str, IndependentV4Agent] = {}
        incident_ids = tuple(self.incidents)
        for index, role in enumerate(self.profile.roles, start=1):
            prefix = ROLE_PREFIX[role]
            agent_id = "%s_%02d" % (prefix, index)
            angle = 2.0 * math.pi * (index - 1) / len(self.profile.roles)
            scope = (incident_ids[(index - 1) % len(incident_ids)],)
            identity = V4Identity(
                agent_id=agent_id,
                role=role,
                application=self.application,
                organization="org_%02d" % index,
                location=(float(math.cos(angle)), float(math.sin(angle))),
                incident_scope=scope,
            )
            utility = V4AgentUtility(
                service_weight=0.75 + 0.07 * (index % 4),
                cost_weight=0.18 + 0.05 * (index % 3),
                safety_weight=0.35 + 0.08 * (index % 2),
                disclosure_cost=0.04 + 0.03 * (index % 3),
                priority_weight=0.30 + 0.04 * (index % 5),
                reservation_value=0.65 + 0.06 * (index % 4),
            )
            agents[agent_id] = IndependentV4Agent(identity, utility, self.seed + 97 * index)
        return agents

    def _build_layers(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        ids = sorted(self.agents)
        ring = [(ids[index], ids[(index + 1) % len(ids)]) for index in range(len(ids))]
        chords = [(ids[index], ids[(index + 3) % len(ids)]) for index in range(0, len(ids), 2)]
        communication = sorted(set(tuple(sorted(edge)) for edge in ring + chords))
        if self.profile.utility_layered:
            service_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.identity.role in {"distribution_zone", "substation", "microgrid", "critical_load"}
            ]
            logistics_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.identity.role in {"crew_dispatch", "parts_depot", "mobile_generation", "incident_coordinator"}
            ]
            service = [tuple(sorted((service_agents[index], service_agents[index + 1]))) for index in range(len(service_agents) - 1)]
            logistics = [tuple(sorted((left, right))) for left in logistics_agents for right in service_agents if (sum(map(ord, left + right)) % 3 == 0)]
        else:
            demand_roles = {"retailer", "clinic", "community"}
            sources = [agent_id for agent_id, agent in self.agents.items() if agent.identity.role not in demand_roles]
            demands = [agent_id for agent_id, agent in self.agents.items() if agent.identity.role in demand_roles]
            service = [tuple(sorted((sources[index % len(sources)], demand))) for index, demand in enumerate(demands)]
            logistics = [tuple(sorted((source, demand))) for source in sources[:4] for demand in demands]
        return sorted(set(service)), communication, sorted(set(logistics))

    def _initial_resources(self) -> Dict[str, float]:
        if self.application == V4Application.UTILITY.value:
            return {
                "switch_module": 3.0,
                "telemetry_module": 3.0,
                "transformer_module": 3.0,
                "crew_units": 3.0,
                "crew_hours": 48.0,
                "fuel": 120.0,
                "mobile_generators": 2.0,
            }
        return {
            self.profile.resource_names[0]: 10.0,
            self.profile.resource_names[1]: 10.0,
            self.profile.resource_names[2]: 10.0,
            "transport_units": 4.0,
            "transport_hours": 56.0,
            "fuel": 100.0,
        }

    def _configure_disruption(self) -> None:
        if self.regime == "nominal":
            return
        active_count = 1 if self.regime in {"isolated_physical", "telemetry_integrity", "partition"} else 2 if self.regime == "correlated" else 3
        for index, incident in enumerate(self.incidents.values()):
            if index >= active_count:
                continue
            incident.active = True
            # Active incidents begin with matched visible KPI severity.  Their
            # private belief coherence differs, allowing a prospective test of
            # information value rather than a relabeling of severity.
            incident.service_deficit = 0.48
            incident.backlog = 0.42
            incident.lateness = 0.28
            incident.safety_stress = 0.38
            incident.commitment_strain = 0.30
            incident.congestion = 0.24
            if self.regime == "isolated_physical":
                incident.true_mode = "physical_failure"
            elif self.regime in {"telemetry_integrity", "partition"}:
                incident.true_mode = "telemetry_integrity_loss"
                incident.ambiguous = True
            elif self.regime == "correlated":
                incident.true_mode = "physical_failure" if index == 0 else "contradictory_telemetry"
                incident.ambiguous = index == 1
            else:
                modes = ("cyber_physical_compound", "resource_database_inconsistency", "command_unavailable")
                incident.true_mode = modes[index]
                incident.ambiguous = index != 1 or self.regime == "ood"
            if incident.ambiguous:
                # Sequential panel seeds are deliberately balanced across the
                # four fragmentation levels.  The incident-specific phase
                # shift prevents every simultaneous incident from sharing one
                # coherence state while remaining independent of outcomes.
                phase_token = "%s|%s|fragmentation-phase-v1" % (
                    self.application, incident.incident_id
                )
                phase = int(hashlib.sha256(phase_token.encode("utf-8")).hexdigest()[:8], 16) % 4
                bucket = (self.seed + phase) % 4
                incident.belief_fragmentation = (0.12, 0.42, 0.74, 1.0)[bucket]
            if self.information_condition == InformationCondition.GLOBALLY_PUBLIC:
                incident.public_information = True
                incident.verified = True
        if self.regime in {"partition", "compound", "ood"}:
            midpoint = len(self.agents) // 2
            left = set(sorted(self.agents)[:midpoint])
            self.communication_edges = [
                edge for edge in self.communication_edges
                if (edge[0] in left) == (edge[1] in left)
            ]
        self.ledger.append(
            self.step_index,
            "disruption",
            "simulator",
            {
                "regime": self.regime,
                "abstract_only": True,
                "active_incidents": [
                    {
                        "incident_id": value.incident_id,
                        "visible_service_deficit": value.service_deficit,
                        "event_class": value.true_mode,
                    }
                    for value in self.incidents.values() if value.active
                ],
                "communication_partition": self.regime in {"partition", "compound", "ood"},
            },
        )

    def _agent_incident(self, agent: IndependentV4Agent) -> IncidentState:
        scoped = [self.incidents[value] for value in agent.identity.incident_scope]
        return max(scoped, key=lambda incident: (incident.service_deficit * incident.criticality, incident.incident_id))

    def private_observation(self, agent_id: str) -> V4PrivateObservation:
        agent = self.agents[agent_id]
        incident = self._agent_incident(agent)
        index = sorted(self.agents).index(agent_id)
        if not incident.active:
            belief_operational, belief_unreliable, confidence = 0.86, 0.05, 0.90
        elif incident.verified or incident.public_information or not incident.ambiguous:
            belief_operational, belief_unreliable, confidence = 0.08, 0.08, 0.90
        else:
            # Panels have matched visible service severity but prospectively
            # varied coherence of private evidence.  This is the intended
            # identification contrast: KPIs describe magnitude, whereas the
            # distributed belief sketch describes fragmentation.
            patterns = ((0.86, 0.08, 0.30), (0.05, 0.88, 0.32), (0.06, 0.10, 0.36))
            incident_observers = sorted(
                value.agent_id for value in self.agents.values()
                if self._agent_incident(value).incident_id == incident.incident_id
            )
            pattern_index = incident_observers.index(agent_id) % len(patterns)
            discordant_operational, discordant_unreliable, discordant_confidence = patterns[pattern_index]
            fragmentation = incident.belief_fragmentation
            belief_operational = (1.0 - fragmentation) * 0.05 + fragmentation * discordant_operational
            belief_unreliable = (1.0 - fragmentation) * 0.88 + fragmentation * discordant_unreliable
            confidence = (1.0 - fragmentation) * 0.78 + fragmentation * discordant_confidence
        local_noise = 0.0
        actionability = 0.5
        if self.application == V4Application.COMMERCIAL.value and incident.active:
            # Commercial local contracts/routes expose most actionable state;
            # this is the prospectively specified boundary condition.
            actionability = 0.92 if incident.ambiguous else 0.72
        observation = V4PrivateObservation(
            step=self.step_index,
            incident_id=incident.incident_id,
            local_service_deficit=bounded_probability(incident.service_deficit + local_noise),
            local_backlog=bounded_probability(incident.backlog + local_noise),
            local_lateness=bounded_probability(incident.lateness),
            local_resource_scarcity=self.resource_scarcity(incident.resource_required),
            local_commitment_strain=bounded_probability(incident.commitment_strain),
            local_safety_stress=bounded_probability(incident.safety_stress),
            local_disruption_risk=bounded_probability(0.15 + 0.75 * incident.service_deficit),
            local_actionability_flag=actionability,
            belief_operational=belief_operational,
            belief_telemetry_unreliable=belief_unreliable,
            telemetry_confidence=confidence,
            observed_resource_available=bounded_probability(self.resources.get(incident.resource_required, 0.0) / 3.0),
            communication_reliability=1.0 if self._agent_degree(agent_id) > 0 else 0.10,
            private_cost=0.45 + 0.04 * (index % 5),
            private_priority=bounded_probability(incident.criticality + 0.03 * ((index % 2) - 0.5)),
            authorized_actions=tuple(sorted(self.registry.allowed(agent.identity.role))),
        )
        return observation

    def deliver_observations(self) -> None:
        for agent_id, agent in self.agents.items():
            observation = self.private_observation(agent_id)
            agent.deliver_observation(observation, self.ledger)
            if self.application == V4Application.UTILITY.value:
                self.ledger.append(
                    self.step_index,
                    "telemetry_observation",
                    "abstract_telemetry_layer",
                    {
                        "recipient": agent_id,
                        "incident_id": observation.incident_id,
                        "confidence": observation.telemetry_confidence,
                        "status": "available" if observation.telemetry_confidence >= 0.45 else "uncertain",
                    },
                    private_to=agent_id,
                )

    def _agent_degree(self, agent_id: str) -> int:
        return sum(agent_id in edge for edge in self.communication_edges)

    def _communication_component(self, agent_id: str) -> set:
        component = {agent_id}
        changed = True
        while changed:
            changed = False
            for left, right in self.communication_edges:
                if left in component and right not in component:
                    component.add(right)
                    changed = True
                elif right in component and left not in component:
                    component.add(left)
                    changed = True
        return component

    def resource_scarcity(self, resource: str) -> float:
        available = float(self.resources.get(resource, 0.0))
        return bounded_probability(1.0 - available / 3.0)

    def exchange_sketches(self, gossip_rounds: int = 3) -> Dict[str, ThermodynamicFeaturesV4]:
        sketches = {agent_id: agent.coarse_sketch() for agent_id, agent in self.agents.items()}
        output: Dict[str, ThermodynamicFeaturesV4] = {}
        for sketch in sketches.values():
            encoded = json.dumps(sketch, sort_keys=True, separators=(",", ":"))
            self.ledger.append(self.step_index, "thermodynamic_sketch", sketch["agent_id"], sketch)
            recipients = self._agent_degree(sketch["agent_id"]) if self.communication_enabled else 0
            self.metric_counters["thermodynamic_sketch_messages"] += recipients
            self.metric_counters["thermodynamic_sketch_bytes"] += len(encoded.encode("utf-8")) * recipients
        for incident_id, incident in self.incidents.items():
            scoped = {
                agent_id: agent.private_beliefs[agent.vault.observation(agent_id).incident_id]
                for agent_id, agent in self.agents.items()
                if agent.vault.observation(agent_id).incident_id == incident_id
            }
            if not scoped:
                continue
            edges = [edge for edge in self.communication_edges if edge[0] in scoped and edge[1] in scoped]
            rounds = [edges if self.communication_enabled else [] for _ in range(gossip_rounds)]
            estimates, trace = gossip_distributions_with_trace(scoped, rounds)
            exact = np.mean(np.asarray(list(scoped.values()), dtype=float), axis=0)
            exact /= exact.sum()
            residuals = local_consensus_residuals(estimates, edges if self.communication_enabled else [])
            exact_rmse = consensus_rmse(estimates, exact)
            # Disagreement is computed over the contributed local belief
            # sketches before consensus averaging.  Computing it over the
            # post-gossip estimates would erase the very conflict the
            # operator must triage while still reporting a precise consensus.
            js = jensen_shannon_disagreement(list(scoped.values()))
            local_entropy = float(np.mean([normalized_entropy(value) for value in estimates.values()]))
            distributed_entropy = normalized_entropy(np.mean(np.asarray(list(estimates.values())), axis=0))
            energy = self.operational_energy(incident)
            energy_center, energy_scale = 0.10, 0.05
            entropy_center, entropy_scale = 0.32, 0.08
            previous = self.thermodynamic_history[incident_id][-1] if self.thermodynamic_history[incident_id] else None
            entropy_residual = (distributed_entropy - entropy_center) / entropy_scale
            slope = 0.0 if previous is None else distributed_entropy - previous.distributed_entropy
            acceleration = 0.0 if previous is None else slope - previous.entropy_slope
            confidence = bounded_probability(
                np.mean([1.0 - min(1.0, value) for value in residuals.values()])
                * (1.0 if edges else 0.35)
            )
            temperature = bounded_probability(0.15 + 0.65 * incident.lateness + 0.20 * js)
            free_energy = energy - temperature * distributed_entropy
            features = ThermodynamicFeaturesV4(
                raw_service_deficit=incident.service_deficit,
                raw_backlog=incident.backlog,
                raw_lateness=incident.lateness,
                raw_safety_stress=incident.safety_stress,
                raw_commitment_strain=incident.commitment_strain,
                raw_resource_scarcity=self.resource_scarcity(incident.resource_required),
                operational_energy=energy,
                standardized_energy=(energy - energy_center) / energy_scale,
                belief_entropy=local_entropy,
                alternative_entropy=normalized_entropy((
                    max(0.01, 1.0 - incident.congestion),
                    max(0.01, incident.congestion),
                    max(0.01, self.resource_scarcity(incident.resource_required)),
                )),
                commitment_entropy=normalized_entropy((
                    max(0.01, 1.0 - incident.commitment_strain),
                    max(0.01, incident.commitment_strain),
                )),
                distributed_entropy=distributed_entropy,
                entropy_residual=entropy_residual,
                entropy_anomaly=abs(entropy_residual),
                entropy_slope=slope,
                entropy_acceleration=acceleration,
                belief_disagreement=js,
                consensus_confidence=confidence,
                consensus_error=exact_rmse,
                temperature_diagnostic=temperature,
                free_energy_diagnostic=free_energy,
                free_energy_residual=(free_energy - (energy_center - 0.25 * entropy_center)) / 0.08,
                sketch_contributors=len(scoped),
                sketch_messages=int(len(edges) * 2 * gossip_rounds) if self.communication_enabled else 0,
                sketch_bytes=int(len(scoped) * 96 * gossip_rounds) if self.communication_enabled else 0,
            )
            self.thermodynamic_history[incident_id].append(features)
            for agent_id in scoped:
                self.ledger.append(
                    self.step_index,
                    "thermodynamic_state",
                    agent_id,
                    {
                        "incident_id": incident_id,
                        "distributed_estimate": estimates[agent_id].tolist(),
                        "features": features.as_dict(),
                        "trace_rounds": len(trace),
                        "privacy_boundary": "coarse sketches only",
                    },
                    private_to=agent_id,
                )
            output[incident_id] = features
        return output

    def operational_energy(self, incident: IncidentState) -> float:
        values = {
            "service_deficit": incident.service_deficit,
            "backlog": incident.backlog,
            "lateness": incident.lateness,
            "unsafe_stress": incident.safety_stress,
            "failed_commitments": incident.commitment_strain,
            "congestion": incident.congestion,
            "resource_scarcity": self.resource_scarcity(incident.resource_required),
        }
        return float(sum(ENERGY_WEIGHTS[key] * values[key] for key in ENERGY_WEIGHTS))

    def send_message(self, sender: str, recipient: str, kind: str, payload: Mapping[str, Any]) -> bool:
        if sender not in self.agents or recipient not in self.agents:
            return False
        if not self.communication_enabled or tuple(sorted((sender, recipient))) not in self.communication_edges:
            return False
        self.message_counter += 1
        message = Message(
            message_id="V4M%07d" % self.message_counter,
            sender=sender,
            recipient=recipient,
            kind=kind,
            payload=dict(payload),
            sent_step=self.step_index,
            deliver_step=self.step_index,
        )
        self.agents[recipient].receive_message(message)
        encoded = json.dumps(asdict(message), sort_keys=True, separators=(",", ":"))
        self.metric_counters["messages"] += 1
        self.metric_counters["message_bytes"] += len(encoded.encode("utf-8"))
        self.ledger.append(self.step_index, "message", sender, asdict(message))
        self.ledger.append(self.step_index, "message_delivery", "communication_network", asdict(message), private_to=recipient)
        return True

    def validate_and_execute_plan(
        self,
        agent_id: str,
        plan: PlanOutput,
        repair_plan: Optional[PlanOutput] = None,
        intervention_id: Optional[str] = None,
    ) -> ToolResult:
        agent = self.agents[agent_id]
        self.metric_counters["structured_attempts"] += 1
        validation = self.registry.validate(agent.identity.role, plan)
        first_pass = validation.ok
        if first_pass:
            self.metric_counters["first_pass_valid"] += 1
        elif repair_plan is not None:
            repaired = self.registry.validate(agent.identity.role, repair_plan)
            if repaired.ok:
                plan = repair_plan
                validation = repaired
        if validation.ok:
            self.metric_counters["valid_after_repair"] += 1
        self.metric_counters["tool_calls"] += 1
        self.ledger.append(
            self.step_index,
            "tool_call",
            agent_id,
            {"plan": plan.as_dict(), "first_pass_valid": first_pass, "repair_used": not first_pass and repair_plan is not None},
            private_to=agent_id,
        )
        if not validation.ok:
            self.ledger.append(self.step_index, "tool_result", "simulator", validation.as_dict(), private_to=agent_id)
            agent.reflect(self.step_index, plan, validation)
            return validation
        result = self._execute_semantics(agent_id, plan, intervention_id)
        self.ledger.append(self.step_index, "tool_result", "simulator", result.as_dict(), private_to=agent_id)
        agent.reflect(self.step_index, plan, result)
        return result

    def _execute_semantics(self, agent_id: str, plan: PlanOutput, intervention_id: Optional[str]) -> ToolResult:
        observation = self.agents[agent_id].vault.observation(agent_id)
        incident_id = str(plan.arguments.get("target_zone", observation.incident_id))
        incident = self.incidents.get(incident_id)
        if incident is None:
            return ToolResult(False, "unknown_incident", "target is not a simulated incident")
        if plan.tool in {"request_telemetry_verification", "request_cross_agent_consensus", "request_info", "disclose_summary", "report_local_need", "request_priority", "request_emergency_logistics_edge", "defer_due_to_uncertainty", "no_op"}:
            return ToolResult(True, "bounded_nonmaterial_action", "validated information or abstention action", {"incident_id": incident_id})
        resource = incident.resource_required
        quantity = 1.0
        if plan.tool == "allocate_spare_component":
            resource = str(plan.arguments["component"])
            quantity = float(plan.arguments["quantity"])
        elif plan.tool in {"schedule_shipment", "transfer_resource"}:
            resource = incident.resource_required
            quantity = min(float(plan.arguments["quantity"]), 2.0)
        elif plan.tool == "route_mobile_generator":
            resource = "mobile_generators"
        elif plan.tool == "allocate_fuel":
            resource = "fuel"
            quantity = float(plan.arguments["quantity"])
        elif plan.tool == "dispatch_field_crew":
            resource = "crew_units"
        elif plan.tool == "prioritize_critical_load":
            incident.priority_multiplier = 1.15 if plan.arguments["priority"] == "critical" else 1.05
            self.metric_counters["commitment_changes"] += 1
            return ToolResult(True, "priority_commitment_changed", "critical-load priority changed within authority", {"incident_id": incident_id})
        elif plan.tool == "temporarily_isolate_service_zone":
            incident.isolated = True
            return ToolResult(True, "abstract_zone_isolated", "simulated service zone isolated", {"incident_id": incident_id})
        elif plan.tool in {"negotiate_repair_commitment", "revise_resource_commitment", "accept_offer", "reject_offer", "counter_offer", "propose_coalition", "join_coalition", "refuse_coalition", "withdraw_coalition"}:
            self.metric_counters["commitment_changes"] += 1
            return ToolResult(True, "commitment_state_changed", "independent commitment authority exercised", {"incident_id": incident_id})
        if resource not in self.resources:
            return ToolResult(False, "resource_unavailable", "resource type is unavailable in this application", {"resource": resource})
        if self.resources[resource] + 1e-12 < quantity:
            return ToolResult(False, "insufficient_resource", "available resource is below requested quantity", {"resource": resource, "available": self.resources[resource]})
        if resource == "crew_units" and plan.arguments.get("crew_id") in self.crew_assignments:
            return ToolResult(False, "crew_already_assigned", "a crew cannot be assigned twice", {"crew_id": plan.arguments.get("crew_id")})
        if resource == "mobile_generators" and plan.arguments.get("generator_id") in self.generator_assignments:
            return ToolResult(False, "generator_already_assigned", "a generator cannot be assigned twice", {"generator_id": plan.arguments.get("generator_id")})
        self.resources[resource] -= quantity
        self.action_counter += 1
        action_id = "V4A%07d" % self.action_counter
        if resource == "crew_units":
            self.crew_assignments[str(plan.arguments.get("crew_id"))] = incident_id
        if resource == "mobile_generators":
            self.generator_assignments[str(plan.arguments.get("generator_id"))] = incident_id
        known_correct = incident.verified or incident.public_information or not incident.ambiguous
        proposed_resource = resource
        correct = known_correct or proposed_resource == incident.resource_required
        if incident.ambiguous and not known_correct:
            private = self.agents[agent_id].private_beliefs[incident_id]
            recognized_disruption = int(np.argmax(private)) != 0
            if incident_id in self.coordinated_incidents:
                component = self._communication_component(agent_id)
                peer_beliefs = [
                    value.private_beliefs[incident_id]
                    for value in self.agents.values()
                    if value.agent_id in component and incident_id in value.private_beliefs
                ]
                recognized_disruption = any(int(np.argmax(value)) != 0 for value in peer_beliefs)
            supports_restoration = proposed_resource in {"crew_units", "mobile_generators", "fuel"}
            # Peer exchange improves action selection but cannot fully resolve
            # potentially corrupted telemetry; this leaves a legitimate role
            # for bounded verification rather than treating communication as
            # an oracle.
            if incident_id in self.coordinated_incidents:
                disagreement = jensen_shannon_disagreement(peer_beliefs)
                # Coordinated inference is reliable when private reports are
                # coherent and fails sharply as mutually inconsistent reports
                # accumulate.  This smooth, fixed response curve creates the
                # prospective matched-KPI mechanism test; it does not use an
                # evaluator disruption label.
                if disagreement <= 0.02:
                    probability = 1.0
                elif disagreement >= 0.06:
                    probability = 0.0
                else:
                    probability = (0.06 - disagreement) / 0.04
                if self.application == V4Application.COMMERCIAL.value:
                    # Pre-specified boundary application: ordinary local
                    # contracts and route evidence remain highly actionable.
                    probability = max(0.85, probability)
            else:
                probability = 0.20
            token = "%d|%s|%s|%s" % (self.seed, incident_id, agent_id, "coordinated" if incident_id in self.coordinated_incidents else "local")
            draw = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) / float(16 ** 8)
            correct = bool(recognized_disruption and draw < probability)
            if not supports_restoration and proposed_resource != incident.resource_required:
                correct = False
        arrival = self.step_index + (1 if self._edge_authorized(agent_id, incident_id) else 2)
        transit = TransitRecord(
            action_id=action_id,
            actor=agent_id,
            incident_id=incident_id,
            resource=resource,
            quantity=quantity,
            accepted_step=self.step_index,
            arrival_step=arrival,
            correct_resource=correct,
            intervention_id=intervention_id,
        )
        self.transit.append(transit)
        self.metric_counters["material_actions_accepted"] += 1
        self.ledger.append(self.step_index, "restoration_action", agent_id, asdict(transit))
        self.ledger.append(self.step_index, "resource_assignment", agent_id, {
            "action_id": action_id, "resource": resource, "quantity": quantity,
            "incident_id": incident_id, "feasible": True,
        })
        return ToolResult(True, "material_action_accepted", "resource entered the simulated logistics pipeline", asdict(transit))

    def _edge_authorized(self, source: str, incident_id: str) -> bool:
        return any(
            source == left and incident_id == right and expires >= self.step_index
            for (left, right), expires in self.authorized_edges.items()
        )

    def apply_operator_intervention(self, intervention: OperatorInterventionV4) -> ToolResult:
        if intervention.incident_id not in self.incidents:
            return ToolResult(False, "unknown_incident", "operator intervention target is unknown")
        incident = self.incidents[intervention.incident_id]
        action = intervention.action
        if action in {"authorize_verification", "authorize_information_sharing", "resolve_conflicting_reports"}:
            incident.verified = True
            incident.public_information = action == "authorize_information_sharing"
            self.metric_counters["commitment_changes"] += 1
            self.ledger.append(self.step_index, "telemetry_verification", "simulated_operator", {
                "incident_id": incident.incident_id,
                "action": action,
                "abstract_only": True,
                "result": "authorized evidence reconciliation",
            })
        elif action == "authorize_emergency_resource":
            resource = incident.resource_required
            self.resources[resource] += 1.0
            self.emergency_additions[resource] += 1.0
        elif action == "authorize_emergency_logistics_edge":
            self.authorized_edges[(intervention.target_agent, incident.incident_id)] = self.step_index + 5
        elif action == "adjust_priority":
            incident.priority_multiplier = min(1.35, incident.priority_multiplier + 0.20)
            self.metric_counters["commitment_changes"] += 1
        elif action == "temporary_override":
            incident.verified = True
            self.authorized_edges[(intervention.target_agent, incident.incident_id)] = self.step_index + 3
            self.metric_counters["commitment_changes"] += 1
        elif action == "abstain":
            return ToolResult(True, "operator_abstained", "operator retained scarce attention", {"incident_id": incident.incident_id})
        else:
            return ToolResult(False, "operator_action_not_allowed", "intervention is outside bounded authority")
        if intervention.target_agent in self.agents:
            self.commitment_counter += 1
            commitment = V4Commitment(
                commitment_id="V4C%07d" % self.commitment_counter,
                proposer="simulated_operator",
                recipient=intervention.target_agent,
                incident_id=incident.incident_id,
                resource=incident.resource_required,
                quantity=1.0,
                due_step=self.step_index + max(1, intervention.service_steps),
                status="accepted",
            )
            self.agents[intervention.target_agent].commitments[commitment.commitment_id] = commitment
            self.metric_counters["commitment_changes"] += 1
            self.ledger.append(
                self.step_index,
                "commitment",
                intervention.target_agent,
                {**asdict(commitment), "source": "bounded_operator_directive"},
            )
        self.ledger.append(self.step_index, "operator_action", "simulated_operator", intervention.as_dict())
        return ToolResult(True, "bounded_intervention_applied", "authorized information or feasible action space changed", {"incident_id": incident.incident_id, "action": action})

    def _progress_transit(self) -> None:
        remaining: List[TransitRecord] = []
        for record in self.transit:
            if record.arrival_step > self.step_index:
                remaining.append(record)
                continue
            record.stage = "next_stage"
            self.metric_counters["material_actions_next_stage"] += 1
            incident = self.incidents[record.incident_id]
            reached = False
            if record.correct_resource:
                gain = 0.42 * record.quantity
                if record.resource == "mobile_generators":
                    gain = 0.34
                elif record.resource == "crew_units":
                    gain = 0.22
                elif record.resource == "fuel":
                    gain = 0.08 * min(record.quantity, 3.0)
                    self.consumed["fuel"] += record.quantity
                incident.restored_fraction = bounded_probability(incident.restored_fraction + gain)
                incident.service_deficit = bounded_probability(incident.service_deficit - gain)
                incident.backlog = bounded_probability(incident.backlog - 0.45 * gain)
                incident.commitment_strain = bounded_probability(incident.commitment_strain - 0.25 * gain)
                reached = True
                record.stage = "reached_service"
                self.metric_counters["material_actions_reached_service"] += 1
            self.completed_transit.append(record)
            self.ledger.append(self.step_index, "material_progress", "simulator", {
                **asdict(record),
                "reached_demand_or_critical_service": reached,
            })
            self.ledger.append(self.step_index, "service_transition", "simulator", {
                "incident_id": incident.incident_id,
                "action_id": record.action_id,
                "reached_service": reached,
                "service_deficit_after": incident.service_deficit,
                "restored_fraction": incident.restored_fraction,
            })
        self.transit = remaining

    def _natural_dynamics(self) -> None:
        for incident in self.incidents.values():
            if not incident.active:
                continue
            if incident.service_deficit > 0.02:
                incident.backlog = bounded_probability(incident.backlog + 0.035 * incident.criticality)
                incident.lateness = bounded_probability(incident.lateness + 0.025)
                incident.commitment_strain = bounded_probability(incident.commitment_strain + 0.018)
            incident.visible_collapse = incident.service_deficit >= 0.58 and incident.backlog >= 0.52

    def current_loss(self) -> float:
        return float(sum(
            incident.criticality * incident.priority_multiplier
            * (0.68 * incident.service_deficit + 0.32 * incident.backlog)
            for incident in self.incidents.values()
        ))

    def step(self) -> Dict[str, Any]:
        before = self.state_payload()
        if self.step_index == self.disruption_step:
            self._configure_disruption()
        self._progress_transit()
        self._natural_dynamics()
        loss = self.current_loss()
        self.loss_history.append(loss)
        after = self.state_payload()
        self.ledger.append(self.step_index, "v4_state_transition", "simulator", {
            "before": before,
            "before_digest": self._payload_digest(before),
            "after": after,
            "after_digest": self._payload_digest(after),
            "loss": loss,
            "conservation": self.conservation_report(),
            "rng_digest": self.rng_digest(),
        })
        self.ledger.append(self.step_index, "environment_transition", "simulator", {
            "state_digest": self._payload_digest(after),
            "loss": loss,
            "v4": True,
        })
        self._last_transition_state = after
        self.step_index += 1
        return {"step": self.step_index - 1, "loss": loss, "state_digest": self._payload_digest(after)}

    def automated_response(
        self,
        incident_id: str,
        intervention_id: Optional[str] = None,
        coordinated: bool = False,
    ) -> ToolResult:
        incident = self.incidents[incident_id]
        candidates = [
            agent for agent in self.agents.values()
            if incident_id in agent.identity.incident_scope
            and agent.vault.observation(agent.agent_id).incident_id == incident_id
            and agent.identity.role in {
                "supplier", "manufacturer", "warehouse", "ngo", "agency", "depot",
                "crew_dispatch", "parts_depot", "mobile_generation",
            }
        ]
        if not candidates:
            candidates = [next(iter(self.agents.values()))]
        if coordinated:
            self.coordinated_incidents.add(incident_id)
            actor = max(
                candidates,
                key=lambda value: 1.0 - value.private_beliefs.get(incident_id, [1.0])[0],
            )
            for candidate in candidates:
                if candidate.agent_id != actor.agent_id:
                    self.send_message(
                        candidate.agent_id,
                        actor.agent_id,
                        "bounded_incident_summary",
                        {"incident_id": incident_id, "belief_band": candidate.coarse_sketch()["severity_band"]},
                    )
        else:
            actor = candidates[0]
        if self.application == V4Application.UTILITY.value:
            if actor.identity.role == "crew_dispatch":
                plan = PlanOutput(
                    "Dispatch an available simulated field crew after local assessment.",
                    "dispatch_field_crew",
                    {"crew_id": "crew_%d" % (1 + len(self.crew_assignments)), "target_zone": incident_id, "skill": "electrical"},
                    "Independent local action under bounded authority.",
                    0.8,
                )
            elif actor.identity.role == "mobile_generation":
                plan = PlanOutput(
                    "Route one available mobile generator to protect critical service.",
                    "route_mobile_generator",
                    {
                        "generator_id": "generator_%d" % (1 + len(self.generator_assignments)),
                        "target_zone": incident_id,
                    },
                    "Independent critical-service support under local authority.",
                    0.77,
                )
            else:
                if incident.verified or incident.public_information:
                    component = incident.resource_required
                else:
                    belief_class = int(np.argmax(actor.private_beliefs[incident_id]))
                    # The inferred class, not the evaluator truth, determines
                    # the spare choice: operational fallback, physical repair,
                    # or telemetry replacement respectively.
                    component = {
                        0: "switch_module",
                        1: "transformer_module",
                        2: "telemetry_module",
                    }[belief_class]
                plan = PlanOutput(
                    "Allocate one locally selected restoration component.",
                    "allocate_spare_component",
                    {"component": component, "quantity": 1, "target_zone": incident_id},
                    "Choice uses only the actor's current private/authorized information.",
                    0.78,
                )
        elif self.application == V4Application.HUMANITARIAN.value:
            plan = PlanOutput(
                "Transfer one relief resource to the locally selected affected region.",
                "transfer_resource",
                {"target": incident_id, "quantity": 1.0, "arrival_step": self.step_index + 2},
                "Independent allocation based on current authorized need evidence.",
                0.78,
            )
        else:
            plan = PlanOutput(
                "Schedule one bounded shipment to the selected retailer incident.",
                "schedule_shipment",
                {"target": incident_id, "quantity": 1.0, "arrival_step": self.step_index + 2},
                "Independent shipment within local inventory authority.",
                0.78,
            )
        return self.validate_and_execute_plan(actor.agent_id, plan, intervention_id=intervention_id)

    def state_payload(self) -> Dict[str, Any]:
        return {
            "step": self.step_index,
            "application": self.application,
            "regime": self.regime,
            "information_condition": self.information_condition.value,
            "incidents": {key: asdict(value) for key, value in sorted(self.incidents.items())},
            "resources": {key: float(value) for key, value in sorted(self.resources.items())},
            "emergency_additions": {key: float(value) for key, value in sorted(self.emergency_additions.items())},
            "consumed": {key: float(value) for key, value in sorted(self.consumed.items())},
            "transit": [asdict(value) for value in sorted(self.transit, key=lambda row: row.action_id)],
            "completed_transit": [asdict(value) for value in sorted(self.completed_transit, key=lambda row: row.action_id)],
            "crew_assignments": dict(sorted(self.crew_assignments.items())),
            "generator_assignments": dict(sorted(self.generator_assignments.items())),
            "authorized_edges": [
                [left, right, expires]
                for (left, right), expires in sorted(self.authorized_edges.items())
            ],
            "coordinated_incidents": sorted(self.coordinated_incidents),
            "communication_edges": [list(edge) for edge in sorted(self.communication_edges)],
            "metric_counters": {key: float(value) for key, value in sorted(self.metric_counters.items())},
        }

    @staticmethod
    def _payload_digest(payload: Mapping[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def state_digest(self) -> str:
        return self._payload_digest(self.state_payload())

    def rng_digest(self) -> str:
        state = self.rng.get_state()
        payload = {
            "name": state[0],
            "keys": state[1].tolist(),
            "position": int(state[2]),
            "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        }
        return self._payload_digest(payload)

    def conservation_report(self) -> Dict[str, Any]:
        in_transit: Dict[str, float] = {key: 0.0 for key in self.resources}
        deployed: Dict[str, float] = {key: 0.0 for key in self.resources}
        for record in self.transit:
            in_transit[record.resource] = in_transit.get(record.resource, 0.0) + record.quantity
        for record in self.completed_transit:
            if record.resource != "fuel":
                deployed[record.resource] = deployed.get(record.resource, 0.0) + record.quantity
        residuals: Dict[str, float] = {}
        for resource, initial in self.initial_resources.items():
            residuals[resource] = (
                initial + self.emergency_additions.get(resource, 0.0)
                - self.resources.get(resource, 0.0)
                - in_transit.get(resource, 0.0)
                - deployed.get(resource, 0.0)
                - self.consumed.get(resource, 0.0)
            )
        max_residual = max((abs(value) for value in residuals.values()), default=0.0)
        impossible_assignments = len(self.crew_assignments) > int(self.initial_resources.get("crew_units", 10 ** 6))
        impossible_assignments = impossible_assignments or len(self.generator_assignments) > int(self.initial_resources.get("mobile_generators", 10 ** 6))
        negative_inventory = any(value < -1e-12 for value in self.resources.values())
        return {
            "residuals": residuals,
            "maximum_residual": float(max_residual),
            "negative_inventory": bool(negative_inventory),
            "impossible_assignment": bool(impossible_assignments),
            "feasible": not negative_inventory and not impossible_assignments and max_residual <= 1e-9,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "incidents": deepcopy(self.incidents),
            "resources": deepcopy(self.resources),
            "emergency_additions": deepcopy(self.emergency_additions),
            "consumed": deepcopy(self.consumed),
            "transit": deepcopy(self.transit),
            "completed_transit": deepcopy(self.completed_transit),
            "crew_assignments": deepcopy(self.crew_assignments),
            "generator_assignments": deepcopy(self.generator_assignments),
            "authorized_edges": deepcopy(self.authorized_edges),
            "coordinated_incidents": deepcopy(self.coordinated_incidents),
            "communication_edges": deepcopy(self.communication_edges),
            "metric_counters": deepcopy(self.metric_counters),
            "loss_history": deepcopy(self.loss_history),
            "rng_state": deepcopy(self.rng.get_state()),
            "agents": deepcopy(self.agents),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.step_index = int(snapshot["step_index"])
        self.incidents = deepcopy(snapshot["incidents"])
        self.resources = deepcopy(snapshot["resources"])
        self.emergency_additions = deepcopy(snapshot["emergency_additions"])
        self.consumed = deepcopy(snapshot["consumed"])
        self.transit = deepcopy(snapshot["transit"])
        self.completed_transit = deepcopy(snapshot["completed_transit"])
        self.crew_assignments = deepcopy(snapshot["crew_assignments"])
        self.generator_assignments = deepcopy(snapshot["generator_assignments"])
        self.authorized_edges = deepcopy(snapshot["authorized_edges"])
        self.coordinated_incidents = deepcopy(snapshot["coordinated_incidents"])
        self.communication_edges = deepcopy(snapshot["communication_edges"])
        self.metric_counters = deepcopy(snapshot["metric_counters"])
        self.loss_history = deepcopy(snapshot["loss_history"])
        self.rng.set_state(deepcopy(snapshot["rng_state"]))
        self.agents = deepcopy(snapshot["agents"])


class UtilityRestorationEnvironment(FragmentedOversightEnvironment):
    """Explicit defensive utility-restoration application entry point."""

    def __init__(
        self,
        regime: str,
        information_condition: str,
        seed: int,
        horizon: int = 20,
        disruption_step: int = 6,
        communication_enabled: bool = True,
    ) -> None:
        super().__init__(
            V4Application.UTILITY.value,
            regime,
            information_condition,
            seed,
            horizon,
            disruption_step,
            communication_enabled,
        )
