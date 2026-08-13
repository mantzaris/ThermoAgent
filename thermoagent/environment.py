"""Deterministic multi-echelon logistics simulators shared by both applications."""

from __future__ import annotations

import math
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .agents import AutonomousAgent
from .events import EventLedger
from .types import (
    Application,
    Commitment,
    Identity,
    Message,
    PrivateObservation,
    Shipment,
    ToolResult,
    UtilityWeights,
)


COMMERCIAL_ROLES = [
    "supplier", "supplier", "manufacturer", "manufacturer", "carrier", "carrier",
    "warehouse", "warehouse", "retailer", "retailer", "retailer",
]
HUMANITARIAN_ROLES = [
    "ngo", "ngo", "agency", "transport", "transport", "depot", "depot", "clinic", "clinic", "community",
]
DEMAND_ROLES = {"retailer", "clinic", "community"}
# Transport organizations hold an explicit bounded operating stock in this
# abstract model, so they need executable outbound arcs just like other
# resource owners. Production remains restricted by PRODUCTION_ROLES.
SOURCE_ROLES = {
    "supplier", "manufacturer", "carrier", "warehouse",
    "ngo", "agency", "transport", "depot",
}
PRODUCTION_ROLES = {"supplier", "manufacturer", "ngo", "agency"}
RNG_STREAM_OFFSETS = {
    "initialization": 0,
    "exogenous_dynamics": 1_000_003,
    "observation_noise": 2_000_003,
    "communication": 3_000_017,
    "monitor_link_sampling": 4_000_037,
    "monitor_noise": 5_000_011,
}


def derived_rng_seed(base_seed: int, stream: str) -> int:
    if stream not in RNG_STREAM_OFFSETS:
        raise ValueError("unknown RNG stream: %s" % stream)
    return (int(base_seed) + RNG_STREAM_OFFSETS[stream]) % (2 ** 32)


@dataclass
class ScenarioConfig:
    application: str
    seed: int
    horizon: int = 20
    n_agents: Optional[int] = None
    private_information: float = 0.5
    objective_misalignment: float = 0.5
    communication: str = "reliable"
    disruption: str = "moderate"
    decision_interval: int = 4
    communication_budget: int = 12
    random_gate_probability: float = 0.5
    topology: str = "ring_plus_hubs"


@dataclass
class NodeState:
    inventory: float
    base_capacity: float
    capacity: float
    base_demand: float
    demand: float = 0.0
    backlog: float = 0.0
    fulfilled: float = 0.0
    cumulative_demand: float = 0.0
    impairment: float = 0.0
    delay: float = 0.0
    service_shortfall: float = 0.0
    commitment_strain: float = 0.0
    private_cost: float = 1.0
    local_forecast: float = 1.0
    priority_weight: float = 1.0


@dataclass
class Coalition:
    coalition_id: str
    proposer: str
    invited: List[str]
    members: Set[str]
    purpose: str
    expires_step: int
    refusals: Set[str] = field(default_factory=set)


class LogisticsEnvironment:
    """Material-conserving abstract logistics environment.

    Production is recorded as an exogenous material inflow. Inventory can only
    move through validated shipments and is never teleported between nodes.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.application = Application(config.application)
        # Exogenous demand/production must be paired across methods. Keep it
        # independent from action-dependent message counts and from forecast
        # sampling. The initialization stream preserves topology/state pairing.
        self.rng = np.random.RandomState(
            derived_rng_seed(config.seed, "initialization")
        )
        self.exogenous_rng = np.random.RandomState(
            derived_rng_seed(config.seed, "exogenous_dynamics")
        )
        self.observation_rng = np.random.RandomState(
            derived_rng_seed(config.seed, "observation_noise")
        )
        self.communication_rng = np.random.RandomState(
            derived_rng_seed(config.seed, "communication")
        )
        self.ledger = EventLedger()
        self.step_index = 0
        self.agents: Dict[str, AutonomousAgent] = {}
        self.states: Dict[str, NodeState] = {}
        self.shipments: Dict[str, Shipment] = {}
        self.completed_shipments: Dict[str, Shipment] = {}
        self.commitments: Dict[str, Commitment] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.pending_messages: List[Message] = []
        self.interaction_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self.physical_edges: Set[Tuple[str, str]] = set()
        self.communication_edges: Set[Tuple[str, str]] = set()
        self.initial_material = 0.0
        self.produced_material = 0.0
        self.delivered_material = 0.0
        self.total_cost = 0.0
        self.message_attempts = 0
        self.messages_delivered = 0
        self.message_bytes = 0
        self.information_disclosures = 0
        self.disclosures_by_agent: Dict[str, int] = defaultdict(int)
        self.dispatch_used: Dict[str, float] = defaultdict(float)
        self.tool_calls = 0
        self.valid_tool_calls = 0
        self.offers_submitted = 0
        self.offers_accepted = 0
        self.individually_rational_acceptances = 0
        self.shipment_material_dispatched = 0.0
        self.shipment_material_arrived = 0.0
        self.shipments_arrived = 0
        self.shipments_on_time = 0
        self.commitment_breaches = 0
        self.plan_revisions = 0
        self._message_counter = 0
        self._commitment_counter = 0
        self._shipment_counter = 0
        self._coalition_counter = 0
        self._disruption_applied = False
        self.closed_physical_edges: Set[Tuple[str, str]] = set()
        self.route_lead_time_penalty = 0
        self._build()
        self.initial_physical_edges = set(self.physical_edges)
        self.initial_communication_edges = set(self.communication_edges)
        self.ledger.append(0, "topology_snapshot", "simulator", {
            "topology": self.config.topology,
            "communication_regime": self.config.communication,
            "agents": {
                agent_id: {
                    "role": agent.identity.role,
                    "location": list(agent.identity.location),
                }
                for agent_id, agent in self.agents.items()
            },
            "physical_edges": [list(edge) for edge in sorted(self.initial_physical_edges)],
            "communication_edges": [list(edge) for edge in sorted(self.initial_communication_edges)],
        })

    def _build(self) -> None:
        roles = COMMERCIAL_ROLES if self.application == Application.COMMERCIAL else HUMANITARIAN_ROLES
        n_agents = self.config.n_agents or len(roles)
        if self.application == Application.COMMERCIAL and not 5 <= n_agents <= len(roles):
            raise ValueError("commercial configuration supports 5 to 11 agents")
        if self.application == Application.HUMANITARIAN and not 6 <= n_agents <= len(roles):
            raise ValueError("humanitarian configuration supports 6 to 10 agents")
        roles = roles[:n_agents]
        # Ensure small topologies retain at least one demand node.
        if not any(role in DEMAND_ROLES for role in roles):
            roles[-1] = "retailer" if self.application == Application.COMMERCIAL else "clinic"
        for index, role in enumerate(roles):
            agent_id = "%s_%02d" % (role, index + 1)
            angle = 2.0 * math.pi * index / len(roles)
            utility_noise = self.config.objective_misalignment * self.rng.uniform(-0.45, 0.45, size=4)
            risk_noise = self.config.objective_misalignment * float(self.rng.uniform(-0.15, 0.15))
            utility = UtilityWeights(
                service=float(max(0.3, 1.0 + utility_noise[0])),
                cost=float(max(0.05, 0.3 + utility_noise[1])),
                fairness=float(max(0.02, 0.3 + utility_noise[2])),
                disclosure=float(max(0.0, 0.08 + utility_noise[3] * 0.25)),
                risk=float(0.4 + risk_noise),
                reservation_price=float(1.3 + utility_noise[0] - utility_noise[1]),
            )
            identity = Identity(
                agent_id=agent_id,
                role=role,
                application=self.application.value,
                organization=("enterprise" if self.application == Application.COMMERCIAL else "relief") + "_%02d" % (index + 1),
                location=(round(math.cos(angle), 4), round(math.sin(angle), 4)),
            )
            agent = AutonomousAgent(identity, utility, risk_tolerance=utility.risk, rng_seed=self.config.seed * 100 + index)
            agent.communication_budget = self.config.communication_budget
            self.agents[agent_id] = agent
            is_demand = role in DEMAND_ROLES
            inventory = float(self.rng.uniform(3, 8) if is_demand else self.rng.uniform(35, 60))
            capacity = float(self.rng.uniform(7, 12) if not is_demand else self.rng.uniform(2, 5))
            demand = float(self.rng.uniform(5, 9) if is_demand else 0.0)
            # Information regime changes who can observe a cost, never the
            # underlying economic state. Keeping the draw invariant is required
            # for causal comparisons across privacy levels.
            hidden_cost = float(self.rng.uniform(0.7, 1.5))
            self.states[agent_id] = NodeState(
                inventory=inventory,
                base_capacity=capacity,
                capacity=capacity,
                base_demand=demand,
                private_cost=hidden_cost,
                local_forecast=max(1.0, demand * self.config.horizon / 4.0),
                priority_weight=(
                    1.5 if self.application == Application.HUMANITARIAN and role == "clinic"
                    else 1.2 if self.application == Application.HUMANITARIAN and role == "community"
                    else 1.0
                ),
            )
            self.initial_material += inventory

        ids = list(self.agents)
        if self.config.topology == "ring_plus_hubs":
            for i, left in enumerate(ids):
                right = ids[(i + 1) % len(ids)]
                self.communication_edges.add(tuple(sorted((left, right))))
            hubs = [a for a, agent in self.agents.items() if agent.identity.role in ("carrier", "warehouse", "agency", "transport", "ngo")]
            for hub in hubs:
                for target in ids:
                    if hub != target:
                        self.communication_edges.add(tuple(sorted((hub, target))))
        elif self.config.topology == "holdout_nine_agent":
            # Unseen sparse dual-region topology: two local paths joined by a
            # single bridge rather than a ring with organization hubs.
            midpoint = len(ids) // 2
            regions = (ids[:midpoint], ids[midpoint:])
            for region in regions:
                for left, right in zip(region[:-1], region[1:]):
                    self.communication_edges.add(tuple(sorted((left, right))))
            self.communication_edges.add(tuple(sorted((ids[midpoint - 1], ids[midpoint]))))
        else:
            raise ValueError("unknown topology: %s" % self.config.topology)
        sources = [a for a, agent in self.agents.items() if agent.identity.role in SOURCE_ROLES]
        demands = [a for a, agent in self.agents.items() if agent.identity.role in DEMAND_ROLES]
        if self.config.topology == "ring_plus_hubs":
            for source in sources:
                for target in demands:
                    self.physical_edges.add((source, target))
        else:
            # Every demand location retains two sources where possible, but
            # source--destination reachability differs from training.
            for demand_index, target in enumerate(demands):
                degree = min(2, len(sources))
                for offset in range(degree):
                    source = sources[(2 * demand_index + offset) % len(sources)]
                    self.physical_edges.add((source, target))

    @property
    def agent_ids(self) -> List[str]:
        return list(self.agents)

    def public_identities(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for agent_id, agent in self.agents.items():
            state = self.states[agent_id]
            row: Dict[str, Any] = {
                "agent_id": agent.agent_id,
                "role": agent.identity.role,
                "organization": agent.identity.organization,
            }
            if self.config.private_information <= 0.1:
                row["shared_operational_state"] = {
                    "inventory": round(state.inventory, 2),
                    "available_capacity": round(state.capacity, 2),
                    "backlog": round(state.backlog, 2),
                    "impairment": round(state.impairment, 2),
                    "marginal_cost": round(state.private_cost, 2),
                    "local_forecast": round(state.local_forecast, 2),
                }
            elif self.config.private_information < 0.9:
                row["shared_operational_state"] = {
                    "inventory": "high" if state.inventory > state.base_capacity * 2 else ("nominal" if state.inventory > state.base_capacity * 0.5 else "low"),
                    "backlog": "high" if state.backlog > max(2.0, state.base_demand) else ("nominal" if state.backlog > 0 else "low"),
                    "impairment": "high" if state.impairment > 0.5 else ("nominal" if state.impairment > 0 else "low"),
                }
            rows.append(row)
        return rows

    def _communication_probability(self, sender: str, recipient: str) -> float:
        if tuple(sorted((sender, recipient))) not in self.communication_edges:
            return 0.0
        regime = self.config.communication
        if regime == "reliable":
            return 0.98
        if regime == "intermittent":
            return 0.65
        if regime == "partition":
            if self.step_index < max(2, self.config.horizon // 3):
                return 0.98
            ids = self.agent_ids
            midpoint = len(ids) // 2
            same_side = (sender in ids[:midpoint]) == (recipient in ids[:midpoint])
            return 0.85 if same_side else 0.0
        return 0.8

    def active_communication_edges(self) -> Set[Tuple[str, str]]:
        if (
            self.config.communication != "partition"
            or self.step_index < max(2, self.config.horizon // 3)
        ):
            return set(self.communication_edges)
        ids = self.agent_ids
        midpoint = len(ids) // 2
        return {edge for edge in self.communication_edges if ((edge[0] in ids[:midpoint]) == (edge[1] in ids[:midpoint]))}

    def _next_id(self, kind: str) -> str:
        if kind == "message":
            self._message_counter += 1
            return "M%06d" % self._message_counter
        if kind == "commitment":
            self._commitment_counter += 1
            return "C%06d" % self._commitment_counter
        if kind == "shipment":
            self._shipment_counter += 1
            return "S%06d" % self._shipment_counter
        self._coalition_counter += 1
        return "K%06d" % self._coalition_counter

    def _send(self, sender: str, recipient: str, kind: str, payload: Dict[str, Any], public: bool = False) -> ToolResult:
        self.message_attempts += 1
        if sender not in self.agents or recipient not in self.agents:
            return ToolResult(False, "invalid_recipient", "recipient does not exist")
        if sender == recipient:
            return ToolResult(False, "self_target", "an agent may not send an inter-agent message to itself")
        if self.agents[sender].communication_budget <= 0:
            return ToolResult(False, "communication_budget_exhausted", "no messages remain")
        self.agents[sender].communication_budget -= 1
        probability = self._communication_probability(sender, recipient)
        dropped = self.communication_rng.rand() > probability
        delay = 1 if self.config.communication == "reliable" else int(self.communication_rng.randint(1, 3))
        message = Message(
            message_id=self._next_id("message"), sender=sender, recipient=recipient, kind=kind,
            payload=deepcopy(payload), sent_step=self.step_index, deliver_step=self.step_index + delay, public=public,
        )
        self.message_bytes += len(json.dumps(asdict(message), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self.interaction_weights[(sender, recipient)] += 1.0
        self.ledger.append(self.step_index, "message", sender, {**asdict(message), "dropped": dropped}, None if public else recipient)
        # The sender remembers every explicit attempt, including one that the
        # communication channel drops. Recipients see only delivered messages.
        self.agents[sender].outbox.append(message)
        if not dropped:
            self.pending_messages.append(message)
        return ToolResult(True, "sent" if not dropped else "packet_dropped", "message processed", {"message_id": message.message_id, "dropped": dropped})

    def _deliver_messages(self) -> None:
        remaining: List[Message] = []
        for message in self.pending_messages:
            if message.deliver_step <= self.step_index:
                if message.kind in ("offer", "counteroffer"):
                    commitment_id = str(message.payload.get("commitment_id", ""))
                    if commitment_id in self.commitments:
                        self.agents[message.recipient].commitments[commitment_id] = deepcopy(
                            self.commitments[commitment_id]
                        )
                elif message.kind in ("offer_accepted", "offer_rejected"):
                    commitment_id = str(message.payload.get("commitment_id", ""))
                    if commitment_id in self.commitments:
                        self.agents[message.recipient].commitments[commitment_id] = deepcopy(
                            self.commitments[commitment_id]
                        )
                self.agents[message.recipient].deliver_message(message)
                self.ledger.append(
                    self.step_index,
                    "message_delivery",
                    "communication_network",
                    {
                        "message_id": message.message_id,
                        "sender": message.sender,
                        "recipient": message.recipient,
                        "kind": message.kind,
                    },
                    None if message.public else message.recipient,
                )
                self.messages_delivered += 1
            else:
                remaining.append(message)
        self.pending_messages = remaining

    def private_observation(self, agent_id: str) -> PrivateObservation:
        state = self.states[agent_id]
        reliability = np.mean([
            self._communication_probability(agent_id, other)
            for other in self.agents if other != agent_id
        ])
        # Every agent has the same local forecast-quality process across the
        # privacy factor. Privacy changes disclosure to others, not the quality
        # of the agent's own private observation.
        forecast_noise = self.observation_rng.normal(0, 0.10)
        return PrivateObservation(
            step=self.step_index,
            inventory=round(state.inventory, 4),
            capacity=round(state.capacity, 4),
            impairment=round(state.impairment, 4),
            demand=round(state.demand, 4),
            backlog=round(state.backlog, 4),
            delay=round(state.delay, 4),
            service_shortfall=round(state.service_shortfall, 4),
            commitment_strain=round(state.commitment_strain, 4),
            communication_reliability=float(reliability),
            private_cost=round(state.private_cost, 4),
            local_forecast=round(max(1.0, state.local_forecast * (1.0 + forecast_noise)), 4),
        )

    def deliver_observations(self) -> None:
        for agent_id, agent in self.agents.items():
            agent.deliver_observation(self.private_observation(agent_id), self.ledger)
            agent.update_beliefs()

    def apply_disruption(self) -> None:
        if self._disruption_applied or self.config.disruption == "nominal":
            return
        if self.step_index != max(2, self.config.horizon // 3):
            return
        ids = self.agent_ids
        affected_count = 1 if self.config.disruption == "moderate" else max(2, len(ids) // 3)
        sources = [a for a in ids if self.agents[a].identity.role in SOURCE_ROLES]
        logistics = [
            a for a in ids
            if self.agents[a].identity.role in ("carrier", "warehouse", "transport", "depot")
        ]
        if affected_count == 1:
            affected = sources[:1]
        else:
            affected = list(dict.fromkeys(sources[: affected_count - 1] + logistics[:1]))
        if not affected:
            affected = ids[:affected_count]
        severity = 0.45 if self.config.disruption == "moderate" else 0.8
        for agent_id in affected:
            state = self.states[agent_id]
            state.impairment = severity
            state.capacity = state.base_capacity * (1.0 - severity)
            state.delay = severity
        if self.config.disruption in ("correlated", "compound"):
            self.route_lead_time_penalty = 1 if self.config.disruption == "correlated" else 2
            delay_floor = 0.5 if self.config.disruption == "correlated" else 1.0
            for state in self.states.values():
                state.delay = max(state.delay, delay_floor)
            for agent_id, agent in self.agents.items():
                if agent.identity.role in DEMAND_ROLES:
                    self.states[agent_id].base_demand *= 1.7
        closure_count = 1
        if self.config.disruption == "correlated":
            closure_count = max(1, len(self.physical_edges) // 4)
        elif self.config.disruption == "compound":
            closure_count = max(2, len(self.physical_edges) // 3)
        candidate_edges = sorted(
            self.physical_edges,
            key=lambda edge: (edge[0] not in affected, edge[0], edge[1]),
        )
        for edge in candidate_edges[:closure_count]:
            self.physical_edges.remove(edge)
            self.closed_physical_edges.add(edge)
        outage_agents: List[str] = []
        if self.config.disruption == "compound":
            facility_role = "warehouse" if self.application == Application.COMMERCIAL else "depot"
            facility = next((a for a in ids if self.agents[a].identity.role == facility_role), None)
            if facility is not None:
                outage_agents.append(facility)
            if self.application == Application.HUMANITARIAN:
                coordinator = next((a for a in ids if self.agents[a].identity.role == "agency"), None)
                if coordinator is not None:
                    outage_agents.append(coordinator)
                    self.communication_edges = {
                        edge for edge in self.communication_edges if coordinator not in edge
                    }
            for outage_agent in outage_agents:
                self.states[outage_agent].impairment = 1.0
                self.states[outage_agent].capacity = 0.0
                if outage_agent not in affected:
                    affected.append(outage_agent)
        self._disruption_applied = True
        self.ledger.append(self.step_index, "disruption", "simulator", {
            "regime": self.config.disruption,
            "affected": affected,
            "severity": severity,
            "route_closures": [list(edge) for edge in sorted(self.closed_physical_edges)],
            "lead_time_penalty": self.route_lead_time_penalty,
            "facility_outages": outage_agents,
            "coordinator_loss": next((
                agent_id for agent_id in outage_agents
                if self.agents[agent_id].identity.role == "agency"
            ), None),
            "demand_surge": self.config.disruption in ("correlated", "compound"),
        })

    def execute_tool(self, agent_id: str, tool: str, args: Mapping[str, Any]) -> ToolResult:
        self.tool_calls += 1
        self.ledger.append(self.step_index, "tool_call", agent_id, {"tool": tool, "arguments": dict(args)}, private_to=agent_id)
        state = self.states[agent_id]
        result: ToolResult
        if tool == "no_op":
            result = ToolResult(True, "no_op", "no external action")
        elif tool == "inspect_private_inventory":
            result = ToolResult(True, "private_inventory", "private inventory inspected", {"inventory": state.inventory})
        elif tool == "forecast_local_demand":
            result = ToolResult(True, "local_forecast", "local forecast calculated", {"forecast": state.local_forecast})
        elif tool == "request_info":
            result = self._send(agent_id, args["target"], "information_request", {"topic": args["topic"]})
        elif tool == "request_quote":
            if not self.step_index < int(args["due_step"]) <= self.step_index + 6:
                result = ToolResult(False, "invalid_deadline", "quote deadline must be 1 to 6 periods ahead")
            else:
                result = self._send(agent_id, args["target"], "quote_request", {"quantity": args["quantity"], "due_step": args["due_step"]})
        elif tool in ("disclose_summary", "report_local_need", "request_priority", "challenge_allocation"):
            kind = {"disclose_summary": "summary", "report_local_need": "need", "request_priority": "priority_request", "challenge_allocation": "challenge"}[tool]
            result = self._send(agent_id, args["target"], kind, dict(args), public=tool == "disclose_summary")
            if result.ok and tool in ("disclose_summary", "report_local_need"):
                self.information_disclosures += 1
                self.disclosures_by_agent[agent_id] += 1
        elif tool in ("submit_offer", "pledge_resource"):
            if not self.step_index < int(args["due_step"]) <= self.step_index + 6:
                result = ToolResult(False, "invalid_deadline", "commitment deadline must be 1 to 6 periods ahead")
            elif args["target"] not in self.agents:
                result = ToolResult(False, "invalid_recipient", "offer target does not exist")
            elif args["target"] == agent_id:
                result = ToolResult(False, "self_target", "an agent may not offer to itself")
            elif self.agents[agent_id].communication_budget <= 0:
                result = ToolResult(False, "communication_budget_exhausted", "no messages remain to deliver the offer")
            elif float(args["quantity"]) > state.inventory + (
                state.capacity * (int(args["due_step"]) - self.step_index)
                if self.agents[agent_id].identity.role in PRODUCTION_ROLES else 0.0
            ):
                result = ToolResult(False, "infeasible_offer", "offered quantity exceeds available plus producible capacity")
            else:
                commitment = Commitment(
                    commitment_id=self._next_id("commitment"), proposer=agent_id, partner=args["target"],
                    quantity=float(args["quantity"]), unit_price=float(args.get("unit_price", 0.0)),
                    due_step=int(args["due_step"]), kind="pledge" if tool == "pledge_resource" else "shipment",
                    resource_owner=agent_id, resource_recipient=args["target"],
                )
                self.commitments[commitment.commitment_id] = commitment
                self.agents[agent_id].commitments[commitment.commitment_id] = deepcopy(commitment)
                self.ledger.append(self.step_index, "offer", agent_id, asdict(commitment), private_to=commitment.partner)
                self._send(agent_id, commitment.partner, "offer", asdict(commitment))
                self.offers_submitted += 1
                result = ToolResult(True, "offer_submitted", "offer created", {"commitment_id": commitment.commitment_id})
        elif tool in ("accept_offer", "reject_offer", "counter_offer"):
            commitment_id = args["commitment_id"]
            commitment = self.commitments.get(commitment_id)
            if commitment is None or commitment.partner != agent_id or commitment.status != "proposed":
                result = ToolResult(False, "invalid_commitment", "offer is not pending for this agent")
            elif tool == "accept_offer":
                commitment.status = "accepted"
                self.offers_accepted += 1
                if self._agreement_is_individually_rational(commitment):
                    self.individually_rational_acceptances += 1
                self.agents[agent_id].commitments[commitment_id] = deepcopy(commitment)
                self.ledger.append(self.step_index, "commitment", agent_id, asdict(commitment))
                self._send(agent_id, commitment.proposer, "offer_accepted", {"commitment_id": commitment_id})
                result = ToolResult(True, "offer_accepted", "commitment accepted", {"commitment_id": commitment_id})
            elif tool == "reject_offer":
                commitment.status = "rejected"
                self.agents[agent_id].commitments[commitment_id] = deepcopy(commitment)
                self._send(agent_id, commitment.proposer, "offer_rejected", {"commitment_id": commitment_id, "reason": args["reason"]})
                result = ToolResult(True, "offer_rejected", "offer rejected", {"commitment_id": commitment_id})
            else:
                if not self.step_index < int(args["due_step"]) <= self.step_index + 6:
                    result = ToolResult(False, "invalid_deadline", "counteroffer deadline must be 1 to 6 periods ahead")
                elif float(args["quantity"]) > commitment.quantity + 1e-9:
                    result = ToolResult(False, "counter_quantity", "counteroffer may not increase the offered quantity")
                else:
                    commitment.status = "countered"
                    counter = Commitment(
                        commitment_id=self._next_id("commitment"), proposer=agent_id, partner=commitment.proposer,
                        quantity=float(args["quantity"]), unit_price=float(args["unit_price"]), due_step=int(args["due_step"]),
                        kind=commitment.kind,
                        resource_owner=commitment.resource_owner or commitment.proposer,
                        resource_recipient=commitment.resource_recipient or commitment.partner,
                        parent_commitment_id=commitment.commitment_id,
                        negotiation_round=commitment.negotiation_round + 1,
                    )
                    self.commitments[counter.commitment_id] = counter
                    self.agents[counter.proposer].commitments[counter.commitment_id] = deepcopy(counter)
                    self.ledger.append(self.step_index, "counteroffer", agent_id, asdict(counter), private_to=counter.partner)
                    self._send(agent_id, counter.partner, "counteroffer", asdict(counter))
                    result = ToolResult(True, "offer_countered", "counteroffer created", {"commitment_id": counter.commitment_id})
        elif tool in ("schedule_shipment", "transfer_resource"):
            target = args["target"]
            quantity = float(args["quantity"])
            arrival = int(args["arrival_step"])
            minimum_arrival = self.step_index + 1 + self.route_lead_time_penalty
            if target not in self.agents:
                result = ToolResult(False, "invalid_recipient", "shipment target does not exist")
            elif (agent_id, target) not in self.physical_edges and not self._coalition_route_available(agent_id, target):
                result = ToolResult(False, "no_route", "no physical route to target")
            elif arrival < minimum_arrival:
                result = ToolResult(False, "lead_time_infeasible", "arrival is earlier than current route conditions allow", {"earliest_arrival_step": minimum_arrival})
            elif quantity > state.inventory + 1e-9:
                result = ToolResult(False, "insufficient_inventory", "shipment exceeds private inventory", {"available": state.inventory})
            elif self.dispatch_used[agent_id] + quantity > state.capacity + 1e-9:
                result = ToolResult(False, "handling_capacity_exceeded", "dispatch exceeds this period's available handling capacity", {"available_capacity": max(0.0, state.capacity - self.dispatch_used[agent_id])})
            elif arrival <= self.step_index:
                result = ToolResult(False, "invalid_arrival", "arrival must be in the future")
            elif arrival > self.step_index + 6:
                result = ToolResult(False, "invalid_arrival", "arrival must be no more than 6 periods ahead")
            else:
                shipment = Shipment(
                    self._next_id("shipment"), agent_id, target, quantity,
                    self.step_index, arrival, promised_arrival_step=arrival,
                )
                matching = [
                    commitment for commitment in self.commitments.values()
                    if (commitment.resource_owner or commitment.proposer) == agent_id
                    and (commitment.resource_recipient or commitment.partner) == target
                    and commitment.status in ("accepted", "breached")
                    and quantity <= commitment.quantity + 1e-9
                ]
                if matching:
                    matching[0].status = "in_transit"
                    shipment.commitment_id = matching[0].commitment_id
                    for owner in (matching[0].resource_owner or matching[0].proposer, matching[0].resource_recipient or matching[0].partner):
                        self.agents[owner].commitments[matching[0].commitment_id] = deepcopy(matching[0])
                state.inventory -= quantity
                self.dispatch_used[agent_id] += quantity
                self.shipment_material_dispatched += quantity
                self.shipments[shipment.shipment_id] = shipment
                self.total_cost += quantity * state.private_cost
                result = ToolResult(True, "shipment_scheduled", "inventory moved into transit", {"shipment_id": shipment.shipment_id})
        elif tool == "propose_coalition":
            requested_members = list(args["members"])
            if not all(isinstance(member, str) for member in requested_members):
                result = ToolResult(False, "invalid_member_type", "coalition member IDs must be strings")
            elif len(set(requested_members)) != len(requested_members):
                result = ToolResult(False, "duplicate_member", "coalition invitees must be unique")
            elif agent_id in requested_members:
                result = ToolResult(False, "self_member", "the proposer is already a coalition member")
            elif any(member not in self.agents for member in requested_members):
                result = ToolResult(False, "invalid_member", "coalition contains an unknown agent")
            elif not self.step_index + 2 <= int(args["expires_step"]) <= self.step_index + 8:
                result = ToolResult(False, "invalid_expiry", "coalition expiry must be 2 to 8 periods ahead")
            elif not requested_members:
                result = ToolResult(False, "empty_coalition", "no valid coalition invitees")
            else:
                coalition = Coalition(
                    self._next_id("coalition"), agent_id, requested_members,
                    {agent_id}, args["purpose"], int(args["expires_step"]),
                )
                self.coalitions[coalition.coalition_id] = coalition
                self.agents[agent_id].coalition_ledger[coalition.coalition_id] = {
                    "status": "member", "expires_step": coalition.expires_step,
                    "purpose": coalition.purpose, "proposer": coalition.proposer,
                }
                self.ledger.append(self.step_index, "coalition_event", agent_id, {"action": "propose", **self.coalition_dict(coalition)})
                for member in requested_members:
                    self._send(agent_id, member, "coalition_proposal", self.coalition_dict(coalition))
                result = ToolResult(True, "coalition_proposed", "temporary coalition proposed", {"coalition_id": coalition.coalition_id})
        elif tool in ("join_coalition", "refuse_coalition", "withdraw_coalition"):
            coalition = self.coalitions.get(args["coalition_id"])
            if coalition is None:
                result = ToolResult(False, "invalid_coalition", "coalition does not exist")
            elif self.step_index > coalition.expires_step:
                result = ToolResult(False, "coalition_expired", "coalition contract has expired")
            elif (
                tool == "join_coalition"
                and agent_id in coalition.invited
                and agent_id not in coalition.members
                and agent_id not in coalition.refusals
            ):
                coalition.members.add(agent_id)
                self.agents[agent_id].coalition_ledger[coalition.coalition_id] = {
                    "status": "member", "expires_step": coalition.expires_step,
                    "purpose": coalition.purpose, "proposer": coalition.proposer,
                }
                self._send(agent_id, coalition.proposer, "coalition_joined", {
                    "coalition_id": coalition.coalition_id,
                })
                result = ToolResult(True, "coalition_joined", "agent joined coalition")
            elif (
                tool == "refuse_coalition"
                and agent_id in coalition.invited
                and agent_id not in coalition.members
                and agent_id not in coalition.refusals
            ):
                coalition.refusals.add(agent_id)
                self.agents[agent_id].coalition_ledger[coalition.coalition_id] = {
                    "status": "refused", "expires_step": coalition.expires_step,
                    "purpose": coalition.purpose, "proposer": coalition.proposer,
                }
                self._send(agent_id, coalition.proposer, "coalition_refused", {
                    "coalition_id": coalition.coalition_id,
                    "reason": args["reason"],
                })
                result = ToolResult(True, "coalition_refused", "agent refused coalition")
            elif tool == "withdraw_coalition" and agent_id in coalition.members and agent_id != coalition.proposer:
                coalition.members.remove(agent_id)
                self.agents[agent_id].coalition_ledger[coalition.coalition_id] = {
                    "status": "withdrawn", "expires_step": coalition.expires_step,
                    "purpose": coalition.purpose, "proposer": coalition.proposer,
                }
                self._send(agent_id, coalition.proposer, "coalition_withdrawn", {
                    "coalition_id": coalition.coalition_id,
                    "reason": args["reason"],
                })
                result = ToolResult(True, "coalition_withdrawn", "agent withdrew from coalition")
            else:
                result = ToolResult(False, "coalition_authority", "agent lacks authority for coalition action")
            self.ledger.append(self.step_index, "coalition_event", agent_id, {"action": tool, "coalition_id": args["coalition_id"], "ok": result.ok})
        elif tool in ("verify_delivery", "reroute_shipment", "expedite_shipment"):
            result = self._shipment_admin(agent_id, tool, args)
        else:
            result = ToolResult(False, "unimplemented_tool", "tool has no environment implementation")
        if result.ok:
            self.valid_tool_calls += 1
        self.ledger.append(self.step_index, "tool_result", agent_id, {"tool": tool, **result.as_dict()}, private_to=agent_id)
        return result

    def _shipment_admin(self, agent_id: str, tool: str, args: Mapping[str, Any]) -> ToolResult:
        shipment_id = args["shipment_id"]
        shipment = self.shipments.get(shipment_id) or self.completed_shipments.get(shipment_id)
        if shipment is None:
            return ToolResult(False, "invalid_shipment", "shipment does not exist")
        if tool == "verify_delivery":
            if agent_id not in (shipment.sender, shipment.recipient):
                return ToolResult(False, "shipment_privacy", "only shipment parties may verify delivery")
            delivered = shipment_id in self.completed_shipments
            return ToolResult(
                True,
                "delivery_status",
                "delivery status read",
                {
                    "shipment_id": shipment_id,
                    "delivered": delivered,
                    "recipient": shipment.recipient,
                    "quantity": shipment.quantity,
                },
            )
        if self.agents[agent_id].identity.role not in ("carrier", "transport"):
            return ToolResult(False, "permission_denied", "only transport roles may alter shipments")
        if shipment_id in self.completed_shipments:
            return ToolResult(False, "shipment_completed", "a delivered shipment cannot be altered")
        if shipment.sender != agent_id:
            return ToolResult(False, "shipment_authority", "only the dispatching transport agent may alter this shipment")
        if tool == "reroute_shipment":
            target = args["new_target"]
            if target not in self.agents:
                return ToolResult(False, "invalid_recipient", "new target does not exist")
            if target == shipment.sender:
                return ToolResult(False, "self_target", "shipment may not be rerouted to its sender")
            if (
                (shipment.sender, target) not in self.physical_edges
                and not self._coalition_route_available(shipment.sender, target)
            ):
                return ToolResult(False, "no_route", "no physical or coalition route to new target")
            shipment.recipient = target
            shipment.arrival_step += 1
            return ToolResult(True, "shipment_rerouted", "shipment route updated")
        shipment.expedited = True
        shipment.arrival_step = max(self.step_index + 1, shipment.arrival_step - 1)
        self.total_cost += 0.5 * shipment.quantity
        return ToolResult(True, "shipment_expedited", "shipment arrival advanced")

    def _coalition_route_available(self, source: str, target: str) -> bool:
        """An active coalition can pool temporary recovery route authority."""

        return any(
            self.step_index <= coalition.expires_step
            and source in coalition.members
            and target in coalition.members
            for coalition in self.coalitions.values()
        )

    def _agreement_is_individually_rational(self, commitment: Commitment) -> bool:
        """Evaluator-only two-party rationality check for an accepted offer.

        Priced shipment agreements must cover the resource owner's private
        marginal cost and stay below the recipient's private reservation
        price.  A zero-price humanitarian pledge is voluntarily initiated by
        its resource owner, so owner consent is represented by the pledge
        itself; the recipient's nonnegative reservation constraint still
        applies.  This value never reaches either execution-time actor.
        """

        resource_owner = commitment.resource_owner or commitment.proposer
        resource_recipient = commitment.resource_recipient or commitment.partner
        if resource_owner not in self.agents or resource_recipient not in self.agents:
            return False
        owner_ok = (
            commitment.kind == "pledge"
            or commitment.unit_price + 1e-9 >= self.states[resource_owner].private_cost
        )
        recipient_ok = (
            commitment.unit_price
            <= self.agents[resource_recipient].utility.reservation_price + 1e-9
        )
        return bool(owner_ok and recipient_ok)

    def formed_coalitions(self) -> List[Coalition]:
        """Return coalitions with at least one consenting invited member."""

        return [coalition for coalition in self.coalitions.values() if len(coalition.members) >= 2]

    @staticmethod
    def coalition_dict(coalition: Coalition) -> Dict[str, Any]:
        return {
            "coalition_id": coalition.coalition_id,
            "proposer": coalition.proposer,
            "invited": list(coalition.invited),
            "members": sorted(coalition.members),
            "purpose": coalition.purpose,
            "expires_step": coalition.expires_step,
            "refusals": sorted(coalition.refusals),
        }

    def _arrivals(self) -> None:
        for shipment_id, shipment in list(self.shipments.items()):
            if shipment.arrival_step <= self.step_index:
                self.states[shipment.recipient].inventory += shipment.quantity
                self.shipment_material_arrived += shipment.quantity
                self.shipments_arrived += 1
                promised = shipment.promised_arrival_step if shipment.promised_arrival_step is not None else shipment.arrival_step
                self.shipments_on_time += int(self.step_index <= promised)
                if shipment.commitment_id and shipment.commitment_id in self.commitments:
                    commitment = self.commitments[shipment.commitment_id]
                    late = commitment.status == "breached" or self.step_index > commitment.due_step
                    if late and commitment.status != "breached":
                        self.commitment_breaches += 1
                    commitment.status = "late_delivered" if late else "honored"
                    resource_owner = commitment.resource_owner or commitment.proposer
                    resource_recipient = commitment.resource_recipient or commitment.partner
                    for owner in (resource_owner, resource_recipient):
                        self.agents[owner].commitments[commitment.commitment_id] = deepcopy(commitment)
                    self._send(
                        resource_owner, resource_recipient,
                        "late_delivery" if commitment.status == "late_delivered" else "commitment_honored",
                        {"commitment_id": commitment.commitment_id},
                    )
                self.ledger.append(self.step_index, "environment_transition", "simulator", {"transition": "shipment_arrival", **asdict(shipment)})
                self.completed_shipments[shipment_id] = deepcopy(shipment)
                del self.shipments[shipment_id]

    def _production_and_demand(self) -> None:
        for commitment in self.commitments.values():
            if commitment.status == "proposed" and commitment.due_step < self.step_index:
                commitment.status = "expired"
                parties = {
                    commitment.proposer,
                    commitment.partner,
                    commitment.resource_owner or commitment.proposer,
                    commitment.resource_recipient or commitment.partner,
                }
                for party in parties:
                    if party in self.agents:
                        self.agents[party].commitments[commitment.commitment_id] = deepcopy(commitment)
        for agent_id, agent in self.agents.items():
            state = self.states[agent_id]
            role = agent.identity.role
            if role in PRODUCTION_ROLES:
                produced = state.capacity * float(
                    self.exogenous_rng.uniform(0.85, 1.05)
                )
                state.inventory += produced
                self.produced_material += produced
            if role in DEMAND_ROLES:
                shock = 1.0 + (0.25 if self._disruption_applied and self.config.disruption == "compound" else 0.0)
                state.demand = max(
                    0.0,
                    state.base_demand
                    * shock
                    * float(self.exogenous_rng.lognormal(0.0, 0.12)),
                )
                state.cumulative_demand += state.demand
                state.backlog += state.demand
                fulfilled = min(state.inventory, state.backlog)
                state.inventory -= fulfilled
                state.backlog -= fulfilled
                state.fulfilled += fulfilled
                self.delivered_material += fulfilled
                state.service_shortfall = state.backlog / max(state.backlog + fulfilled, 1.0)
                state.local_forecast = 0.8 * state.local_forecast + 0.2 * max(state.demand * 3.0, 1.0)
            else:
                state.demand = 0.0
                state.service_shortfall = 0.0
            accepted = [
                c for c in self.commitments.values()
                if c.status in ("accepted", "in_transit")
                and (c.resource_owner or c.proposer) == agent_id
            ]
            overdue = [c for c in accepted if c.due_step < self.step_index]
            state.commitment_strain = min(1.0, len(overdue) / max(1, len(accepted)))
            for commitment in overdue:
                if commitment.status in ("accepted", "in_transit"):
                    commitment.status = "breached"
                    self.commitment_breaches += 1
                    resource_owner = commitment.resource_owner or commitment.proposer
                    resource_recipient = commitment.resource_recipient or commitment.partner
                    for owner in (resource_owner, resource_recipient):
                        self.agents[owner].commitments[commitment.commitment_id] = deepcopy(commitment)
                    self._send(resource_owner, resource_recipient, "commitment_breach", {"commitment_id": commitment.commitment_id})

    def transition(self) -> None:
        self.dispatch_used.clear()
        for edge in list(self.interaction_weights):
            self.interaction_weights[edge] *= 0.8
            if self.interaction_weights[edge] < 1e-4:
                del self.interaction_weights[edge]
        self.apply_disruption()
        self._deliver_messages()
        self._arrivals()
        self._production_and_demand()
        self.ledger.append(self.step_index, "environment_transition", "simulator", self.public_metrics())

    def advance(self) -> None:
        self.step_index += 1

    def total_material(self) -> float:
        return sum(state.inventory for state in self.states.values()) + sum(s.quantity for s in self.shipments.values()) + self.delivered_material

    def conservation_error(self) -> float:
        return self.total_material() - (self.initial_material + self.produced_material)

    def public_metrics(self) -> Dict[str, Any]:
        demands = [state for agent_id, state in self.states.items() if self.agents[agent_id].identity.role in DEMAND_ROLES]
        cumulative_demand = sum(state.cumulative_demand for state in demands)
        fulfilled = sum(state.fulfilled for state in demands)
        backlog = sum(state.backlog for state in demands)
        weighted_backlog = sum(state.priority_weight * state.backlog for state in demands)
        fulfillment = fulfilled / max(cumulative_demand, 1e-9)
        per_location = [state.fulfilled / max(state.cumulative_demand, 1e-9) for state in demands]
        fairness = (sum(per_location) ** 2) / (len(per_location) * sum(v * v for v in per_location) + 1e-9) if per_location else 1.0
        on_time = self.shipments_on_time / max(self.shipments_arrived, 1)
        transport_efficiency = self.shipment_material_arrived / max(self.shipment_material_dispatched, 1e-9)
        inventory_efficiency = self.delivered_material / max(self.initial_material + self.produced_material, 1e-9)
        return {
            "step": self.step_index,
            "cumulative_demand": cumulative_demand,
            "fulfilled": fulfilled,
            "backlog": backlog,
            "weighted_backlog": weighted_backlog,
            "fulfillment_rate": fulfillment,
            "service_loss": 1.0 - fulfillment,
            "fairness": fairness,
            "total_cost": self.total_cost,
            "commitment_breaches": self.commitment_breaches,
            "messages": self.message_attempts,
            "delivered_messages": self.messages_delivered,
            "message_bytes": self.message_bytes,
            "information_disclosures": self.information_disclosures,
            "tool_calls": self.tool_calls,
            "valid_tool_calls": self.valid_tool_calls,
            "offers_submitted": self.offers_submitted,
            "offers_accepted": self.offers_accepted,
            "individually_rational_acceptances": self.individually_rational_acceptances,
            "coalitions": len(self.coalitions),
            "coalitions_formed": len(self.formed_coalitions()),
            "on_time_delivery_rate": on_time,
            "transport_efficiency": transport_efficiency,
            "inventory_efficiency": inventory_efficiency,
            "conservation_error": self.conservation_error(),
        }

    def full_state_for_evaluator(self) -> Dict[str, Any]:
        """Never passed to execution-time autonomous actors."""
        return {
            "step": self.step_index,
            "states": {agent_id: asdict(state) for agent_id, state in self.states.items()},
            "shipments": {sid: asdict(shipment) for sid, shipment in self.shipments.items()},
            "completed_shipments": {
                sid: asdict(shipment)
                for sid, shipment in self.completed_shipments.items()
            },
            "commitments": {cid: asdict(commitment) for cid, commitment in self.commitments.items()},
            "metrics": self.public_metrics(),
        }
