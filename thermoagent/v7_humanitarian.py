"""Coupled multi-commodity humanitarian disaster-logistics environment for V7."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .types import MemoryRecord
from .v7_agents import IndependentV7Agent, role_authority
from .v7_base import LEVEL_VALUES, V7CoupledEnvironment
from .v7_topology import apply_partition, generate_graph, restore_edges, topology_diagnostics
from .v7_types import (
    V7ActionResult, V7Commitment, V7Identity, V7PrivateObservation,
    V7ResourceAccount, V7StructuredDecision, V7Utility,
)


COMMODITIES = ("food", "water", "medical", "fuel", "shelter_material")
BELIEF_MODES = (
    "nominal", "demand_surge", "route_failure", "stock_shortage",
    "access_uncertain", "commitment_conflict",
)
NODE_ROLES = ("depot", "hub", "shelter", "clinic")
AGENT_ROLES = (
    "ngo", "transport", "local_authority", "assessment", "clinic",
    "shelter", "depot", "hub",
)


class HumanitarianV7Environment(V7CoupledEnvironment):
    belief_modes = BELIEF_MODES

    def _stable_rng(self, *values: object) -> np.random.RandomState:
        text = "|".join(str(value) for value in (self.environment_seed,) + values)
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.RandomState(seed)

    def _initialize_domain(self) -> None:
        if self.topology_family not in (
            "random_geometric", "small_world", "modular", "chain",
        ):
            raise ValueError("humanitarian topology is not domain appropriate")
        self.operational_graph = generate_graph(
            self.topology_family, self.spec.operational_nodes,
            self.environment_seed + 70001,
        )
        for first, second, data in self.operational_graph.edges(data=True):
            data["travel_time"] = int(self.rng.randint(1, 5))
            data["capacity"] = float(self.rng.uniform(3.0, 8.0))
            data["route_available"] = True
            data["failed_until"] = -1
        node_count = self.spec.operational_nodes
        depot_count = max(1, node_count // 8)
        hub_count = max(2, node_count // 4)
        clinic_count = max(1, node_count // 5)
        self.node_role: Dict[str, str] = {}
        for index in range(node_count):
            if index < depot_count:
                role = "depot"
            elif index < depot_count + hub_count:
                role = "hub"
            elif index >= node_count - clinic_count:
                role = "clinic"
            else:
                role = "shelter"
            node = "H%02d" % index
            self.node_role[node] = role
            self.operational_graph.nodes[index]["asset_id"] = node
            self.operational_graph.nodes[index]["role"] = role
            self.operational_graph.nodes[index]["priority"] = float(
                1.8 if role == "clinic" else (1.3 if role == "shelter" else 0.7)
            )
        self.asset_to_node = {
            data["asset_id"]: int(index)
            for index, data in self.operational_graph.nodes(data=True)
        }
        self.inventory: Dict[str, Dict[str, float]] = {}
        self.initial_inventory: Dict[str, float] = {commodity: 0.0 for commodity in COMMODITIES}
        for asset, role in self.node_role.items():
            multiplier = 4.5 if role == "depot" else (2.2 if role == "hub" else 0.8)
            self.inventory[asset] = {}
            for commodity in COMMODITIES:
                value = float(multiplier * self.rng.uniform(2.0, 4.5))
                self.inventory[asset][commodity] = value
                self.initial_inventory[commodity] += value
        self.emergency_reserve: Dict[str, float] = {
            commodity: float(self.rng.uniform(5.0, 9.0)) for commodity in COMMODITIES
        }
        for commodity, value in self.emergency_reserve.items():
            self.initial_inventory[commodity] += value
        self.consumed: Dict[str, float] = {commodity: 0.0 for commodity in COMMODITIES}
        self.losses: Dict[str, float] = {commodity: 0.0 for commodity in COMMODITIES}
        self.demand: Dict[str, Dict[str, float]] = {
            asset: {commodity: 0.0 for commodity in COMMODITIES}
            for asset in self.node_role
        }
        self.unmet: Dict[str, Dict[str, float]] = {
            asset: {commodity: 0.0 for commodity in COMMODITIES}
            for asset in self.node_role
        }
        self.priority_multiplier: Dict[str, float] = {
            asset: float(self.operational_graph.nodes[node]["priority"])
            for asset, node in self.asset_to_node.items()
        }
        self.initial_vehicles = max(4, self.spec.agent_count // 4)
        self.available_vehicles = self.initial_vehicles
        self.busy_vehicles = 0
        self.initial_fuel = float(self.initial_vehicles * self.spec.horizon * 0.65)
        self.remaining_fuel = self.initial_fuel
        self.consumed_fuel = 0.0
        self.shipments: List[Dict[str, Any]] = []
        self.pending_actions: List[Dict[str, Any]] = []
        self.completed_actions: List[Dict[str, Any]] = []
        self.commitments: Dict[str, V7Commitment] = {}
        self.service_loss_auc = 0.0
        self.critical_shortage_exposure = 0.0
        self.time_to_first_critical_delivery: Optional[int] = None
        self.total_delivered = 0.0
        self.waste = 0.0
        self.commitment_failures = 0
        self.harmful_actions = 0
        self.beneficial_actions = 0
        self.neutral_actions = 0
        self.autonomous_harmful_actions = 0
        self.autonomous_beneficial_actions = 0
        self.autonomous_neutral_actions = 0
        self.operator_executed_actions = 0
        self.physical_actions = 0
        self.service_reaching_actions = 0
        self.information_actions = 0
        self.actionable_opportunities = 0
        self.net_causal_utility = 0.0
        self.maximum_cascade_depth = 0
        self.disabled_communication_edges: List[Tuple[int, int]] = []
        self.disruption_step = max(4, int(round(0.18 * self.spec.horizon)))
        self.recovery_step = min(
            self.spec.horizon - 2,
            self.disruption_step + max(5, int(0.30 * self.spec.horizon)),
        )
        self.failed_routes: List[Tuple[int, int]] = []
        self._build_agents()
        self._seed_commitments()
        diagnostics = topology_diagnostics(self.operational_graph, self.topology_family)
        self.ledger.append(
            0, "v7_topology_snapshot", "humanitarian_simulator",
            {"layer": "logistics", "diagnostics": asdict(diagnostics)},
        )

    def _build_agents(self) -> None:
        assets = sorted(self.node_role)
        for index in range(self.spec.agent_count):
            role = AGENT_ROLES[index % len(AGENT_ROLES)]
            # Each persistent organization controls or observes at least two
            # locations, and scopes overlap so information is complementary.
            scope_size = 2 if self.complexity == "small" else 3
            scope = tuple(
                assets[(index * 3 + offset * max(2, len(assets) // 3)) % len(assets)]
                for offset in range(scope_size)
            )
            identity = V7Identity(
                agent_id="HUM-%02d-%s" % (index, role),
                application="humanitarian",
                role=role,
                asset_scope=tuple(dict.fromkeys(scope)),
                location_scope=tuple(dict.fromkeys(scope)),
                physical_authority=role_authority("humanitarian", role),
            )
            utility_rng = self._stable_rng("utility", index)
            utility = V7Utility(
                service_weight=float(utility_rng.uniform(0.85, 1.25)),
                safety_weight=float(utility_rng.uniform(0.65, 1.20)),
                equity_weight=float(utility_rng.uniform(0.35, 1.10)),
                cost_weight=float(utility_rng.uniform(0.15, 0.55)),
                disclosure_cost=float(utility_rng.uniform(0.03, 0.18)),
                commitment_weight=float(utility_rng.uniform(0.25, 0.80)),
                risk_tolerance=float(utility_rng.uniform(0.30, 0.72)),
            )
            self.agents[identity.agent_id] = IndependentV7Agent(
                identity, utility, self.environment_seed + 1000 + index,
            )

    def _seed_commitments(self) -> None:
        agent_ids = sorted(self.agents)
        for index in range(max(2, self.spec.agent_count // 6)):
            proposer = agent_ids[(2 * index) % len(agent_ids)]
            recipient = agent_ids[(2 * index + 1) % len(agent_ids)]
            shared = sorted(
                set(self.agents[proposer].identity.asset_scope)
                | set(self.agents[recipient].identity.asset_scope)
            )
            commitment = V7Commitment(
                commitment_id="HCOM-%04d" % index,
                proposer=proposer,
                recipient=recipient,
                action="allocate_shipment",
                resource=shared[index % len(shared)],
                quantity=float(0.6 + 0.2 * (index % 3)),
                due_step=self.disruption_step + 4 + index,
            )
            # Commitments remain private to their two parties. Counteroffers
            # and rejection are exercised during initialization, before any
            # evaluator truth is available to an agent.
            for agent_id in (proposer, recipient):
                self.agents[agent_id].commitments[commitment.commitment_id] = deepcopy(commitment)
            self.commitments[commitment.commitment_id] = commitment
            self.ledger.append(
                0, "offer", proposer, asdict(commitment), private_to=recipient,
            )

    def _true_local_mode(self, asset: str, step: int) -> str:
        node = self.asset_to_node[asset]
        incident_edges = list(self.operational_graph.edges(node, data=True))
        if any(not bool(data.get("route_available", True)) for _, _, data in incident_edges):
            return "route_failure"
        inventory = sum(self.inventory[asset].values())
        unmet = sum(self.unmet[asset].values())
        if unmet > 3.5 and inventory < 2.5:
            return "stock_shortage"
        if unmet > 4.5:
            return "demand_surge"
        if any(
            commitment.status in ("proposed", "countered", "breached")
            and asset in (commitment.resource, commitment.action)
            for commitment in self.commitments.values()
        ):
            return "commitment_conflict"
        if step >= self.disruption_step and self.fragmentation > 0.5:
            local_rng = self._stable_rng("access", asset, step)
            if local_rng.uniform() < 0.18 * self.fragmentation:
                return "access_uncertain"
        return "nominal"

    def _belief_for(
        self, agent_id: str, asset: str, true_mode: str, step: int,
    ) -> Tuple[float, ...]:
        rng = self._stable_rng("belief", agent_id, asset, step)
        true_index = BELIEF_MODES.index(true_mode)
        evidence = rng.dirichlet(np.ones(len(BELIEF_MODES)) * 1.15)
        if self.information_condition == "public_shared":
            signal = 0.57
            shared_rng = self._stable_rng("public", asset, step)
            evidence = shared_rng.dirichlet(np.ones(len(BELIEF_MODES)) * 1.2)
        else:
            signal = 0.64 - 0.42 * self.fragmentation
            # Roles observe distinct aspects, creating complementary rather
            # than duplicated private evidence.
            role = self.agents[agent_id].identity.role
            role_offset = AGENT_ROLES.index(role) % len(BELIEF_MODES)
            if rng.uniform() < 0.24 * self.fragmentation:
                true_index = (true_index + 1 + role_offset) % len(BELIEF_MODES)
        evidence *= 1.0 - signal
        evidence[true_index] += signal
        evidence /= evidence.sum()
        return tuple(float(value) for value in evidence)

    def _feasible_actions(self, agent: IndependentV7Agent, asset: str) -> Tuple[str, ...]:
        actions = set(agent.identity.physical_authority)
        if self.available_vehicles <= 0 or self.remaining_fuel < 0.5:
            actions.discard("allocate_shipment")
            actions.discard("redirect_vehicle")
        if max(self.emergency_reserve.values()) <= 1e-9:
            actions.discard("release_emergency_reserve")
        actions.add("no_operational_action")
        return tuple(sorted(actions))

    def deliver_private_observations(self, step: int) -> None:
        for agent_id, agent in sorted(self.agents.items()):
            for asset in agent.identity.asset_scope:
                true_mode = self._true_local_mode(asset, step)
                unmet = sum(self.unmet[asset].values())
                demand = sum(self.demand[asset].values())
                inventory = sum(self.inventory[asset].values())
                node = self.asset_to_node[asset]
                route_failures = sum(
                    not bool(data.get("route_available", True))
                    for _, _, data in self.operational_graph.edges(node, data=True)
                )
                role_rng = self._stable_rng("observation", agent_id, asset, step)
                noise = float(role_rng.normal(0.0, 0.06 + 0.16 * self.fragmentation))
                severity = float(np.clip(
                    0.45 * unmet / max(demand + unmet, 1.0)
                    + 0.25 * route_failures / max(self.operational_graph.degree(node), 1)
                    + 0.30 * (1.0 - min(inventory / 8.0, 1.0)) + noise,
                    0.0, 1.0,
                ))
                confidence = float(np.clip(
                    0.93 - 0.55 * self.fragmentation - 0.25 * route_failures + role_rng.normal(0, 0.04),
                    0.08, 0.99,
                ))
                if self.information_condition == "public_shared":
                    confidence = max(confidence, 0.74)
                commitments = tuple(
                    key for key, value in self.commitments.items()
                    if value.proposer == agent_id or value.recipient == agent_id
                )
                observation = V7PrivateObservation(
                    step=step,
                    focal_asset=asset,
                    local_kpis={
                        "severity": severity,
                        "unmet_need": float(unmet),
                        "local_inventory": float(inventory),
                        "route_failures": float(route_failures),
                        "resource_scarcity": float(1.0 - min(inventory / 12.0, 1.0)),
                        "safety_risk": float(np.clip(0.55 * severity + 0.15 * route_failures + noise, 0, 1)),
                        "delay": float(1.0 + route_failures),
                    },
                    belief_distribution=self._belief_for(agent_id, asset, true_mode, step),
                    telemetry_confidence=confidence,
                    message_age=float(max(0, step - self.disruption_step)) if route_failures else 0.0,
                    communication_reliability=float(1.0 - 0.55 * self.network_disruption),
                    available_resources={
                        "vehicles": float(self.available_vehicles),
                        "fuel": float(self.remaining_fuel),
                        "emergency_reserve": float(sum(self.emergency_reserve.values())),
                    },
                    feasible_physical_actions=self._feasible_actions(agent, asset),
                    active_commitments=commitments,
                )
                agent.deliver_observation(observation, self.ledger)

    def _apply_disruption(self, step: int) -> None:
        if step == self.disruption_step:
            edges = list(self.operational_graph.edges())
            self.rng.shuffle(edges)
            count = max(1, int(round(self.network_disruption * len(edges) * 0.35)))
            self.failed_routes = [(int(a), int(b)) for a, b in edges[:count]]
            for first, second in self.failed_routes:
                data = self.operational_graph.edges[first, second]
                data["route_available"] = False
                data["failed_until"] = self.recovery_step
            self.disabled_communication_edges = apply_partition(
                self.communication_graph,
                min(0.50, self.network_disruption * 0.45),
                self.rng,
            )
            self.ledger.append(
                step, "disruption", "humanitarian_simulator",
                {
                    "kind": "correlated_aftershock_and_partition",
                    "failed_routes": self.failed_routes,
                    "disabled_communication_edges": self.disabled_communication_edges,
                },
            )
        if step == self.recovery_step:
            for first, second in self.failed_routes:
                self.operational_graph.edges[first, second]["route_available"] = True
            restore_edges(self.communication_graph, self.disabled_communication_edges)
            self.ledger.append(
                step, "v7_domain_transition", "humanitarian_simulator",
                {"kind": "route_and_communication_recovery"},
            )

    def _active_route_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.operational_graph.nodes())
        graph.add_edges_from(
            (first, second, data)
            for first, second, data in self.operational_graph.edges(data=True)
            if bool(data.get("route_available", True))
        )
        return graph

    def _choose_shipment(self, target: str) -> Optional[Tuple[str, str, float, int]]:
        shortages = sorted(
            COMMODITIES,
            key=lambda commodity: self.unmet[target][commodity] - self.inventory[target][commodity],
            reverse=True,
        )
        route_graph = self._active_route_graph()
        target_node = self.asset_to_node[target]
        for commodity in shortages:
            sources = sorted(
                self.node_role,
                key=lambda asset: self.inventory[asset][commodity],
                reverse=True,
            )
            for source in sources:
                if source == target or self.inventory[source][commodity] < 0.5:
                    continue
                source_node = self.asset_to_node[source]
                if not nx.has_path(route_graph, source_node, target_node):
                    continue
                path = nx.shortest_path(route_graph, source_node, target_node)
                travel = sum(
                    int(route_graph.edges[a, b].get("travel_time", 1))
                    for a, b in zip(path[:-1], path[1:])
                )
                quantity = min(2.0, self.inventory[source][commodity] * 0.35)
                return source, commodity, float(quantity), max(1, travel)
        return None

    def _schedule_chain(
        self, decision: V7StructuredDecision, step: int, payload: Dict[str, Any],
    ) -> str:
        chain_id = "HC%08d" % (len(self.causal_chains) + 1)
        payload.update({
            "chain_id": chain_id,
            "actor": decision.proposal.agent_id,
            "action": decision.proposal.proposed_operational_action,
            "scheduled_step": step,
            "delegation_action": decision.delegation_action,
        })
        self.pending_actions.append(payload)
        self.causal_chains[chain_id].append({
            "step": step, "stage": "action_scheduled",
            "action": decision.proposal.proposed_operational_action,
        })
        self.ledger.append(
            step, "v7_action_scheduled", decision.proposal.agent_id, deepcopy(payload),
        )
        return chain_id

    def validate_and_schedule(
        self, decision: V7StructuredDecision, step: int,
    ) -> Mapping[str, Any]:
        proposal = decision.proposal
        if proposal.agent_id not in self.agents:
            return asdict(V7ActionResult(False, False, proposal.proposed_operational_action, proposal.agent_id, proposal.target_asset_or_location, None, None, "unknown_agent"))
        agent = self.agents[proposal.agent_id]
        target = str(proposal.target_asset_or_location)
        if target not in agent.identity.asset_scope:
            return asdict(V7ActionResult(False, False, proposal.proposed_operational_action, proposal.agent_id, target, None, None, "target_outside_scope"))
        action = proposal.proposed_operational_action
        observation = agent.vault.observation(agent.agent_id, target)
        if action not in observation.feasible_physical_actions:
            return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "action_masked"))
        if decision.information_action != "no_information_action":
            self.information_actions += 1
            self.ledger.append(
                step, "v7_information_action", agent.agent_id,
                {"action": decision.information_action, "target": target},
            )
        resolved_communication = self.operational_communication_action(
            decision.communication_action, agent.agent_id, target,
            float(observation.local_kpis["severity"]), step,
        )
        if resolved_communication != "no_communication_action":
            node = self.agent_nodes[agent.agent_id]
            neighbors = sorted(self.communication_graph.neighbors(node))
            if neighbors:
                recipient = self.node_agents[int(neighbors[step % len(neighbors)])]
                self.send_message(
                    agent.agent_id, recipient, resolved_communication,
                    {
                        "target": target,
                        "severity": observation.local_kpis["severity"],
                        "belief_distribution": list(agent.private_beliefs[target]),
                    },
                    step, sketch=False,
                )
            self.ledger.append(
                step, "v7_communication_action", agent.agent_id,
                {
                    "requested_action": decision.communication_action,
                    "resolved_action": resolved_communication,
                    "policy": self.operational_communication_policy,
                    "target": target,
                },
            )
        self.ledger.append(
            step, "v7_delegation_decision", agent.agent_id,
            {"delegation_action": decision.delegation_action, "target": target},
        )
        if action == "no_operational_action" or decision.delegation_action not in (
            "execute_autonomously", "escalate_operator",
        ):
            return asdict(V7ActionResult(True, False, action, agent.agent_id, target, None, None, "nonphysical_or_withheld"))
        self.actionable_opportunities += 1
        if action in ("allocate_shipment", "redirect_vehicle"):
            selection = self._choose_shipment(target)
            if selection is None or self.available_vehicles <= 0 or self.remaining_fuel < 0.5:
                return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "no_feasible_shipment"))
            source, commodity, quantity, travel = selection
            fuel = max(0.25, 0.12 * travel * quantity)
            if fuel > self.remaining_fuel:
                return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "insufficient_fuel"))
            self.inventory[source][commodity] -= quantity
            self.remaining_fuel -= fuel
            self.consumed_fuel += fuel
            self.available_vehicles -= 1
            self.busy_vehicles += 1
            chain_id = self._schedule_chain(decision, step, {
                "kind": "shipment", "source": source, "target": target,
                "commodity": commodity, "quantity": quantity,
                "complete_step": step + travel,
                "loss_probability": 0.04 + 0.18 * self.network_disruption,
                "vehicle": 1, "fuel": fuel,
            })
        elif action == "release_emergency_reserve":
            commodity = max(COMMODITIES, key=lambda value: self.unmet[target][value])
            quantity = min(1.5, self.emergency_reserve[commodity])
            if quantity <= 1e-9:
                return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "reserve_exhausted"))
            self.emergency_reserve[commodity] -= quantity
            chain_id = self._schedule_chain(decision, step, {
                "kind": "reserve_transfer", "source": "emergency_reserve",
                "target": target, "commodity": commodity, "quantity": quantity,
                "complete_step": step + 2,
            })
        elif action == "revise_delivery_priority":
            chain_id = self._schedule_chain(decision, step, {
                "kind": "priority_change", "target": target,
                "quantity": min(max(proposal.quantity_or_capacity, 0.1), 1.5),
                "complete_step": step + 1,
            })
        elif action == "cancel_risky_dispatch":
            matching = next((value for value in self.pending_actions if value.get("target") == target and value.get("kind") == "shipment"), None)
            if matching is None:
                chain_id = self._schedule_chain(decision, step, {
                    "kind": "unnecessary_cancellation", "target": target,
                    "complete_step": step + 1,
                })
            else:
                matching["cancelled"] = True
                chain_id = self._schedule_chain(decision, step, {
                    "kind": "cancellation", "target": target,
                    "parent_chain": matching["chain_id"], "complete_step": step + 1,
                })
        else:
            return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "unsupported_action"))
        self.physical_actions += 1
        return asdict(V7ActionResult(
            True, True, action, agent.agent_id, target, step,
            int(self.pending_actions[-1]["complete_step"]), "scheduled",
            causal_chain_id=chain_id,
        ))

    def _complete_actions(self, step: int) -> None:
        due = [value for value in self.pending_actions if int(value["complete_step"]) <= step]
        self.pending_actions = [value for value in self.pending_actions if int(value["complete_step"]) > step]
        for action in due:
            kind = action["kind"]
            effect = 0.0
            reached = False
            if kind in ("shipment", "reserve_transfer"):
                quantity = float(action["quantity"])
                lost = 0.0
                if kind == "shipment":
                    self.available_vehicles += 1
                    self.busy_vehicles -= 1
                    tape = self.stochastic_tape["loss"][step % len(self.stochastic_tape["loss"])]
                    if tape < float(action.get("loss_probability", 0.0)):
                        lost = 0.25 * quantity
                        self.losses[action["commodity"]] += lost
                delivered = quantity - lost
                if bool(action.get("cancelled", False)):
                    self.inventory[action["source"]][action["commodity"]] += delivered
                    effect = -0.05 * delivered
                else:
                    target = action["target"]
                    before = self.unmet[target][action["commodity"]]
                    self.inventory[target][action["commodity"]] += delivered
                    effect = min(before, delivered) * self.priority_multiplier[target] - 0.08 * delivered
                    self.total_delivered += delivered
                    reached = delivered > 0.0
                    if reached and self.node_role[target] == "clinic" and self.time_to_first_critical_delivery is None:
                        self.time_to_first_critical_delivery = step
                self.causal_chains[action["chain_id"]].append({
                    "step": step, "stage": "resource_arrived",
                    "quantity": delivered, "target": action.get("target"),
                })
            elif kind == "priority_change":
                target = action["target"]
                previous = self.priority_multiplier[target]
                self.priority_multiplier[target] = min(2.5, previous + 0.25)
                # Raising one priority consumes scarce attention and slightly
                # lowers all other priorities: a wrong choice can cause harm.
                for asset in self.priority_multiplier:
                    if asset != target:
                        self.priority_multiplier[asset] = max(
                            0.5, self.priority_multiplier[asset] - 0.02 * self.coupling_strength,
                        )
                true_need = sum(self.unmet[target].values())
                effect = 0.12 * true_need - 0.10 * self.coupling_strength
            elif kind == "cancellation":
                effect = 0.08 if any(
                    not self.operational_graph.edges[a, b].get("route_available", True)
                    for a, b in self.failed_routes
                ) else -0.12
            elif kind == "unnecessary_cancellation":
                effect = -0.16
            beneficial = effect > 1e-9
            harmful = effect < -1e-9
            self.beneficial_actions += int(beneficial)
            self.harmful_actions += int(harmful)
            self.neutral_actions += int(not beneficial and not harmful)
            if action.get("delegation_action") == "execute_autonomously":
                self.autonomous_beneficial_actions += int(beneficial)
                self.autonomous_harmful_actions += int(harmful)
                self.autonomous_neutral_actions += int(not beneficial and not harmful)
            elif action.get("delegation_action") == "escalate_operator":
                self.operator_executed_actions += 1
            self.service_reaching_actions += int(reached)
            self.net_causal_utility += effect
            record = dict(action)
            record.update({
                "completed_step": step, "causal_effect": effect,
                "beneficial": beneficial, "harmful": harmful,
                "reached_service": reached,
            })
            self.completed_actions.append(record)
            self.causal_chains[action["chain_id"]].append({
                "step": step, "stage": "outcome_change",
                "causal_effect": effect, "reached_service": reached,
            })
            self.maximum_cascade_depth = max(
                self.maximum_cascade_depth, len(self.causal_chains[action["chain_id"]]),
            )
            self.ledger.append(step, "v7_action_completed", record["actor"], record)

    def _demand_and_service(self, step: int) -> None:
        step_unmet = 0.0
        step_critical = 0.0
        for asset, role in self.node_role.items():
            if role not in ("shelter", "clinic"):
                continue
            node_rng = self._stable_rng("demand", asset, step)
            disruption_multiplier = 1.0 + (
                0.65 * self.coupling_strength
                if step >= self.disruption_step else 0.0
            )
            for commodity_index, commodity in enumerate(COMMODITIES):
                base = 0.10 + 0.04 * commodity_index
                if role == "clinic" and commodity in ("medical", "water", "fuel"):
                    base *= 1.7
                demand = max(0.0, disruption_multiplier * base * (1.0 + 0.18 * node_rng.normal()))
                # Short water supplies increase later medical need: a genuine
                # cross-commodity delayed cascade.
                if commodity == "medical" and self.unmet[asset]["water"] > 1.0:
                    demand += 0.10 * self.coupling_strength * self.unmet[asset]["water"]
                    self.ledger.append(
                        step, "v7_cascade_transition", "humanitarian_simulator",
                        {"source": asset + ":water", "target": asset + ":medical", "depth": 2},
                    )
                    self.maximum_cascade_depth = max(self.maximum_cascade_depth, 2)
                self.demand[asset][commodity] = demand
                available = self.inventory[asset][commodity]
                served = min(available, demand + self.unmet[asset][commodity] * 0.35)
                self.inventory[asset][commodity] -= served
                self.consumed[commodity] += served
                current_unmet = max(0.0, self.unmet[asset][commodity] + demand - served)
                self.unmet[asset][commodity] = current_unmet
                weighted = current_unmet * self.priority_multiplier[asset]
                step_unmet += weighted
                if role == "clinic":
                    step_critical += weighted
        self.service_loss_auc += step_unmet
        self.critical_shortage_exposure += step_critical
        self.ledger.append(
            step, "v7_service_transition", "humanitarian_simulator",
            {
                "weighted_unmet_need": step_unmet,
                "critical_shortage_exposure": step_critical,
                "service_loss_auc": self.service_loss_auc,
            },
        )

    def advance_domain(self, step: int) -> None:
        self._apply_disruption(step)
        self._complete_actions(step)
        self._demand_and_service(step)
        self.ledger.append(
            step, "v7_domain_transition", "humanitarian_simulator",
            {
                "available_vehicles": self.available_vehicles,
                "busy_vehicles": self.busy_vehicles,
                "remaining_fuel": self.remaining_fuel,
                "pending_actions": len(self.pending_actions),
            },
        )
        self._record_resource_state(step)

    def _record_resource_state(self, step: int) -> None:
        for commodity in COMMODITIES:
            remaining = sum(value[commodity] for value in self.inventory.values()) + self.emergency_reserve[commodity]
            in_transit = sum(
                float(value["quantity"])
                for value in self.pending_actions
                if value.get("kind") in ("shipment", "reserve_transfer")
                and value.get("commodity") == commodity
            )
            self.ledger.append(
                step, "v7_resource_transition", "humanitarian_simulator",
                {
                    "resource": commodity,
                    "initial": self.initial_inventory[commodity],
                    "remaining": remaining,
                    "consumed": self.consumed[commodity],
                    "in_transit": in_transit,
                    "delivered": 0.0,
                    "losses": self.losses[commodity],
                },
                private_to="evaluator",
            )
        for resource, initial, remaining, consumed, in_transit in (
            ("vehicles", self.initial_vehicles, self.available_vehicles, 0.0, self.busy_vehicles),
            ("fuel", self.initial_fuel, self.remaining_fuel, self.consumed_fuel, 0.0),
        ):
            self.ledger.append(
                step, "v7_resource_transition", "humanitarian_simulator",
                {
                    "resource": resource, "initial": initial,
                    "remaining": remaining, "consumed": consumed,
                    "in_transit": in_transit, "delivered": 0.0, "losses": 0.0,
                },
                private_to="evaluator",
            )

    def conservation_report(self) -> Dict[str, Any]:
        residuals: Dict[str, float] = {}
        accounts: Dict[str, Dict[str, float]] = {}
        for commodity in COMMODITIES:
            remaining = sum(value[commodity] for value in self.inventory.values()) + self.emergency_reserve[commodity]
            in_transit = sum(
                float(value["quantity"])
                for value in self.pending_actions
                if value.get("kind") in ("shipment", "reserve_transfer")
                and value.get("commodity") == commodity
            )
            account = V7ResourceAccount(
                initial=self.initial_inventory[commodity],
                remaining=remaining,
                consumed=self.consumed[commodity],
                in_transit=in_transit,
                delivered=0.0,
                losses=self.losses[commodity],
            )
            accounts[commodity] = account.as_dict()
            residuals[commodity] = account.residual()
        vehicle_residual = float(
            self.initial_vehicles - self.available_vehicles - self.busy_vehicles
        )
        fuel_residual = float(
            self.initial_fuel - self.remaining_fuel - self.consumed_fuel
        )
        residuals["vehicles"] = vehicle_residual
        residuals["fuel"] = fuel_residual
        maximum = max(abs(value) for value in residuals.values())
        report = {
            "feasible": maximum <= 1e-9
            and min(self.available_vehicles, self.busy_vehicles, self.remaining_fuel) >= -1e-9
            and all(min(values.values()) >= -1e-9 for values in self.inventory.values()),
            "maximum_residual": maximum,
            "residuals": residuals,
            "accounts": accounts,
        }
        return report

    def inject_conservation_fault_for_test(self, commodity: str, quantity: float) -> None:
        asset = sorted(self.inventory)[0]
        self.inventory[asset][commodity] += float(quantity)

    def metrics(self) -> Dict[str, Any]:
        conservation = self.conservation_report()
        allocations = [sum(self.consumed.values())]
        total_unmet_by_asset = [sum(value.values()) for value in self.unmet.values()]
        sorted_unmet = np.sort(np.asarray(total_unmet_by_asset, dtype=float))
        if float(sorted_unmet.sum()) > 1e-12:
            count = len(sorted_unmet)
            ranks = np.arange(1, count + 1, dtype=float)
            equity_gini = float(
                np.sum((2 * ranks - count - 1) * sorted_unmet)
                / (count * sorted_unmet.sum())
            )
        else:
            equity_gini = 0.0
        return {
            "service_loss": float(self.service_loss_auc),
            "weighted_unmet_critical_need": float(sum(
                self.unmet[asset][commodity] * self.priority_multiplier[asset]
                for asset in self.unmet for commodity in COMMODITIES
            )),
            "critical_shortage_exposure": float(self.critical_shortage_exposure),
            "time_to_first_critical_delivery": self.time_to_first_critical_delivery,
            "delivery_completion": float(self.total_delivered),
            "resource_waste": float(sum(self.losses.values())),
            "allocation_inequality_gini": equity_gini,
            "commitment_failures": int(self.commitment_failures),
            "harmful_actions": int(self.harmful_actions),
            "beneficial_actions": int(self.beneficial_actions),
            "neutral_actions": int(self.neutral_actions),
            "autonomous_harmful_actions": int(self.autonomous_harmful_actions),
            "autonomous_beneficial_actions": int(self.autonomous_beneficial_actions),
            "autonomous_neutral_actions": int(self.autonomous_neutral_actions),
            "operator_executed_actions": int(self.operator_executed_actions),
            "physical_actions": int(self.physical_actions),
            "service_reaching_actions": int(self.service_reaching_actions),
            "information_actions": int(self.information_actions),
            "actionable_opportunities": int(self.actionable_opportunities),
            "net_causal_utility": float(self.net_causal_utility),
            "maximum_cascade_depth": int(self.maximum_cascade_depth),
            "maximum_conservation_residual": float(conservation["maximum_residual"]),
            "conservation_feasible": bool(conservation["feasible"]),
            "operational_messages": int(self.operational_messages),
            "sketch_messages": int(self.sketch_messages),
            "dropped_messages": int(self.dropped_messages),
            "total_messages": int(self.total_messages),
            "operational_bytes": int(self.operational_bytes),
            "sketch_bytes": int(self.sketch_bytes),
            "total_bytes": int(self.total_bytes),
            "cross_community_messages": int(self.cross_community_messages),
        }
