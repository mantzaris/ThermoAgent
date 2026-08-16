"""Defensive abstract cyber-physical utility restoration environment for V7.

Cyber events are stochastic simulator state transitions (corrupted telemetry,
disabled links, and defensive isolation), never operational attack procedures.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np

from .v7_agents import IndependentV7Agent, role_authority
from .v7_base import V7CoupledEnvironment
from .v7_topology import apply_partition, generate_graph, restore_edges, topology_diagnostics
from .v7_types import (
    V7ActionResult, V7Commitment, V7Identity, V7PrivateObservation,
    V7ResourceAccount, V7StructuredDecision, V7Utility,
)


BELIEF_MODES = (
    "nominal", "physical_failure", "telemetry_corrupt",
    "communication_failure", "capacity_overload", "cascading_risk",
)
AGENT_ROLES = (
    "zone_operator", "crew_dispatch", "cyber_defense", "communications",
    "resource_allocation", "critical_load",
)


class UtilityRestorationV7Environment(V7CoupledEnvironment):
    belief_modes = BELIEF_MODES

    def _stable_rng(self, *values: object) -> np.random.RandomState:
        text = "|".join(str(value) for value in (self.environment_seed,) + values)
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.RandomState(seed)

    def _initialize_domain(self) -> None:
        if self.topology_family not in (
            "grid", "scale_free", "modular", "small_world",
        ):
            raise ValueError("utility topology is not domain appropriate")
        self.physical_graph = generate_graph(
            self.topology_family, self.spec.operational_nodes,
            self.environment_seed + 80001,
        )
        count = self.spec.operational_nodes
        source_count = max(1, count // 12)
        critical_count = max(2, count // 5)
        self.asset_to_node: Dict[str, int] = {}
        self.node_role: Dict[str, str] = {}
        self.demand: Dict[str, float] = {}
        self.critical_weight: Dict[str, float] = {}
        self.failed: Dict[str, bool] = {}
        self.isolated: Dict[str, bool] = {}
        self.isolation_release_step: Dict[str, int] = {}
        self.telemetry_confidence: Dict[str, float] = {}
        self.telemetry_corrupt: Dict[str, bool] = {}
        self.mobile_service: Dict[str, float] = {}
        for index in range(count):
            if index < source_count:
                role = "source"
            elif index >= count - critical_count:
                role = "critical_load"
            else:
                role = "distribution"
            asset = "U%02d" % index
            self.asset_to_node[asset] = index
            self.node_role[asset] = role
            self.physical_graph.nodes[index]["asset_id"] = asset
            self.physical_graph.nodes[index]["role"] = role
            self.demand[asset] = 0.0 if role == "source" else float(self.rng.uniform(0.7, 1.7))
            self.critical_weight[asset] = 3.0 if role == "critical_load" else 1.0
            self.failed[asset] = False
            self.isolated[asset] = False
            self.telemetry_confidence[asset] = float(self.rng.uniform(0.88, 0.99))
            self.telemetry_corrupt[asset] = False
            self.mobile_service[asset] = 0.0
        for first, second, data in self.physical_graph.edges(data=True):
            data["service_available"] = True
            data["capacity"] = float(self.rng.uniform(1.8, 4.5))
            data["base_capacity"] = data["capacity"]
            data["emergency_authorized"] = False
        self.failed_service_edges: List[Tuple[int, int]] = []
        self.source_assets = tuple(
            asset for asset, role in self.node_role.items() if role == "source"
        )
        self.critical_assets = tuple(
            asset for asset, role in self.node_role.items() if role == "critical_load"
        )
        self.initial_crews = max(3, self.spec.agent_count // 8)
        self.available_crews = self.initial_crews
        self.busy_crews = 0
        self.initial_spares = max(5, self.spec.agent_count // 5)
        self.remaining_spares = float(self.initial_spares)
        self.assigned_spares = 0.0
        self.consumed_spares = 0.0
        self.initial_generators = max(2, self.spec.agent_count // 12)
        self.available_generators = self.initial_generators
        self.deployed_generators = 0
        self.initial_fuel = float(self.initial_generators * self.spec.horizon * 0.45)
        self.remaining_fuel = self.initial_fuel
        self.consumed_fuel = 0.0
        self.pending_actions: List[Dict[str, Any]] = []
        self.completed_actions: List[Dict[str, Any]] = []
        self.service_loss_auc = 0.0
        self.unserved_critical_load_auc = 0.0
        self.current_served: Dict[str, float] = {asset: 0.0 for asset in self.node_role}
        self.restoration_times: Dict[str, int] = {}
        self.cascade_count = 0
        self.maximum_cascade_depth = 0
        self.unsafe_switching = 0
        self.repeated_work_orders = 0
        self.crew_travel = 0.0
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
        self.commitments: Dict[str, V7Commitment] = {}
        self.disabled_communication_edges: List[Tuple[int, int]] = []
        self.disruption_step = max(4, int(round(0.16 * self.spec.horizon)))
        self.recovery_step = min(
            self.spec.horizon - 2,
            self.disruption_step + max(6, int(0.32 * self.spec.horizon)),
        )
        self._build_agents()
        self._seed_commitments()
        diagnostics = topology_diagnostics(self.physical_graph, self.topology_family)
        self.ledger.append(
            0, "v7_topology_snapshot", "utility_simulator",
            {"layer": "physical_service", "diagnostics": asdict(diagnostics)},
        )

    def _build_agents(self) -> None:
        assets = sorted(self.node_role)
        for index in range(self.spec.agent_count):
            role = AGENT_ROLES[index % len(AGENT_ROLES)]
            scope_size = 2 if self.complexity == "small" else 3
            scope = tuple(
                assets[(index * 5 + offset * max(2, len(assets) // 4)) % len(assets)]
                for offset in range(scope_size)
            )
            identity = V7Identity(
                agent_id="UTL-%02d-%s" % (index, role),
                application="utility_restoration",
                role=role,
                asset_scope=tuple(dict.fromkeys(scope)),
                location_scope=tuple(dict.fromkeys(scope)),
                physical_authority=role_authority("utility_restoration", role),
            )
            utility_rng = self._stable_rng("utility", index)
            utility = V7Utility(
                service_weight=float(utility_rng.uniform(0.75, 1.25)),
                safety_weight=float(utility_rng.uniform(0.85, 1.40)),
                equity_weight=float(utility_rng.uniform(0.10, 0.55)),
                cost_weight=float(utility_rng.uniform(0.18, 0.60)),
                disclosure_cost=float(utility_rng.uniform(0.02, 0.15)),
                commitment_weight=float(utility_rng.uniform(0.20, 0.75)),
                risk_tolerance=float(utility_rng.uniform(0.25, 0.68)),
            )
            self.agents[identity.agent_id] = IndependentV7Agent(
                identity, utility, self.environment_seed + 2000 + index,
            )

    def _seed_commitments(self) -> None:
        agents = sorted(self.agents)
        for index in range(max(2, self.spec.agent_count // 7)):
            proposer = agents[(3 * index) % len(agents)]
            recipient = agents[(3 * index + 1) % len(agents)]
            asset = self.agents[recipient].identity.asset_scope[index % len(self.agents[recipient].identity.asset_scope)]
            commitment = V7Commitment(
                commitment_id="UCOM-%04d" % index,
                proposer=proposer,
                recipient=recipient,
                action="dispatch_repair_crew",
                resource=asset,
                quantity=1.0,
                due_step=self.disruption_step + 5 + index,
            )
            for agent_id in (proposer, recipient):
                self.agents[agent_id].commitments[commitment.commitment_id] = deepcopy(commitment)
            self.commitments[commitment.commitment_id] = commitment
            self.ledger.append(0, "offer", proposer, asdict(commitment), private_to=recipient)

    def _true_local_mode(self, asset: str, step: int) -> str:
        if self.failed[asset]:
            return "physical_failure"
        if self.telemetry_corrupt[asset]:
            return "telemetry_corrupt"
        controlling_nodes = [
            self.agent_nodes[agent_id]
            for agent_id, agent in self.agents.items()
            if asset in agent.identity.asset_scope
        ]
        communication_isolated = bool(controlling_nodes) and all(
            not any(
                bool(self.communication_graph.edges[node, neighbor].get("available", True))
                for neighbor in self.communication_graph.neighbors(node)
            )
            for node in controlling_nodes
        )
        if communication_isolated or self.telemetry_confidence[asset] < 0.45:
            return "communication_failure"
        node = self.asset_to_node[asset]
        if any(
            not bool(data.get("service_available", True))
            for _, _, data in self.physical_graph.edges(node, data=True)
        ):
            return "capacity_overload"
        if self._downstream_cascade_risk(asset) > 0.55:
            return "cascading_risk"
        return "nominal"

    def _downstream_cascade_risk(self, asset: str) -> float:
        node = self.asset_to_node[asset]
        failed_neighbors = sum(
            self.failed[self.physical_graph.nodes[neighbor]["asset_id"]]
            for neighbor in self.physical_graph.neighbors(node)
        )
        degree = max(self.physical_graph.degree(node), 1)
        return float(failed_neighbors / degree)

    def _belief_for(
        self, agent_id: str, asset: str, true_mode: str, step: int,
    ) -> Tuple[float, ...]:
        rng = self._stable_rng("belief", agent_id, asset, step)
        true_index = BELIEF_MODES.index(true_mode)
        evidence = rng.dirichlet(np.ones(len(BELIEF_MODES)) * 1.08)
        if self.information_condition == "public_shared":
            shared = self._stable_rng("public", asset, step)
            evidence = shared.dirichlet(np.ones(len(BELIEF_MODES)) * 1.18)
            signal = 0.56
        else:
            signal = 0.66 - 0.46 * self.fragmentation
            role = self.agents[agent_id].identity.role
            if rng.uniform() < 0.30 * self.fragmentation:
                # Telemetry and field agents receive different plausible but
                # conflicting modes under corruption.
                role_offset = AGENT_ROLES.index(role)
                true_index = (true_index + 1 + role_offset) % len(BELIEF_MODES)
        evidence *= 1.0 - signal
        evidence[true_index] += signal
        evidence /= evidence.sum()
        return tuple(float(value) for value in evidence)

    def _feasible_actions(self, agent: IndependentV7Agent, asset: str) -> Tuple[str, ...]:
        actions = set(agent.identity.physical_authority)
        if self.available_crews <= 0:
            actions.discard("dispatch_repair_crew")
        if self.remaining_spares < 1.0:
            actions.discard("allocate_spare_component")
        if self.available_generators <= 0 or self.remaining_fuel < 1.0:
            actions.discard("deploy_mobile_generation")
        actions.add("no_operational_action")
        return tuple(sorted(actions))

    def deliver_private_observations(self, step: int) -> None:
        for agent_id, agent in sorted(self.agents.items()):
            for asset in agent.identity.asset_scope:
                true_mode = self._true_local_mode(asset, step)
                node = self.asset_to_node[asset]
                served = self.current_served.get(asset, 0.0)
                demand = self.demand[asset]
                unserved = max(0.0, demand - served)
                adjacent_failures = sum(
                    self.failed[self.physical_graph.nodes[neighbor]["asset_id"]]
                    for neighbor in self.physical_graph.neighbors(node)
                )
                rng = self._stable_rng("observation", agent_id, asset, step)
                local_confidence = self.telemetry_confidence[asset]
                reported_unserved = unserved
                if self.telemetry_corrupt[asset]:
                    reported_unserved = max(0.0, demand * float(rng.uniform(0.0, 1.5)))
                noise = float(rng.normal(0.0, 0.05 + 0.14 * self.fragmentation))
                severity = float(np.clip(
                    0.55 * reported_unserved / max(demand, 0.5)
                    + 0.25 * adjacent_failures / max(self.physical_graph.degree(node), 1)
                    + 0.20 * self._downstream_cascade_risk(asset) + noise,
                    0.0, 1.0,
                ))
                if self.information_condition == "public_shared":
                    local_confidence = max(local_confidence, 0.72)
                observation = V7PrivateObservation(
                    step=step,
                    focal_asset=asset,
                    local_kpis={
                        "severity": severity,
                        "reported_unserved_load": float(reported_unserved),
                        "adjacent_failures": float(adjacent_failures),
                        "resource_scarcity": float(1.0 - min(self.available_crews / max(self.initial_crews, 1), 1.0)),
                        "safety_risk": float(np.clip(0.45 * severity + 0.35 * self._downstream_cascade_risk(asset) + noise, 0, 1)),
                        "delay": float(1.0 + self.busy_crews / max(self.initial_crews, 1)),
                    },
                    belief_distribution=self._belief_for(agent_id, asset, true_mode, step),
                    telemetry_confidence=float(local_confidence),
                    message_age=float(0 if local_confidence > 0.6 else max(1, step - self.disruption_step)),
                    communication_reliability=float(1.0 - 0.60 * self.network_disruption),
                    available_resources={
                        "crews": float(self.available_crews),
                        "spares": float(self.remaining_spares),
                        "mobile_generators": float(self.available_generators),
                        "fuel": float(self.remaining_fuel),
                    },
                    feasible_physical_actions=self._feasible_actions(agent, asset),
                    active_commitments=tuple(
                        key for key, value in self.commitments.items()
                        if value.proposer == agent_id or value.recipient == agent_id
                    ),
                )
                agent.deliver_observation(observation, self.ledger)

    def _apply_disruption(self, step: int) -> None:
        if step == self.disruption_step:
            candidates = [asset for asset in sorted(self.node_role) if self.node_role[asset] != "source"]
            self.rng.shuffle(candidates)
            failure_count = max(1, int(round(self.coupling_strength * len(candidates) * 0.27)))
            corruption_count = max(1, int(round(self.fragmentation * len(candidates) * 0.26)))
            for asset in candidates[:failure_count]:
                self.failed[asset] = True
            for asset in candidates[failure_count:failure_count + corruption_count]:
                self.telemetry_corrupt[asset] = True
                self.telemetry_confidence[asset] = float(self.rng.uniform(0.08, 0.38))
            edge_candidates = list(self.physical_graph.edges())
            self.rng.shuffle(edge_candidates)
            edge_failure_count = max(
                1, int(round(self.network_disruption * len(edge_candidates) * 0.16)),
            )
            self.failed_service_edges = [
                (int(first), int(second))
                for first, second in edge_candidates[:edge_failure_count]
            ]
            for first, second in self.failed_service_edges:
                self.physical_graph.edges[first, second]["service_available"] = False
            self.disabled_communication_edges = apply_partition(
                self.communication_graph,
                min(0.55, 0.50 * self.network_disruption), self.rng,
            )
            self.ledger.append(
                step, "disruption", "utility_simulator",
                {
                    "kind": "abstract_correlated_cyber_physical",
                    "physical_failures": candidates[:failure_count],
                    "telemetry_integrity_loss_count": corruption_count,
                    "service_edges_disabled": self.failed_service_edges,
                    "communication_edges_disabled": len(self.disabled_communication_edges),
                },
                private_to="evaluator",
            )
        if step == self.recovery_step:
            restore_edges(self.communication_graph, self.disabled_communication_edges)
            self.ledger.append(
                step, "v7_domain_transition", "utility_simulator",
                {"kind": "exogenous_communication_reconnection"},
            )

    def _service_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for node, data in self.physical_graph.nodes(data=True):
            asset = data["asset_id"]
            if not self.failed[asset] and not self.isolated[asset]:
                graph.add_node(node, **data)
        for first, second, data in self.physical_graph.edges(data=True):
            if first in graph and second in graph and bool(data.get("service_available", True)):
                graph.add_edge(first, second, **data)
        return graph

    def _compute_service(self, step: int) -> Dict[str, float]:
        graph = self._service_graph()
        sources = [self.asset_to_node[value] for value in self.source_assets if self.asset_to_node[value] in graph]
        served: Dict[str, float] = {}
        for asset, node in self.asset_to_node.items():
            if self.node_role[asset] == "source":
                served[asset] = 0.0
                continue
            reachable = any(nx.has_path(graph, source, node) for source in sources) if node in graph else False
            network_service = self.demand[asset] if reachable else 0.0
            # Low-integrity telemetry forces a conservative capacity derating;
            # restoring communications can therefore have a real service
            # effect without revealing evaluator truth.
            if self.telemetry_confidence[asset] < 0.30 and not self.isolated[asset]:
                network_service *= 0.45
            served[asset] = min(self.demand[asset], network_service + self.mobile_service[asset])
        return served

    def _cascade(self, step: int) -> None:
        if step <= self.disruption_step:
            return
        service_graph = self._service_graph()
        failed_now: List[Tuple[str, str]] = []
        edge_failures: List[Tuple[str, str]] = []
        for asset, node in self.asset_to_node.items():
            if self.failed[asset] or self.isolated[asset] or self.node_role[asset] == "source":
                continue
            failed_neighbors = [
                self.physical_graph.nodes[neighbor]["asset_id"]
                for neighbor in self.physical_graph.neighbors(node)
                if self.failed[self.physical_graph.nodes[neighbor]["asset_id"]]
            ]
            risk = self.coupling_strength * len(failed_neighbors) / max(self.physical_graph.degree(node), 1)
            tape_index = (step + node) % len(self.stochastic_tape["failure"])
            if failed_neighbors and self.stochastic_tape["failure"][tape_index] < 0.08 * risk:
                self.failed[asset] = True
                source = failed_neighbors[0]
                failed_now.append((source, asset))
                self.cascade_count += 1
                self.maximum_cascade_depth = max(self.maximum_cascade_depth, 2)
            if self.telemetry_corrupt[asset] and not self.isolated[asset]:
                corrupt_risk = 0.10 + 0.12 * self.coupling_strength
                corrupt_tape = self.stochastic_tape["failure"][(tape_index + 5) % len(self.stochastic_tape["failure"])]
                available_edges = [
                    (node, neighbor)
                    for neighbor in self.physical_graph.neighbors(node)
                    if self.physical_graph.edges[node, neighbor].get("service_available", True)
                ]
                if available_edges and corrupt_tape < corrupt_risk:
                    first, second = available_edges[0]
                    self.physical_graph.edges[first, second]["service_available"] = False
                    edge_failures.append((
                        self.physical_graph.nodes[first]["asset_id"],
                        self.physical_graph.nodes[second]["asset_id"],
                    ))
                    self.cascade_count += 1
        for source, target in failed_now:
            self.ledger.append(
                step, "v7_cascade_transition", "utility_simulator",
                {"source": source, "target": target, "depth": 2, "kind": "dependent_service_failure"},
            )
        for source, target in edge_failures:
            self.ledger.append(
                step, "v7_cascade_transition", "utility_simulator",
                {"source": source, "target": target, "depth": 2, "kind": "telemetry_induced_defensive_lockout"},
            )

    def _schedule_action(
        self, decision: V7StructuredDecision, step: int, payload: Dict[str, Any],
    ) -> str:
        chain_id = "UC%08d" % (len(self.causal_chains) + 1)
        payload.update({
            "chain_id": chain_id,
            "actor": decision.proposal.agent_id,
            "action": decision.proposal.proposed_operational_action,
            "scheduled_step": step,
            "delegation_action": decision.delegation_action,
        })
        self.pending_actions.append(payload)
        self.causal_chains[chain_id].append({
            "step": step, "stage": "action_scheduled", "action": payload["action"],
        })
        self.ledger.append(step, "v7_action_scheduled", payload["actor"], deepcopy(payload))
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
            return asdict(V7ActionResult(False, False, proposal.proposed_operational_action, agent.agent_id, target, None, None, "target_outside_scope"))
        observation = agent.vault.observation(agent.agent_id, target)
        action = proposal.proposed_operational_action
        if action not in observation.feasible_physical_actions:
            return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "action_masked"))
        if decision.information_action != "no_information_action":
            self.information_actions += 1
            self.ledger.append(step, "v7_information_action", agent.agent_id, {"action": decision.information_action, "target": target})
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
        self.ledger.append(step, "v7_delegation_decision", agent.agent_id, {"delegation_action": decision.delegation_action, "target": target})
        if action == "no_operational_action" or decision.delegation_action not in ("execute_autonomously", "escalate_operator"):
            return asdict(V7ActionResult(True, False, action, agent.agent_id, target, None, None, "nonphysical_or_withheld"))
        self.actionable_opportunities += 1
        before = self.current_served.get(target, 0.0)
        if action == "dispatch_repair_crew":
            if self.available_crews <= 0 or self.remaining_spares < 1.0:
                return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "crew_or_spare_unavailable"))
            if any(value.get("target") == target and value.get("kind") == "repair" for value in self.pending_actions):
                self.repeated_work_orders += 1
            self.available_crews -= 1
            self.busy_crews += 1
            self.remaining_spares -= 1.0
            self.assigned_spares += 1.0
            travel = 2 + (self.asset_to_node[target] % 4)
            self.crew_travel += travel
            chain = self._schedule_action(decision, step, {
                "kind": "repair", "target": target, "complete_step": step + travel + 2,
                "service_before": before, "crew": 1, "spare": 1.0,
            })
        elif action == "allocate_spare_component":
            if self.remaining_spares < 1.0:
                return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "spare_unavailable"))
            self.remaining_spares -= 1.0
            self.assigned_spares += 1.0
            chain = self._schedule_action(decision, step, {
                "kind": "staged_spare", "target": target, "complete_step": step + 2,
                "service_before": before, "spare": 1.0,
            })
        elif action == "reconfigure_service_edge":
            chain = self._schedule_action(decision, step, {
                "kind": "reconfigure", "target": target, "complete_step": step + 1,
                "service_before": before,
            })
        elif action == "isolate_component":
            chain = self._schedule_action(decision, step, {
                "kind": "isolate", "target": target, "complete_step": step + 1,
                "service_before": before,
            })
        elif action == "deploy_mobile_generation":
            if self.available_generators <= 0 or self.remaining_fuel < 1.0:
                return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "generator_or_fuel_unavailable"))
            self.available_generators -= 1
            self.deployed_generators += 1
            chain = self._schedule_action(decision, step, {
                "kind": "mobile_generation", "target": target, "complete_step": step + 2,
                "service_before": before, "generator": 1,
            })
        elif action == "restore_communication_relay":
            chain = self._schedule_action(decision, step, {
                "kind": "relay_restore", "target": target, "complete_step": step + 2,
                "service_before": before,
            })
        else:
            return asdict(V7ActionResult(False, False, action, agent.agent_id, target, None, None, "unsupported_action"))
        self.physical_actions += 1
        return asdict(V7ActionResult(True, True, action, agent.agent_id, target, step, int(self.pending_actions[-1]["complete_step"]), "scheduled", causal_chain_id=chain))

    def _complete_actions(self, step: int) -> None:
        due = [value for value in self.pending_actions if int(value["complete_step"]) <= step]
        self.pending_actions = [value for value in self.pending_actions if int(value["complete_step"]) > step]
        for action in due:
            kind = action["kind"]
            target = action["target"]
            effect = 0.0
            reached = False
            if kind == "repair":
                was_failed = self.failed[target]
                self.failed[target] = False
                self.telemetry_corrupt[target] = False
                self.telemetry_confidence[target] = max(self.telemetry_confidence[target], 0.82)
                self.available_crews += 1
                self.busy_crews -= 1
                self.assigned_spares -= 1.0
                self.consumed_spares += 1.0
                self.restoration_times[target] = step
                effect = self.demand[target] * self.critical_weight[target] if was_failed else -0.25
                reached = was_failed
            elif kind == "staged_spare":
                self.assigned_spares -= 1.0
                self.consumed_spares += 1.0
                effect = 0.18 if self.failed[target] else -0.12
            elif kind == "reconfigure":
                node = self.asset_to_node[target]
                unavailable = [
                    (first, second) for first, second, data in self.physical_graph.edges(node, data=True)
                    if not bool(data.get("service_available", True))
                ]
                if unavailable:
                    first, second = unavailable[0]
                    self.physical_graph.edges[first, second]["service_available"] = True
                    self.physical_graph.edges[first, second]["emergency_authorized"] = True
                    effect = 0.45 * self.demand[target]
                    reached = True
                else:
                    # Unnecessary switching carries bounded safety harm.
                    self.unsafe_switching += 1
                    effect = -0.22
            elif kind == "isolate":
                compromised = self.telemetry_corrupt[target] or self._downstream_cascade_risk(target) > 0.45
                self.isolated[target] = True
                # Emergency isolation is a bounded protective action, not a
                # permanent deletion of service.  Automatic expiry preserves
                # the possibility of both short-term harm and delayed cascade
                # prevention without requiring evaluator intervention.
                self.isolation_release_step[target] = step + 4
                if compromised:
                    effect = 0.28 * self.coupling_strength
                else:
                    effect = -self.demand[target] * self.critical_weight[target] * 0.35
                    self.unsafe_switching += 1
            elif kind == "mobile_generation":
                amount = min(self.demand[target], 1.2)
                self.mobile_service[target] += amount
                effect = amount * self.critical_weight[target] - 0.18
                reached = amount > 0
            elif kind == "relay_restore":
                restored = 0
                controller_nodes = {
                    self.agent_nodes[agent_id]
                    for agent_id, agent in self.agents.items()
                    if target in agent.identity.asset_scope
                }
                ordered_edges = sorted(
                    self.disabled_communication_edges,
                    key=lambda edge: (
                        0 if edge[0] in controller_nodes or edge[1] in controller_nodes else 1,
                        int(edge[0]), int(edge[1]),
                    ),
                )
                for first, second in ordered_edges:
                    if not self.communication_graph.edges[first, second].get("available", True):
                        self.communication_graph.edges[first, second]["available"] = True
                        restored += 1
                        if restored >= 2:
                            break
                self.telemetry_confidence[target] = min(0.9, self.telemetry_confidence[target] + 0.35)
                effect = 0.08 * restored
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
            self.causal_chains[action["chain_id"]].extend([
                {"step": step, "stage": "resource_or_authority_change", "target": target},
                {"step": step, "stage": "service_outcome_change", "causal_effect": effect, "reached_service": reached},
            ])
            self.maximum_cascade_depth = max(self.maximum_cascade_depth, len(self.causal_chains[action["chain_id"]]))
            self.ledger.append(step, "v7_action_completed", action["actor"], record)

    def _release_expired_isolations(self, step: int) -> None:
        due = sorted(
            asset for asset, release_step in self.isolation_release_step.items()
            if int(release_step) <= int(step)
        )
        for asset in due:
            self.isolated[asset] = False
            del self.isolation_release_step[asset]
            self.ledger.append(
                step, "v7_domain_transition", "utility_simulator",
                {
                    "kind": "bounded_defensive_isolation_expired",
                    "asset": asset,
                },
            )

    def _consume_mobile_fuel(self) -> None:
        active = sum(value > 0.0 for value in self.mobile_service.values())
        required = min(self.remaining_fuel, 0.12 * active)
        self.remaining_fuel -= required
        self.consumed_fuel += required
        if required + 1e-12 < 0.12 * active:
            for asset in self.mobile_service:
                self.mobile_service[asset] = 0.0

    def advance_domain(self, step: int) -> None:
        self._apply_disruption(step)
        self._complete_actions(step)
        self._release_expired_isolations(step)
        self._consume_mobile_fuel()
        self._cascade(step)
        self.current_served = self._compute_service(step)
        step_loss = 0.0
        critical_loss = 0.0
        for asset in self.node_role:
            unserved = max(0.0, self.demand[asset] - self.current_served[asset])
            weighted = unserved * self.critical_weight[asset]
            step_loss += weighted
            if self.node_role[asset] == "critical_load":
                critical_loss += weighted
        self.service_loss_auc += step_loss
        self.unserved_critical_load_auc += critical_loss
        self.ledger.append(
            step, "v7_service_transition", "utility_simulator",
            {
                "unserved_load": step_loss,
                "unserved_critical_load": critical_loss,
                "service_loss_auc": self.service_loss_auc,
            },
        )
        self._record_resource_state(step)

    def _record_resource_state(self, step: int) -> None:
        values = (
            ("crews", self.initial_crews, self.available_crews, 0.0, self.busy_crews),
            ("spares", self.initial_spares, self.remaining_spares, self.consumed_spares, self.assigned_spares),
            ("generators", self.initial_generators, self.available_generators, 0.0, self.deployed_generators),
            ("fuel", self.initial_fuel, self.remaining_fuel, self.consumed_fuel, 0.0),
        )
        for resource, initial, remaining, consumed, in_transit in values:
            self.ledger.append(
                step, "v7_resource_transition", "utility_simulator",
                {
                    "resource": resource, "initial": initial,
                    "remaining": remaining, "consumed": consumed,
                    "in_transit": in_transit, "delivered": 0.0, "losses": 0.0,
                },
                private_to="evaluator",
            )
        self.ledger.append(
            step, "v7_domain_transition", "utility_simulator",
            {
                "available_crews": self.available_crews,
                "busy_crews": self.busy_crews,
                "remaining_spares": self.remaining_spares,
                "failed_components": sum(self.failed.values()),
            },
        )

    def conservation_report(self) -> Dict[str, Any]:
        residuals = {
            "crews": float(self.initial_crews - self.available_crews - self.busy_crews),
            "spares": float(self.initial_spares - self.remaining_spares - self.assigned_spares - self.consumed_spares),
            "generators": float(self.initial_generators - self.available_generators - self.deployed_generators),
            "fuel": float(self.initial_fuel - self.remaining_fuel - self.consumed_fuel),
        }
        maximum = max(abs(value) for value in residuals.values())
        feasible = (
            maximum <= 1e-9
            and min(
                self.available_crews, self.busy_crews, self.remaining_spares,
                self.assigned_spares, self.available_generators,
                self.deployed_generators, self.remaining_fuel,
            ) >= -1e-9
        )
        return {"feasible": feasible, "maximum_residual": maximum, "residuals": residuals}

    def inject_conservation_fault_for_test(self, resource: str, quantity: float) -> None:
        if resource == "spares":
            self.remaining_spares += float(quantity)
        elif resource == "fuel":
            self.remaining_fuel += float(quantity)
        else:
            raise ValueError("unsupported deliberate fault resource")

    def metrics(self) -> Dict[str, Any]:
        conservation = self.conservation_report()
        return {
            "service_loss": float(self.service_loss_auc),
            "unserved_critical_load_auc": float(self.unserved_critical_load_auc),
            "restoration_time_mean": float(np.mean(list(self.restoration_times.values()))) if self.restoration_times else float(self.spec.horizon),
            "cascaded_failures": int(self.cascade_count),
            "maximum_cascade_depth": int(self.maximum_cascade_depth),
            "unsafe_switching_or_isolation": int(self.unsafe_switching),
            "crew_utilization": float(self.busy_crews / max(self.initial_crews, 1)),
            "crew_travel": float(self.crew_travel),
            "repeated_work_orders": int(self.repeated_work_orders),
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
            "belief_consensus_recovery": float(np.mean(list(self.telemetry_confidence.values()))),
        }
