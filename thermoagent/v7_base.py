"""Shared ledger, communication, privacy, and estimator infrastructure for V7.

Domain transitions are intentionally *not* implemented here. Humanitarian and
utility simulators each own their state evolution and action consequences.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .events import EventLedger
from .types import Message
from .v6_entropy import entropy_spectrum, generalized_disagreement, shannon_entropy
from .v7_agents import IndependentV7Agent
from .v7_entropy import (
    V7EntropySketch, encoded_sketch_bytes, graph_disagreement_from_active_edges,
    weighted_information_state,
)
from .v7_topology import generate_graph, topology_diagnostics
from .v7_types import (
    DEFAULT_COMPLEXITY, V7DistributedState, V7PrivateObservation,
    V7RiskContext, V7StructuredDecision,
)


LEVEL_VALUES = {"low": 0.20, "medium": 0.55, "high": 0.85}


class V7CoupledEnvironment(ABC):
    """Abstract multi-agent shell; subclasses implement every domain transition."""

    belief_modes: Tuple[str, ...] = ()

    def __init__(
        self,
        application: str,
        complexity: str,
        coupling: str,
        fragmentation: str,
        network_disruption: str,
        topology_family: str,
        environment_seed: int,
        information_condition: str = "private_fragmented",
        sketch_policy: str = "event_triggered",
        operational_communication_policy: str = "agent_event_triggered",
    ) -> None:
        if complexity not in DEFAULT_COMPLEXITY:
            raise ValueError("unknown complexity level")
        for value in (coupling, fragmentation, network_disruption):
            if value not in LEVEL_VALUES:
                raise ValueError("complexity factor must be low, medium, or high")
        if information_condition not in ("private_fragmented", "public_shared"):
            raise ValueError("invalid information condition")
        if sketch_policy not in ("none", "periodic", "event_triggered", "always_on"):
            raise ValueError("invalid sketch policy")
        if operational_communication_policy not in (
            "none", "periodic", "always_on", "kpi_event_triggered",
            "agent_event_triggered",
        ):
            raise ValueError("invalid operational communication policy")
        self.application = application
        self.spec = DEFAULT_COMPLEXITY[complexity]
        self.complexity = complexity
        self.coupling_level = coupling
        self.fragmentation_level = fragmentation
        self.network_disruption_level = network_disruption
        self.coupling_strength = LEVEL_VALUES[coupling]
        self.fragmentation = LEVEL_VALUES[fragmentation]
        self.network_disruption = LEVEL_VALUES[network_disruption]
        self.topology_family = topology_family
        self.environment_seed = int(environment_seed)
        self.information_condition = information_condition
        self.sketch_policy = sketch_policy
        self.operational_communication_policy = operational_communication_policy
        self.rng = np.random.RandomState(self.environment_seed)
        self.ledger = EventLedger()
        self.agents: Dict[str, IndependentV7Agent] = {}
        self.agent_nodes: Dict[str, int] = {}
        self.node_agents: Dict[int, str] = {}
        self.pending_messages: DefaultDict[int, List[Message]] = defaultdict(list)
        self.sketch_cache: DefaultDict[str, Dict[Tuple[str, str], V7EntropySketch]] = defaultdict(dict)
        self.last_sent_entropy: Dict[Tuple[str, str], float] = {}
        self.last_operational_kpi: Dict[Tuple[str, str], float] = {}
        self.forwarded_sketches: set = set()
        self.entropy_history: DefaultDict[Tuple[str, str, float], List[float]] = defaultdict(list)
        self.disagreement_history: DefaultDict[Tuple[str, str, float], List[float]] = defaultdict(list)
        self.message_counter = 0
        self.operational_messages = 0
        self.sketch_messages = 0
        self.dropped_messages = 0
        self.operational_bytes = 0
        self.sketch_bytes = 0
        self.edge_message_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.cross_community_messages = 0
        self.sketch_messages_by_step: DefaultDict[int, int] = defaultdict(int)
        self.evaluator_estimation_errors: Dict[Tuple[int, str, str], float] = {}
        self.causal_chains: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.stochastic_tape = self._make_stochastic_tape()
        self.stochastic_tape_digest = hashlib.sha256(
            json.dumps(self.stochastic_tape, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._initialize_domain()
        if len(self.agents) != self.spec.agent_count:
            raise RuntimeError(
                "domain initialized %d agents; complexity requires %d"
                % (len(self.agents), self.spec.agent_count)
            )
        self.communication_graph = generate_graph(
            topology_family, self.spec.agent_count, self.environment_seed + 7107,
        )
        for index, agent_id in enumerate(sorted(self.agents)):
            self.agent_nodes[agent_id] = index
            self.node_agents[index] = agent_id
            self.communication_graph.nodes[index]["agent_id"] = agent_id
        diagnostics = topology_diagnostics(self.communication_graph, topology_family)
        self.ledger.append(
            0, "v7_topology_snapshot", "simulator",
            {
                "application": application,
                "complexity": complexity,
                "coupling": coupling,
                "fragmentation": fragmentation,
                "network_disruption": network_disruption,
                "information_condition": information_condition,
                "sketch_policy": sketch_policy,
                "operational_communication_policy": operational_communication_policy,
                "diagnostics": asdict(diagnostics),
                "stochastic_tape_digest": self.stochastic_tape_digest,
            },
        )

    def _make_stochastic_tape(self) -> Dict[str, List[float]]:
        horizon = self.spec.horizon + 12
        return {
            "demand": [float(value) for value in self.rng.normal(0.0, 1.0, horizon)],
            "failure": [float(value) for value in self.rng.uniform(0.0, 1.0, horizon)],
            "loss": [float(value) for value in self.rng.uniform(0.0, 1.0, horizon)],
            "recovery": [float(value) for value in self.rng.uniform(0.0, 1.0, horizon)],
            "message": [float(value) for value in self.rng.uniform(0.0, 1.0, horizon)],
        }

    @abstractmethod
    def _initialize_domain(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def deliver_private_observations(self, step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def advance_domain(self, step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_and_schedule(
        self, decision: V7StructuredDecision, step: int,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def conservation_report(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _message_available(self, sender: str, recipient: str) -> bool:
        first = self.agent_nodes[sender]
        second = self.agent_nodes[recipient]
        if not self.communication_graph.has_edge(first, second):
            return False
        return bool(self.communication_graph.edges[first, second].get("available", True))

    def _edge_data(self, sender: str, recipient: str) -> Mapping[str, Any]:
        return self.communication_graph.edges[
            self.agent_nodes[sender], self.agent_nodes[recipient]
        ]

    def send_message(
        self,
        sender: str,
        recipient: str,
        kind: str,
        payload: Mapping[str, Any],
        step: int,
        sketch: bool = False,
    ) -> bool:
        if sender not in self.agents or recipient not in self.agents:
            raise ValueError("V7 message endpoint does not exist")
        self.message_counter += 1
        encoded = len(json.dumps(dict(payload), sort_keys=True).encode("utf-8")) + 40
        if sketch:
            self.sketch_messages += 1
            self.sketch_bytes += encoded
            self.sketch_messages_by_step[int(step)] += 1
        else:
            self.operational_messages += 1
            self.operational_bytes += encoded
        first_node = self.agent_nodes[sender]
        second_node = self.agent_nodes[recipient]
        first_community = self.communication_graph.nodes[first_node].get("community")
        second_community = self.communication_graph.nodes[second_node].get("community")
        if first_community != second_community:
            self.cross_community_messages += 1
        self.edge_message_counts[(sender, recipient)] += 1
        if not self._message_available(sender, recipient):
            self.dropped_messages += 1
            self.ledger.append(
                step, "v7_message_dropped", sender,
                {
                    "recipient": recipient, "kind": kind,
                    "encoded_bytes": encoded, "reason": "partition_or_no_edge",
                },
            )
            return False
        data = self._edge_data(sender, recipient)
        reliability = float(data.get("reliability", 1.0))
        tape = self.stochastic_tape["message"][step % len(self.stochastic_tape["message"])]
        if tape > reliability:
            self.dropped_messages += 1
            self.ledger.append(
                step, "v7_message_dropped", sender,
                {
                    "recipient": recipient, "kind": kind,
                    "encoded_bytes": encoded, "reason": "stochastic_loss",
                },
            )
            return False
        latency = int(data.get("latency", 1))
        message = Message(
            message_id="V7M%08d" % self.message_counter,
            sender=sender,
            recipient=recipient,
            kind=kind,
            payload=deepcopy(dict(payload)),
            sent_step=int(step),
            deliver_step=int(step + latency),
            public=False,
        )
        self.pending_messages[message.deliver_step].append(message)
        self.ledger.append(
            step, "v7_message_sent", sender,
            {
                "message_id": message.message_id, "recipient": recipient,
                "kind": kind, "deliver_step": message.deliver_step,
                "encoded_bytes": encoded,
            },
        )
        return True

    def operational_communication_action(
        self,
        requested_action: str,
        agent_id: str,
        asset: str,
        severity: float,
        step: int,
    ) -> str:
        """Resolve a deployable operational-message policy.

        This is deliberately separate from thermodynamic sketch exchange so
        that every traffic class can be costed and ablated independently.
        The KPI trigger sees only the local severity available to the sender.
        """
        policy = self.operational_communication_policy
        key = (str(agent_id), str(asset))
        previous = self.last_operational_kpi.get(key)
        self.last_operational_kpi[key] = float(severity)
        requested = str(requested_action)
        if policy == "none":
            return "no_communication_action"
        if policy == "always_on":
            return (
                requested if requested != "no_communication_action"
                else "send_targeted_summary"
            )
        if policy == "periodic":
            if step % max(2 * self.spec.decision_interval, 1) != 0:
                return "no_communication_action"
            return (
                requested if requested != "no_communication_action"
                else "send_targeted_summary"
            )
        if policy == "kpi_event_triggered":
            # The first value calibrates the local reference but does not
            # count as a disruption-triggered operational alert.
            if previous is None or abs(float(severity) - previous) < 0.12:
                return "no_communication_action"
            return (
                requested if requested != "no_communication_action"
                else "send_targeted_summary"
            )
        return requested

    def deliver_messages(self, step: int) -> None:
        for message in self.pending_messages.pop(step, []):
            self.agents[message.recipient].receive(message)
            evidence_integrated = self.agents[message.recipient].integrate_delivered_evidence(message)
            if message.kind == "entropy_sketch":
                payload = message.payload
                sketch = V7EntropySketch(
                    sender=str(payload.get("origin_sender", message.sender)),
                    focal_asset=str(payload["focal_asset"]),
                    belief_distribution=tuple(payload["belief_distribution"]),
                    telemetry_confidence=float(payload["telemetry_confidence"]),
                    sent_step=int(payload["sent_step"]),
                    encoded_bytes=int(payload["encoded_bytes"]),
                    hop_count=int(payload.get("hop_count", 0)),
                )
                self.sketch_cache[message.recipient][
                    (sketch.sender, sketch.focal_asset)
                ] = sketch
            self.ledger.append(
                step, "v7_message_delivered", message.sender,
                {
                    "message_id": message.message_id,
                    "recipient": message.recipient,
                    "kind": message.kind,
                    "evidence_integrated": evidence_integrated,
                },
                private_to=message.recipient,
            )

    def _should_send_sketch(
        self, agent_id: str, asset: str, belief: Sequence[float], step: int,
    ) -> bool:
        if self.sketch_policy == "none":
            return False
        if self.sketch_policy == "always_on":
            return True
        if self.sketch_policy == "periodic":
            return step % max(2 * self.spec.decision_interval, 1) == 0
        current = shannon_entropy(belief)
        previous = self.last_sent_entropy.get((agent_id, asset))
        return previous is None or abs(current - previous) >= 0.045

    def exchange_entropy_sketches(self, step: int) -> None:
        for agent_id in sorted(self.agents):
            agent = self.agents[agent_id]
            if not agent.private_beliefs:
                continue
            # Persistent agents rotate among their multiple assets; sketches
            # therefore exercise cross-location information over the episode.
            assets = sorted(agent.private_beliefs)
            asset = assets[(step // max(self.spec.decision_interval, 1)) % len(assets)]
            belief = agent.private_beliefs[asset]
            if not self._should_send_sketch(agent_id, asset, belief, step):
                continue
            observation = agent.vault.observation(agent_id, asset)
            self.last_sent_entropy[(agent_id, asset)] = shannon_entropy(belief)
            node = self.agent_nodes[agent_id]
            for neighbor_node in sorted(self.communication_graph.neighbors(node)):
                recipient = self.node_agents[int(neighbor_node)]
                size = encoded_sketch_bytes(len(belief))
                payload = {
                    "origin_sender": agent_id,
                    "focal_asset": asset,
                    "belief_distribution": list(belief),
                    "telemetry_confidence": observation.telemetry_confidence,
                    "sent_step": step,
                    "encoded_bytes": size,
                    "hop_count": 0,
                }
                self.send_message(
                    agent_id, recipient, "entropy_sketch", payload, step, sketch=True,
                )
                self.ledger.append(
                    step, "v7_entropy_sketch", agent_id,
                    {
                        "recipient": recipient, "focal_asset": asset,
                        "encoded_bytes": size,
                    },
                )

        # Bounded two-hop gossip. Intermediaries may forward only an already
        # delivered sketch, never a peer's private vault. Each hop is counted,
        # delayed, and can be blocked by a partition.
        for intermediary in sorted(self.agents):
            node = self.agent_nodes[intermediary]
            cache = list(sorted(self.sketch_cache[intermediary].items()))
            for (origin, asset), sketch in cache:
                if (
                    sketch.hop_count >= 2
                    or step - sketch.sent_step > 3 * self.spec.decision_interval
                ):
                    continue
                for neighbor_node in sorted(self.communication_graph.neighbors(node)):
                    recipient = self.node_agents[int(neighbor_node)]
                    if recipient == origin:
                        continue
                    forward_key = (
                        intermediary, origin, asset, sketch.sent_step, recipient,
                    )
                    if forward_key in self.forwarded_sketches:
                        continue
                    self.forwarded_sketches.add(forward_key)
                    payload = {
                        "origin_sender": origin,
                        "focal_asset": asset,
                        "belief_distribution": list(sketch.belief_distribution),
                        "telemetry_confidence": sketch.telemetry_confidence,
                        "sent_step": sketch.sent_step,
                        "encoded_bytes": sketch.encoded_bytes,
                        "hop_count": sketch.hop_count + 1,
                    }
                    self.send_message(
                        intermediary, recipient, "entropy_sketch", payload,
                        step, sketch=True,
                    )
                    self.ledger.append(
                        step, "v7_entropy_sketch", intermediary,
                        {
                            "recipient": recipient,
                            "origin_sender": origin,
                            "focal_asset": asset,
                            "encoded_bytes": sketch.encoded_bytes,
                            "hop_count": sketch.hop_count + 1,
                        },
                    )

    def distributed_state(
        self, agent_id: str, asset: str, step: int, q: float = 1.0,
    ) -> V7DistributedState:
        agent = self.agents[agent_id]
        own_belief = agent.private_beliefs[asset]
        own_observation = agent.vault.observation(agent_id, asset)
        beliefs: List[Sequence[float]] = [own_belief]
        confidences = [own_observation.telemetry_confidence]
        ages = [0.0]
        contributors = [agent_id]
        known_scoped_agents = [
            candidate_id for candidate_id, candidate in self.agents.items()
            if asset in candidate.identity.asset_scope
        ]
        for (sender, focal_asset), sketch in sorted(self.sketch_cache[agent_id].items()):
            if focal_asset != asset or sender == agent_id:
                continue
            beliefs.append(sketch.belief_distribution)
            confidences.append(sketch.telemetry_confidence)
            ages.append(float(step - sketch.sent_step))
            contributors.append(sender)
        state = weighted_information_state(beliefs, confidences, ages, q=q)
        belief_map = {identifier: belief for identifier, belief in zip(contributors, beliefs)}
        graph_disagreement = graph_disagreement_from_active_edges(
            belief_map, self.communication_graph, q=q,
        )
        key = (agent_id, asset, float(q))
        entropy = float(state["pooled_uncertainty"])
        disagreement = float(state["generalized_disagreement"])
        previous_entropy = self.entropy_history[key][-1] if self.entropy_history[key] else entropy
        previous_disagreement = (
            self.disagreement_history[key][-1]
            if self.disagreement_history[key] else disagreement
        )
        self.entropy_history[key].append(entropy)
        self.disagreement_history[key].append(disagreement)
        missing = sorted(set(known_scoped_agents) - set(contributors))
        consensus_residual = float(
            len(missing) / max(len(known_scoped_agents), 1)
            + np.mean([min(age / 10.0, 1.0) for age in ages]) * 0.25
        )
        output = V7DistributedState(
            q=q,
            local_entropy=shannon_entropy(own_belief),
            average_local_uncertainty=float(state["average_local_uncertainty"]),
            pooled_uncertainty=entropy,
            generalized_disagreement=disagreement,
            graph_disagreement=graph_disagreement,
            consensus=float(state["consensus"]),
            consensus_residual=consensus_residual,
            entropy_slope=entropy - previous_entropy,
            disagreement_slope=disagreement - previous_disagreement,
            contributors=tuple(contributors),
            missing_agents=tuple(missing),
            sketch_messages=self.sketch_messages,
            sketch_bytes=self.sketch_bytes,
            dropped_sketch_messages=self.dropped_messages,
            maximum_message_age=max(ages),
        )
        self.ledger.append(
            step, "v7_distributed_state", "distributed_estimator",
            {"recipient": agent_id, "asset": asset, "state": asdict(output)},
            private_to=agent_id,
        )
        return output

    def risk_context(
        self, decision: V7StructuredDecision, step: int,
    ) -> V7RiskContext:
        proposal = decision.proposal
        agent = self.agents[proposal.agent_id]
        asset = str(proposal.target_asset_or_location)
        observation = agent.vault.observation(proposal.agent_id, asset)
        belief = agent.private_beliefs[asset]
        states = {
            q: self.distributed_state(proposal.agent_id, asset, step, q=q)
            for q in (0.5, 1.0, 1.5, 2.0, 3.0)
        }
        spectrum = entropy_spectrum(belief)
        primary = states[1.0]
        scoped = [
            candidate for candidate in self.agents.values()
            if asset in candidate.identity.asset_scope
            and asset in candidate.private_beliefs
        ]
        global_beliefs = [candidate.private_beliefs[asset] for candidate in scoped]
        global_confidences = [
            candidate.vault.observation(candidate.agent_id, asset).telemetry_confidence
            for candidate in scoped
        ]
        global_state = weighted_information_state(
            global_beliefs, global_confidences,
            [0.0 for _ in global_beliefs], q=1.0,
        )
        estimation_error = abs(
            primary.generalized_disagreement
            - float(global_state["generalized_disagreement"])
        )
        self.evaluator_estimation_errors[(int(step), proposal.agent_id, asset)] = float(estimation_error)
        self.ledger.append(
            step, "v7_distributed_state", "evaluator",
            {
                "recipient": proposal.agent_id, "asset": asset,
                "distributed_estimation_error": estimation_error,
                "global_contributor_count": len(global_beliefs),
            },
            private_to="evaluator",
        )
        return V7RiskContext(
            step=step,
            proposal=proposal,
            local_kpis=dict(observation.local_kpis),
            predictive_uncertainty=float(1.0 - proposal.action_probability),
            action_value_margin=proposal.value_margin,
            shannon_local=float(spectrum["q_1_0"]),
            tsallis_0_5_local=float(spectrum["q_0_5"]),
            tsallis_1_5_local=float(spectrum["q_1_5"]),
            tsallis_2_local=float(spectrum["q_2_0"]),
            tsallis_3_local=float(spectrum["q_3_0"]),
            gini_simpson_local=float(spectrum["q_2_0"]),
            average_local_uncertainty=primary.average_local_uncertainty,
            pooled_uncertainty=primary.pooled_uncertainty,
            js_disagreement=primary.generalized_disagreement,
            jt_disagreement_0_5=states[0.5].generalized_disagreement,
            jt_disagreement_1_5=states[1.5].generalized_disagreement,
            jt_disagreement_2=states[2.0].generalized_disagreement,
            jt_disagreement_3=states[3.0].generalized_disagreement,
            graph_disagreement=primary.graph_disagreement,
            consensus=primary.consensus,
            consensus_residual=primary.consensus_residual,
            entropy_slope=primary.entropy_slope,
            disagreement_slope=primary.disagreement_slope,
            communication_reliability=observation.communication_reliability,
            coupling_strength=self.coupling_strength,
            fragmentation=self.fragmentation,
            size_normalized=self.spec.agent_count / 80.0,
            contributors=primary.contributors,
            missing_agents=primary.missing_agents,
        )

    def privacy_audit(self) -> Dict[str, Any]:
        violations = []
        for event in self.ledger.events:
            if event.kind in ("v7_private_observation", "v7_belief_update"):
                if event.private_to is None:
                    violations.append(event.event_id + ":private_event_public")
            if event.kind == "v7_distributed_state":
                if event.private_to is None:
                    violations.append(event.event_id + ":distributed_state_public")
        return {"pass": not violations, "violations": violations}

    def process_commitments(self, step: int) -> None:
        """Let recipients independently accept, counter, or reject offers."""
        commitments = getattr(self, "commitments", {})
        for commitment_id in sorted(commitments):
            commitment = commitments[commitment_id]
            if commitment.status not in ("proposed", "countered"):
                continue
            recipient = self.agents[commitment.recipient]
            decision = recipient.evaluate_commitment(commitment)
            if decision == "counter":
                commitment.revision += 1
                commitment.quantity *= 0.72
                commitment.status = "countered"
                kind = "counteroffer"
            elif decision == "accept":
                commitment.status = "accepted"
                kind = "commitment"
            else:
                commitment.status = "rejected"
                kind = "commitment"
            for agent_id in (commitment.proposer, commitment.recipient):
                if commitment_id in self.agents[agent_id].commitments:
                    self.agents[agent_id].commitments[commitment_id] = deepcopy(commitment)
            self.ledger.append(
                step, kind, commitment.recipient,
                {
                    "commitment_id": commitment_id,
                    "status": commitment.status,
                    "revision": commitment.revision,
                    "quantity": commitment.quantity,
                    "proposer": commitment.proposer,
                    "recipient": commitment.recipient,
                },
                private_to=commitment.proposer,
            )
            self.ledger.append(
                step, "v7_commitment_transition", commitment.recipient,
                {
                    "commitment_id": commitment_id,
                    "status": commitment.status,
                    "revision": commitment.revision,
                },
            )

    @property
    def total_messages(self) -> int:
        return self.operational_messages + self.sketch_messages

    @property
    def total_bytes(self) -> int:
        return self.operational_bytes + self.sketch_bytes
