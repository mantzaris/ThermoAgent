"""Actual-wire distributed belief monitoring over the frozen V7 domains.

V8 inherits domain dynamics and agent privacy from V7 but never invokes V7's
formula-based sketch accounting.  This module owns scheduling, serialization,
delivery, forwarding, local reconstruction, and evaluator-only error scoring.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Message
from .v6_entropy import generalized_disagreement, probability_vector, weighted_pooled_belief
from .v7_base import V7CoupledEnvironment
from .v8_trigger import LocalBeliefScheduler, TriggerConfig, TriggerDecision
from .v8_wire import DecodedBeliefSketch, decode_belief_sketch, encode_belief_sketch


@dataclass(frozen=True)
class WireTransit:
    message_id: str
    origin: str
    transmitter: str
    recipient: str
    asset: str
    transmission_step: int
    delivery_step: int
    wire_bytes: bytes


@dataclass(frozen=True)
class CachedBeliefSketch:
    origin: str
    asset: str
    belief: Tuple[float, ...]
    confidence: float
    sent_step: int
    delivered_step: int
    hop_count: int
    wire_bytes: int


class V8BeliefNetwork:
    """Distributed belief exchange with exact simulated wire accounting."""

    def __init__(
        self,
        environment: V7CoupledEnvironment,
        trigger_config: TriggerConfig,
        *,
        encoding: str = "fp16",
        maximum_hops: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        if environment.sketch_policy != "none":
            raise ValueError("V8 requires the inherited V7 sketch policy to be disabled")
        self.environment = environment
        self.trigger_config = trigger_config
        self.scheduler = LocalBeliefScheduler(
            trigger_config,
            deterministic_seed=(environment.environment_seed if seed is None else int(seed)),
        )
        self.encoding = str(encoding)
        self.maximum_hops = int(maximum_hops)
        if not 0 <= self.maximum_hops <= 8:
            raise ValueError("maximum_hops must be between zero and eight")
        self.agent_ids = tuple(sorted(environment.agents))
        self.agent_registry = {identifier: index + 1 for index, identifier in enumerate(self.agent_ids)}
        assets = sorted({
            asset
            for agent in environment.agents.values()
            for asset in agent.identity.asset_scope
        })
        self.asset_registry = {identifier: index + 1 for index, identifier in enumerate(assets)}
        self.agent_from_registry = {value: key for key, value in self.agent_registry.items()}
        self.asset_from_registry = {value: key for key, value in self.asset_registry.items()}
        self.pending: DefaultDict[int, List[WireTransit]] = defaultdict(list)
        self.cache: DefaultDict[str, Dict[Tuple[str, str], CachedBeliefSketch]] = defaultdict(dict)
        self.forwarded: set = set()
        self.message_counter = 0
        self.attempted_messages = 0
        self.transmitted_messages = 0
        self.delivered_messages = 0
        self.dropped_messages = 0
        self.forwarded_messages = 0
        self.retries = 0
        self.header_bytes = 0
        self.payload_bytes = 0
        self.integrity_bytes = 0
        self.total_on_wire_bytes = 0
        self.delivered_useful_bytes = 0
        self.redundant_bytes = 0
        self.stale_bytes = 0
        self.quantization_l1_error_sum = 0.0
        self.edge_attempts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.edge_transmissions: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.edge_bytes: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.trigger_evaluations = 0
        self.trigger_activations = 0
        self.trigger_compute_seconds = 0.0
        self.trigger_reasons: DefaultDict[str, int] = defaultdict(int)
        self.transmissions_by_step: DefaultDict[int, int] = defaultdict(int)
        self.bytes_by_step: DefaultDict[int, int] = defaultdict(int)
        self.last_connectivity: Dict[str, bool] = {
            agent_id: self._has_available_neighbor(agent_id) for agent_id in self.agent_ids
        }
        self.estimation_rows: List[Dict[str, Any]] = []
        self.delivery_rows: List[Dict[str, Any]] = []
        self.trigger_rows: List[Dict[str, Any]] = []
        self.action_belief_updates = 0

    def _has_available_neighbor(self, agent_id: str) -> bool:
        node = self.environment.agent_nodes[agent_id]
        return any(
            bool(self.environment.communication_graph.edges[node, neighbor].get("available", True))
            for neighbor in self.environment.communication_graph.neighbors(node)
        )

    def _loss_draw(
        self, transmitter: str, recipient: str, origin: str, asset: str,
        sent_step: int, hop_count: int,
    ) -> float:
        value = "%d|%s|%s|%s|%s|%d|%d" % (
            self.environment.environment_seed, transmitter, recipient,
            origin, asset, int(sent_step), int(hop_count),
        )
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    def _edge_available(self, transmitter: str, recipient: str) -> bool:
        first = self.environment.agent_nodes[transmitter]
        second = self.environment.agent_nodes[recipient]
        return bool(
            self.environment.communication_graph.has_edge(first, second)
            and self.environment.communication_graph.edges[first, second].get("available", True)
        )

    def _send(
        self,
        *,
        origin: str,
        transmitter: str,
        recipient: str,
        asset: str,
        belief: Sequence[float],
        confidence: float,
        original_sent_step: int,
        transmission_step: int,
        hop_count: int,
    ) -> bool:
        self.attempted_messages += 1
        self.edge_attempts[(transmitter, recipient)] += 1
        encoded = encode_belief_sketch(
            origin_sender_id=self.agent_registry[origin],
            transmitter_id=self.agent_registry[transmitter],
            target_asset_id=self.asset_registry[asset],
            sent_step=int(original_sent_step),
            confidence=float(confidence),
            belief=belief,
            encoding=self.encoding,
            hop_count=int(hop_count),
            flags=(1 if hop_count else 0),
        )
        if not self._edge_available(transmitter, recipient):
            self.dropped_messages += 1
            self.environment.ledger.append(
                transmission_step, "v8_sketch_attempt_blocked", transmitter,
                {
                    "origin": origin, "recipient": recipient, "asset": asset,
                    "reason": "partition_or_unavailable_edge", "hop_count": hop_count,
                    "would_be_wire_bytes": encoded.total_bytes,
                },
            )
            return False
        self.transmitted_messages += 1
        self.forwarded_messages += int(hop_count > 0)
        self.header_bytes += encoded.header_bytes
        self.payload_bytes += encoded.payload_bytes
        self.integrity_bytes += encoded.integrity_bytes
        self.total_on_wire_bytes += encoded.total_bytes
        self.quantization_l1_error_sum += encoded.quantization_l1_error
        self.edge_transmissions[(transmitter, recipient)] += 1
        self.edge_bytes[(transmitter, recipient)] += encoded.total_bytes
        self.transmissions_by_step[int(transmission_step)] += 1
        self.bytes_by_step[int(transmission_step)] += encoded.total_bytes
        edge = self.environment.communication_graph.edges[
            self.environment.agent_nodes[transmitter],
            self.environment.agent_nodes[recipient],
        ]
        reliability = float(edge.get("reliability", 1.0))
        dropped = self._loss_draw(
            transmitter, recipient, origin, asset, original_sent_step, hop_count,
        ) > reliability
        self.message_counter += 1
        message_id = "V8B%09d" % self.message_counter
        if dropped:
            self.dropped_messages += 1
            self.environment.ledger.append(
                transmission_step, "v8_sketch_dropped", transmitter,
                {
                    "message_id": message_id, "origin": origin,
                    "recipient": recipient, "asset": asset,
                    "reason": "stochastic_packet_loss", "hop_count": hop_count,
                    "wire_bytes": encoded.total_bytes,
                },
            )
            return True
        latency = max(1, int(edge.get("latency", 1)))
        transit = WireTransit(
            message_id=message_id,
            origin=origin,
            transmitter=transmitter,
            recipient=recipient,
            asset=asset,
            transmission_step=int(transmission_step),
            delivery_step=int(transmission_step + latency),
            wire_bytes=encoded.wire_bytes,
        )
        self.pending[transit.delivery_step].append(transit)
        self.environment.ledger.append(
            transmission_step, "v8_sketch_transmitted", transmitter,
            {
                "message_id": message_id, "origin": origin,
                "recipient": recipient, "asset": asset,
                "delivery_step": transit.delivery_step, "hop_count": hop_count,
                "encoding": self.encoding, "header_bytes": encoded.header_bytes,
                "payload_bytes": encoded.payload_bytes,
                "integrity_bytes": encoded.integrity_bytes,
                "wire_bytes": encoded.total_bytes,
            },
        )
        return True

    def _integrate(
        self, recipient: str, decoded: DecodedBeliefSketch, asset: str,
        delivered_step: int,
    ) -> bool:
        agent = self.environment.agents[recipient]
        if asset not in agent.identity.asset_scope:
            return False
        synthetic = Message(
            message_id="V8-INTEGRATE",
            sender=self.agent_from_registry[decoded.origin_sender_id],
            recipient=recipient,
            kind="v8_belief_sketch",
            payload={"target": asset, "belief_distribution": list(decoded.belief)},
            sent_step=int(decoded.sent_step),
            deliver_step=int(delivered_step),
            public=False,
        )
        agent.receive(synthetic)
        before = tuple(agent.private_beliefs.get(asset, ()))
        integrated = agent.integrate_delivered_evidence(synthetic)
        after = tuple(agent.private_beliefs.get(asset, ()))
        self.action_belief_updates += int(integrated and before != after)
        return bool(integrated)

    def deliver(self, step: int) -> None:
        """Deliver frames due now, then perform bounded logged forwarding."""
        for transit in sorted(
            self.pending.pop(int(step), []),
            key=lambda value: (value.message_id, value.recipient),
        ):
            decoded = decode_belief_sketch(transit.wire_bytes)
            origin = self.agent_from_registry[decoded.origin_sender_id]
            asset = self.asset_from_registry[decoded.target_asset_id]
            key = (origin, asset)
            previous = self.cache[transit.recipient].get(key)
            redundant = bool(
                previous is not None
                and previous.sent_step >= decoded.sent_step
                and np.allclose(previous.belief, decoded.belief, atol=1e-12)
            )
            stale = int(step) - int(decoded.sent_step) > self.trigger_config.maximum_silence_steps
            if redundant:
                self.redundant_bytes += decoded.total_bytes
            else:
                self.delivered_useful_bytes += decoded.total_bytes
            if stale:
                self.stale_bytes += decoded.total_bytes
            cached = CachedBeliefSketch(
                origin=origin,
                asset=asset,
                belief=decoded.belief,
                confidence=decoded.confidence,
                sent_step=decoded.sent_step,
                delivered_step=int(step),
                hop_count=decoded.hop_count,
                wire_bytes=decoded.total_bytes,
            )
            if previous is None or decoded.sent_step >= previous.sent_step:
                self.cache[transit.recipient][key] = cached
            integrated = self._integrate(transit.recipient, decoded, asset, int(step))
            self.delivered_messages += 1
            row = {
                "step": int(step), "message_id": transit.message_id,
                "origin": origin, "transmitter": transit.transmitter,
                "recipient": transit.recipient, "asset": asset,
                "sent_step": decoded.sent_step, "hop_count": decoded.hop_count,
                "wire_bytes": decoded.total_bytes, "encoding": decoded.encoding,
                "redundant": redundant, "stale": stale,
                "belief_integrated": integrated,
            }
            self.delivery_rows.append(row)
            self.environment.ledger.append(
                step, "v8_sketch_delivered", transit.transmitter,
                row, private_to=transit.recipient,
            )
            if decoded.hop_count >= self.maximum_hops:
                continue
            node = self.environment.agent_nodes[transit.recipient]
            for neighbor_node in sorted(self.environment.communication_graph.neighbors(node)):
                neighbor = self.environment.node_agents[int(neighbor_node)]
                if neighbor in (origin, transit.transmitter):
                    continue
                forward_key = (
                    transit.recipient, neighbor, origin, asset,
                    decoded.sent_step, decoded.hop_count + 1,
                )
                if forward_key in self.forwarded:
                    continue
                self.forwarded.add(forward_key)
                self._send(
                    origin=origin,
                    transmitter=transit.recipient,
                    recipient=neighbor,
                    asset=asset,
                    belief=decoded.belief,
                    confidence=decoded.confidence,
                    original_sent_step=decoded.sent_step,
                    transmission_step=int(step),
                    hop_count=decoded.hop_count + 1,
                )

    def exchange(self, step: int) -> None:
        """Evaluate one local sender trigger per persistent agent."""
        for agent_id in self.agent_ids:
            agent = self.environment.agents[agent_id]
            assets = sorted(agent.private_beliefs)
            if not assets:
                continue
            asset = assets[(int(step) // max(self.environment.spec.decision_interval, 1)) % len(assets)]
            observation = agent.vault.observation(agent_id, asset)
            # The transmitted posterior is the sender's current private local
            # evidence distribution. Received peer sketches may affect its
            # operational belief, but are not recursively retransmitted as if
            # they were independent evidence.
            belief = observation.belief_distribution
            connected = self._has_available_neighbor(agent_id)
            healed = bool(connected and not self.last_connectivity.get(agent_id, connected))
            self.last_connectivity[agent_id] = connected
            trigger_started = time.perf_counter()
            decision = self.scheduler.evaluate(
                sender=agent_id,
                asset=asset,
                belief=belief,
                confidence=observation.telemetry_confidence,
                local_kpi=float(observation.local_kpis.get("severity", 0.0)),
                step=int(step),
                partition_healed=healed,
                # Only the explicitly labeled oracle gets evaluator assistance.
                offline_oracle_change=(
                    self._oracle_change(agent_id, asset, belief)
                    if self.trigger_config.method == "offline_oracle" else None
                ),
            )
            self.trigger_compute_seconds += time.perf_counter() - trigger_started
            self.trigger_evaluations += 1
            self.trigger_activations += int(decision.transmit)
            self.trigger_reasons[decision.reason] += 1
            self.trigger_rows.append({
                "step": int(step), "sender": agent_id, "asset": asset,
                "method": self.trigger_config.method,
                **asdict(decision),
            })
            self.environment.ledger.append(
                step, "v8_trigger_evaluated", agent_id,
                {
                    "asset": asset, "method": self.trigger_config.method,
                    **asdict(decision),
                    "information_boundary": (
                        "evaluator_only_oracle" if self.trigger_config.method == "offline_oracle"
                        else "sender_private_local"
                    ),
                },
                private_to=("evaluator" if self.trigger_config.method == "offline_oracle" else agent_id),
            )
            if not decision.transmit:
                continue
            node = self.environment.agent_nodes[agent_id]
            transmitted_any = False
            for neighbor_node in sorted(self.environment.communication_graph.neighbors(node)):
                recipient = self.environment.node_agents[int(neighbor_node)]
                transmitted_any = self._send(
                    origin=agent_id,
                    transmitter=agent_id,
                    recipient=recipient,
                    asset=asset,
                    belief=belief,
                    confidence=observation.telemetry_confidence,
                    original_sent_step=int(step),
                    transmission_step=int(step),
                    hop_count=0,
                ) or transmitted_any
            if transmitted_any:
                self.scheduler.mark_transmitted(
                    sender=agent_id, asset=asset, belief=belief,
                    confidence=observation.telemetry_confidence,
                    local_kpi=float(observation.local_kpis.get("severity", 0.0)),
                    step=int(step),
                )

    def _oracle_change(self, agent_id: str, asset: str, belief: Sequence[float]) -> float:
        """Evaluator-only drift from the pooled current beliefs (upper bound)."""
        scoped = [
            candidate.private_beliefs[asset]
            for candidate in self.environment.agents.values()
            if asset in candidate.private_beliefs
        ]
        if not scoped:
            return 0.0
        pooled = weighted_pooled_belief(scoped, np.ones(len(scoped)))
        return float(np.sum(np.abs(probability_vector(belief) - pooled)))

    def distributed_estimate(
        self, recipient: str, asset: str, step: int,
    ) -> Dict[str, Any]:
        agent = self.environment.agents[recipient]
        observation = agent.vault.observation(recipient, asset)
        own = probability_vector(observation.belief_distribution)
        beliefs: List[np.ndarray] = [own]
        confidences: List[float] = [float(observation.telemetry_confidence)]
        ages: List[int] = [0]
        contributors = [recipient]
        for (origin, focal_asset), sketch in sorted(self.cache[recipient].items()):
            if focal_asset != asset or origin == recipient:
                continue
            beliefs.append(probability_vector(sketch.belief))
            confidences.append(float(sketch.confidence))
            ages.append(max(0, int(step) - sketch.sent_step))
            contributors.append(origin)
        weights = np.asarray([
            max(1e-6, confidence) * np.exp(-age / max(self.trigger_config.maximum_silence_steps, 1))
            for confidence, age in zip(confidences, ages)
        ])
        pooled = weighted_pooled_belief(beliefs, weights)
        disagreement = generalized_disagreement(beliefs, weights, 1.0)
        scoped_agents = [
            candidate
            for candidate in self.environment.agents.values()
            if asset in candidate.private_beliefs
        ]
        # Evaluator target: reliability-weighted pooled *independent current
        # private evidence*. It is fixed by the panel observation process and
        # does not move merely because one communication scheduler delivered
        # more peer messages than another.
        global_beliefs = [
            probability_vector(
                candidate.vault.observation(candidate.agent_id, asset).belief_distribution
            )
            for candidate in scoped_agents
        ]
        global_weights = np.asarray([
            max(
                1e-6,
                candidate.vault.observation(candidate.agent_id, asset).telemetry_confidence,
            )
            for candidate in scoped_agents
        ])
        global_pooled = weighted_pooled_belief(global_beliefs, global_weights)
        global_disagreement = generalized_disagreement(global_beliefs, global_weights, 1.0)
        mae = float(np.mean(np.abs(pooled - global_pooled)))
        disagreement_error = abs(float(disagreement) - float(global_disagreement))
        disrupted_probability = float(1.0 - pooled[0])
        global_disrupted_probability = float(1.0 - global_pooled[0])
        return {
            "step": int(step), "recipient": recipient, "asset": asset,
            "contributors": len(contributors),
            "scoped_agents": len(scoped_agents),
            "missing_agents": max(0, len(scoped_agents) - len(set(contributors))),
            "maximum_age": max(ages),
            "mean_age": float(np.mean(ages)),
            "distributed_pooled_belief": tuple(float(value) for value in pooled),
            "evaluator_global_pooled_belief": tuple(float(value) for value in global_pooled),
            "distributed_disagreement": float(disagreement),
            "evaluator_global_disagreement": float(global_disagreement),
            "belief_mae": mae,
            "disagreement_absolute_error": disagreement_error,
            "distributed_disrupted_probability": disrupted_probability,
            "evaluator_global_disrupted_probability": global_disrupted_probability,
        }

    def record_estimates(self, step: int) -> None:
        for agent_id in self.agent_ids:
            agent = self.environment.agents[agent_id]
            assets = sorted(agent.private_beliefs)
            if not assets:
                continue
            asset = assets[(int(step) // max(self.environment.spec.decision_interval, 1)) % len(assets)]
            row = self.distributed_estimate(agent_id, asset, step)
            # The domain's latent incident mode is used only to score detection
            # offline.  It is excluded from the recipient-visible ledger event.
            true_mode = str(self.environment._true_local_mode(asset, int(step)))
            row["evaluator_true_mode"] = true_mode
            row["evaluator_disrupted"] = bool(true_mode != "nominal")
            self.estimation_rows.append(row)
            self.environment.ledger.append(
                step, "v8_distributed_estimate", "distributed_estimator",
                {
                    **{key: value for key, value in row.items() if not key.startswith("evaluator_")},
                    "information_boundary": "recipient_local_plus_delivered_sketches",
                },
                private_to=agent_id,
            )
            self.environment.ledger.append(
                step, "v8_estimation_error", "evaluator",
                {
                    "recipient": agent_id, "asset": asset,
                    "belief_mae": row["belief_mae"],
                    "disagreement_absolute_error": row["disagreement_absolute_error"],
                    "evaluator_global_disagreement": row["evaluator_global_disagreement"],
                },
                private_to="evaluator",
            )

    def accounting(self) -> Dict[str, Any]:
        quantization_mean = (
            self.quantization_l1_error_sum / self.transmitted_messages
            if self.transmitted_messages else 0.0
        )
        return {
            "scheduler": self.trigger_config.method,
            "encoding": self.encoding,
            "attempted_sketch_messages": self.attempted_messages,
            "transmitted_sketch_messages": self.transmitted_messages,
            "delivered_sketch_messages": self.delivered_messages,
            "dropped_sketch_messages": self.dropped_messages,
            "forwarded_sketch_messages": self.forwarded_messages,
            "retries": self.retries,
            "sketch_header_bytes": self.header_bytes,
            "sketch_payload_bytes": self.payload_bytes,
            "sketch_integrity_bytes": self.integrity_bytes,
            "sketch_on_wire_bytes": self.total_on_wire_bytes,
            "delivered_useful_sketch_bytes": self.delivered_useful_bytes,
            "redundant_sketch_bytes": self.redundant_bytes,
            "stale_sketch_bytes": self.stale_bytes,
            "operational_messages": int(self.environment.operational_messages),
            "operational_bytes": int(self.environment.operational_bytes),
            "fully_counted_messages": int(self.transmitted_messages + self.environment.operational_messages),
            "fully_counted_bytes": int(self.total_on_wire_bytes + self.environment.operational_bytes),
            "mean_quantization_l1_error": float(quantization_mean),
            "trigger_evaluations": self.trigger_evaluations,
            "trigger_activations": self.trigger_activations,
            "trigger_activation_rate": float(
                self.trigger_activations / max(self.trigger_evaluations, 1)
            ),
            "trigger_compute_seconds": float(self.trigger_compute_seconds),
            "trigger_compute_microseconds_per_evaluation": float(
                1e6 * self.trigger_compute_seconds / max(self.trigger_evaluations, 1)
            ),
            "trigger_reasons": dict(sorted(self.trigger_reasons.items())),
            "belief_updates_from_delivered_sketches": self.action_belief_updates,
        }
