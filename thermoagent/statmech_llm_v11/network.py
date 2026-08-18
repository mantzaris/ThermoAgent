"""Persistent decentralized LLM-agent network used after qualification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    EvidenceGroundedDecision,
    EvidencePacket,
    IndependentEvidenceAgent,
    StructuredProvider,
    build_evidence_prompt,
    generate_private_evidence,
    serialize_evidence_packet,
)


@dataclass(frozen=True)
class StepTape:
    scheduled_agent: int
    private_uniform: float
    reliability_uniform: float
    recipient_uniform: float
    inference_seed: int
    option_right_first: bool
    paraphrase: int


def generate_step_tape(n_agents: int, turns: int, seed: int) -> List[StepTape]:
    rng = np.random.default_rng(int(seed))
    permutation: List[int] = []
    output: List[StepTape] = []
    for turn in range(int(turns)):
        if not permutation:
            permutation = list(rng.permutation(int(n_agents)).astype(int))
        output.append(
            StepTape(
                scheduled_agent=permutation.pop(),
                private_uniform=float(rng.random()),
                reliability_uniform=float(rng.random()),
                recipient_uniform=float(rng.random()),
                inference_seed=int(seed) * 10000 + turn,
                option_right_first=bool(rng.integers(0, 2)),
                paraphrase=int(rng.integers(0, 3)),
            )
        )
    return output


def undirected_skeleton(n_agents: int, topology: str, seed: int) -> np.ndarray:
    n = int(n_agents)
    if n < 4:
        raise ValueError("network needs at least four agents")
    adjacency = np.zeros((n, n), dtype=int)
    if topology == "ring":
        for index in range(n):
            adjacency[index, (index + 1) % n] = 1
            adjacency[(index + 1) % n, index] = 1
    elif topology == "modular":
        split = n // 2
        for start, stop in ((0, split), (split, n)):
            for index in range(start, stop):
                neighbor = start + ((index - start + 1) % (stop - start))
                adjacency[index, neighbor] = adjacency[neighbor, index] = 1
        adjacency[split - 1, split] = adjacency[split, split - 1] = 1
        adjacency[0, n - 1] = adjacency[n - 1, 0] = 1
    elif topology == "small_world":
        for index in range(n):
            for offset in (1, 2):
                neighbor = (index + offset) % n
                adjacency[index, neighbor] = adjacency[neighbor, index] = 1
        rng = np.random.default_rng(int(seed))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i, j]]
        for i, j in edges[::4]:
            candidates = [value for value in range(n) if value != i and not adjacency[i, value]]
            if candidates:
                replacement = int(rng.choice(candidates))
                adjacency[i, j] = adjacency[j, i] = 0
                adjacency[i, replacement] = adjacency[replacement, i] = 1
    else:
        raise ValueError("unsupported topology")
    if np.any(adjacency != adjacency.T) or np.any(np.diag(adjacency)):
        raise AssertionError("communication skeleton must be simple and undirected")
    return adjacency


def oriented_edge_signs(adjacency: np.ndarray, orientation_seed: int) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=int)
    rng = np.random.default_rng(int(orientation_seed))
    sign = np.zeros_like(matrix, dtype=int)
    ring_edges = np.zeros_like(matrix, dtype=int)
    for index in range(matrix.shape[0]):
        ring_edges[index, (index + 1) % matrix.shape[0]] = 1
        ring_edges[(index + 1) % matrix.shape[0], index] = 1
    if np.array_equal(matrix, ring_edges):
        orientation = 1 if int(orientation_seed) % 2 == 0 else -1
        for index in range(matrix.shape[0]):
            neighbor = (index + 1) % matrix.shape[0]
            sign[index, neighbor] = orientation
            sign[neighbor, index] = -orientation
        return sign
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if not matrix[i, j]:
                continue
            direction = 1 if rng.random() < 0.5 else -1
            sign[i, j] = direction
            sign[j, i] = -direction
    return sign


def choose_recipient(
    sender: int,
    adjacency: np.ndarray,
    orientation: np.ndarray,
    alpha: float,
    uniform: float,
) -> int:
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("nonreciprocity alpha must be in [0,1)")
    neighbors = np.flatnonzero(np.asarray(adjacency)[int(sender)] > 0)
    if neighbors.size == 0:
        raise ValueError("sender has no communication neighbor")
    weights = np.asarray([1.0 + float(alpha) * float(orientation[int(sender), int(recipient)]) for recipient in neighbors])
    weights /= weights.sum()
    cumulative = np.cumsum(weights)
    selected = min(int(np.searchsorted(cumulative, float(uniform), side="right")), neighbors.size - 1)
    return int(neighbors[selected])


def _transport_packet(packet: EvidencePacket, delivery_time: int, control: str) -> EvidencePacket:
    observation = packet.observation
    reliability = packet.reliability
    packet_kind = packet.packet_kind
    explanation = packet.explanation
    if control == "content_reversal":
        observation = "right" if observation == "left" else "left"
    elif control == "reliability_permutation":
        # The formal design uses these three declared reliability levels.  A
        # cyclic relabeling destroys calibration without changing observation
        # text, packet count, or packet length.
        levels = np.asarray([0.55, 0.70, 0.85], dtype=float)
        nearest = int(np.argmin(np.abs(levels - float(reliability))))
        reliability = float(levels[(nearest + 1) % levels.size])
    elif control == "reliability_destroyed":
        reliability = 0.5
        observation = "unknown"
        packet_kind = "placebo"
        explanation = "A format-matched packet with no usable reliability information."
    elif control == "natural_language_placebo":
        reliability = 0.5
        observation = "unknown"
        packet_kind = "placebo"
        explanation = "A peer sent a format-matched acknowledgment without an observation."
    return EvidencePacket(
        source_id=packet.source_id,
        observation=observation,
        reliability=float(reliability),
        observation_time=packet.observation_time,
        delivery_time=int(delivery_time),
        freshness=max(0.0, 1.0 - 0.1 * max(0, int(delivery_time) - packet.observation_time)),
        evidence_domain=packet.evidence_domain,
        explanation=explanation,
        packet_kind=packet_kind,
    )


def coarse_macrostate(agents: Sequence[IndependentEvidenceAgent]) -> int:
    n = len(agents)
    right_beliefs = sum(agent.belief_choice == "right" for agent in agents)
    right_actions = sum(agent.action == "select_right" for agent in agents)
    deferred = sum(agent.action == "defer" for agent in agents)
    return int((right_beliefs * (n + 1) + right_actions) * (n + 1) + deferred)


class DecentralizedEvidenceNetwork:
    """The scheduler offers updates and transports packets but never chooses decisions."""

    def __init__(
        self,
        agents: Sequence[IndependentEvidenceAgent],
        adjacency: np.ndarray,
        orientation: np.ndarray,
        latent_state: str,
        service_deficit: float = 0.5,
    ) -> None:
        self.agents = [agent.clone() for agent in agents]
        self.adjacency = np.asarray(adjacency, dtype=int).copy()
        self.orientation = np.asarray(orientation, dtype=int).copy()
        self.latent_state = str(latent_state)
        self.service_deficit = float(service_deficit)
        if self.adjacency.shape != (len(self.agents), len(self.agents)):
            raise ValueError("network dimensions do not match agents")
        if self.latent_state not in ("left", "right"):
            raise ValueError("latent state must be binary")

    def private_fingerprints(self) -> Tuple[str, ...]:
        return tuple(agent.private_fingerprint() for agent in self.agents)

    def offered_step(
        self,
        provider: StructuredProvider,
        tape: StepTape,
        alpha: float,
        reliability: float,
        domain: str,
        turn: int,
        control: str = "unaltered",
        prompt_mode: str = "formal",
    ) -> Dict[str, object]:
        index = int(tape.scheduled_agent)
        agent = self.agents[index]
        before_peers = [item for item in self.private_fingerprints()]
        inbox_count_before = agent.inbox_size
        probability_before = agent.probability_right
        belief_before = agent.belief_choice
        action_state_before = agent.action
        commitment_before = agent.commitment
        observation = self.latent_state if tape.private_uniform < float(reliability) else (
            "left" if self.latent_state == "right" else "right"
        )
        private = EvidencePacket(
            source_id="private_agent_%d" % index,
            observation=observation,
            reliability=float(reliability),
            observation_time=int(turn),
            delivery_time=int(turn),
            freshness=1.0,
            evidence_domain=domain,
            explanation="A new local observation sampled from the shared reliability model.",
        )
        if agent.private_evidence.source_id != private.source_id:
            raise AssertionError("agent source identity changed")
        agent.replace_private_evidence(private)
        order = ("right", "left") if tape.option_right_first else ("left", "right")
        prompt = build_evidence_prompt(agent, prompt_mode, order, tape.paraphrase, turn)
        result = provider.decide(prompt, tape.inference_seed)
        decision = EvidenceGroundedDecision.from_mapping(result.payload)
        outgoing_accepted = agent.apply_decision(decision, turn)
        message_attempted = int(decision.outgoing_evidence_action == "send_private_evidence")
        message_transmitted = 0
        message_bytes = 0
        recipient: Optional[int] = None
        if outgoing_accepted and control != "no_message":
            recipient_uniform = (tape.recipient_uniform + 0.5) % 1.0 if control == "message_permutation" else tape.recipient_uniform
            recipient = choose_recipient(index, self.adjacency, self.orientation, alpha, recipient_uniform)
            transported = _transport_packet(agent.private_evidence, turn + 1, control)
            self.agents[recipient].receive(transported)
            message_transmitted = 1
            message_bytes = len(serialize_evidence_packet(transported))
        agent.clear_inbox()
        action_before = self.service_deficit
        if decision.action_choice == "defer":
            causal_change = 0.01
        else:
            selected = "right" if decision.action_choice == "select_right" else "left"
            causal_change = -0.04 if selected == self.latent_state else 0.07
        self.service_deficit = float(np.clip(self.service_deficit + causal_change, 0.0, 1.0))
        after_peers = [item for item in self.private_fingerprints()]
        peer_mutations = sum(
            before_peers[peer] != after_peers[peer]
            for peer in range(len(self.agents))
            if peer not in (index, recipient)
        )
        return {
            "turn": int(turn),
            "scheduled_agent": index,
            "recipient": -1 if recipient is None else int(recipient),
            "alpha": float(alpha),
            "control": control,
            "local_evidence_reliability": float(reliability),
            "inbox_count_before": int(inbox_count_before),
            "probability_right_before": float(probability_before),
            "belief_right_before": int(belief_before == "right"),
            "action_choice_before": action_state_before,
            "commitment_before": commitment_before,
            "probability_right": decision.probability_right,
            "belief_right": int(decision.belief_choice == "right"),
            "action_choice": decision.action_choice,
            "commitment_status": decision.commitment_status,
            "reason_code": decision.reason_code,
            "message_attempted": message_attempted,
            "message_transmitted": message_transmitted,
            "message_wire_bytes": message_bytes,
            "outgoing_packet_accepted": int(outgoing_accepted),
            "service_before": action_before,
            "service_after": self.service_deficit,
            "causal_service_change": causal_change,
            "coarse_macrostate": coarse_macrostate(self.agents),
            "belief_macrostate": int(sum(item.belief_choice == "right" for item in self.agents)),
            "action_macrostate": int(
                sum(item.action == "select_right" for item in self.agents)
                - sum(item.action == "select_left" for item in self.agents)
                + len(self.agents)
            ),
            "first_pass_valid": int(result.first_pass_valid),
            "repaired": int(result.repaired),
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "latency_seconds": result.latency_seconds,
            "raw_artifact_sha256": result.raw_artifact_sha256,
            "unrelated_peer_private_mutations": int(peer_mutations),
            "network_state_sha256": hashlib.sha256(
                "|".join(self.private_fingerprints()).encode("utf-8")
            ).hexdigest(),
        }


def make_network_agents(n_agents: int, domain: str, latent_state: str, reliability: float, seed: int) -> List[IndependentEvidenceAgent]:
    rng = np.random.default_rng(int(seed))
    output: List[IndependentEvidenceAgent] = []
    for index in range(int(n_agents)):
        packet = generate_private_evidence(
            latent_state,
            reliability,
            rng,
            "private_agent_%d" % index,
            domain,
        )
        output.append(IndependentEvidenceAgent(index, "local_coordinator_%d" % index, packet))
    return output
