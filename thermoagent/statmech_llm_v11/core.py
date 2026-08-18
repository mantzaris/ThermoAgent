"""Typed evidence, independent agent state, and the V11 prompt boundary.

The latent state is evaluator-only. Agents see only their own sampled packet,
their bounded local state when the protocol permits it, and explicitly
delivered packets. The scheduler offers turns and validates typed outputs but
never selects an agent's belief, action, commitment, or outgoing evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Protocol, Sequence, Tuple

import numpy as np


SIDES = ("left", "right")
DOMAINS = ("route_viability", "repair_hypothesis")
PACKET_KINDS = ("evidence", "placebo")
ACTIONS = ("select_left", "select_right", "defer")
COMMITMENTS = ("uncommitted", "provisional", "committed", "revised", "deferred")
OUTGOING_EVIDENCE_ACTIONS = ("send_private_evidence", "abstain")
REASON_CODES = (
    "private_evidence",
    "delivered_evidence",
    "combined_evidence",
    "conflicting_evidence",
    "insufficient_evidence",
    "stale_evidence",
)


def clipped_logit(probability: float, epsilon: float = 1e-4) -> float:
    value = float(np.clip(float(probability), epsilon, 1.0 - epsilon))
    return float(math.log(value / (1.0 - value)))


def logistic(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -700.0, 700.0)))))


@dataclass(frozen=True)
class EvidencePacket:
    """One observation with an operationally defined likelihood."""

    source_id: str
    observation: str
    reliability: float
    observation_time: int
    delivery_time: int
    freshness: float
    evidence_domain: str
    explanation: str = ""
    packet_kind: str = "evidence"

    def validate(self) -> None:
        if not self.source_id or len(self.source_id.encode("utf-8")) > 64:
            raise ValueError("source_id must contain at most 64 UTF-8 bytes")
        if self.observation not in ("left", "right", "unknown"):
            raise ValueError("invalid evidence observation")
        if self.packet_kind not in PACKET_KINDS:
            raise ValueError("invalid packet kind")
        if self.evidence_domain not in DOMAINS:
            raise ValueError("invalid evidence domain")
        if not np.isfinite(self.reliability) or not 0.5 <= self.reliability < 1.0:
            raise ValueError("reliability must be finite and in [0.5,1)")
        if self.packet_kind == "evidence" and self.observation == "unknown":
            raise ValueError("an evidence packet must report left or right")
        if self.packet_kind == "placebo" and (self.observation != "unknown" or self.reliability != 0.5):
            raise ValueError("a placebo packet must be unknown with reliability 0.5")
        if self.observation_time < 0 or self.delivery_time < self.observation_time:
            raise ValueError("invalid observation or delivery time")
        if not np.isfinite(self.freshness) or not 0.0 <= self.freshness <= 1.0:
            raise ValueError("freshness must be finite and in [0,1]")
        if len(self.explanation.encode("utf-8")) > 160:
            raise ValueError("explanation exceeds 160 UTF-8 bytes")

    @property
    def effective_reliability(self) -> float:
        """Freshness contracts evidence reliability toward an uninformative 0.5."""

        self.validate()
        return float(0.5 + self.freshness * (self.reliability - 0.5))

    @property
    def log_likelihood_ratio(self) -> float:
        """Normative log P(packet|right) / P(packet|left)."""

        if self.packet_kind == "placebo":
            return 0.0
        reliability = self.effective_reliability
        magnitude = math.log(reliability / (1.0 - reliability))
        return float(magnitude if self.observation == "right" else -magnitude)

    def to_mapping(self) -> Dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvidencePacket":
        expected = {
            "source_id",
            "observation",
            "reliability",
            "observation_time",
            "delivery_time",
            "freshness",
            "evidence_domain",
            "explanation",
            "packet_kind",
        }
        if set(value) != expected:
            raise ValueError("evidence packet schema mismatch")
        packet = cls(
            source_id=str(value["source_id"]),
            observation=str(value["observation"]),
            reliability=float(value["reliability"]),
            observation_time=int(value["observation_time"]),
            delivery_time=int(value["delivery_time"]),
            freshness=float(value["freshness"]),
            evidence_domain=str(value["evidence_domain"]),
            explanation=str(value["explanation"]),
            packet_kind=str(value["packet_kind"]),
        )
        packet.validate()
        return packet


_WIRE_HEADER = "!BBBBffIIHH"


def serialize_evidence_packet(packet: EvidencePacket) -> bytes:
    """Serialize the exact V11 wire representation used for byte accounting."""

    packet.validate()
    source = packet.source_id.encode("utf-8")
    explanation = packet.explanation.encode("utf-8")
    header = struct.pack(
        _WIRE_HEADER,
        1,
        PACKET_KINDS.index(packet.packet_kind),
        ("left", "right", "unknown").index(packet.observation),
        DOMAINS.index(packet.evidence_domain),
        float(packet.reliability),
        float(packet.freshness),
        int(packet.observation_time),
        int(packet.delivery_time),
        len(source),
        len(explanation),
    )
    return header + source + explanation


def deserialize_evidence_packet(payload: bytes) -> EvidencePacket:
    size = struct.calcsize(_WIRE_HEADER)
    if len(payload) < size:
        raise ValueError("truncated evidence header")
    version, kind, observation, domain, reliability, freshness, observed, delivered, source_n, text_n = struct.unpack(
        _WIRE_HEADER, payload[:size]
    )
    if version != 1 or kind >= len(PACKET_KINDS) or observation >= 3 or domain >= len(DOMAINS):
        raise ValueError("unsupported evidence packet")
    if len(payload) != size + source_n + text_n:
        raise ValueError("evidence payload length mismatch")
    source = payload[size : size + source_n].decode("utf-8")
    explanation = payload[size + source_n :].decode("utf-8")
    packet = EvidencePacket(
        source_id=source,
        observation=("left", "right", "unknown")[observation],
        reliability=float(reliability),
        observation_time=int(observed),
        delivery_time=int(delivered),
        freshness=float(freshness),
        evidence_domain=DOMAINS[domain],
        explanation=explanation,
        packet_kind=PACKET_KINDS[kind],
    )
    packet.validate()
    return packet


def generate_private_evidence(
    latent_state: str,
    reliability: float,
    rng: np.random.Generator,
    source_id: str,
    domain: str,
    observation_time: int = 0,
) -> EvidencePacket:
    if latent_state not in SIDES:
        raise ValueError("latent state must be left or right")
    if not 0.5 < float(reliability) < 1.0:
        raise ValueError("signal reliability must be in (0.5,1)")
    correct = bool(rng.random() < float(reliability))
    observation = latent_state if correct else ("left" if latent_state == "right" else "right")
    return EvidencePacket(
        source_id=str(source_id),
        observation=observation,
        reliability=float(reliability),
        observation_time=int(observation_time),
        delivery_time=int(observation_time),
        freshness=1.0,
        evidence_domain=str(domain),
        explanation="A conditionally independent local observation.",
    )


def bayesian_probability_right(packets: Sequence[EvidencePacket], prior_right: float = 0.5) -> float:
    odds = clipped_logit(prior_right)
    for packet in packets:
        odds += packet.log_likelihood_ratio
    return logistic(odds)


@dataclass(frozen=True)
class EvidenceGroundedDecision:
    probability_right: float
    belief_choice: str
    action_choice: str
    commitment_status: str
    outgoing_evidence_action: str
    reason_code: str
    explanation: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvidenceGroundedDecision":
        expected = {
            "probability_right",
            "belief_choice",
            "action_choice",
            "commitment_status",
            "outgoing_evidence_action",
            "reason_code",
            "explanation",
        }
        if set(value) != expected:
            raise ValueError("decision schema mismatch")
        decision = cls(
            probability_right=float(value["probability_right"]),
            belief_choice=str(value["belief_choice"]),
            action_choice=str(value["action_choice"]),
            commitment_status=str(value["commitment_status"]),
            outgoing_evidence_action=str(value["outgoing_evidence_action"]),
            reason_code=str(value["reason_code"]),
            explanation=str(value["explanation"]),
        )
        decision.validate()
        return decision

    def validate(self) -> None:
        if not np.isfinite(self.probability_right) or not 0.0 <= self.probability_right <= 1.0:
            raise ValueError("probability_right must be finite and in [0,1]")
        if self.belief_choice not in SIDES:
            raise ValueError("belief_choice must be left or right")
        derived = "right" if self.probability_right >= 0.5 else "left"
        if self.belief_choice != derived:
            raise ValueError("belief_choice must equal the frozen 0.5 threshold of probability_right")
        if self.action_choice not in ACTIONS or self.commitment_status not in COMMITMENTS:
            raise ValueError("invalid action or commitment")
        if self.outgoing_evidence_action not in OUTGOING_EVIDENCE_ACTIONS:
            raise ValueError("invalid outgoing evidence action")
        if self.reason_code not in REASON_CODES:
            raise ValueError("invalid reason code")
        if len(self.explanation.encode("utf-8")) > 240:
            raise ValueError("explanation exceeds 240 UTF-8 bytes")


def decision_schema() -> Dict[str, object]:
    properties = {
        "probability_right": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "belief_choice": {"enum": list(SIDES)},
        "action_choice": {"enum": list(ACTIONS)},
        "commitment_status": {"enum": list(COMMITMENTS)},
        "outgoing_evidence_action": {"enum": list(OUTGOING_EVIDENCE_ACTIONS)},
        "reason_code": {"enum": list(REASON_CODES)},
        "explanation": {"type": "string", "maxLength": 240},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


@dataclass(frozen=True)
class ProviderResult:
    payload: Mapping[str, object]
    first_pass_valid: bool
    repaired: bool
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    raw_artifact_sha256: str


class StructuredProvider(Protocol):
    def decide(self, prompt: str, seed: int) -> ProviderResult:
        ...


@dataclass
class IndependentEvidenceAgent:
    identifier: int
    role: str
    private_evidence: EvidencePacket
    _probability_right: float = 0.5
    _action: str = "defer"
    _commitment: str = "uncommitted"
    _memory: List[str] = field(default_factory=list)
    _inbox: List[EvidencePacket] = field(default_factory=list)
    _outbox: List[EvidencePacket] = field(default_factory=list)

    def clone(self) -> "IndependentEvidenceAgent":
        return copy.deepcopy(self)

    @property
    def probability_right(self) -> float:
        return float(self._probability_right)

    @property
    def action(self) -> str:
        return str(self._action)

    @property
    def commitment(self) -> str:
        return str(self._commitment)

    @property
    def inbox_size(self) -> int:
        return len(self._inbox)

    @property
    def memory(self) -> Tuple[str, ...]:
        return tuple(self._memory)

    def replace_private_evidence(self, packet: EvidencePacket) -> None:
        packet.validate()
        if packet.source_id != self.private_evidence.source_id:
            raise ValueError("private evidence source identity cannot change")
        self.private_evidence = packet

    def clear_inbox(self) -> None:
        self._inbox.clear()

    @property
    def belief_choice(self) -> str:
        return "right" if self._probability_right >= 0.5 else "left"

    def receive(self, packet: EvidencePacket) -> None:
        packet.validate()
        self._inbox.append(packet)

    def private_fingerprint(self) -> str:
        state = {
            "private_evidence": self.private_evidence.to_mapping(),
            "probability_right": self._probability_right,
            "action": self._action,
            "commitment": self._commitment,
            "memory": list(self._memory),
            "inbox": [packet.to_mapping() for packet in self._inbox],
            "outbox": [packet.to_mapping() for packet in self._outbox],
        }
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()

    def prompt_view(self, mode: str, display_order: Tuple[str, str], time_step: int) -> Dict[str, object]:
        if set(display_order) != set(SIDES):
            raise ValueError("display order must contain left and right")
        view: Dict[str, object] = {
            "agent_id": self.identifier,
            "role": self.role,
            "decision_mode": mode,
            "time_step": int(time_step),
            "display_order": list(display_order),
            "private_evidence": self.private_evidence.to_mapping(),
            "delivered_evidence": [packet.to_mapping() for packet in self._inbox],
        }
        if mode != "qualification_unanchored":
            if mode != "formal_no_reported_probability":
                view["previous_probability_right"] = self._probability_right
            view["previous_action"] = self._action
            if mode != "formal_no_commitment":
                view["previous_commitment"] = self._commitment
            view["bounded_memory"] = list(self._memory[-4:])
        return view

    def apply_decision(self, decision: EvidenceGroundedDecision, time_step: int) -> bool:
        """Apply the model's decision and reject, never replace, invalid outgoing evidence."""

        decision.validate()
        self._probability_right = float(decision.probability_right)
        self._action = decision.action_choice
        self._commitment = decision.commitment_status
        self._memory.append(
            "t=%d p_right=%.4f action=%s commitment=%s reason=%s"
            % (time_step, decision.probability_right, decision.action_choice, decision.commitment_status, decision.reason_code)
        )
        self._memory[:] = self._memory[-6:]
        if decision.outgoing_evidence_action == "abstain":
            return False
        self._outbox.append(self.private_evidence)
        return True


TASK_TEXT = {
    "route_viability": {
        "left": "the western route is more likely to remain viable",
        "right": "the eastern route is more likely to remain viable",
    },
    "repair_hypothesis": {
        "left": "the local hardware-fault hypothesis is more likely",
        "right": "the telemetry-integrity hypothesis is more likely",
    },
}


PROMPT_PARAPHRASES = (
    "Estimate the hidden state from your own observation and the packets actually delivered to you.",
    "Independently assess which alternative is more probable using only the listed local evidence.",
    "Combine the available conditionally independent observations into your best local probability estimate.",
)


def build_evidence_prompt(
    agent: IndependentEvidenceAgent,
    mode: str,
    display_order: Tuple[str, str],
    paraphrase: int,
    time_step: int,
) -> str:
    domain = agent.private_evidence.evidence_domain
    ordered = [{"choice": side, "meaning": TASK_TEXT[domain][side]} for side in display_order]
    view = agent.prompt_view(mode, display_order, time_step)
    instructions = (
        "You are one independent decentralized agent. You cannot inspect the hidden state, global simulator, "
        "undelivered messages, peer memory, future observations, or counterfactual outcomes. "
        + PROMPT_PARAPHRASES[int(paraphrase) % len(PROMPT_PARAPHRASES)]
        + " In this controlled environment, an evidence reliability r means P(observation equals hidden state)=r. "
        "Packets are conditionally independent unless the packet list explicitly says otherwise. Freshness contracts "
        "reliability toward 0.5. A placebo packet with observation unknown and reliability 0.5 carries no directional evidence. "
        "First state probability_right, then derive belief_choice at the fixed 0.5 threshold, and only then choose an action "
        "and commitment. Do not preserve a prior merely because it existed, and do not follow a message merely because it was sent. "
        "Your role includes forwarding usable local evidence: choose send_private_evidence unless the private packet is "
        "malformed or unavailable; choose abstain otherwise. The typed tool, not you, serializes the immutable packet. "
        "Return exactly one JSON object with no prose."
    )
    envelope = {
        "alternatives_in_display_order": ordered,
        "authorized_local_view": view,
        "required_schema": decision_schema(),
    }
    return instructions + "\nCONTROLLED_TASK=" + json.dumps(envelope, sort_keys=True, separators=(",", ":"))
