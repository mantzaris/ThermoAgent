"""Typed V12 microstates, independent agents, prompts, and wire messages.

The scheduler can offer a turn and transport a validated packet.  It never
chooses a belief, action, commitment, memory state, or outgoing signal.  Every
piece of prompt state is included in the observable microstate or the delivered
inbox; the latent decoding map and global network are evaluator-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


LABELS = ("amber", "cobalt")
ACTIONS = ("amber", "cobalt")
COMMITMENTS = ("uncommitted", "provisional", "committed", "revised", "deferred")
MEMORY_STATES = ("stable", "evidence_amber", "evidence_cobalt", "conflict", "uncertain")
SIGNALS = ("amber", "cobalt", "uncertain")
TOOLS = ("execute_selected", "no_action")
REASONS = (
    "private_observation",
    "neighbor_messages",
    "belief_action_consistency",
    "conflicting_information",
    "persistence",
    "insufficient_information",
)


@dataclass(frozen=True)
class LatentMapping:
    """Counterbalanced mapping between displayed labels and latent spins."""

    plus_label: str
    display_order: Tuple[str, str]

    def validate(self) -> None:
        if self.plus_label not in LABELS or set(self.display_order) != set(LABELS):
            raise ValueError("invalid latent-label mapping")

    @property
    def minus_label(self) -> str:
        return LABELS[1] if self.plus_label == LABELS[0] else LABELS[0]

    def label(self, spin: int) -> str:
        self.validate()
        if int(spin) not in (-1, 1):
            raise ValueError("spin must be -1 or +1")
        return self.plus_label if int(spin) == 1 else self.minus_label

    def spin(self, label: str) -> int:
        self.validate()
        if label not in LABELS:
            raise ValueError("unknown displayed label")
        return 1 if label == self.plus_label else -1

    @classmethod
    def balanced(cls, seed: int) -> "LatentMapping":
        rng = np.random.default_rng(int(seed))
        plus = LABELS[int(rng.integers(0, 2))]
        order = LABELS if int(rng.integers(0, 2)) == 0 else tuple(reversed(LABELS))
        return cls(plus, tuple(order))


@dataclass(frozen=True)
class AgentDecision:
    belief_choice: str
    action_choice: str
    confidence: float
    commitment_status: str
    memory_state: str
    outgoing_signal: str
    tool_action: str
    reason_code: str

    def validate(self) -> None:
        if self.belief_choice not in LABELS:
            raise ValueError("invalid belief choice")
        if self.action_choice not in ACTIONS:
            raise ValueError("invalid action choice")
        if not np.isfinite(self.confidence) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")
        if self.commitment_status not in COMMITMENTS:
            raise ValueError("invalid commitment status")
        if self.memory_state not in MEMORY_STATES:
            raise ValueError("invalid bounded memory state")
        if self.outgoing_signal not in SIGNALS:
            raise ValueError("invalid outgoing signal")
        if self.tool_action not in TOOLS:
            raise ValueError("invalid typed tool action")
        if self.reason_code not in REASONS:
            raise ValueError("invalid reason code")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AgentDecision":
        expected = {
            "belief_choice",
            "action_choice",
            "confidence",
            "commitment_status",
            "memory_state",
            "outgoing_signal",
            "tool_action",
            "reason_code",
        }
        if set(value) != expected:
            raise ValueError("decision schema mismatch")
        result = cls(
            belief_choice=str(value["belief_choice"]),
            action_choice=str(value["action_choice"]),
            confidence=float(value["confidence"]),
            commitment_status=str(value["commitment_status"]),
            memory_state=str(value["memory_state"]),
            outgoing_signal=str(value["outgoing_signal"]),
            tool_action=str(value["tool_action"]),
            reason_code=str(value["reason_code"]),
        )
        result.validate()
        return result


def decision_schema() -> Dict[str, object]:
    properties: Dict[str, object] = {
        "belief_choice": {"enum": list(LABELS)},
        "action_choice": {"enum": list(ACTIONS)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "commitment_status": {"enum": list(COMMITMENTS)},
        "memory_state": {"enum": list(MEMORY_STATES)},
        "outgoing_signal": {"enum": list(SIGNALS)},
        "tool_action": {"enum": list(TOOLS)},
        "reason_code": {"enum": list(REASONS)},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


@dataclass(frozen=True)
class SignalPacket:
    sender_id: int
    time_step: int
    belief_spin: int
    action_spin: int
    signal_spin: int
    confidence: float
    commitment_code: int
    memory_code: int

    def validate(self) -> None:
        if not 0 <= int(self.sender_id) <= 65535 or int(self.time_step) < 0:
            raise ValueError("invalid packet identity or time")
        if int(self.belief_spin) not in (-1, 1) or int(self.action_spin) not in (-1, 1):
            raise ValueError("belief and action packet states must be binary")
        if int(self.signal_spin) not in (-1, 0, 1):
            raise ValueError("signal packet state must be -1, 0, or +1")
        if not np.isfinite(self.confidence) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("invalid packet confidence")
        if not 0 <= int(self.commitment_code) < len(COMMITMENTS):
            raise ValueError("invalid commitment code")
        if not 0 <= int(self.memory_code) < len(MEMORY_STATES):
            raise ValueError("invalid memory code")


# version, sender, step, belief, action, signal, confidence_uint16,
# commitment, memory.  A four-byte CRC is included as simulated framing.
_PACKET_HEADER = "!BHIbbbHBB"


def serialize_signal_packet(packet: SignalPacket) -> bytes:
    packet.validate()
    confidence = int(round(float(packet.confidence) * 65535.0))
    body = struct.pack(
        _PACKET_HEADER,
        1,
        int(packet.sender_id),
        int(packet.time_step),
        int(packet.belief_spin),
        int(packet.action_spin),
        int(packet.signal_spin),
        confidence,
        int(packet.commitment_code),
        int(packet.memory_code),
    )
    return body + struct.pack("!I", zlib.crc32(body) & 0xFFFFFFFF)


def deserialize_signal_packet(payload: bytes) -> SignalPacket:
    body_size = struct.calcsize(_PACKET_HEADER)
    if len(payload) != body_size + 4:
        raise ValueError("signal packet length mismatch")
    body, checksum = payload[:body_size], payload[body_size:]
    if struct.unpack("!I", checksum)[0] != (zlib.crc32(body) & 0xFFFFFFFF):
        raise ValueError("signal packet checksum mismatch")
    version, sender, step, belief, action, signal, confidence, commitment, memory = struct.unpack(
        _PACKET_HEADER, body
    )
    if version != 1:
        raise ValueError("unsupported signal packet version")
    packet = SignalPacket(
        int(sender),
        int(step),
        int(belief),
        int(action),
        int(signal),
        float(confidence) / 65535.0,
        int(commitment),
        int(memory),
    )
    packet.validate()
    return packet


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
    def decide(self, prompt: str, seed: int, sampling_temperature: Optional[float] = None) -> ProviderResult:
        ...


@dataclass
class IndependentStatmechAgent:
    identifier: int
    role: str
    private_field: int
    _belief: int
    _action: int
    _confidence: float = 0.5
    _commitment: str = "uncommitted"
    _memory_state: str = "uncertain"
    _workload: int = 0
    _memory_history: List[str] = field(default_factory=list)
    _inbox: List[SignalPacket] = field(default_factory=list)
    _outbox: List[SignalPacket] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.private_field not in (-1, 0, 1) or self._belief not in (-1, 1) or self._action not in (-1, 1):
            raise ValueError("agent fields, beliefs, and actions must use the declared discrete states")

    def clone(self) -> "IndependentStatmechAgent":
        return copy.deepcopy(self)

    @property
    def belief(self) -> int:
        return int(self._belief)

    @property
    def action(self) -> int:
        return int(self._action)

    @property
    def confidence(self) -> float:
        return float(self._confidence)

    @property
    def commitment(self) -> str:
        return str(self._commitment)

    @property
    def memory_state(self) -> str:
        return str(self._memory_state)

    @property
    def memory(self) -> Tuple[str, ...]:
        return tuple(self._memory_history)

    @property
    def workload(self) -> int:
        return int(self._workload)

    @property
    def inbox(self) -> Tuple[SignalPacket, ...]:
        return tuple(self._inbox)

    @property
    def outbox(self) -> Tuple[SignalPacket, ...]:
        return tuple(self._outbox)

    def receive(self, packet: SignalPacket) -> None:
        packet.validate()
        self._inbox[:] = [existing for existing in self._inbox if existing.sender_id != packet.sender_id]
        self._inbox.append(packet)

    def clear_inbox(self) -> None:
        self._inbox.clear()

    def private_fingerprint(self) -> str:
        value = {
            "id": self.identifier,
            "role": self.role,
            "private_field": self.private_field,
            "belief": self._belief,
            "action": self._action,
            "confidence": self._confidence,
            "commitment": self._commitment,
            "memory_state": self._memory_state,
            "workload": self._workload,
            "memory": list(self._memory_history),
            "inbox": [packet.__dict__ for packet in self._inbox],
            "outbox": [packet.__dict__ for packet in self._outbox],
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()

    def apply_decision(
        self,
        decision: AgentDecision,
        mapping: LatentMapping,
        time_step: int,
        persistent_memory: bool,
    ) -> SignalPacket:
        """Apply only the validated model decision and create its chosen packet."""

        decision.validate()
        self._belief = mapping.spin(decision.belief_choice)
        self._action = mapping.spin(decision.action_choice)
        self._confidence = float(decision.confidence)
        self._commitment = decision.commitment_status
        self._memory_state = decision.memory_state
        if persistent_memory:
            self._memory_history.append(
                "t=%d belief=%s action=%s memory=%s"
                % (int(time_step), mapping.label(self._belief), mapping.label(self._action), self._memory_state)
            )
            self._memory_history[:] = self._memory_history[-3:]
        else:
            self._memory_history.clear()
        signal = 0 if decision.outgoing_signal == "uncertain" else mapping.spin(decision.outgoing_signal)
        packet = SignalPacket(
            sender_id=self.identifier,
            time_step=int(time_step),
            belief_spin=self._belief,
            action_spin=self._action,
            signal_spin=signal,
            confidence=self._confidence,
            commitment_code=COMMITMENTS.index(self._commitment),
            memory_code=MEMORY_STATES.index(self._memory_state),
        )
        self._outbox.append(packet)
        return packet

    def apply_tool_consequence(self, tool_action: str) -> int:
        """Apply a bounded local consequence without selecting the tool action."""

        if tool_action not in TOOLS:
            raise ValueError("invalid tool action")
        before = self._workload
        if tool_action != "no_action" and self.private_field != 0:
            if self._action == self.private_field:
                self._workload = max(-1, self._workload - 1)
            else:
                self._workload = min(1, self._workload + 1)
        return int(self._workload - before)


def _display_packet(packet: SignalPacket, mapping: LatentMapping) -> Dict[str, object]:
    return {
        "source_id": int(packet.sender_id),
        "observed_at_step": int(packet.time_step),
        "belief": mapping.label(packet.belief_spin),
        "action": mapping.label(packet.action_spin),
        "signal": "uncertain" if packet.signal_spin == 0 else mapping.label(packet.signal_spin),
        "confidence": round(float(packet.confidence), 3),
        "commitment": COMMITMENTS[int(packet.commitment_code)],
        "memory_state": MEMORY_STATES[int(packet.memory_code)],
    }


PROMPT_PARAPHRASES = (
    "Reconsider your local coordination state using only the authorized information below.",
    "Make one independent local update from the displayed private state and delivered neighbor packets.",
)


def build_agent_prompt(
    agent: IndependentStatmechAgent,
    mapping: LatentMapping,
    time_step: int,
    regime: str,
    coupling_strength: float,
    paraphrase: int,
) -> str:
    """Build a state-complete prompt without evaluator or peer-private state."""

    mapping.validate()
    if regime not in ("markovized", "persistent_memory"):
        raise ValueError("unknown agent regime")
    if not 0.0 <= float(coupling_strength) <= 1.0:
        raise ValueError("coupling strength must be in [0,1]")
    private = (
        "balanced/no directional local observation"
        if agent.private_field == 0
        else "local observation supports %s" % mapping.label(agent.private_field)
    )
    view: Dict[str, object] = {
        "agent_id": int(agent.identifier),
        "role": agent.role,
        "time_step": int(time_step),
        "regime": regime,
        "alternatives_in_display_order": list(mapping.display_order),
        "private_observation": private,
        "private_observation_strength": "balanced" if agent.private_field == 0 else "moderate",
        "current_belief": mapping.label(agent.belief),
        "current_action": mapping.label(agent.action),
        "current_confidence": round(agent.confidence, 3),
        "current_commitment": agent.commitment,
        "current_memory_state": agent.memory_state,
        "current_local_workload": agent.workload,
        "delivered_neighbor_packets": [_display_packet(packet, mapping) for packet in agent.inbox],
        "neighbor_relevance": round(float(coupling_strength), 3),
    }
    if regime == "persistent_memory":
        view["bounded_private_memory"] = list(agent.memory[-3:])
    instructions = (
        "You are one independent decentralized coordination agent. Amber and cobalt are symmetric alternatives; neither "
        "label, order, or color is intrinsically preferred. Your categorical belief is which alternative currently seems "
        "locally appropriate. Your categorical action is the alternative you select; choosing the current action retains it. Neighbor relevance states "
        "how strongly their locally generated packets bear on your decision, but you may agree, reject, defer, or revise. "
        "Use no global state, hidden truth, undelivered message, peer memory, future event, or counterfactual. The scheduler "
        "only offered this turn; it does not select your response. The typed tool execute_selected executes your own "
        "action_choice, while no_action changes no workload. Choose the outgoing signal yourself. Do not report a "
        "probability or chain of thought. "
        + PROMPT_PARAPHRASES[int(paraphrase) % len(PROMPT_PARAPHRASES)]
        + " Return exactly one JSON object and no prose."
    )
    envelope = {"authorized_local_state": view, "required_schema": decision_schema()}
    return instructions + "\nLOCAL_UPDATE=" + json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def encode_microstate(agents: Sequence[IndependentStatmechAgent]) -> int:
    """Encode the primary categorical belief-action projection into one integer."""

    value = 0
    for index, agent in enumerate(agents):
        if agent.belief == 1:
            value |= 1 << index
        if agent.action == 1:
            value |= 1 << (len(agents) + index)
    return int(value)


def decode_microstate(value: int, n_agents: int) -> Tuple[np.ndarray, np.ndarray]:
    if int(value) < 0 or int(value) >= 2 ** (2 * int(n_agents)):
        raise ValueError("microstate index out of bounds")
    beliefs = np.asarray([1 if int(value) & (1 << i) else -1 for i in range(int(n_agents))], dtype=int)
    actions = np.asarray(
        [1 if int(value) & (1 << (int(n_agents) + i)) else -1 for i in range(int(n_agents))], dtype=int
    )
    return beliefs, actions


def categorical_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    values = values / values.sum()
    return float(-np.sum(values * np.log(values)))


def binary_choice_entropy(choices: Sequence[int]) -> float:
    values = np.asarray(choices, dtype=int)
    if values.size == 0 or np.any(~np.isin(values, (-1, 1))):
        raise ValueError("choices must be a nonempty binary sequence")
    p = float(np.mean(values == 1))
    return categorical_entropy((p, 1.0 - p))


def bounded_logit(probability: float, epsilon: float = 1e-6) -> float:
    value = float(np.clip(float(probability), epsilon, 1.0 - epsilon))
    return float(math.log(value / (1.0 - value)))
