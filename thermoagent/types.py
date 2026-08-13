"""Typed domain objects shared by simulators and autonomous agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple


class Application(str, Enum):
    COMMERCIAL = "commercial"
    HUMANITARIAN = "humanitarian"


class Method(str, Enum):
    CENTRALIZED = "centralized_lookahead"
    CENTRAL_LLM = "centralized_llm"
    SCRIPTED = "scripted_independent"
    NO_COMM = "autonomous_no_comm"
    FIXED_COMM = "autonomous_fixed_comm"
    LEARNED_NO_ENTROPY = "learned_no_entropy"
    THERMO = "thermoagent"
    RANDOM_GATE = "random_gate"
    ENTROPY_LLM_ONLY = "entropy_llm_only"
    NO_EPISODIC_MEMORY = "no_episodic_memory"
    GLOBAL_ORACLE = "global_entropy_oracle"
    SHUFFLED_ENTROPY = "shuffled_entropy"


class CoordinationOption(IntEnum):
    CONTINUE = 0
    REQUEST_INFO = 1
    DISCLOSE_SUMMARY = 2
    NEGOTIATE = 3
    RESPOND_OFFER = 4
    PROPOSE_COALITION = 5
    REQUEST_REALLOCATION = 6
    EMERGENCY = 7
    SILENT = 8


@dataclass(frozen=True)
class Identity:
    agent_id: str
    role: str
    application: str
    organization: str
    location: Tuple[float, float]


@dataclass
class UtilityWeights:
    service: float = 1.0
    cost: float = 0.25
    fairness: float = 0.25
    disclosure: float = 0.05
    risk: float = 0.25
    reservation_price: float = 1.0

    def vector(self) -> List[float]:
        return [self.service, self.cost, self.fairness, self.disclosure, self.risk]


@dataclass
class PrivateObservation:
    step: int
    inventory: float
    capacity: float
    impairment: float
    demand: float
    backlog: float
    delay: float
    service_shortfall: float
    commitment_strain: float
    communication_reliability: float
    private_cost: float
    local_forecast: float

    def macro_features(self) -> List[float]:
        pressure = max(self.backlog / max(self.local_forecast, 1.0), self.service_shortfall)
        return [
            min(1.0, max(0.0, pressure)),
            min(1.0, max(0.0, self.impairment)),
            min(1.0, max(0.0, 0.6 * self.commitment_strain + 0.4 * (1.0 - self.communication_reliability))),
        ]


@dataclass
class MemoryRecord:
    step: int
    kind: str
    summary: str
    importance: float


@dataclass
class Message:
    message_id: str
    sender: str
    recipient: str
    kind: str
    payload: Dict[str, Any]
    sent_step: int
    deliver_step: int
    public: bool = False


@dataclass
class Commitment:
    commitment_id: str
    proposer: str
    partner: str
    quantity: float
    unit_price: float
    due_step: int
    status: str = "proposed"
    kind: str = "shipment"
    coalition_id: Optional[str] = None
    resource_owner: Optional[str] = None
    resource_recipient: Optional[str] = None
    parent_commitment_id: Optional[str] = None
    negotiation_round: int = 0


@dataclass
class PlanOutput:
    plan_summary: str
    tool: str
    arguments: Dict[str, Any]
    justification: str
    confidence: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    ok: bool
    code: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntropySummary:
    local_entropy: float = 0.0
    local_free_energy: float = 0.0
    delta_free_energy: float = 0.0
    local_surprisal: float = 0.0
    interaction_entropy: float = 0.0
    consensus_error: float = 0.0
    delayed: bool = False
    noisy: bool = False


@dataclass
class AgentDecision:
    agent_id: str
    option: int
    plan: PlanOutput
    tool_result: Optional[ToolResult] = None
    valid_output: bool = True
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_seconds: float = 0.0


@dataclass
class Shipment:
    shipment_id: str
    sender: str
    recipient: str
    quantity: float
    sent_step: int
    arrival_step: int
    promised_arrival_step: Optional[int] = None
    expedited: bool = False
    commitment_id: Optional[str] = None
