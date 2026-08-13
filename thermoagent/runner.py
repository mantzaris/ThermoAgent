"""Episode orchestration, monitoring, baselines, and experiment metrics."""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .consensus import (
    consensus_rmse,
    gossip_distributions_with_trace,
    local_consensus_residuals,
    one_hot_sketch,
)
from .doet import (
    CommunicationMode,
    DistributedEntropyTrigger,
    TriggerConfig,
)
from .environment import (
    DEMAND_ROLES,
    SOURCE_ROLES,
    LogisticsEnvironment,
    ScenarioConfig,
    derived_rng_seed,
)
from .events import sha256_file
from .mechanics import (
    ENERGY_WEIGHT_SENSITIVITY,
    K_MACROSTATES,
    MacrostateCalibration,
    RollingMacrostateMonitor,
    free_energy_gap,
    interaction_entropy,
    normalized_entropy,
    occupancy_distribution,
    operational_energy,
)
from .planners import MockPlanner, PlannerRequest, PlannerResponse, validate_request_plan
from .policy import CoordinationPolicy
from .tools import ToolRegistry
from .types import CoordinationOption, EntropySummary, Method, PlanOutput, ToolResult


ROLE_INDEX = {
    role: index
    for index, role in enumerate(
        ["supplier", "manufacturer", "carrier", "warehouse", "retailer", "ngo", "agency", "transport", "depot", "clinic", "community"]
    )
}

BUDGET_MATCH_ACTIVATION_INTERVAL = 2
QUIET_BASELINE_DECISION_INTERVAL = 8


DOET_TRIGGER_METHODS = {
    Method.DOET_RULE,
    Method.DOET_RL,
    Method.KPI_CUSUM_TRIGGER,
    Method.GLOBAL_ENTROPY_TRIGGER_ORACLE,
    Method.DISRUPTION_LABEL_ORACLE,
}

DISTRIBUTED_ENTROPY_METHODS = {
    Method.THERMO,
    Method.ENTROPY_LLM_ONLY,
    Method.NO_EPISODIC_MEMORY,
    Method.SHUFFLED_ENTROPY,
    Method.DOET_RULE,
    Method.DOET_RL,
}

V2_COMMUNICATION_METHODS = DOET_TRIGGER_METHODS | {
    Method.FIXED_ALWAYS_ON,
    Method.PERIODIC_COMMUNICATION,
    Method.RANDOM_BUDGET_MATCHED,
}


@dataclass
class EpisodeResult:
    run_id: str
    application: str
    method: str
    scenario: str
    seed: int
    metrics: Dict[str, Any]
    time_series: List[Dict[str, Any]]
    agent_metrics: Dict[str, Any]
    planner_metrics: Dict[str, Any]
    completion_status: str
    wall_clock_seconds: float
    trajectory: List[Dict[str, Any]]


def default_calibration() -> MacrostateCalibration:
    return MacrostateCalibration(
        thresholds=np.asarray([[0.20, 0.55], [0.15, 0.50], [0.20, 0.55]], dtype=float),
        alpha=0.1,
        temperature=0.35,
    )


def calibration_from_json(path: Optional[Path]) -> MacrostateCalibration:
    if path is None:
        return default_calibration()
    value = json.loads(path.read_text(encoding="utf-8"))
    return MacrostateCalibration(
        thresholds=np.asarray(value["thresholds"], dtype=float),
        alpha=float(value.get("alpha", 0.1)),
        temperature=float(value.get("temperature", 0.35)),
        energy_weights=tuple(value.get("energy_weights", [0.35, 0.20, 0.30, 0.15])),
        role_references={
            str(role): list(reference)
            for role, reference in value.get("role_references", {}).items()
        },
    )


class EpisodeRunner:
    def __init__(
        self,
        config: ScenarioConfig,
        method: str,
        planner: Optional[Any] = None,
        policy: Optional[CoordinationPolicy] = None,
        calibration: Optional[MacrostateCalibration] = None,
        deterministic_policy: bool = True,
        gossip_rounds: int = 3,
        monitor_window: int = 3,
        monitor_formulation: str = "pooled",
        trigger_config: Optional[Mapping[str, Any]] = None,
        trigger_normalizers: Optional[Mapping[str, Mapping[str, float]]] = None,
        periodic_interval: Optional[int] = None,
        fixed_broadcast_fanout: int = 3,
    ) -> None:
        self.config = config
        self.method = Method(method)
        self.planner = planner or MockPlanner()
        self.policy = policy
        self.calibration = calibration or default_calibration()
        self.deterministic_policy = deterministic_policy
        self.gossip_rounds = gossip_rounds
        self.periodic_interval = int(
            periodic_interval or max(config.decision_interval, 1) * 2
        )
        if self.periodic_interval < 1:
            raise ValueError("periodic_interval must be positive")
        self.fixed_broadcast_fanout = int(fixed_broadcast_fanout)
        if self.fixed_broadcast_fanout < 1:
            raise ValueError("fixed_broadcast_fanout must be positive")
        self.registry = ToolRegistry()
        self.env = LogisticsEnvironment(config)
        self.monitor = RollingMacrostateMonitor(
            self.calibration, window=monitor_window, formulation=monitor_formulation
        )
        self.previous_free_energy: Dict[str, float] = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        self.previous_distributed_free_energy: Dict[str, float] = {
            agent_id: 0.0 for agent_id in self.env.agent_ids
        }
        self.shuffled_monitor_history: Dict[str, EntropySummary] = {
            agent_id: EntropySummary() for agent_id in self.env.agent_ids
        }
        uniform = np.full(K_MACROSTATES, 1.0 / K_MACROSTATES, dtype=float)
        self.previous_gossip_estimates: Dict[str, np.ndarray] = {
            agent_id: uniform.copy() for agent_id in self.env.agent_ids
        }
        # Evaluator-only perturbations use an independent deterministic stream,
        # so measuring estimator robustness cannot alter environment dynamics.
        self.monitor_link_rng = np.random.RandomState(
            derived_rng_seed(self.config.seed, "monitor_link_sampling")
        )
        self.monitor_noise_rng = np.random.RandomState(
            derived_rng_seed(self.config.seed, "monitor_noise")
        )
        self.trajectory: List[Dict[str, Any]] = []
        self._active_decision_indices: Dict[str, int] = {}
        self.valid_structured_outputs = 0
        self.total_structured_outputs = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.llm_calls = 0
        self.llm_latency = 0.0
        self.rejections = 0
        self.counteroffers = 0
        self.acceptances = 0
        self.failed_actions = 0
        self.revisions = 0
        self.proposed_tool_calls = 0
        self.successful_tool_proposals = 0
        self.monitor_sketch_messages = 0
        self.monitor_sketch_bytes = 0
        self.central_history: List[Dict[str, Any]] = []
        self.option_counts: Dict[int, int] = {index: 0 for index in range(9)}
        self.macro_features: List[List[float]] = []
        self.trigger: Optional[DistributedEntropyTrigger] = None
        self.latest_trigger_decisions: Dict[str, Dict[str, Any]] = {}
        self.processed_alert_ids: set[str] = set()
        self.trigger_alert_attempts = 0
        self.trigger_alert_successes = 0
        self.communication_active_decision_epochs = 0
        self.mode_step_counts: Dict[int, int] = {
            int(CommunicationMode.QUIET): 0,
            int(CommunicationMode.TARGETED): 0,
            int(CommunicationMode.CRISIS): 0,
        }
        self.random_active_agents: set[str] = set()
        self.previous_distributed_entropy: Dict[str, float] = {
            agent_id: 0.0 for agent_id in self.env.agent_ids
        }
        if self.method in DOET_TRIGGER_METHODS:
            parsed = TriggerConfig.from_mapping(trigger_config)
            normalizers: Dict[str, Mapping[str, float]] = {}
            supplied = dict(trigger_normalizers or {})
            application_values = dict(
                supplied.get("applications", {}).get(
                    self.config.application, {}
                )
            )
            default_normalizer = application_values.get(
                "default", supplied.get("default")
            )
            role_normalizers = dict(supplied.get("roles", {}))
            role_normalizers.update(application_values.get("roles", {}))
            for agent_id, agent in self.env.agents.items():
                selected = role_normalizers.get(agent.identity.role, default_normalizer)
                if selected is not None:
                    normalizers[agent_id] = selected
            self.trigger = DistributedEntropyTrigger(
                self.env.agent_ids,
                parsed,
                normalizers=normalizers,
            )

    def _gossip_round_edges(self) -> List[set[Tuple[str, str]]]:
        """Sample the sketch channel from its own paired RNG stream."""

        base_edges = sorted(self.env.active_communication_edges())
        if self.config.communication == "reliable":
            probability = 0.98
        elif self.config.communication == "intermittent":
            probability = 0.65
        elif self.config.communication == "partition":
            probability = (
                0.98
                if self.env.step_index < max(2, self.config.horizon // 3)
                else 0.85
            )
        else:
            probability = 0.80
        if self.trigger is None or self.method not in (
            Method.DOET_RULE,
            Method.DOET_RL,
        ):
            desired_rounds = {
                agent_id: self.gossip_rounds for agent_id in self.env.agent_ids
            }
            maximum_rounds = max(desired_rounds.values())
            return [
                {
                    edge for edge in base_edges
                    if self.monitor_link_rng.rand() <= probability
                }
                for _ in range(maximum_rounds)
            ]
        if self.trigger.config.disable_gossip:
            return []
        due_agents = {
            agent_id for agent_id in self.env.agent_ids
            if self.env.step_index % self.trigger.gossip_period(agent_id) == 0
        }
        if not due_agents:
            return []
        desired_rounds = {
            agent_id: (
                self.trigger.gossip_rounds(agent_id)
                if agent_id in due_agents else 0
            )
            for agent_id in self.env.agent_ids
        }
        maximum_rounds = max(desired_rounds.values())
        sampled: List[set[Tuple[str, str]]] = []
        for round_index in range(maximum_rounds):
            candidates = [
                edge for edge in base_edges
                if max(desired_rounds[edge[0]], desired_rounds[edge[1]]) > round_index
            ]
            if candidates:
                offset = (self.env.step_index + round_index) % len(candidates)
                candidates = candidates[offset:] + candidates[:offset]
            used = set()
            matching: set[Tuple[str, str]] = set()
            for edge in candidates:
                if edge[0] in used or edge[1] in used:
                    continue
                used.update(edge)
                if self.monitor_link_rng.rand() <= probability:
                    matching.add(edge)
            sampled.append(matching)
        return sampled

    def _local_interaction_entropy(self, agent_id: str) -> float:
        """Entropy of only the messages this agent sent or received."""

        agent = self.env.agents[agent_id]
        weights: Dict[Tuple[str, str], float] = {}
        for message in list(agent.outbox) + list(agent.inbox):
            age = max(0, self.env.step_index - int(message.sent_step))
            edge = (message.sender, message.recipient)
            weights[edge] = weights.get(edge, 0.0) + 0.8 ** age
        return interaction_entropy(weights)

    def _update_monitor(self) -> Dict[str, Any]:
        states: Dict[str, int] = {}
        roles = {agent_id: agent.identity.role for agent_id, agent in self.env.agents.items()}
        for agent_id, agent in self.env.agents.items():
            features = agent.vault.observation(agent_id).macro_features()
            self.macro_features.append(features)
            states[agent_id] = self.calibration.encode(features)
        exact = self.monitor.update(states, roles)
        local_sketches = {
            agent_id: one_hot_sketch(state, alpha=self.calibration.alpha, population_size=len(states))
            for agent_id, state in states.items()
        }
        if self.method in (Method.DOET_RULE, Method.DOET_RL):
            # Between sparse exchanges, every agent refreshes its own sketch
            # locally and retains the distributed estimate from prior gossip.
            # This is a compressed temporal consensus filter, not a free global
            # recomputation.
            sketches = {
                agent_id: (
                    0.75 * self.previous_gossip_estimates[agent_id]
                    + 0.25 * local_sketches[agent_id]
                )
                for agent_id in states
            }
        else:
            sketches = local_sketches
        edges_by_round = self._gossip_round_edges()
        if edges_by_round:
            estimates, gossip_trace = gossip_distributions_with_trace(
                sketches, edges_by_round
            )
        else:
            estimates = {
                agent_id: estimate.copy()
                for agent_id, estimate in sketches.items()
            }
            gossip_trace = []
        if self.method in DISTRIBUTED_ENTROPY_METHODS:
            # Gossip uses a low-bandwidth monitoring channel separate from the
            # bounded negotiation budget, but it is not free: count every
            # directed edge-round transmission and its deterministic compact
            # serialized payload for communication/Pareto reporting.
            for round_index, round_edges in enumerate(edges_by_round):
                round_estimates = gossip_trace[round_index]
                for left, right in sorted(round_edges):
                    for sender, recipient in ((left, right), (right, left)):
                        payload = {
                            "sender": sender,
                            "recipient": recipient,
                            "step": self.env.step_index,
                            "round": round_index + 1,
                            "distribution": np.round(
                                round_estimates[sender], 8
                            ).tolist(),
                        }
                        self.monitor_sketch_messages += 1
                        self.monitor_sketch_bytes += len(
                            json.dumps(
                                payload, sort_keys=True, separators=(",", ":")
                            ).encode("utf-8")
                        )
        for round_index, (round_edges, round_estimates) in enumerate(
            zip(edges_by_round, gossip_trace), start=1
        ):
            neighbors: Dict[str, List[str]] = {
                agent_id: [] for agent_id in self.env.agent_ids
            }
            for left, right in round_edges:
                neighbors[left].append(right)
                neighbors[right].append(left)
            for agent_id in sorted(self.env.agent_ids):
                self.env.ledger.append(
                    self.env.step_index,
                    "macrostate_sketch",
                    agent_id,
                    {
                        "round": round_index,
                        "neighbors": sorted(neighbors[agent_id]),
                        "distribution": np.round(
                            round_estimates[agent_id], 8
                        ).tolist(),
                    },
                    private_to=agent_id,
                )
        if not edges_by_round and self.method in (
            Method.DOET_RULE,
            Method.DOET_RL,
        ):
            for agent_id in sorted(self.env.agent_ids):
                self.env.ledger.append(
                    self.env.step_index,
                    "macrostate_sketch",
                    agent_id,
                    {
                        "round": 0,
                        "neighbors": [],
                        "local_update_only": True,
                        "distribution": np.round(
                            estimates[agent_id], 8
                        ).tolist(),
                    },
                    private_to=agent_id,
                )
        final_edges = edges_by_round[-1] if edges_by_round else set()
        local_residuals = local_consensus_residuals(estimates, final_edges)
        exact_p = occupancy_distribution(list(states.values()), self.calibration.alpha)
        error = consensus_rmse(estimates, exact_p)
        q = self.calibration.healthy_reference()
        delayed_estimates = {
            agent_id: self.previous_gossip_estimates[agent_id].copy()
            for agent_id in sorted(estimates)
        }
        noisy_estimates: Dict[str, np.ndarray] = {}
        for agent_id, estimate in estimates.items():
            perturbed = np.clip(
                estimate + self.monitor_noise_rng.normal(0.0, 0.01, size=estimate.shape),
                1e-9,
                None,
            )
            noisy_estimates[agent_id] = perturbed / perturbed.sum()
        self.previous_gossip_estimates = {
            agent_id: estimate.copy() for agent_id, estimate in estimates.items()
        }
        interaction = interaction_entropy(self.env.interaction_weights)
        local_interactions = {
            agent_id: self._local_interaction_entropy(agent_id)
            for agent_id in self.env.agent_ids
        }
        distributed_summaries: Dict[str, EntropySummary] = {}
        for agent_id, agent in self.env.agents.items():
            estimate = estimates[agent_id]
            local_entropy = normalized_entropy(estimate)
            local_free = free_energy_gap(estimate, q, self.calibration.temperature)
            distributed_summaries[agent_id] = EntropySummary(
                local_entropy=local_entropy,
                local_free_energy=local_free,
                delta_free_energy=local_free - self.previous_distributed_free_energy[agent_id],
                local_surprisal=float(exact["surprisal"][agent_id]),
                interaction_entropy=local_interactions[agent_id],
                consensus_error=local_residuals[agent_id],
                delayed=self.config.communication != "reliable",
                noisy=False,
            )
            self.previous_distributed_free_energy[agent_id] = local_free
        ordered_ids = sorted(self.env.agent_ids)
        next_shuffled_history: Dict[str, EntropySummary] = {
            agent_id: EntropySummary(**asdict(distributed_summaries[agent_id]))
            for agent_id in ordered_ids
        }
        for index, agent_id in enumerate(ordered_ids):
            agent = self.env.agents[agent_id]
            if self.method in (
                Method.CENTRALIZED,
                Method.CENTRAL_LLM,
                Method.NO_COMM,
                Method.FIXED_COMM,
                Method.LEARNED_NO_ENTROPY,
                Method.SCRIPTED,
                Method.RANDOM_GATE,
                Method.FIXED_ALWAYS_ON,
                Method.PERIODIC_COMMUNICATION,
                Method.RANDOM_BUDGET_MATCHED,
                Method.KPI_CUSUM_TRIGGER,
                Method.DISRUPTION_LABEL_ORACLE,
            ):
                agent.entropy = EntropySummary()
            elif self.method in (
                Method.GLOBAL_ORACLE,
                Method.GLOBAL_ENTROPY_TRIGGER_ORACLE,
            ):
                agent.entropy = EntropySummary(
                    local_entropy=float(exact["entropy"]),
                    local_free_energy=float(exact["free_energy"]),
                    delta_free_energy=float(exact["free_energy"]) - self.previous_free_energy[agent_id],
                    local_surprisal=float(exact["surprisal"][agent_id]),
                    interaction_entropy=interaction,
                    consensus_error=0.0,
                )
            elif self.method == Method.SHUFFLED_ENTROPY:
                # Causal parameter-matched negative control: each agent gets
                # another agent's prior-period monitor vector. This preserves
                # scale and dimensionality without future leakage while
                # breaking identity and current-event alignment.
                source_id = ordered_ids[(index + 1) % len(ordered_ids)]
                agent.entropy = EntropySummary(**asdict(self.shuffled_monitor_history[source_id]))
                agent.entropy.noisy = True
                agent.entropy.delayed = True
            else:
                agent.entropy = distributed_summaries[agent_id]
            self.previous_free_energy[agent_id] = agent.entropy.local_free_energy
        self.shuffled_monitor_history = next_shuffled_history
        trigger_metrics = self._update_coordination_triggers(
            distributed_summaries,
            exact,
        )
        evaluator_sensitivity: Dict[str, float] = {}
        exact_distribution = np.asarray(exact["p"], dtype=float)
        for name, weights in ENERGY_WEIGHT_SENSITIVITY.items():
            alternative_energy = self.calibration.energies(weights)
            alternative_reference = self.calibration.healthy_reference(weights)
            evaluator_sensitivity["exact_energy_sensitivity_" + name] = operational_energy(
                exact_distribution, alternative_energy
            )
            evaluator_sensitivity["exact_free_energy_sensitivity_" + name] = free_energy_gap(
                exact_distribution, alternative_reference, self.calibration.temperature
            )
        surprisal_ranking = sorted(
            exact["surprisal"], key=exact["surprisal"].get, reverse=True
        )
        return {
            "states": states,
            "exact_entropy": float(exact["entropy"]),
            "exact_energy": float(exact["energy"]),
            "exact_free_energy": float(exact["free_energy"]),
            "distributed_entropy_mean": float(np.mean([normalized_entropy(p) for p in estimates.values()])),
            "distributed_free_energy_mean": float(np.mean([free_energy_gap(p, q, self.calibration.temperature) for p in estimates.values()])),
            "delayed_entropy_mean": float(np.mean([normalized_entropy(p) for p in delayed_estimates.values()])),
            "delayed_free_energy_mean": float(np.mean([free_energy_gap(p, q, self.calibration.temperature) for p in delayed_estimates.values()])),
            "delayed_consensus_rmse": consensus_rmse(delayed_estimates, exact_p),
            "noisy_entropy_mean": float(np.mean([normalized_entropy(p) for p in noisy_estimates.values()])),
            "noisy_free_energy_mean": float(np.mean([free_energy_gap(p, q, self.calibration.temperature) for p in noisy_estimates.values()])),
            "noisy_consensus_rmse": consensus_rmse(noisy_estimates, exact_p),
            "consensus_rmse": error,
            "mean_local_consensus_residual": float(np.mean(list(local_residuals.values()))),
            "interaction_entropy": interaction,
            "mean_local_interaction_entropy": float(np.mean(list(local_interactions.values()))),
            "monitor_sketch_messages": self.monitor_sketch_messages,
            "monitor_sketch_bytes": self.monitor_sketch_bytes,
            "max_surprisal_agent": surprisal_ranking[0],
            "surprisal_ranked_agents": ";".join(surprisal_ranking),
            "max_surprisal": float(max(exact["surprisal"].values())),
            **trigger_metrics,
            **evaluator_sensitivity,
        }

    def _prepare_step_modes(self) -> None:
        """Sample paired random-baseline activation once per simulator step."""

        self.random_active_agents = set()
        if (
            self.method == Method.RANDOM_BUDGET_MATCHED
            and self.env.step_index % BUDGET_MATCH_ACTIVATION_INTERVAL == 0
        ):
            for agent_id, agent in self.env.agents.items():
                if agent.rng.rand() < self.config.random_gate_probability:
                    self.random_active_agents.add(agent_id)

    def _communication_mode(self, agent_id: str) -> CommunicationMode:
        if self.trigger is not None:
            return self.trigger.mode(agent_id)
        if self.method == Method.FIXED_ALWAYS_ON:
            return CommunicationMode.CRISIS
        if self.method == Method.PERIODIC_COMMUNICATION:
            return (
                CommunicationMode.CRISIS
                if self.env.step_index % self.periodic_interval == 0
                else CommunicationMode.QUIET
            )
        if self.method == Method.RANDOM_BUDGET_MATCHED:
            return (
                CommunicationMode.CRISIS
                if agent_id in self.random_active_agents
                else CommunicationMode.QUIET
            )
        return CommunicationMode.QUIET

    def _new_delivered_alerts(self, agent_id: str) -> int:
        count = 0
        for message in self.env.agents[agent_id].inbox:
            if (
                message.kind == "entropy_alert"
                and message.message_id not in self.processed_alert_ids
            ):
                self.processed_alert_ids.add(message.message_id)
                count += 1
        return count

    def _alert_neighbors(self, agent_id: str) -> List[str]:
        if self.trigger is None:
            return []
        guidance = self._material_action_guidance(agent_id)
        direct = list(guidance["direct_message_ids"])
        material = set(guidance["known_outbound_material_ids"]) | set(
            guidance["known_inbound_material_ids"]
        )
        agent = self.env.agents[agent_id]
        ranked = sorted(
            direct,
            key=lambda target: (
                target not in material,
                -float(agent.partner_trust.get(target, 0.5)),
                target,
            ),
        )
        return ranked[: self.trigger.config.max_alert_neighbors]

    def _update_coordination_triggers(
        self,
        distributed: Mapping[str, EntropySummary],
        exact: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Update local trigger state and emit only explicit bounded alerts."""

        activated: List[str] = []
        if self.trigger is not None:
            for agent_id in sorted(self.env.agent_ids):
                observation = self.env.agents[agent_id].vault.observation(agent_id)
                local_summary = distributed[agent_id]
                if self.method in (Method.DOET_RULE, Method.DOET_RL):
                    signal = float(local_summary.local_entropy)
                    if self.trigger.config.signal_noise_std > 0:
                        signal += float(self.monitor_noise_rng.normal(
                            0.0, self.trigger.config.signal_noise_std
                        ))
                    surprisal = float(local_summary.local_surprisal)
                    disagreement = float(local_summary.consensus_error)
                    signal_source = "distributed_operational_entropy"
                elif self.method == Method.KPI_CUSUM_TRIGGER:
                    pressure, impairment, strain = observation.macro_features()
                    signal = float(max(pressure, impairment, strain))
                    surprisal = 0.0
                    disagreement = 0.0
                    signal_source = "private_local_kpi_composite"
                elif self.method == Method.GLOBAL_ENTROPY_TRIGGER_ORACLE:
                    signal = float(exact["entropy"])
                    surprisal = float(exact["surprisal"][agent_id])
                    disagreement = 0.0
                    signal_source = "evaluator_global_entropy_oracle"
                else:
                    signal = float(self.env._disruption_applied)
                    surprisal = 0.0
                    disagreement = 0.0
                    signal_source = "evaluator_disruption_label_oracle"
                delivered_alerts = (
                    self._new_delivered_alerts(agent_id)
                    if self.trigger.config.propagation == "neighbor"
                    else 0
                )
                decision = self.trigger.update(
                    agent_id=agent_id,
                    step=self.env.step_index,
                    entropy=signal,
                    local_surprisal=surprisal,
                    consensus_disagreement=disagreement,
                    communication_availability=float(
                        observation.communication_reliability
                    ),
                    delivered_alerts=delivered_alerts,
                )
                row = {"signal_source": signal_source, "signal": signal, **decision.as_dict()}
                self.latest_trigger_decisions[agent_id] = row
                self.env.ledger.append(
                    self.env.step_index,
                    "coordination_trigger",
                    agent_id,
                    row,
                    private_to=agent_id,
                )
                if decision.activated:
                    activated.append(agent_id)

            if self.trigger.config.propagation == "neighbor":
                for agent_id in activated:
                    mode = int(self.trigger.mode(agent_id))
                    level = "severe" if mode == int(CommunicationMode.CRISIS) else "elevated"
                    for recipient in self._alert_neighbors(agent_id):
                        self.trigger_alert_attempts += 1
                        result = self.env.send_entropy_alert(
                            agent_id, recipient, mode, level
                        )
                        self.trigger_alert_successes += int(result.ok)
                        self.env.ledger.append(
                            self.env.step_index,
                            "trigger_alert_result",
                            agent_id,
                            {
                                "recipient": recipient,
                                "mode": mode,
                                **result.as_dict(),
                            },
                            private_to=agent_id,
                        )

        modes = [
            int(self._communication_mode(agent_id))
            for agent_id in self.env.agent_ids
        ]
        for mode in modes:
            self.mode_step_counts[mode] += 1
        statistics = (
            [state.cumulative_statistic for state in self.trigger.states.values()]
            if self.trigger is not None else [0.0]
        )
        activations = (
            sum(state.activation_count for state in self.trigger.states.values())
            if self.trigger is not None else 0
        )
        return {
            "trigger_active_agents": sum(mode > 0 for mode in modes),
            "trigger_crisis_agents": sum(mode == int(CommunicationMode.CRISIS) for mode in modes),
            "mean_trigger_statistic": float(np.mean(statistics)),
            "cumulative_trigger_activations": int(activations),
            "trigger_alert_attempts": self.trigger_alert_attempts,
            "trigger_alert_successes": self.trigger_alert_successes,
        }

    def _heuristic_option(self, agent_id: str) -> int:
        agent = self.env.agents[agent_id]
        obs = agent.vault.observation(agent_id)
        pending = [c for c in agent.commitments.values() if c.status == "proposed" and c.partner == agent_id]
        unresolved_coalitions = [
            message for message in agent.inbox
            if message.kind == "coalition_proposal"
            and message.payload.get("coalition_id") not in agent.coalition_ledger
            and int(message.payload.get("expires_step", self.env.step_index)) >= self.env.step_index
        ]
        accepted_to_honor = [
            c for c in agent.commitments.values()
            if c.status in ("accepted", "breached")
            and (c.resource_owner or c.proposer) == agent_id
        ]
        if pending:
            return int(CoordinationOption.RESPOND_OFFER)
        if accepted_to_honor:
            if not agent.last_tool_ok and agent.communication_budget > 0:
                return int(CoordinationOption.PROPOSE_COALITION)
            return int(CoordinationOption.CONTINUE)
        if unresolved_coalitions:
            return int(CoordinationOption.PROPOSE_COALITION)
        own_pending = [
            commitment for commitment in agent.commitments.values()
            if commitment.status == "proposed"
            and (commitment.resource_owner or commitment.proposer) == agent_id
        ]
        if own_pending:
            return int(CoordinationOption.CONTINUE)
        if not agent.last_tool_ok:
            return int(CoordinationOption.REQUEST_REALLOCATION)
        if obs.impairment > 0.65 and agent.communication_budget > 0:
            return int(CoordinationOption.PROPOSE_COALITION)
        if obs.service_shortfall > 0.55:
            return int(CoordinationOption.EMERGENCY)
        if agent.identity.role in DEMAND_ROLES and obs.backlog > max(2.0, obs.demand):
            return int(CoordinationOption.DISCLOSE_SUMMARY)
        if agent.identity.role in DEMAND_ROLES and obs.backlog > 0:
            return int(CoordinationOption.REQUEST_INFO)
        if agent.identity.role in SOURCE_ROLES and obs.inventory > obs.capacity * 1.4:
            return int(CoordinationOption.NEGOTIATE)
        return int(CoordinationOption.CONTINUE)

    def _local_option_mask(self, agent_id: str) -> np.ndarray:
        """Mask only structurally unavailable choices using private state."""

        agent = self.env.agents[agent_id]
        obs = agent.vault.observation(agent_id)
        mask = np.ones(9, dtype=bool)
        mask[int(CoordinationOption.RESPOND_OFFER)] = any(
            commitment.status == "proposed" and commitment.partner == agent_id
            for commitment in agent.commitments.values()
        )
        own_pending = any(
            commitment.status == "proposed"
            and (commitment.resource_owner or commitment.proposer) == agent_id
            for commitment in agent.commitments.values()
        )
        if own_pending:
            mask[int(CoordinationOption.NEGOTIATE)] = False
        delivered_coalition = any(
            message.kind == "coalition_proposal"
            and message.payload.get("coalition_id") not in agent.coalition_ledger
            and int(message.payload.get("expires_step", self.env.step_index)) >= self.env.step_index
            for message in agent.inbox
        )
        mask[int(CoordinationOption.PROPOSE_COALITION)] = bool(
            delivered_coalition
            or obs.impairment > 0.35
            or obs.service_shortfall > 0.55
            or agent.entropy.delta_free_energy > 0.03
        )
        mask[int(CoordinationOption.REQUEST_REALLOCATION)] = not agent.last_tool_ok
        mask[int(CoordinationOption.EMERGENCY)] = bool(
            obs.service_shortfall > 0.40 or obs.impairment > 0.60
        )
        if agent.communication_budget <= 0:
            for option in (
                CoordinationOption.REQUEST_INFO,
                CoordinationOption.DISCLOSE_SUMMARY,
                CoordinationOption.NEGOTIATE,
                CoordinationOption.PROPOSE_COALITION,
            ):
                mask[int(option)] = False
        if self.method in DOET_TRIGGER_METHODS | {
            Method.PERIODIC_COMMUNICATION,
            Method.RANDOM_BUDGET_MATCHED,
        }:
            mode = self._communication_mode(agent_id)
            if mode == CommunicationMode.QUIET:
                for option in (
                    CoordinationOption.REQUEST_INFO,
                    CoordinationOption.DISCLOSE_SUMMARY,
                    CoordinationOption.NEGOTIATE,
                ):
                    mask[int(option)] = False
                mask[int(CoordinationOption.PROPOSE_COALITION)] = delivered_coalition
            elif mode == CommunicationMode.TARGETED:
                # Bilateral information and negotiation are available, while
                # multi-party coalition formation remains a crisis privilege.
                mask[int(CoordinationOption.PROPOSE_COALITION)] = delivered_coalition
        mask[int(CoordinationOption.CONTINUE)] = True
        mask[int(CoordinationOption.SILENT)] = True
        return mask

    def _intensive_option(self, agent_id: str) -> int:
        """Communication-rich rule used by strong always-on controls."""

        heuristic = self._heuristic_option(agent_id)
        agent = self.env.agents[agent_id]
        has_execution_obligation = any(
            commitment.status in ("accepted", "breached", "proposed")
            and (commitment.resource_owner or commitment.proposer) == agent_id
            for commitment in agent.commitments.values()
        )
        if has_execution_obligation:
            return heuristic
        if heuristic not in (
            int(CoordinationOption.CONTINUE),
            int(CoordinationOption.SILENT),
        ):
            return heuristic
        observation = agent.vault.observation(agent_id)
        guidance = self._material_action_guidance(agent_id)
        epoch = self.env.step_index // max(1, self.config.decision_interval)
        if (
            agent.identity.role in SOURCE_ROLES
            and observation.inventory > 0
            and guidance["eligible_offer_target_ids"]
        ):
            return int(
                CoordinationOption.NEGOTIATE
                if epoch % 2 == 0 else CoordinationOption.DISCLOSE_SUMMARY
            )
        if agent.identity.role in DEMAND_ROLES:
            return int(
                CoordinationOption.REQUEST_INFO
                if epoch % 2 == 0 else CoordinationOption.DISCLOSE_SUMMARY
            )
        if guidance["direct_message_ids"]:
            return int(
                CoordinationOption.DISCLOSE_SUMMARY
                if epoch % 2 == 0 else CoordinationOption.REQUEST_INFO
            )
        return int(CoordinationOption.CONTINUE)

    def _material_action_guidance(self, agent_id: str) -> Dict[str, Any]:
        """Return auditable route affordances from locally legal information.

        Physical and communication topology is public infrastructure.  Closed
        routes remain in the initially advertised physical set so an unseen
        disruption can still produce a failed tool result and genuine
        replanning.  Temporary coalition reachability is added only when the
        agent's own ledger or delivered messages reveal the partner.
        """

        active_edges = self.env.active_communication_edges()
        direct = sorted({
            right if left == agent_id else left
            for left, right in active_edges
            if agent_id in (left, right)
        })
        outbound = {
            target for source, target in self.env.initial_physical_edges
            if source == agent_id
        }
        inbound = {
            source for source, target in self.env.initial_physical_edges
            if target == agent_id
        }
        agent = self.env.agents[agent_id]
        active_coalitions = {
            coalition_id
            for coalition_id, state in agent.coalition_ledger.items()
            if state.get("status") == "member"
            and int(state.get("expires_step", -1)) >= self.env.step_index
        }
        coalition_partners = set()
        for coalition_id in active_coalitions:
            state = agent.coalition_ledger[coalition_id]
            proposer = str(state.get("proposer", ""))
            if proposer and proposer != agent_id:
                coalition_partners.add(proposer)
        for message in agent.inbox:
            coalition_id = str(message.payload.get("coalition_id", ""))
            if coalition_id not in active_coalitions:
                continue
            if message.kind in ("coalition_joined", "coalition_proposal"):
                coalition_partners.add(message.sender)
        outbound.update(coalition_partners)
        direct_set = set(direct)
        return {
            "information_boundary": (
                "public initial routes, locally active communication links, "
                "and own delivered coalition state"
            ),
            "direct_message_ids": direct,
            "known_outbound_material_ids": sorted(outbound),
            "known_inbound_material_ids": sorted(inbound),
            "eligible_quote_source_ids": sorted(inbound & direct_set),
            "eligible_offer_target_ids": sorted(outbound & direct_set),
        }

    def _option(self, agent_id: str) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
        agent = self.env.agents[agent_id]
        include_entropy = self.method not in (
            Method.CENTRALIZED,
            Method.CENTRAL_LLM,
            Method.LEARNED_NO_ENTROPY,
            Method.NO_COMM,
            Method.FIXED_COMM,
            Method.SCRIPTED,
            Method.RANDOM_GATE,
            Method.FIXED_ALWAYS_ON,
            Method.PERIODIC_COMMUNICATION,
            Method.RANDOM_BUDGET_MATCHED,
            Method.KPI_CUSUM_TRIGGER,
            Method.DISRUPTION_LABEL_ORACLE,
        )
        observation = agent.observation_vector(ROLE_INDEX[agent.identity.role], include_entropy=include_entropy)
        action_mask = self._local_option_mask(agent_id)
        if self.method in (
            Method.LEARNED_NO_ENTROPY,
            Method.THERMO,
            Method.NO_EPISODIC_MEMORY,
            Method.GLOBAL_ORACLE,
            Method.SHUFFLED_ENTROPY,
            Method.DOET_RL,
        ):
            if self.policy is None:
                raise ValueError("learned method requires a coordination policy")
            action, logp, value = self.policy.act(
                observation, deterministic=self.deterministic_policy, action_mask=action_mask
            )
            return action, logp, value, observation, action_mask
        if self.method == Method.NO_COMM:
            action = int(CoordinationOption.CONTINUE if agent.identity.role in SOURCE_ROLES else CoordinationOption.SILENT)
        elif self.method == Method.FIXED_COMM:
            epoch = self.env.step_index // max(1, self.config.decision_interval)
            action = int(CoordinationOption.DISCLOSE_SUMMARY if epoch % 2 == 0 else CoordinationOption.CONTINUE)
        elif self.method in (
            Method.FIXED_ALWAYS_ON,
        ):
            action = self._intensive_option(agent_id)
        elif self.method in (
            Method.PERIODIC_COMMUNICATION,
            Method.RANDOM_BUDGET_MATCHED,
        ):
            action = (
                self._intensive_option(agent_id)
                if self._communication_mode(agent_id) > CommunicationMode.QUIET
                else self._heuristic_option(agent_id)
            )
        elif self.method == Method.RANDOM_GATE:
            pending = any(
                commitment.status == "proposed" and commitment.partner == agent_id
                for commitment in agent.commitments.values()
            )
            accepted = any(
                commitment.status in ("accepted", "breached")
                and (commitment.resource_owner or commitment.proposer) == agent_id
                for commitment in agent.commitments.values()
            )
            coalition_proposal = any(
                message.kind == "coalition_proposal"
                and message.payload.get("coalition_id") not in agent.coalition_ledger
                for message in agent.inbox
            )
            if pending:
                action = int(CoordinationOption.RESPOND_OFFER)
            elif accepted:
                action = int(CoordinationOption.CONTINUE)
            elif coalition_proposal:
                action = int(CoordinationOption.PROPOSE_COALITION)
            elif agent.rng.rand() < self.config.random_gate_probability:
                action = int(agent.rng.choice([
                    CoordinationOption.REQUEST_INFO,
                    CoordinationOption.DISCLOSE_SUMMARY,
                    CoordinationOption.NEGOTIATE,
                ]))
            else:
                action = int(CoordinationOption.SILENT)
        elif self.method in DOET_TRIGGER_METHODS:
            action = self._heuristic_option(agent_id)
        else:
            action = self._heuristic_option(agent_id)
        if not action_mask[action]:
            action = int(CoordinationOption.SILENT)
        return action, 0.0, 0.0, observation, action_mask

    def _centralized_actions(self) -> None:
        """Privileged full-information receding-horizon upper bound.

        This controller is deliberately unattainable under private
        information.  It observes exact inventories, demand, costs, routes,
        in-transit material, and disruption lead times, and replans every
        simulator period.  All material movement still passes through the same
        validated shipment tools and capacity constraints as every treatment.
        """

        sources = [a for a in self.env.agent_ids if self.env.agents[a].identity.role in SOURCE_ROLES]
        demands = [a for a in self.env.agent_ids if self.env.agents[a].identity.role in DEMAND_ROLES]
        lead = 1 + self.env.route_lead_time_penalty

        def projected_need(agent_id: str) -> float:
            state = self.env.states[agent_id]
            incoming = sum(
                shipment.quantity for shipment in self.env.shipments.values()
                if shipment.recipient == agent_id
            )
            forecast = max(state.demand, state.base_demand) * lead
            return max(0.0, state.backlog + forecast - state.inventory - incoming)

        for demand in sorted(
            demands,
            key=lambda agent_id: (
                self.env.states[agent_id].priority_weight * projected_need(agent_id),
                agent_id,
            ),
            reverse=True,
        ):
            need = projected_need(demand)
            feasible_sources = [
                source for source in sources
                if (source, demand) in self.env.physical_edges
                or self.env._coalition_route_available(source, demand)
            ]
            for source in sorted(
                feasible_sources,
                key=lambda agent_id: (self.env.states[agent_id].private_cost, agent_id),
            ):
                if need <= 0:
                    break
                available = max(0.0, self.env.states[source].inventory)
                handling_available = max(
                    0.0,
                    self.env.states[source].capacity - self.env.dispatch_used[source],
                )
                quantity = min(need, available, handling_available)
                if quantity > 0.01:
                    result = self.env.execute_tool(source, "schedule_shipment" if self.config.application == "commercial" else "transfer_resource", {
                        "target": demand, "quantity": float(quantity),
                        "arrival_step": self.env.step_index + lead,
                    })
                    if result.ok:
                        need -= quantity

    def _central_llm_action(self) -> None:
        """One coordinator acting only on legally public operational reports.

        The coordinator gets one independently validated dispatch slot per
        publicly reported demand organization.  This avoids an artificial
        one-tool bottleneck relative to a team of autonomous organizations,
        while strongly private conditions still expose no operational report
        and therefore permit only a safe no-op.
        """
        identities: List[Dict[str, Any]] = []
        reported_need_score = 0.0
        for public in self.env.public_identities():
            shared = public.get("shared_operational_state")
            report: Dict[str, Any] = {
                "agent_id": public["agent_id"],
                "role": public["role"],
                "organization": public["organization"],
                "inventory_level": "unreported",
                "need_level": "unreported",
                "impairment_level": "unreported",
            }
            if shared:
                # Preserve the exact legal report at shared-information levels
                # and the already coarse report at moderate levels.  No hidden
                # simulator field is added here.
                report["shared_operational_state"] = shared
                inventory = shared.get("inventory", "unreported")
                backlog = shared.get("backlog", "unreported")
                impairment = shared.get("impairment", "unreported")
                report["inventory_level"] = (
                    "high" if isinstance(inventory, (int, float)) and inventory > 20
                    else "nominal" if isinstance(inventory, (int, float)) and inventory > 3
                    else "low" if isinstance(inventory, (int, float))
                    else inventory
                )
                report["need_level"] = (
                    "high" if isinstance(backlog, (int, float)) and backlog > 5
                    else "nominal" if isinstance(backlog, (int, float)) and backlog > 0
                    else "low" if isinstance(backlog, (int, float))
                    else backlog
                )
                report["impairment_level"] = (
                    "high" if isinstance(impairment, (int, float)) and impairment > 0.5
                    else "nominal" if isinstance(impairment, (int, float)) and impairment > 0
                    else "low" if isinstance(impairment, (int, float))
                    else impairment
                )
            reported_need_score += {
                "high": 2.0, "nominal": 1.0, "low": 0.0,
                "unreported": 0.0,
            }.get(str(report["need_level"]), 0.0)
            identities.append(report)
        self.env.ledger.append(
            self.env.step_index,
            "public_signal",
            "public_information_interface",
            {
                "privacy_level": self.config.private_information,
                "recipient_scope": "legal_central_coordinator",
                "reports": identities,
            },
        )
        base_context = {
            "identity": {"agent_id": "central_coordinator", "role": "coordinator", "application": self.config.application, "organization": "legal_coordinator", "location": [0.0, 0.0]},
            "utility": {"service": 1.0, "cost": 0.25, "fairness": 0.25, "disclosure": 0.0, "risk": 0.2, "reservation_price": 0.0},
            "risk_tolerance": 0.2,
            "observation": {
                "step": self.env.step_index, "inventory": 0.0, "capacity": 0.0, "impairment": 0.0,
                "demand": 0.0, "backlog": reported_need_score, "delay": 0.0,
                "service_shortfall": min(1.0, reported_need_score / 4.0),
                "commitment_strain": 0.0, "communication_reliability": 1.0,
                "private_cost": 0.0, "local_forecast": max(1.0, reported_need_score),
            },
            "beliefs": {}, "working_memory": {},
            "memories": list(self.central_history[-4:]), "commitments": [],
            "partner_trust": {}, "messages": [], "entropy": {}, "communication_budget": 1,
            "last_plan_summary": "Allocate from coarse reports only.", "last_tool_ok": True,
        }
        reported_count = sum(
            identity["inventory_level"] != "unreported"
            or identity["need_level"] != "unreported"
            for identity in identities
        )
        reported_demands = [
            identity for identity in identities
            if identity["role"] in DEMAND_ROLES
            and identity["need_level"] != "unreported"
        ]
        assignments: List[Optional[Dict[str, Any]]] = (
            reported_demands if reported_demands else [None]
        )
        requests: List[PlannerRequest] = []
        for assignment in assignments:
            context = dict(base_context)
            eligible_source_ids: List[str] = []
            if assignment is not None:
                eligible_source_ids = sorted(
                    identity["agent_id"] for identity in identities
                    if identity["role"] in SOURCE_ROLES
                    and (identity["agent_id"], assignment["agent_id"])
                    in self.env.physical_edges
                )
                context["coordinator_assignment"] = {
                    "target": assignment["agent_id"],
                    "need_level": assignment["need_level"],
                    "eligible_source_ids": eligible_source_ids,
                    "instruction": (
                        "Select one legally reported source from eligible_source_ids "
                        "for this exact target; the IDs encode public physical routes."
                    ),
                }
            candidates = identities
            if assignment is not None:
                candidates = [
                    identity for identity in identities
                    if identity["agent_id"] in eligible_source_ids
                    or identity["agent_id"] == assignment["agent_id"]
                ]
            request = PlannerRequest(
                "central_coordinator", "coordinator", self.config.application,
                int(CoordinationOption.EMERGENCY), context, candidates,
            )
            requests.append(request)
            self.env.ledger.append(
                self.env.step_index,
                "llm_request",
                "central_coordinator",
                {
                    "option": int(CoordinationOption.EMERGENCY),
                    "legal_information": "public_operational_reports_only",
                    "reported_agents": reported_count,
                    "assigned_target": (
                        assignment["agent_id"] if assignment is not None else None
                    ),
                    "eligible_source_ids": eligible_source_ids,
                },
            )
        responses = self.planner.plan_batch(requests)
        if len(responses) != len(requests):
            raise RuntimeError("central planner batch response count mismatch")
        for request, response in zip(requests, responses):
            self.proposed_tool_calls += 1
            self.total_structured_outputs += 1
            self.valid_structured_outputs += int(response.valid_json)
            self.prompt_tokens += response.prompt_tokens
            self.generated_tokens += response.generated_tokens
            self.llm_calls += int(not isinstance(self.planner, MockPlanner))
            self.llm_latency += response.latency_seconds
            self.env.ledger.append(
                self.env.step_index,
                "llm_structured_response",
                "central_coordinator",
                {
                    "plan": response.output.as_dict(),
                    "valid_json": response.valid_json,
                    "recovery": response.recovery,
                    "raw_text_sha256": hashlib.sha256(
                        response.raw_text.encode("utf-8")
                    ).hexdigest(),
                    "raw_text_characters": len(response.raw_text),
                    "assigned_target": request.context.get(
                        "coordinator_assignment", {}
                    ).get("target"),
                },
            )
            validation = self.registry.validate("coordinator", response.output)
            if validation.ok and response.output.tool == "central_dispatch":
                args = dict(validation.data)
                source = args.pop("source")
                assigned_target = request.context.get(
                    "coordinator_assignment", {}
                ).get("target")
                eligible_sources = set(request.context.get(
                    "coordinator_assignment", {}
                ).get("eligible_source_ids", []))
                if assigned_target is None:
                    result = ToolResult(
                        False,
                        "coordinator_no_public_demand",
                        "no legal public demand report authorizes dispatch",
                    )
                elif source not in self.env.agents:
                    result = ToolResult(False, "invalid_source", "coordinator selected an unknown source")
                elif source not in eligible_sources:
                    result = ToolResult(
                        False,
                        "coordinator_source_route",
                        "dispatch source is not on a public physical route to the assigned target",
                        {"eligible_source_ids": sorted(eligible_sources)},
                    )
                elif args.get("target") != assigned_target:
                    result = ToolResult(False, "coordinator_target", "dispatch does not match its public demand assignment")
                else:
                    tool = "schedule_shipment" if self.config.application == "commercial" else "transfer_resource"
                    result = self.env.execute_tool(source, tool, args)
            elif validation.ok and response.output.tool == "no_op":
                result = ToolResult(True, "no_op", "coordinator made no dispatch")
            else:
                result = validation if not validation.ok else ToolResult(False, "coordinator_tool", "coordinator may only dispatch or pause")
            self.env.ledger.append(
                self.env.step_index,
                "tool_result",
                "central_coordinator",
                {"tool": response.output.tool, **result.as_dict()},
            )
            self.central_history.append({
                "step": self.env.step_index,
                "kind": "coordinator_result",
                "summary": "%s -> %s" % (response.output.tool, result.code),
                "result": result.as_dict(),
                "importance": 0.6 if not result.ok else 0.3,
            })
            if not result.ok:
                self.failed_actions += 1
            else:
                self.successful_tool_proposals += 1

    def _fixed_status_broadcast(self, agent_id: str) -> None:
        """Run the declared coarse always-on communication protocol."""

        observation = self.env.agents[agent_id].vault.observation(agent_id)
        pressure_value = max(
            observation.backlog / max(observation.local_forecast, 1.0),
            observation.service_shortfall,
        )
        pressure = (
            "high" if pressure_value > 0.55
            else "nominal" if pressure_value > 0.20 else "low"
        )
        capacity = (
            "high_impairment" if observation.impairment > 0.50
            else "impaired" if observation.impairment > 0.15 else "available"
        )
        strain = (
            "high" if observation.commitment_strain > 0.55
            else "nominal" if observation.commitment_strain > 0.20 else "low"
        )
        for recipient in self._material_action_guidance(agent_id)[
            "direct_message_ids"
        ][: self.fixed_broadcast_fanout]:
            self.env.send_fixed_status_summary(
                agent_id,
                recipient,
                pressure,
                capacity,
                strain,
            )

    def _decision_epoch(self, agent_ids: Optional[Sequence[str]] = None) -> None:
        if self.method == Method.CENTRALIZED:
            self._centralized_actions()
            return
        if self.method == Method.CENTRAL_LLM:
            self._central_llm_action()
            return
        requests: List[PlannerRequest] = []
        metadata: List[Tuple[str, int, float, float, np.ndarray, np.ndarray]] = []
        identities = self.env.public_identities()
        self.env.ledger.append(
            self.env.step_index,
            "public_signal",
            "public_information_interface",
            {
                "privacy_level": self.config.private_information,
                "recipient_scope": "all_autonomous_agents",
                "reports": identities,
            },
        )
        selected = list(agent_ids) if agent_ids is not None else list(self.env.agent_ids)
        for agent_id in selected:
            agent = self.env.agents[agent_id]
            if (
                self.method in (
                    Method.FIXED_ALWAYS_ON,
                    Method.PERIODIC_COMMUNICATION,
                    Method.RANDOM_BUDGET_MATCHED,
                )
                and self._communication_mode(agent_id) > CommunicationMode.QUIET
            ):
                self._fixed_status_broadcast(agent_id)
            option, logp, value, observation, action_mask = self._option(agent_id)
            self.option_counts[option] += 1
            context = agent.retrieve_context(
                self.env.step_index,
                self.env.ledger,
                include_episodic_memory=self.method != Method.NO_EPISODIC_MEMORY,
            )
            context["material_action_guidance"] = self._material_action_guidance(
                agent_id
            )
            mode = int(self._communication_mode(agent_id))
            context["communication_mode"] = {
                "value": mode,
                "name": CommunicationMode(mode).name.lower(),
            }
            context["trigger_state"] = self.latest_trigger_decisions.get(agent_id)
            self.communication_active_decision_epochs += int(mode > 0)
            request = PlannerRequest(agent_id, agent.identity.role, self.config.application, option, context, identities)
            requests.append(request)
            metadata.append((agent_id, option, logp, value, observation, action_mask))
            self.env.ledger.append(
                self.env.step_index,
                "llm_request",
                agent_id,
                {
                    "option": option,
                    "communication_mode": mode,
                    "planner_revision": getattr(self.planner, "revision", "unknown"),
                },
                private_to=agent_id,
            )
        responses: List[PlannerResponse] = self.planner.plan_batch(requests)
        if len(responses) != len(requests):
            raise RuntimeError("planner batch response count mismatch")
        reward_anchor = len(self.trajectory)
        for request, response, meta in zip(requests, responses, metadata):
            agent_id, option, logp, value, observation, action_mask = meta
            agent = self.env.agents[agent_id]
            self.proposed_tool_calls += 1
            self.total_structured_outputs += 1
            self.valid_structured_outputs += int(response.valid_json)
            self.prompt_tokens += response.prompt_tokens
            self.generated_tokens += response.generated_tokens
            self.llm_calls += int(not isinstance(self.planner, MockPlanner))
            self.llm_latency += response.latency_seconds
            self.env.ledger.append(
                self.env.step_index,
                "llm_structured_response",
                agent_id,
                {
                    "plan": response.output.as_dict(),
                    "valid_json": response.valid_json,
                    "raw_text_sha256": hashlib.sha256(response.raw_text.encode("utf-8")).hexdigest(),
                    "raw_text_characters": len(response.raw_text),
                    "recovery": response.recovery,
                },
                private_to=agent_id,
            )
            validation = self.registry.validate(agent.identity.role, response.output)
            affordance_error = validate_request_plan(request, response.output)
            if validation.ok and affordance_error is not None:
                validation = affordance_error
            if validation.ok:
                result = self.env.execute_tool(agent_id, response.output.tool, validation.data)
            else:
                result = validation
                self.env.ledger.append(self.env.step_index, "tool_result", agent_id, {"tool": response.output.tool, **result.as_dict()}, private_to=agent_id)
            if result.code == "offer_rejected":
                self.rejections += 1
            if result.code == "offer_countered":
                self.counteroffers += 1
            if result.code == "offer_accepted":
                self.acceptances += 1
            if not result.ok:
                self.failed_actions += 1
            else:
                self.successful_tool_proposals += 1
            if not agent.last_tool_ok and result.ok:
                self.revisions += 1
                self.env.ledger.append(self.env.step_index, "plan_revision", agent_id, {"previous_failure": True, "new_plan": response.output.plan_summary})
            agent.reflect(self.env.step_index, response.output.plan_summary, result.ok, result.code)
            self.trajectory.append({
                "observation": observation.tolist(), "action": option, "log_probability": logp,
                "value": value, "reward": -0.03 if not result.ok else 0.0,
                "done": False, "agent_id": agent_id,
                "step": self.env.step_index,
                "action_mask": action_mask.tolist(),
            })
            self._active_decision_indices[agent_id] = len(self.trajectory) - 1

    def _assign_rewards(self, previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> None:
        if previous is None or not self._active_decision_indices:
            return
        fulfillment_gain = float(current["fulfillment_rate"] - previous["fulfillment_rate"])
        backlog_growth = float(current["backlog"] - previous["backlog"]) / max(float(current["cumulative_demand"]), 1.0)
        message_cost = float(current["messages"] - previous["messages"]) * 0.002
        breach_cost = float(current["commitment_breaches"] - previous["commitment_breaches"]) * 0.05
        reward = 3.0 * fulfillment_gain - backlog_growth - message_cost - breach_cost
        for index in self._active_decision_indices.values():
            self.trajectory[index]["reward"] += reward

    def _agent_evaluator_metrics(self) -> Dict[str, Any]:
        """Evaluator-only summaries; these values never enter an actor input."""

        final_public = self.env.public_metrics()
        global_service = float(final_public["fulfillment_rate"])
        utilities: List[float] = []
        memory_correct = 0
        memory_total = 0
        reliability_errors: List[float] = []
        for agent_id, agent in self.env.agents.items():
            state = self.env.states[agent_id]
            if agent.identity.role in DEMAND_ROLES:
                performance = state.fulfilled / max(state.cumulative_demand, 1e-9)
                local_loss = state.backlog / max(state.cumulative_demand, 1.0)
            else:
                performance = 1.0 - state.impairment
                local_loss = state.impairment
            disclosure = self.env.disclosures_by_agent.get(agent_id, 0) / max(self.config.communication_budget, 1)
            utility = (
                agent.utility.service * performance
                - agent.utility.cost * local_loss
                - agent.utility.fairness * abs(performance - global_service)
                - agent.utility.disclosure * disclosure
                - agent.utility.risk * state.impairment
            )
            utilities.append(float(utility))
            for commitment_id, commitment in self.env.commitments.items():
                if agent_id not in (commitment.proposer, commitment.partner):
                    continue
                memory_total += 1
                local = agent.commitments.get(commitment_id)
                memory_correct += int(local is not None and local.status == commitment.status)
                if commitment.status in ("honored", "breached"):
                    partner = commitment.partner if agent_id == commitment.proposer else commitment.proposer
                    predicted = float(agent.partner_trust.get(partner, 0.5))
                    observed = float(commitment.status == "honored")
                    reliability_errors.append((predicted - observed) ** 2)
        offer_steps = {
            str(event.payload.get("commitment_id")): event.step
            for event in self.env.ledger.events if event.kind in ("offer", "counteroffer")
        }
        agreement_delays = [
            event.step - offer_steps[str(event.payload.get("commitment_id"))]
            for event in self.env.ledger.events
            if event.kind == "commitment" and str(event.payload.get("commitment_id")) in offer_steps
        ]
        return {
            "minimum_agent_utility": min(utilities) if utilities else 0.0,
            "mean_agent_utility": float(np.mean(utilities)) if utilities else 0.0,
            "commitment_memory_accuracy": memory_correct / max(memory_total, 1),
            "commitment_memory_items": memory_total,
            "partner_reliability_brier": float(np.mean(reliability_errors)) if reliability_errors else None,
            "partner_reliability_observations": len(reliability_errors),
            "mean_time_to_agreement": float(np.mean(agreement_delays)) if agreement_delays else None,
        }

    def _scheduled_agent_ids(self, step: int) -> List[str]:
        """Select locally scheduled planners without using a true disruption label."""

        if self.method == Method.CENTRALIZED:
            return list(self.env.agent_ids)
        if self.method == Method.FIXED_ALWAYS_ON:
            return (
                list(self.env.agent_ids)
                if step % max(1, self.config.decision_interval) == 0 else []
            )
        if self.method == Method.PERIODIC_COMMUNICATION:
            return (
                list(self.env.agent_ids)
                if (
                    step % self.periodic_interval == 0
                    or step % QUIET_BASELINE_DECISION_INTERVAL == 0
                ) else []
            )
        if self.method == Method.RANDOM_BUDGET_MATCHED:
            if step % QUIET_BASELINE_DECISION_INTERVAL == 0:
                return list(self.env.agent_ids)
            return sorted(self.random_active_agents)
        if self.method in DOET_TRIGGER_METHODS:
            selected = []
            for agent_id in self.env.agent_ids:
                interval = self.trigger.decision_interval(agent_id) if self.trigger else self.config.decision_interval
                transition = self.latest_trigger_decisions.get(agent_id, {})
                activated_now = bool(
                    transition.get("activated")
                    and int(transition.get("step", -1)) == step
                )
                if step % max(1, interval) == 0 or activated_now:
                    selected.append(agent_id)
            return selected
        return (
            list(self.env.agent_ids)
            if step % max(1, self.config.decision_interval) == 0 else []
        )

    def run(self, run_id: Optional[str] = None) -> EpisodeResult:
        started = time.perf_counter()
        run_id = run_id or "%s-%s-s%03d" % (self.config.application, self.method.value, self.config.seed)
        time_series: List[Dict[str, Any]] = []
        previous_metrics: Optional[Dict[str, Any]] = None
        try:
            event_cursor = len(self.env.ledger.events)
            requested_replans: set[str] = set()
            for step in range(self.config.horizon):
                self.env.transition()
                new_events = self.env.ledger.events[event_cursor:]
                event_cursor = len(self.env.ledger.events)
                self.env.deliver_observations()
                self._prepare_step_modes()
                monitor = self._update_monitor()
                metrics = self.env.public_metrics()
                row = {**metrics, **monitor, "disruption_active": self.env._disruption_applied}
                time_series.append(row)
                # Credit actions chosen at the previous decision epoch with the
                # subsequently observed transition, including all periods until
                # the next decision. This precedes selection of the next option.
                self._assign_rewards(previous_metrics, metrics)
                disruption_trigger = (
                    self.env._disruption_applied
                    and step == max(2, self.config.horizon // 3)
                    and self.method not in V2_COMMUNICATION_METHODS
                )
                # The full-information controller is explicitly an oracle-like
                # upper bound and replans every period. All deployable methods
                # retain the same periodic/event-triggered schedule.
                scheduled_agents = self._scheduled_agent_ids(step)
                if disruption_trigger:
                    scheduled_agents = list(self.env.agent_ids)
                triggered_agents = set(requested_replans)
                for event in new_events:
                    if event.kind != "message_delivery":
                        continue
                    if event.payload.get("kind") in (
                        "offer", "counteroffer", "offer_accepted",
                        "offer_rejected", "coalition_proposal",
                        "coalition_joined", "coalition_refused",
                        "commitment_breach", "late_delivery",
                    ):
                        triggered_agents.add(str(event.payload["recipient"]))
                selected_agents = sorted(set(scheduled_agents) | triggered_agents)
                if selected_agents:
                    self._decision_epoch(selected_agents)
                requested_replans = {
                    agent_id for agent_id, agent in self.env.agents.items()
                    if not agent.last_tool_ok
                }
                for agent_id, agent in self.env.agents.items():
                    state = self.env.states[agent_id]
                    if any(
                        commitment.status in ("accepted", "breached")
                        and (commitment.resource_owner or commitment.proposer) == agent_id
                        and commitment.quantity <= min(state.inventory, state.capacity) + 1e-9
                        for commitment in agent.commitments.values()
                    ):
                        requested_replans.add(agent_id)
                previous_metrics = metrics
                self.env.advance()
            status = "complete"
        except Exception:
            status = "failed"
            raise
        finally:
            if self.trajectory:
                last_by_agent: Dict[str, int] = {}
                for index, row in enumerate(self.trajectory):
                    last_by_agent[row["agent_id"]] = index
                for index in last_by_agent.values():
                    self.trajectory[index]["done"] = True
        wall = time.perf_counter() - started
        service_loss_auc = float(sum(row["service_loss"] for row in time_series))
        cumulative_unmet = float(sum(row["weighted_backlog"] for row in time_series))
        disruption_step = max(2, self.config.horizon // 3)
        recovered = [row["step"] for row in time_series if row["step"] > disruption_step and row["service_loss"] < 0.2]
        recovery_time = float(recovered[0] - disruption_step) if recovered else float(self.config.horizon - disruption_step)
        final = time_series[-1]
        total_agent_steps = max(1, self.config.horizon * len(self.env.agent_ids))
        evaluator_agent_metrics = self._agent_evaluator_metrics()
        coalition_proposal_steps = [
            event.step for event in self.env.ledger.events
            if event.kind == "coalition_event"
            and event.payload.get("action") == "propose"
        ]
        coalition_formation_by_id: Dict[str, int] = {}
        for event in self.env.ledger.events:
            if (
                event.kind == "coalition_event"
                and event.payload.get("action") == "join_coalition"
                and event.payload.get("ok")
            ):
                coalition_formation_by_id.setdefault(
                    str(event.payload.get("coalition_id")), event.step
                )
        coalition_steps = list(coalition_formation_by_id.values())
        useful_coalitions = 0
        for coalition_step in coalition_steps:
            before = next((row["service_loss"] for row in time_series if row["step"] == coalition_step), None)
            after = [row["service_loss"] for row in time_series if row["step"] > coalition_step]
            useful_coalitions += int(before is not None and bool(after) and min(after) <= float(before) - 0.05)
        cooperation_required = self.config.disruption in ("correlated", "compound")
        primary = service_loss_auc if self.config.application == "commercial" else cumulative_unmet
        metrics = {
            "primary_outcome": primary,
            "service_loss_auc": service_loss_auc,
            "cumulative_unmet_weighted_need": cumulative_unmet,
            "recovery_time": recovery_time,
            "fulfillment_rate": final["fulfillment_rate"],
            "final_backlog": final["backlog"],
            "fairness": final["fairness"],
            "total_cost": final["total_cost"],
            "social_welfare": final["fulfillment_rate"] + 0.25 * final["fairness"] - final["total_cost"] / max(final["cumulative_demand"] * 10.0, 1.0),
            "commitment_breaches": final["commitment_breaches"],
            "messages": final["messages"],
            "delivered_messages": final["delivered_messages"],
            "message_delivery_rate": final["delivered_messages"] / max(final["messages"], 1),
            "message_bytes": final["message_bytes"],
            "monitor_sketch_messages": self.monitor_sketch_messages,
            "monitor_sketch_bytes": self.monitor_sketch_bytes,
            "total_communication_messages": (
                final["messages"] + self.monitor_sketch_messages
            ),
            "total_communication_bytes": (
                final["message_bytes"] + self.monitor_sketch_bytes
            ),
            "communication_active_decision_epochs": self.communication_active_decision_epochs,
            "quiet_mode_fraction": self.mode_step_counts[int(CommunicationMode.QUIET)] / total_agent_steps,
            "targeted_mode_fraction": self.mode_step_counts[int(CommunicationMode.TARGETED)] / total_agent_steps,
            "crisis_mode_fraction": self.mode_step_counts[int(CommunicationMode.CRISIS)] / total_agent_steps,
            "trigger_activations": (
                sum(state.activation_count for state in self.trigger.states.values())
                if self.trigger is not None else 0
            ),
            "trigger_alert_attempts": self.trigger_alert_attempts,
            "trigger_alert_successes": self.trigger_alert_successes,
            "information_disclosures": final["information_disclosures"],
            "tool_calls": final["tool_calls"],
            "coalitions": final["coalitions"],
            "coalition_proposals": len(coalition_proposal_steps),
            "coalitions_formed": len(coalition_formation_by_id),
            "agreement_rate": final["offers_accepted"] / max(final["offers_submitted"], 1),
            "individually_rational_agreement_rate": final["individually_rational_acceptances"] / max(final["offers_accepted"], 1),
            "useful_coalition_precision": useful_coalitions / max(len(coalition_steps), 1),
            "coalition_recall_when_required": float(bool(coalition_steps)) if cooperation_required else None,
            "on_time_delivery_rate": final["on_time_delivery_rate"],
            "transport_efficiency": final["transport_efficiency"],
            "inventory_efficiency": final["inventory_efficiency"],
            "conservation_error": final["conservation_error"],
            "max_free_energy": max(row["exact_free_energy"] for row in time_series),
            "mean_consensus_rmse": float(np.mean([row["consensus_rmse"] for row in time_series])),
            **evaluator_agent_metrics,
        }
        agent_metrics = {
            "valid_structured_output_rate": self.valid_structured_outputs / max(self.total_structured_outputs, 1),
            "valid_tool_call_rate": (
                self.successful_tool_proposals / self.proposed_tool_calls
                if self.proposed_tool_calls else
                self.env.valid_tool_calls / max(self.env.tool_calls, 1)
            ),
            "tool_proposals": self.proposed_tool_calls,
            "plan_revisions": self.revisions,
            "rejections": self.rejections,
            "counteroffers": self.counteroffers,
            "acceptances": self.acceptances,
            "failed_actions": self.failed_actions,
            "option_counts": self.option_counts,
            "coalition_formation_rate": float(bool(coalition_formation_by_id)),
            "multi_step_completion_rate": self.revisions / max(self.failed_actions, 1),
            "contradiction_rate": sum(
                event.payload.get("code") == "private_utility_constraint"
                for event in self.env.ledger.events if event.kind == "tool_result"
            ) / max(self.proposed_tool_calls, 1),
        }
        planner_metrics = {
            "planner_revision": getattr(self.planner, "revision", "unknown"),
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "llm_latency_seconds": self.llm_latency,
        }
        scenario = "%s-%s-p%.1f-o%.1f" % (
            self.config.communication, self.config.disruption,
            self.config.private_information, self.config.objective_misalignment,
        )
        return EpisodeResult(
            run_id, self.config.application, self.method.value, scenario, self.config.seed,
            metrics, time_series, agent_metrics, planner_metrics, status, wall, self.trajectory,
        )


def write_episode(result: EpisodeResult, ledger: Any, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode.json"
    events_path = output_dir / "events.jsonl.gz"
    episode_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger.write_jsonl(events_path)
    return {
        "episode.json": sha256_file(episode_path),
        "events.jsonl.gz": sha256_file(events_path),
    }
