"""Sequential decentralized PPO for frozen V8 downstream policies.

The implementation is deliberately small and NumPy-only so that the complete
training path is reproducible on CPU.  It is nevertheless sequential PPO:
each persistent agent contributes a temporally ordered trajectory, rewards are
assigned after delayed actions complete, advantages use terminal-aware GAE,
and role-specific actors are updated with the clipped probability-ratio
objective.  Execution sees only local observations, private beliefs, and the
distributed estimate reconstructed from messages actually delivered.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .v7_types import DELEGATION_ACTIONS, V7StructuredDecision


FEATURE_VERSION = "v8-local-distributed-v1"
ALGORITHM = "role-specific decentralized linear IPPO with per-agent terminal-aware GAE"


def _softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if logits.ndim != 1 or mask.shape != logits.shape:
        raise ValueError("invalid logits or action mask")
    if not bool(mask.any()):
        raise ValueError("at least one action must be available")
    values = np.where(mask, logits, -1.0e9)
    values = values - float(np.max(values))
    probabilities = np.exp(values)
    probabilities = np.where(mask, probabilities, 0.0)
    return probabilities / float(probabilities.sum())


def _role_code(role: str) -> np.ndarray:
    """Stable low-dimensional role encoding without a global role registry."""
    digest = hashlib.sha256(str(role).encode("utf-8")).digest()
    return np.asarray([
        (digest[0] / 127.5) - 1.0,
        (digest[1] / 127.5) - 1.0,
        (digest[2] / 127.5) - 1.0,
    ], dtype=np.float64)


def deployable_feature_vector(
    *, agent: Any, asset: str, proposal: Any,
    distributed_estimate: Mapping[str, Any], step: int,
) -> np.ndarray:
    """Build fixed-width features using no evaluator-only values."""
    observation = agent.vault.observation(agent.agent_id, asset)
    local_belief = np.asarray(agent.private_beliefs[asset], dtype=np.float64)
    distributed = np.asarray(
        distributed_estimate["distributed_pooled_belief"], dtype=np.float64,
    )
    if len(local_belief) > 6 or len(distributed) > 6:
        raise ValueError("V8 feature schema supports at most six incident modes")
    local = np.zeros(6, dtype=np.float64)
    pooled = np.zeros(6, dtype=np.float64)
    local[: len(local_belief)] = local_belief
    pooled[: len(distributed)] = distributed
    kpis = observation.local_kpis
    contributors = float(distributed_estimate.get("contributors", 1))
    scoped = max(float(distributed_estimate.get("scoped_agents", 1)), 1.0)
    maximum_age = float(distributed_estimate.get("maximum_age", 0.0))
    base = np.asarray([
        float(kpis.get("severity", 0.0)),
        float(kpis.get("safety_risk", 0.0)),
        float(kpis.get("resource_scarcity", 0.0)),
        min(float(kpis.get("delay", 0.0)) / 8.0, 1.0),
        float(observation.telemetry_confidence),
        float(observation.communication_reliability),
        float(proposal.action_probability),
        math.tanh(float(proposal.action_value)),
        float(proposal.value_margin),
        contributors / scoped,
        min(maximum_age / 30.0, 1.0),
        float(distributed_estimate.get("distributed_disagreement", 0.0)),
        min(float(step) / 100.0, 1.0),
    ], dtype=np.float64)
    # Interactions give the linear policy enough capacity to distinguish a
    # high-severity confident recommendation from one supported by stale or
    # contradictory peer evidence without using hidden labels.
    interactions = np.asarray([
        base[0] * base[4],
        base[0] * base[11],
        base[7] * base[8],
        base[9] * (1.0 - base[10]),
        float(np.mean(np.abs(local - pooled))),
    ], dtype=np.float64)
    return np.concatenate((base, local, pooled, interactions, _role_code(agent.identity.role)))


def delegation_mask(proposal: Any) -> np.ndarray:
    mask = np.ones(len(DELEGATION_ACTIONS), dtype=bool)
    if not bool(proposal.is_physical):
        mask[:] = False
        mask[DELEGATION_ACTIONS.index("defer")] = True
    return mask


@dataclass
class V8Transition:
    agent_id: str
    role: str
    asset: str
    step: int
    observation: np.ndarray
    mask: np.ndarray
    action: int
    old_log_probability: float
    value: float
    proposal_is_physical: bool
    chain_id: Optional[str] = None
    accepted_physical_action: bool = False
    reward: float = 0.0
    advantage: float = 0.0
    return_value: float = 0.0
    terminal: bool = False


@dataclass
class RoleParameters:
    actor_weights: np.ndarray
    actor_bias: np.ndarray
    critic_weights: np.ndarray
    critic_bias: float


def grouped_terminal_gae(
    transitions: Sequence[V8Transition], *, gamma: float = 0.985,
    gae_lambda: float = 0.94,
) -> None:
    """Assign GAE independently to each persistent decentralized agent."""
    grouped: Dict[str, List[V8Transition]] = {}
    for transition in transitions:
        grouped.setdefault(transition.agent_id, []).append(transition)
    for values in grouped.values():
        ordered = sorted(values, key=lambda value: value.step)
        for value in ordered:
            value.terminal = False
        if ordered:
            ordered[-1].terminal = True
        next_value = 0.0
        next_advantage = 0.0
        for value in reversed(ordered):
            continuation = 0.0 if value.terminal else 1.0
            delta = value.reward + gamma * next_value * continuation - value.value
            value.advantage = float(
                delta + gamma * gae_lambda * next_advantage * continuation
            )
            value.return_value = float(value.value + value.advantage)
            next_value = value.value
            next_advantage = value.advantage


class V8RoleIPPOPolicy:
    """Role-specific independent actors sharing only frozen source code."""

    def __init__(
        self, *, seed: int, stochastic: bool = True,
        parameters: Optional[Mapping[str, RoleParameters]] = None,
        checkpoint_digest: Optional[str] = None,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.stochastic = bool(stochastic)
        self.parameters: Dict[str, RoleParameters] = dict(parameters or {})
        self.checkpoint_digest = checkpoint_digest
        suffix = checkpoint_digest[:10] if checkpoint_digest else "untrained"
        self.policy_id = "v8_ippo_seed_%d_%s" % (self.seed, suffix)
        self.transitions: List[V8Transition] = []
        self._pending_transition: Optional[V8Transition] = None

    @property
    def feature_dim(self) -> int:
        return 33

    def _parameters(self, role: str) -> RoleParameters:
        role = str(role)
        if role not in self.parameters:
            local_seed = int.from_bytes(
                hashlib.sha256((str(self.seed) + "|" + role).encode("utf-8")).digest()[:4],
                "big",
            )
            rng = np.random.RandomState(local_seed)
            self.parameters[role] = RoleParameters(
                actor_weights=rng.normal(
                    0.0, 0.025, (len(DELEGATION_ACTIONS), self.feature_dim),
                ),
                actor_bias=np.asarray([0.20, -0.05, -0.05, -0.12], dtype=np.float64),
                critic_weights=np.zeros(self.feature_dim, dtype=np.float64),
                critic_bias=0.0,
            )
        return self.parameters[role]

    def decide(
        self, *, agent: Any, asset: str,
        distributed_estimate: Mapping[str, Any], step: int,
    ) -> V7StructuredDecision:
        local_decision = agent.propose(asset)
        proposal = local_decision.proposal
        observation = deployable_feature_vector(
            agent=agent, asset=asset, proposal=proposal,
            distributed_estimate=distributed_estimate, step=step,
        )
        if len(observation) != self.feature_dim:
            raise AssertionError("V8 feature dimension changed without a schema revision")
        mask = delegation_mask(proposal)
        parameters = self._parameters(agent.identity.role)
        probabilities = _softmax(
            parameters.actor_weights @ observation + parameters.actor_bias, mask,
        )
        if self.stochastic:
            action = int(self.rng.choice(len(probabilities), p=probabilities))
        else:
            action = int(np.argmax(probabilities))
        value = float(parameters.critic_weights @ observation + parameters.critic_bias)
        transition = V8Transition(
            agent_id=agent.agent_id, role=agent.identity.role, asset=asset,
            step=int(step), observation=observation, mask=mask, action=action,
            old_log_probability=float(math.log(max(probabilities[action], 1e-12))),
            value=value, proposal_is_physical=bool(proposal.is_physical),
        )
        self.transitions.append(transition)
        self._pending_transition = transition
        delegation = DELEGATION_ACTIONS[action]
        return replace(
            local_decision,
            delegation_action=delegation,
            compact_plan_summary="frozen decentralized IPPO delegation over local and delivered evidence",
        )

    def observe_result(self, result: Mapping[str, Any], *, step: int) -> None:
        if self._pending_transition is None:
            raise RuntimeError("action result arrived without a pending decentralized decision")
        self._pending_transition.chain_id = result.get("causal_chain_id")
        self._pending_transition.accepted_physical_action = bool(
            result.get("accepted_physical_action", False)
        )
        self._pending_transition = None

    def finish_episode(
        self, *, completed_actions: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> None:
        effects = {
            str(value.get("chain_id")): float(value.get("causal_effect", 0.0))
            for value in completed_actions if value.get("chain_id") is not None
        }
        service_penalty = 0.001 * float(summary.get("service_loss", 0.0)) / max(
            len(self.transitions), 1,
        )
        for transition in self.transitions:
            delegation = DELEGATION_ACTIONS[transition.action]
            effect = effects.get(str(transition.chain_id), 0.0)
            reward = -service_penalty
            if delegation == "execute_autonomously":
                if transition.accepted_physical_action:
                    reward += effect
                elif transition.proposal_is_physical:
                    reward -= 0.08
            elif delegation == "escalate_operator":
                # V8 has no human operator. Escalation is a bounded withheld
                # action and therefore carries communication/opportunity cost.
                reward -= 0.05 + 0.20 * max(effect, 0.0)
            elif delegation == "defer":
                reward -= 0.015 + 0.30 * max(effect, 0.0)
            else:
                reward -= 0.02 + 0.40 * max(effect, 0.0)
            transition.reward = float(np.clip(reward, -4.0, 4.0))
        grouped_terminal_gae(self.transitions)

    def reset_episode(self) -> None:
        self.transitions = []
        self._pending_transition = None

    def save(self, path: Path, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            "schema_version": 1,
            "feature_version": FEATURE_VERSION,
            "algorithm": ALGORITHM,
            "seed": self.seed,
            "roles": {
                role: {
                    "actor_weights": values.actor_weights.tolist(),
                    "actor_bias": values.actor_bias.tolist(),
                    "critic_weights": values.critic_weights.tolist(),
                    "critic_bias": values.critic_bias,
                }
                for role, values in sorted(self.parameters.items())
            },
            "metadata": dict(metadata),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
            stream.write(encoded)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.checkpoint_digest = digest
        self.policy_id = "v8_ippo_seed_%d_%s" % (self.seed, digest[:10])
        return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

    @classmethod
    def load(cls, path: Path, *, stochastic: bool = False) -> "V8RoleIPPOPolicy":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError("incompatible V8 policy feature version")
        parameters = {
            role: RoleParameters(
                actor_weights=np.asarray(values["actor_weights"], dtype=np.float64),
                actor_bias=np.asarray(values["actor_bias"], dtype=np.float64),
                critic_weights=np.asarray(values["critic_weights"], dtype=np.float64),
                critic_bias=float(values["critic_bias"]),
            )
            for role, values in payload["roles"].items()
        }
        return cls(
            seed=int(payload["seed"]), stochastic=stochastic,
            parameters=parameters, checkpoint_digest=digest,
        )


def ppo_update(
    policy: V8RoleIPPOPolicy, transitions: Sequence[V8Transition], *,
    learning_rate: float = 0.012, clip_ratio: float = 0.18,
    value_weight: float = 0.45, entropy_weight: float = 0.012,
    epochs: int = 4, gradient_clip: float = 0.75,
) -> Dict[str, float]:
    """Batch clipped-PPO update with role-separated gradients."""
    if not transitions:
        raise ValueError("PPO requires nonempty sequential trajectories")
    advantages = np.asarray([value.advantage for value in transitions], dtype=np.float64)
    advantages = (advantages - float(advantages.mean())) / (
        float(advantages.std()) + 1e-8
    )
    diagnostics = {"actor_loss": 0.0, "critic_loss": 0.0, "policy_entropy": 0.0}
    for _ in range(int(epochs)):
        gradients: Dict[str, Dict[str, np.ndarray]] = {}
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []
        for index, transition in enumerate(transitions):
            parameters = policy._parameters(transition.role)
            bucket = gradients.setdefault(transition.role, {
                "actor_weights": np.zeros_like(parameters.actor_weights),
                "actor_bias": np.zeros_like(parameters.actor_bias),
                "critic_weights": np.zeros_like(parameters.critic_weights),
                "critic_bias": np.zeros(1, dtype=np.float64),
                "count": np.zeros(1, dtype=np.float64),
            })
            logits = parameters.actor_weights @ transition.observation + parameters.actor_bias
            probabilities = _softmax(logits, transition.mask)
            log_probability = math.log(max(probabilities[transition.action], 1e-12))
            ratio = math.exp(log_probability - transition.old_log_probability)
            advantage = float(advantages[index])
            clipped_ratio = float(np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio))
            actor_losses.append(-min(ratio * advantage, clipped_ratio * advantage))
            active = not (
                (advantage >= 0.0 and ratio > 1.0 + clip_ratio)
                or (advantage < 0.0 and ratio < 1.0 - clip_ratio)
            )
            one_hot = np.zeros(len(DELEGATION_ACTIONS), dtype=np.float64)
            one_hot[transition.action] = 1.0
            gradient_logits = (
                -advantage * ratio * (one_hot - probabilities)
                if active else np.zeros_like(probabilities)
            )
            entropy = -float(np.sum(
                probabilities[transition.mask]
                * np.log(np.maximum(probabilities[transition.mask], 1e-12))
            ))
            entropies.append(entropy)
            # d(-beta H)/d logits = beta * p * (log(p) + H).
            gradient_logits += entropy_weight * probabilities * (
                np.log(np.maximum(probabilities, 1e-12)) + entropy
            )
            gradient_logits = np.where(transition.mask, gradient_logits, 0.0)
            bucket["actor_weights"] += np.outer(gradient_logits, transition.observation)
            bucket["actor_bias"] += gradient_logits
            predicted = float(
                parameters.critic_weights @ transition.observation + parameters.critic_bias
            )
            error = predicted - transition.return_value
            critic_losses.append(0.5 * error * error)
            bucket["critic_weights"] += value_weight * error * transition.observation
            bucket["critic_bias"][0] += value_weight * error
            bucket["count"][0] += 1.0
        squared_norm = 0.0
        for bucket in gradients.values():
            count = max(float(bucket["count"][0]), 1.0)
            for name in ("actor_weights", "actor_bias", "critic_weights", "critic_bias"):
                bucket[name] /= count
                squared_norm += float(np.sum(np.square(bucket[name])))
        scale = min(1.0, gradient_clip / (math.sqrt(squared_norm) + 1e-12))
        for role, bucket in gradients.items():
            parameters = policy._parameters(role)
            parameters.actor_weights -= learning_rate * scale * bucket["actor_weights"]
            parameters.actor_bias -= learning_rate * scale * bucket["actor_bias"]
            parameters.critic_weights -= learning_rate * scale * bucket["critic_weights"]
            parameters.critic_bias -= learning_rate * scale * float(bucket["critic_bias"][0])
        diagnostics = {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "policy_entropy": float(np.mean(entropies)),
            "gradient_norm_before_clip": float(math.sqrt(squared_norm)),
        }
    return diagnostics


def action_distribution(transitions: Iterable[V8Transition]) -> Dict[str, int]:
    counts = {action: 0 for action in DELEGATION_ACTIONS}
    for transition in transitions:
        counts[DELEGATION_ACTIONS[transition.action]] += 1
    return counts


def _training_panel(index: int, rl_seed: int) -> Dict[str, Any]:
    application = "humanitarian" if index % 2 == 0 else "utility_restoration"
    complexity = "small" if index % 3 != 2 else "medium"
    # Modular graphs are prospectively reserved for validation and holdout.
    # Training may vary fresh instances of familiar families but cannot expose
    # the structurally held-out topology.
    topology = (
        ("random_geometric", "small_world")
        if application == "humanitarian"
        else ("grid", "scale_free")
    )[(index // 2) % 2]
    return {
        "application": application,
        "complexity": complexity,
        "coupling": ("low", "medium", "high")[(index // 2) % 3],
        "fragmentation": ("high", "low", "medium")[(index // 3) % 3],
        "network_disruption": ("medium", "high", "low")[(index // 4) % 3],
        "topology_family": topology,
        "environment_seed": 88200000 + (int(rl_seed) % 10000) * 100 + int(index),
    }


def _trigger_from_mapping(configuration: Mapping[str, Any]) -> Any:
    from .v8_trigger import TriggerConfig
    fields = set(TriggerConfig.__dataclass_fields__)
    return TriggerConfig(**{
        key: value for key, value in configuration.items() if key in fields
    })


def _training_trigger(
    index: int, *, generalized_configuration: Mapping[str, Any],
    comparator_configuration: Mapping[str, Any],
) -> Any:
    from .v8_trigger import TriggerConfig
    method = (
        "always_on", "generalized_information", "comparator",
        "periodic", "none",
    )[index % 5]
    if method == "generalized_information":
        return _trigger_from_mapping(generalized_configuration)
    if method == "comparator":
        return _trigger_from_mapping(comparator_configuration)
    return TriggerConfig(
        method=method, tau_on=0.11, tau_off=0.04,
        cooldown_steps=2, maximum_silence_steps=30,
        kpi_threshold=0.12, periodic_interval_steps=8,
        weights={"js": 0.45, "spectrum": 0.25, "confidence": 0.15, "age": 0.15},
    )


def train_v8_seed(
    *, rl_seed: int, repository: Path, results_root: Path,
    training_episodes: int = 18, update_epochs: int = 4,
    stage_label: str = "formal",
    generalized_configuration: Optional[Mapping[str, Any]] = None,
    comparator_configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one retained decentralized seed on a scheduler mixture."""
    from .v5_experiments import atomic_json, write_csv
    from .v8_experiments import run_v8_episode

    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    policy = V8RoleIPPOPolicy(seed=int(rl_seed), stochastic=True)
    generalized_configuration = dict(generalized_configuration or {
        "method": "generalized_information", "tau_on": 0.11,
        "tau_off": 0.04, "cooldown_steps": 2,
        "maximum_silence_steps": 30,
    })
    comparator_configuration = dict(comparator_configuration or {
        "method": "kpi_change", "kpi_threshold": 0.12,
        "maximum_silence_steps": 30,
    })
    curve: List[Dict[str, Any]] = []
    total_transitions = 0
    all_actions = {action: 0 for action in DELEGATION_ACTIONS}
    for episode_index in range(int(training_episodes)):
        policy.reset_episode()
        panel = _training_panel(episode_index, int(rl_seed))
        trigger = _training_trigger(
            episode_index,
            generalized_configuration=generalized_configuration,
            comparator_configuration=comparator_configuration,
        )
        output = run_v8_episode(
            **panel, trigger_config=trigger, action_policy=policy,
            information_condition="private_fragmented", encoding="uint8_simplex",
            maximum_hops=1, operational_communication_policy="agent_event_triggered",
            results_root=None, stage="training_%s" % stage_label, resume=False,
        )
        diagnostics = ppo_update(
            policy, policy.transitions, epochs=int(update_epochs),
        )
        counts = action_distribution(policy.transitions)
        for action, count in counts.items():
            all_actions[action] += count
        total_transitions += len(policy.transitions)
        curve.append({
            "rl_seed": int(rl_seed), "episode": episode_index + 1,
            "application": panel["application"], "complexity": panel["complexity"],
            "environment_seed": panel["environment_seed"],
            "scheduler": trigger.method,
            "transitions": len(policy.transitions),
            "mean_transition_reward": float(np.mean([
                value.reward for value in policy.transitions
            ])),
            "service_loss": float(output["summary"]["service_loss"]),
            "harmful_actions": int(output["summary"]["autonomous_harmful_actions"]),
            "beneficial_actions": int(output["summary"]["autonomous_beneficial_actions"]),
            "accepted_physical_actions": int(output["summary"]["accepted_physical_actions_v8"]),
            "action_diversity": int(sum(count > 0 for count in counts.values())),
            **{"delegation_%s" % key: value for key, value in counts.items()},
            **diagnostics,
        })
    checkpoint = results_root / "training" / "checkpoints" / (
        "v8-ippo-seed-%d.json.gz" % int(rl_seed)
    )
    checkpoint_record = policy.save(checkpoint, {
        "stage": stage_label,
        "training_episodes": int(training_episodes),
        "training_schedule": "balanced application and mixed communication scheduler",
        "source_parent": "V7 environments with V8 actual-wire monitoring",
    })
    curve_path = results_root / "training" / "curves" / (
        "v8-ippo-seed-%d.csv" % int(rl_seed)
    )
    write_csv(curve_path, curve)
    report = {
        "algorithm": ALGORITHM,
        "rl_seed": int(rl_seed),
        "status": "complete",
        "training_episodes": int(training_episodes),
        "sequential_transitions": int(total_transitions),
        "delegation_counts": all_actions,
        "delegation_diversity": int(sum(count > 0 for count in all_actions.values())),
        "collapsed": bool(
            all_actions["execute_autonomously"] == 0
            or sum(count > 0 for count in all_actions.values()) < 2
        ),
        "checkpoint": str(checkpoint.relative_to(repository)),
        "checkpoint_sha256": checkpoint_record["sha256"],
        "checkpoint_bytes": checkpoint_record["bytes"],
        "curve": str(curve_path.relative_to(repository)),
        "feature_version": FEATURE_VERSION,
        "execution_information": "strictly decentralized local plus delivered messages",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": float(time.perf_counter() - started_clock),
        "gpu_hours": 0.0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
    }
    atomic_json(
        results_root / "training" / "manifests" / (
            "v8-ippo-seed-%d.json" % int(rl_seed)
        ), report,
    )
    return report


def train_v8_multiseed(
    *, repository: Path, results_root: Path,
    seeds: Sequence[int], training_episodes: int = 18,
    update_epochs: int = 4, stage_label: str = "formal",
) -> Dict[str, Any]:
    """Retain every prospectively selected seed and every failure."""
    from .v5_experiments import atomic_json, write_csv

    protocol = json.loads(
        (results_root / "protocol" / "v8_frozen_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    generalized_configuration = dict(protocol["primary_trigger_configuration"])
    comparator_configuration = dict(
        protocol["strongest_nonentropic_comparator_configuration"]
    )

    reports: List[Dict[str, Any]] = []
    for seed in seeds:
        try:
            reports.append(train_v8_seed(
                rl_seed=int(seed), repository=repository, results_root=results_root,
                training_episodes=int(training_episodes),
                update_epochs=int(update_epochs), stage_label=stage_label,
                generalized_configuration=generalized_configuration,
                comparator_configuration=comparator_configuration,
            ))
        except Exception as error:
            reports.append({
                "rl_seed": int(seed), "status": "failed",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
            })
    write_csv(results_root / "training" / "seed_manifest.csv", reports)
    failures = [value for value in reports if value["status"] != "complete"]
    if failures:
        write_csv(results_root / "negative_results" / "rl_failed_seeds.csv", failures)
    ensemble_record = None
    if not failures and reports:
        checkpoints = [repository / value["checkpoint"] for value in reports]
        loaded = [V8RoleIPPOPolicy.load(path) for path in checkpoints]
        roles = sorted(set.intersection(*[
            set(value.parameters) for value in loaded
        ]))
        if not roles:
            raise RuntimeError("trained seeds have no common decentralized roles")
        averaged = {}
        for role in roles:
            averaged[role] = RoleParameters(
                actor_weights=np.mean([
                    value.parameters[role].actor_weights for value in loaded
                ], axis=0),
                actor_bias=np.mean([
                    value.parameters[role].actor_bias for value in loaded
                ], axis=0),
                critic_weights=np.mean([
                    value.parameters[role].critic_weights for value in loaded
                ], axis=0),
                critic_bias=float(np.mean([
                    value.parameters[role].critic_bias for value in loaded
                ])),
            )
        ensemble = V8RoleIPPOPolicy(seed=0, stochastic=False, parameters=averaged)
        ensemble_path = results_root / "training" / "checkpoints" / "v8-ippo-five-seed-ensemble.json.gz"
        ensemble_record = ensemble.save(ensemble_path, {
            "aggregation": "unweighted parameter mean over every prospectively selected seed",
            "member_seeds": [int(value) for value in seeds],
            "member_checkpoint_sha256": [value["checkpoint_sha256"] for value in reports],
            "selection": "no seed selected or excluded",
        })
        ensemble_record["path"] = str(ensemble_path.relative_to(repository))
    summary = {
        "algorithm": ALGORITHM,
        "seeds": [int(value) for value in seeds],
        "completed_seeds": len(reports) - len(failures),
        "failed_seeds": len(failures),
        "training_episodes_per_seed": int(training_episodes),
        "total_sequential_transitions": int(sum(
            int(value.get("sequential_transitions", 0)) for value in reports
        )),
        "total_seed_wall_seconds": float(sum(
            float(value.get("wall_seconds", 0.0)) for value in reports
        )),
        "gpu_hours": 0.0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "incremental_cloud_cost_usd": 0.0,
        "all_seeds_retained": True,
        "frozen_ensemble": ensemble_record,
    }
    atomic_json(results_root / "training" / "training_summary.json", summary)
    return summary
