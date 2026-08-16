"""Sequential decentralized PPO training for V6 delegation policies.

Actors share no observations or recurrent state.  Each role has its own actor
and critic parameters, and execution features are restricted to the selected
agent's deployable context.  Training uses full dynamic trajectories,
discounted returns, GAE, clipping, and action masks; it is not the contextual
bandit used in V5.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .events import sha256_file
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v6_environment import APP_ROLES, V6PanelEnvironment
from .v6_policies import binary_entropy
from .v6_types import DELEGATION_ACTIONS, OPERATIONAL_ACTIONS, V6DecisionContext


METHODS = (
    "ppo_kpi_only",
    "ppo_predictive_uncertainty",
    "ppo_shannon_js",
    "ppo_generalized_tsallis_gini",
    "ppo_combined_generalized_entropic",
)
ROLE_NAMES = tuple(sorted({role for roles in APP_ROLES.values() for role in roles}))
TRAIN_REGIMES = (
    "isolated_physical", "telemetry_integrity", "partition",
    "correlated", "compound", "ood",
)


def _base_features(context: V6DecisionContext) -> List[float]:
    action_one_hot = [
        float(context.proposal.action == value) for value in OPERATIONAL_ACTIONS
    ]
    return [
        context.local_kpis["visible_severity"],
        context.local_kpis["visible_backlog"],
        context.local_kpis["visible_delay"],
        context.local_kpis["resource_scarcity"],
        context.local_kpis["safety_risk"],
        context.local_kpis["commitment_strain"],
        context.proposal.action_probability,
        context.proposal.action_value,
        context.proposal.value_margin,
        context.communication_reliability,
        *action_one_hot,
    ]


def feature_vector(method: str, context: V6DecisionContext) -> np.ndarray:
    if method not in METHODS:
        raise ValueError("unknown V6 PPO method: %s" % method)
    values = _base_features(context)
    if method in (
        "ppo_predictive_uncertainty", "ppo_shannon_js",
        "ppo_generalized_tsallis_gini", "ppo_combined_generalized_entropic",
    ):
        values.append(binary_entropy(context.proposal.action_probability))
    if method in ("ppo_shannon_js", "ppo_combined_generalized_entropic"):
        values.extend([
            context.shannon_local,
            context.pooled_uncertainty - context.average_local_uncertainty,
            context.js_disagreement,
            context.graph_disagreement,
            context.consensus_residual,
            context.disagreement_slope,
        ])
    if method in (
        "ppo_generalized_tsallis_gini", "ppo_combined_generalized_entropic",
    ):
        values.extend([
            context.tsallis_0_5_local - context.tsallis_3_local,
            context.jt_disagreement_0_5 - context.jt_disagreement_3,
            context.gini_simpson_local,
            context.jt_disagreement_2,
        ])
    return np.asarray(values, dtype=np.float32)


def delegation_mask(context: V6DecisionContext) -> np.ndarray:
    """Mask unavailable delegation choices instead of rewarding violations."""
    mask = np.ones(len(DELEGATION_ACTIONS), dtype=bool)
    if context.proposal.action in ("no_action", "defer"):
        mask[DELEGATION_ACTIONS.index("execute_autonomously")] = False
    if context.communication_reliability <= 0.0:
        mask[DELEGATION_ACTIONS.index("communicate")] = False
        mask[DELEGATION_ACTIONS.index("request_evidence")] = False
    if not mask.any():
        mask[DELEGATION_ACTIONS.index("abstain")] = True
    return mask


class RoleActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(values)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class DecentralizedRolePolicies(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.roles = nn.ModuleDict({
            role: RoleActorCritic(input_dim, len(DELEGATION_ACTIONS))
            for role in ROLE_NAMES
        })

    def evaluate(
        self, role: str, observation: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        logits, value = self.roles[role](observation)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        return Categorical(logits=masked_logits), value


@dataclass
class Transition:
    role: str
    agent_id: str
    step: int
    incident_id: str
    observation: np.ndarray
    mask: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float = 0.0
    advantage: float = 0.0
    return_value: float = 0.0


class TrajectoryController:
    def __init__(
        self, model: DecentralizedRolePolicies, method: str,
        device: torch.device, stochastic: bool,
    ) -> None:
        self.model = model
        self.method = method
        self.device = device
        self.stochastic = stochastic
        self.transitions: List[Transition] = []

    def __call__(
        self, contexts: Sequence[V6DecisionContext], step: int,
    ) -> Mapping[str, str]:
        decisions: Dict[str, str] = {}
        for context in contexts:
            observation = feature_vector(self.method, context)
            mask = delegation_mask(context)
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                distribution, value = self.model.evaluate(
                    context.proposal.role, tensor, mask_tensor,
                )
                action = distribution.sample() if self.stochastic else torch.argmax(distribution.logits)
                log_probability = distribution.log_prob(action)
            index = int(action.item())
            decisions[context.proposal.incident_id] = DELEGATION_ACTIONS[index]
            self.transitions.append(Transition(
                role=context.proposal.role,
                agent_id=context.proposal.agent_id,
                step=int(step),
                incident_id=context.proposal.incident_id,
                observation=observation,
                mask=mask,
                action=index,
                log_probability=float(log_probability.item()),
                value=float(value.item()),
            ))
        return decisions


def assign_trajectory_rewards(
    transitions: List[Transition], environment: V6PanelEnvironment,
    gamma: float = 0.97, gae_lambda: float = 0.92,
) -> None:
    candidate_effect = {
        (int(value["step"]), str(value["incident_id"])): float(
            value["evaluator_causal_utility_if_executed"]
        )
        for value in environment.candidate_records
    }
    operator_effects: Dict[str, List[float]] = {}
    for value in environment.action_records:
        if value["source"] == "bounded_simulated_operator":
            operator_effects.setdefault(str(value["proposal"]["incident_id"]), []).append(
                float(value["causal_effect"])
            )
    operator_positions: Dict[str, int] = {}
    per_decision_service_penalty = 0.006 * float(
        sum(incident.cumulative_loss for incident in environment.incidents.values())
    ) / max(len(transitions), 1)
    for transition in transitions:
        delegation = DELEGATION_ACTIONS[transition.action]
        reward = -per_decision_service_penalty
        if delegation == "execute_autonomously":
            reward += candidate_effect[(transition.step, transition.incident_id)]
        elif delegation == "escalate_operator":
            position = operator_positions.get(transition.incident_id, 0)
            effects = operator_effects.get(transition.incident_id, [])
            if position < len(effects):
                reward += effects[position]
                operator_positions[transition.incident_id] = position + 1
            reward -= 0.018
        elif delegation in ("communicate", "request_evidence"):
            reward -= 0.010
        elif delegation == "defer":
            reward -= 0.006
        transition.reward = float(reward)
    assign_agent_grouped_gae(transitions, gamma, gae_lambda)


def assign_agent_grouped_gae(
    transitions: Sequence[Transition], gamma: float = 0.97,
    gae_lambda: float = 0.92,
) -> None:
    """Compute GAE within each independent agent trajectory.

    Decision records are interleaved by incident at each simulator epoch. A
    value from another organization's actor/critic is therefore not a valid
    bootstrap target. Grouping by persistent agent identity preserves temporal
    credit assignment without crossing private-policy boundaries.
    """
    by_agent: Dict[str, List[Transition]] = {}
    for transition in transitions:
        by_agent.setdefault(transition.agent_id, []).append(transition)
    for agent_transitions in by_agent.values():
        next_value = 0.0
        next_advantage = 0.0
        for transition in reversed(agent_transitions):
            delta = transition.reward + gamma * next_value - transition.value
            transition.advantage = float(
                delta + gamma * gae_lambda * next_advantage
            )
            transition.return_value = float(
                transition.advantage + transition.value
            )
            next_value = transition.value
            next_advantage = transition.advantage


def _ppo_update(
    model: DecentralizedRolePolicies,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
    epochs: int = 4,
) -> Dict[str, float]:
    advantages = torch.as_tensor(
        [value.advantage for value in transitions], dtype=torch.float32, device=device,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    returns = torch.as_tensor(
        [value.return_value for value in transitions], dtype=torch.float32, device=device,
    )
    old_log = torch.as_tensor(
        [value.log_probability for value in transitions], dtype=torch.float32, device=device,
    )
    final_actor = final_critic = final_entropy = 0.0
    for _ in range(int(epochs)):
        log_values: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for transition in transitions:
            observation = torch.as_tensor(
                transition.observation, dtype=torch.float32, device=device,
            )
            mask = torch.as_tensor(transition.mask, dtype=torch.bool, device=device)
            distribution, value = model.evaluate(transition.role, observation, mask)
            action = torch.as_tensor(transition.action, dtype=torch.long, device=device)
            log_values.append(distribution.log_prob(action))
            values.append(value)
            entropies.append(distribution.entropy())
        log_probability = torch.stack(log_values)
        predicted_values = torch.stack(values)
        entropy = torch.stack(entropies).mean()
        ratio = torch.exp(log_probability - old_log)
        clipped = torch.clamp(ratio, 0.80, 1.20)
        actor_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        critic_loss = 0.5 * torch.square(predicted_values - returns).mean()
        loss = actor_loss + 0.50 * critic_loss - 0.02 * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        optimizer.step()
        final_actor = float(actor_loss.item())
        final_critic = float(critic_loss.item())
        final_entropy = float(entropy.item())
    return {
        "actor_loss": final_actor,
        "critic_loss": final_critic,
        "policy_entropy": final_entropy,
    }


def _episode_spec(index: int, seed_offset: int) -> Tuple[str, str, str, int]:
    applications = ("commercial", "humanitarian", "utility_restoration")
    application = applications[index % len(applications)]
    regime = TRAIN_REGIMES[(index // len(applications)) % len(TRAIN_REGIMES)]
    condition = "private_fragmented" if index % 4 else "public_shared"
    environment_seed = 66301 + ((index + seed_offset) % 24)
    return application, regime, condition, environment_seed


def train_seed(
    method: str,
    rl_seed: int,
    results_root: Path,
    training_episodes: int = 200,
    evaluation_episodes: int = 60,
) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError("unknown V6 training method")
    torch.manual_seed(int(rl_seed))
    np.random.seed(int(rl_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 66301,
    ).decision_context("humanitarian_incident_01", 2)
    input_dim = int(len(feature_vector(method, probe)))
    model = DecentralizedRolePolicies(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    curve_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    pending: List[Transition] = []
    decisions_seen = 0
    for episode_index in range(int(training_episodes)):
        application, regime, condition, environment_seed = _episode_spec(
            episode_index, int(rl_seed) % 24,
        )
        environment = V6PanelEnvironment(
            application, regime, condition, environment_seed, "event_triggered",
        )
        controller = TrajectoryController(model, method, device, stochastic=True)
        summary = environment.run(controller, method)
        assign_trajectory_rewards(controller.transitions, environment)
        pending.extend(controller.transitions)
        decisions_seen += len(controller.transitions)
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == training_episodes:
            diagnostics = _ppo_update(model, optimizer, pending, device)
            curve_rows.append({
                "method": method,
                "rl_seed": int(rl_seed),
                "training_episode": episode_index + 1,
                "decision_epochs": decisions_seen,
                "mean_trajectory_reward": float(np.mean([value.reward for value in pending])),
                "service_loss": float(summary["service_loss"]),
                "action_diversity": len({value.action for value in pending}),
                **diagnostics,
            })
            pending = []
    evaluation_rows: List[Dict[str, Any]] = []
    for episode_index in range(int(evaluation_episodes)):
        application, regime, condition, environment_seed = _episode_spec(
            episode_index, 100 + int(rl_seed) % 10,
        )
        environment_seed = 66401 + (environment_seed - 66301) % 10
        environment = V6PanelEnvironment(
            application, regime, condition, environment_seed, "event_triggered",
        )
        controller = TrajectoryController(model, method, device, stochastic=False)
        summary = environment.run(controller, method)
        assign_trajectory_rewards(controller.transitions, environment)
        counts = {
            action: sum(
                DELEGATION_ACTIONS[value.action] == action
                for value in controller.transitions
            )
            for action in DELEGATION_ACTIONS
        }
        evaluation_rows.append({
            "method": method,
            "rl_seed": int(rl_seed),
            "episode_index": episode_index,
            "application": application,
            "regime": regime,
            "information_condition": condition,
            "environment_seed": environment_seed,
            "service_loss": summary["service_loss"],
            "net_causal_utility": summary["net_causal_utility"],
            "harmful_actions": summary["harmful_actions"],
            "beneficial_actions": summary["beneficial_actions"],
            "autonomous_completed_actions": summary["autonomous_completed_actions"],
            "autonomous_harmful_actions": summary["autonomous_harmful_actions"],
            "autonomous_harm_rate": (
                summary["autonomous_harmful_actions"]
                / max(summary["autonomous_completed_actions"], 1)
            ),
            "operator_minutes": summary["operator_minutes"],
            "total_messages": summary["total_messages"],
            "total_bytes": summary["total_bytes"],
            "trajectory_reward": float(sum(value.reward for value in controller.transitions)),
            **{"delegation_%s" % key: value for key, value in counts.items()},
        })
    actions_used = {
        action for row in evaluation_rows for action in DELEGATION_ACTIONS
        if row["delegation_%s" % action] > 0
    }
    checkpoint = results_root / "training" / "checkpoints" / (
        "%s-seed-%d.pt" % (method, rl_seed)
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "method": method,
        "rl_seed": int(rl_seed),
        "input_dim": input_dim,
        "delegation_actions": list(DELEGATION_ACTIONS),
        "role_names": list(ROLE_NAMES),
    }, checkpoint)
    curve_path = results_root / "training" / "curves" / (
        "%s-seed-%d.csv" % (method, rl_seed)
    )
    evaluation_path = results_root / "training" / "evaluation" / (
        "%s-seed-%d.csv" % (method, rl_seed)
    )
    write_csv(curve_path, curve_rows)
    write_csv(evaluation_path, evaluation_rows)
    result = {
        "method": method,
        "rl_seed": int(rl_seed),
        "status": "complete",
        "algorithm": "sequential decentralized PPO with GAE",
        "centralized_training": False,
        "decentralized_execution": True,
        "training_episodes": int(training_episodes),
        "training_decision_epochs": int(training_episodes * 24),
        "evaluation_episodes": int(evaluation_episodes),
        "evaluation_decision_epochs": int(evaluation_episodes * 24),
        "evaluation_mean_reward": float(np.mean([value["trajectory_reward"] for value in evaluation_rows])),
        "evaluation_mean_service_loss": float(np.mean([value["service_loss"] for value in evaluation_rows])),
        "evaluation_mean_causal_utility": float(np.mean([value["net_causal_utility"] for value in evaluation_rows])),
        "evaluation_harmful_actions_per_episode": float(np.mean([value["harmful_actions"] for value in evaluation_rows])),
        "evaluation_autonomous_harm_rate": float(np.mean([
            value["autonomous_harm_rate"] for value in evaluation_rows
        ])),
        "evaluation_action_diversity": len(actions_used),
        "evaluation_actions_used": sorted(actions_used),
        "collapsed_to_no_action": bool(actions_used.issubset({"defer", "abstain"})),
        "wall_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "checkpoint": str(checkpoint.relative_to(results_root)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "curve": str(curve_path.relative_to(results_root)),
        "evaluation": str(evaluation_path.relative_to(results_root)),
    }
    atomic_json(
        results_root / "training" / "manifests" / (
            "%s-seed-%d.json" % (method, rl_seed)
        ), result,
    )
    return result


def train_multiseed(
    repository: Path,
    results_root: Path,
    seeds: Sequence[int] = (66201, 66202, 66203, 66204, 66205),
    training_episodes: int = 200,
    evaluation_episodes: int = 60,
) -> Dict[str, Any]:
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for method in METHODS:
        for seed in seeds:
            try:
                completed.append(train_seed(
                    method, int(seed), results_root,
                    int(training_episodes), int(evaluation_episodes),
                ))
            except Exception as error:
                failures.append({
                    "method": method,
                    "rl_seed": int(seed),
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                })
    write_csv(results_root / "training" / "seed_manifest.csv", completed + failures)
    if failures:
        write_csv(results_root / "negative_results" / "rl_failed_seeds.csv", failures)
    by_method: Dict[str, Any] = {}
    for method in METHODS:
        rows = [value for value in completed if value["method"] == method]
        rewards = np.asarray([value["evaluation_mean_reward"] for value in rows], dtype=float)
        harms = np.asarray([
            value["evaluation_autonomous_harm_rate"] for value in rows
        ], dtype=float)
        by_method[method] = {
            "completed_seeds": len(rows),
            "failed_seeds": sum(value["method"] == method for value in failures),
            "mean_reward": float(rewards.mean()) if len(rewards) else None,
            "between_seed_reward_sd": float(rewards.std(ddof=1)) if len(rewards) > 1 else None,
            "mean_autonomous_harm_rate": float(harms.mean()) if len(harms) else None,
            "between_seed_harm_sd": float(harms.std(ddof=1)) if len(harms) > 1 else None,
            "minimum_action_diversity": min(
                [value["evaluation_action_diversity"] for value in rows] or [0]
            ),
            "collapsed_seeds": sum(value["collapsed_to_no_action"] for value in rows),
        }
    report = {
        "study": "Generalized Entropic Consensus V6",
        "algorithm": "sequential decentralized PPO with clipped objective, GAE, and role actors",
        "training_seeds": list(map(int, seeds)),
        "methods": by_method,
        "completed_runs": len(completed),
        "failed_runs": len(failures),
        "source_checksum": source_checksum(repository),
        "generated_at": utc_now(),
    }
    atomic_json(results_root / "training" / "training_summary.json", report)
    return report
