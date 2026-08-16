"""Sequential decentralized PPO for V7 Level-2 delegation.

Training may use evaluator rewards, but execution features contain only the
selected agent's authorized context and explicitly delivered distributed
summaries. Role-conditioned actors never receive another agent's private vault.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .events import sha256_file
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v7_experiments import run_episode
from .v7_policies import decision_key
from .v7_types import (
    DELEGATION_ACTIONS, HUMANITARIAN_ACTIONS, UTILITY_ACTIONS, V7RiskContext,
)


METHODS = (
    "ppo_kpi_only",
    "ppo_predictive_uncertainty",
    "ppo_shannon_js",
    "ppo_generalized_tsallis_gini",
    "ppo_combined_generalized_entropic",
)
ROLE_NAMES = (
    "assessment", "clinic", "critical_load", "cyber_defense", "depot",
    "hub", "local_authority", "ngo", "resource_allocation", "shelter",
    "transport", "zone_operator", "crew_dispatch", "communications",
)


def _one_hot_action(context: V7RiskContext) -> List[float]:
    actions = HUMANITARIAN_ACTIONS + UTILITY_ACTIONS
    return [
        float(context.proposal.proposed_operational_action == action)
        for action in actions
    ]


def feature_vector(method: str, context: V7RiskContext) -> np.ndarray:
    if method not in METHODS:
        raise ValueError("unknown V7 PPO method")
    values = [
        float(context.local_kpis.get("severity", 0.0)),
        float(context.local_kpis.get("safety_risk", 0.0)),
        float(context.local_kpis.get("resource_scarcity", 0.0)),
        min(float(context.local_kpis.get("delay", 0.0)) / 5.0, 1.0),
        context.proposal.action_probability,
        np.tanh(context.proposal.action_value),
        context.proposal.value_margin,
        context.communication_reliability,
        context.coupling_strength,
        context.fragmentation,
        context.size_normalized,
        *_one_hot_action(context),
    ]
    if method != "ppo_kpi_only":
        values.append(context.predictive_uncertainty)
    if method in ("ppo_shannon_js", "ppo_combined_generalized_entropic"):
        values.extend([
            context.shannon_local, context.pooled_uncertainty,
            context.js_disagreement, context.graph_disagreement,
            context.consensus_residual, context.disagreement_slope,
        ])
    if method in (
        "ppo_generalized_tsallis_gini", "ppo_combined_generalized_entropic",
    ):
        values.extend([
            context.tsallis_0_5_local, context.tsallis_2_local,
            context.tsallis_3_local, context.gini_simpson_local,
            context.jt_disagreement_0_5, context.jt_disagreement_2,
        ])
    return np.asarray(values, dtype=np.float32)


def delegation_mask(context: V7RiskContext, operator_slot_available: bool = True) -> np.ndarray:
    mask = np.ones(len(DELEGATION_ACTIONS), dtype=bool)
    if not context.proposal.is_physical:
        mask[:] = False
        mask[DELEGATION_ACTIONS.index("defer")] = True
    if not operator_slot_available:
        mask[DELEGATION_ACTIONS.index("escalate_operator")] = False
    if not mask.any():
        mask[DELEGATION_ACTIONS.index("abstain")] = True
    return mask


class RoleActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(),
            nn.Linear(96, 64), nn.Tanh(),
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
        return Categorical(logits=logits.masked_fill(~mask, -1.0e9)), value


@dataclass
class Transition:
    role: str
    agent_id: str
    target: str
    step: int
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
        self,
        model: DecentralizedRolePolicies,
        method: str,
        device: torch.device,
        stochastic: bool,
        operator_slots_per_epoch: int = 1,
    ) -> None:
        self.model = model
        self.method = method
        self.device = device
        self.stochastic = bool(stochastic)
        self.operator_slots_per_epoch = int(operator_slots_per_epoch)
        self.transitions: List[Transition] = []

    def __call__(
        self, contexts: Sequence[V7RiskContext], step: int,
    ) -> Mapping[str, str]:
        decisions: Dict[str, str] = {}
        escalations = 0
        for context in contexts:
            observation = feature_vector(self.method, context)
            mask = delegation_mask(
                context, escalations < self.operator_slots_per_epoch,
            )
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                distribution, value = self.model.evaluate(
                    context.proposal.role, tensor, mask_tensor,
                )
                action = (
                    distribution.sample()
                    if self.stochastic else torch.argmax(distribution.logits)
                )
                log_probability = distribution.log_prob(action)
            index = int(action.item())
            delegation = DELEGATION_ACTIONS[index]
            escalations += int(delegation == "escalate_operator")
            decisions[decision_key(context)] = delegation
            self.transitions.append(Transition(
                role=context.proposal.role,
                agent_id=context.proposal.agent_id,
                target=str(context.proposal.target_asset_or_location),
                step=int(step), observation=observation, mask=mask,
                action=index, log_probability=float(log_probability.item()),
                value=float(value.item()),
            ))
        return decisions


def assign_agent_grouped_gae(
    transitions: Sequence[Transition], gamma: float = 0.985,
    gae_lambda: float = 0.94,
) -> None:
    per_agent: Dict[str, List[Transition]] = {}
    for transition in transitions:
        per_agent.setdefault(transition.agent_id, []).append(transition)
    for trajectory in per_agent.values():
        next_value = 0.0
        next_advantage = 0.0
        for transition in reversed(trajectory):
            delta = transition.reward + gamma * next_value - transition.value
            transition.advantage = float(
                delta + gamma * gae_lambda * next_advantage
            )
            transition.return_value = float(transition.advantage + transition.value)
            next_value = transition.value
            next_advantage = transition.advantage


def assign_rewards(
    transitions: Sequence[Transition],
    candidates: Sequence[Mapping[str, Any]],
    service_loss: float,
) -> None:
    values = {
        (int(row["step"]), str(row["agent_id"]), str(row["target"])): row
        for row in candidates
    }
    service_penalty = 0.002 * float(service_loss) / max(len(transitions), 1)
    for transition in transitions:
        row = values[(transition.step, transition.agent_id, transition.target)]
        utility = float(row["counterfactual_causal_utility"])
        evaluated = bool(row["counterfactual_evaluated"])
        action = DELEGATION_ACTIONS[transition.action]
        reward = -service_penalty
        if action == "execute_autonomously":
            reward += utility if evaluated else 0.0
        elif action == "escalate_operator":
            reward += (max(utility, 0.0) if evaluated else 0.0) - 0.06
        elif action == "defer":
            reward += (-0.30 * max(utility, 0.0) if evaluated else 0.0) - 0.008
        else:
            reward += -0.45 * max(utility, 0.0) if evaluated else 0.0
        transition.reward = float(np.clip(reward, -4.0, 4.0))
    assign_agent_grouped_gae(transitions)


def _ppo_update(
    model: DecentralizedRolePolicies,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
    epochs: int = 4,
) -> Dict[str, float]:
    if not transitions:
        raise ValueError("PPO update requires sequential transitions")
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
    diagnostics = {"actor_loss": 0.0, "critic_loss": 0.0, "policy_entropy": 0.0}
    for _ in range(int(epochs)):
        logs: List[torch.Tensor] = []
        predictions: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for transition in transitions:
            observation = torch.as_tensor(
                transition.observation, dtype=torch.float32, device=device,
            )
            mask = torch.as_tensor(transition.mask, dtype=torch.bool, device=device)
            distribution, value = model.evaluate(transition.role, observation, mask)
            action = torch.as_tensor(transition.action, dtype=torch.long, device=device)
            logs.append(distribution.log_prob(action))
            predictions.append(value)
            entropies.append(distribution.entropy())
        log_probability = torch.stack(logs)
        predicted = torch.stack(predictions)
        entropy = torch.stack(entropies).mean()
        ratio = torch.exp(log_probability - old_log)
        clipped = torch.clamp(ratio, 0.82, 1.18)
        actor_loss = -torch.minimum(
            ratio * advantages, clipped * advantages,
        ).mean()
        critic_loss = 0.5 * torch.square(predicted - returns).mean()
        loss = actor_loss + 0.45 * critic_loss - 0.018 * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.75)
        optimizer.step()
        diagnostics = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "policy_entropy": float(entropy.item()),
        }
    return diagnostics


def _training_spec(index: int, seed: int) -> Tuple[str, str, str, str, str, str, int]:
    application = "humanitarian" if index % 2 == 0 else "utility_restoration"
    complexity = "small" if index < 12 else "medium"
    coupling = ("low", "medium", "high")[(index // 2) % 3]
    fragmentation = ("high", "low", "medium")[(index // 3) % 3]
    disruption = ("medium", "high", "low")[(index // 4) % 3]
    if application == "humanitarian":
        topology = ("random_geometric", "small_world", "modular")[index % 3]
    else:
        topology = ("grid", "scale_free", "modular")[index % 3]
    environment_seed = 775000 + (int(seed) % 100) * 100 + index
    return application, complexity, coupling, fragmentation, disruption, topology, environment_seed


def train_seed(
    method: str,
    rl_seed: int,
    results_root: Path,
    training_episodes: int = 60,
    evaluation_episodes: int = 18,
) -> Dict[str, Any]:
    torch.manual_seed(int(rl_seed))
    np.random.seed(int(rl_seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Construct one deployable context to determine the frozen input width.
    from .v7_experiments import make_environment
    probe_environment = make_environment(
        "humanitarian", "small", "medium", "medium", "medium",
        "small_world", 774999,
    )
    probe_environment.advance_domain(0)
    probe_environment.deliver_private_observations(0)
    probe_agent = probe_environment.agents[sorted(probe_environment.agents)[0]]
    probe_asset = probe_agent.identity.asset_scope[0]
    probe_decision = probe_agent.propose(probe_asset)
    probe = probe_environment.risk_context(probe_decision, 0)
    input_dim = len(feature_vector(method, probe))
    model = DecentralizedRolePolicies(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)
    pending: List[Transition] = []
    curve_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for episode_index in range(int(training_episodes)):
        spec = _training_spec(episode_index, rl_seed)
        controller = TrajectoryController(model, method, device, True)
        output = run_episode(
            *spec[:-1], spec[-1], controller,
            counterfactual_limit_per_epoch=4,
        )
        assign_rewards(
            controller.transitions, output["candidates"],
            output["summary"]["service_loss"],
        )
        pending.extend(controller.transitions)
        if (episode_index + 1) % 5 == 0 or episode_index + 1 == training_episodes:
            diagnostics = _ppo_update(model, optimizer, pending, device)
            curve_rows.append({
                "method": method, "rl_seed": int(rl_seed),
                "training_episode": episode_index + 1,
                "transitions": len(pending),
                "mean_reward": float(np.mean([value.reward for value in pending])),
                "action_diversity": len({value.action for value in pending}),
                **diagnostics,
            })
            pending = []
    evaluation_rows: List[Dict[str, Any]] = []
    for episode_index in range(int(evaluation_episodes)):
        spec = _training_spec(100 + episode_index, rl_seed + 1000)
        controller = TrajectoryController(model, method, device, False)
        output = run_episode(
            *spec[:-1], spec[-1], controller,
            counterfactual_limit_per_epoch=4,
        )
        assign_rewards(controller.transitions, output["candidates"], output["summary"]["service_loss"])
        actions = [DELEGATION_ACTIONS[value.action] for value in controller.transitions]
        evaluation_rows.append({
            "method": method, "rl_seed": int(rl_seed),
            "episode_index": episode_index,
            "application": spec[0], "complexity": spec[1],
            "environment_seed": spec[-1],
            "service_loss": output["summary"]["service_loss"],
            "net_causal_utility": output["summary"]["net_causal_utility"],
            "harmful_actions": output["summary"]["harmful_actions"],
            "physical_actions": output["summary"]["physical_actions"],
            "trajectory_reward": float(sum(value.reward for value in controller.transitions)),
            "delegation_diversity": len(set(actions)),
            **{
                "delegation_%s" % action: actions.count(action)
                for action in DELEGATION_ACTIONS
            },
        })
    checkpoint = results_root / "training" / "checkpoints" / (
        "%s-seed-%d.pt" % (method, rl_seed)
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "method": method,
        "rl_seed": int(rl_seed), "input_dim": input_dim,
        "delegation_actions": list(DELEGATION_ACTIONS),
    }, checkpoint)
    curve_path = results_root / "training" / "curves" / (
        "%s-seed-%d.csv" % (method, rl_seed)
    )
    evaluation_path = results_root / "training" / "evaluation" / (
        "%s-seed-%d.csv" % (method, rl_seed)
    )
    write_csv(curve_path, curve_rows)
    write_csv(evaluation_path, evaluation_rows)
    actions_used = {
        action for row in evaluation_rows for action in DELEGATION_ACTIONS
        if row["delegation_%s" % action] > 0
    }
    report = {
        "method": method, "rl_seed": int(rl_seed), "status": "complete",
        "algorithm": "sequential decentralized PPO with per-agent GAE",
        "training_episodes": int(training_episodes),
        "evaluation_episodes": int(evaluation_episodes),
        "evaluation_mean_reward": float(np.mean([value["trajectory_reward"] for value in evaluation_rows])),
        "evaluation_mean_service_loss": float(np.mean([value["service_loss"] for value in evaluation_rows])),
        "evaluation_mean_harmful_actions": float(np.mean([value["harmful_actions"] for value in evaluation_rows])),
        "evaluation_action_diversity": len(actions_used),
        "collapsed": bool(actions_used.issubset({"defer", "abstain"})),
        "device": str(device), "wall_seconds": float(time.perf_counter() - started),
        "checkpoint": str(checkpoint.relative_to(results_root)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "curve": str(curve_path.relative_to(results_root)),
        "evaluation": str(evaluation_path.relative_to(results_root)),
    }
    atomic_json(
        results_root / "training" / "manifests" / (
            "%s-seed-%d.json" % (method, rl_seed)
        ), report,
    )
    return report


def train_multiseed(
    repository: Path,
    results_root: Path,
    seeds: Sequence[int] = (77301, 77302, 77303, 77304, 77305),
    training_episodes: int = 60,
    evaluation_episodes: int = 18,
) -> Dict[str, Any]:
    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for method in METHODS:
        for seed in seeds:
            try:
                completed.append(train_seed(
                    method, int(seed), results_root,
                    training_episodes, evaluation_episodes,
                ))
            except Exception as error:
                failures.append({
                    "method": method, "rl_seed": int(seed), "status": "failed",
                    "failure_type": type(error).__name__, "failure_reason": str(error),
                })
    write_csv(results_root / "training" / "seed_manifest.csv", completed + failures)
    if failures:
        write_csv(results_root / "negative_results" / "rl_failed_seeds.csv", failures)
    report = {
        "study": "Complexity-dependent entropic coordination V7",
        "methods": list(METHODS), "training_seeds": list(map(int, seeds)),
        "completed_runs": len(completed), "failed_runs": len(failures),
        "source_checksum": source_checksum(repository), "generated_at": utc_now(),
    }
    atomic_json(results_root / "training" / "training_summary.json", report)
    return report
