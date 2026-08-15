"""Five-seed decentralized PPO-style V5 agent training.

Each actor receives one agent's local state. Distributed features are present
only in the entropy/disagreement method. Shared parameters do not merge agent
memory, observations, inboxes, utilities, or action authority.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from .events import sha256_file
from .v5_environment import APP_ROLES, V5PanelEnvironment, stable_seed
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v5_types import OPERATOR_ACTIONS


ROLE_NAMES = tuple(sorted({role for roles in APP_ROLES.values() for role in roles}))
REGIME_NAMES = (
    "isolated_physical", "telemetry_integrity", "partition",
    "correlated", "compound", "ood",
)


class DecentralizedActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48), nn.Tanh(),
            nn.Linear(48, 48), nn.Tanh(),
        )
        self.actor = nn.Linear(48, action_dim)
        self.critic = nn.Linear(48, 1)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(values)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def _agent_vector(
    environment: V5PanelEnvironment,
    agent_id: str,
    include_thermodynamics: bool,
) -> np.ndarray:
    agent = environment.agents[agent_id]
    observation = environment.observations[agent_id]
    belief = np.asarray(agent.private_beliefs[observation.incident_id], dtype=float)
    role = np.asarray([float(agent.identity.role == name) for name in ROLE_NAMES], dtype=float)
    local = np.asarray([
        observation.visible_severity,
        observation.visible_backlog,
        observation.visible_delay,
        observation.resource_scarcity,
        observation.safety_risk,
        observation.commitment_strain,
        observation.telemetry_confidence,
        observation.communication_reliability,
        observation.private_inventory / 2.5,
        observation.private_cost,
        observation.private_priority,
        *belief.tolist(),
        *role.tolist(),
    ], dtype=np.float32)
    thermo = environment.thermodynamics[observation.incident_id]
    distributed = np.asarray([
        thermo.mean_belief_entropy,
        thermo.entropy_dispersion,
        thermo.js_disagreement,
        thermo.entropy_slope,
        thermo.consensus_residual,
        thermo.consensus_confidence,
    ], dtype=np.float32)
    if not include_thermodynamics:
        distributed[:] = 0.0
    return np.concatenate([local, distributed])


def build_agent_dataset(
    seeds: Sequence[int],
    include_thermodynamics: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    features: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    for seed in seeds:
        for application in ("commercial", "humanitarian", "utility_restoration"):
            for regime in REGIME_NAMES:
                environment = V5PanelEnvironment(
                    application, regime, "private_fragmented", int(seed),
                    sketch_policy="event_triggered",
                )
                for agent_id, agent in environment.agents.items():
                    incident_id = agent.identity.incident_scope[0]
                    vector = _agent_vector(environment, agent_id, include_thermodynamics)
                    outcome = np.asarray([
                        environment.action_effect(incident_id, action).causal_effect
                        for action in OPERATOR_ACTIONS
                    ], dtype=np.float32)
                    # A role cannot execute disallowed actions; retain them as
                    # bounded negative validation results rather than silently
                    # replacing the decision centrally.
                    from .v5_tools import V5ToolRegistry
                    allowed = set(V5ToolRegistry().allowed_actions(agent.identity.role))
                    outcome = np.asarray([
                        value if action in allowed else -0.30
                        for action, value in zip(OPERATOR_ACTIONS, outcome)
                    ], dtype=np.float32)
                    features.append(vector)
                    rewards.append(outcome)
                    metadata.append({
                        "application": application,
                        "regime": regime,
                        "environment_seed": int(seed),
                        "agent_id": agent_id,
                        "role": agent.identity.role,
                        "incident_id": incident_id,
                    })
    return np.vstack(features), np.vstack(rewards), metadata


def train_seed(
    method: str,
    seed: int,
    training_steps: int,
    results_root: Path,
    prepared_training: Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]] = None,
    prepared_evaluation: Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if method not in ("ippo_kpi_only", "ippo_entropy_disagreement"):
        raise ValueError("unknown V5 decentralized training method")
    include_thermo = method == "ippo_entropy_disagreement"
    torch.manual_seed(int(seed))
    np_rng = np.random.RandomState(int(seed))
    train_seeds = tuple(range(51301, 51313))
    eval_seeds = tuple(range(52101, 52111))
    training_data = prepared_training or build_agent_dataset(train_seeds, include_thermo)
    evaluation_data = prepared_evaluation or build_agent_dataset(eval_seeds, include_thermo)
    x_train, reward_train, _ = training_data
    x_eval, reward_eval, eval_meta = evaluation_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecentralizedActorCritic(x_train.shape[1], len(OPERATOR_ACTIONS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    feature_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    reward_tensor = torch.as_tensor(reward_train, dtype=torch.float32, device=device)
    batch_size = 256
    updates = int(np.ceil(int(training_steps) / batch_size))
    curves: List[Dict[str, Any]] = []
    started = time.perf_counter()
    decisions = 0
    for update in range(updates):
        positions = np_rng.randint(0, len(x_train), size=batch_size)
        batch = feature_tensor[positions]
        reward_matrix = reward_tensor[positions]
        with torch.no_grad():
            old_logits, old_values = model(batch)
            old_distribution = Categorical(logits=old_logits)
            actions = old_distribution.sample()
            old_log_prob = old_distribution.log_prob(actions)
            observed_reward = reward_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)
            advantages = observed_reward - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        for _ in range(4):
            logits, values = model(batch)
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(actions)
            ratio = torch.exp(log_prob - old_log_prob)
            clipped = torch.clamp(ratio, 0.80, 1.20)
            actor_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            critic_loss = 0.5 * torch.square(values - observed_reward).mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + critic_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
        decisions += batch_size
        if update % 10 == 0 or update == updates - 1:
            curves.append({
                "method": method,
                "rl_seed": int(seed),
                "environment_steps": min(decisions, int(training_steps)),
                "sample_reward": float(observed_reward.mean().item()),
                "policy_entropy": float(entropy.item()),
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "action_diversity": int(actions.unique().numel()),
            })
    with torch.no_grad():
        eval_features = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
        logits, _ = model(eval_features)
        actions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = torch.softmax(logits, dim=1)
        entropy = Categorical(probs=probabilities).entropy().mean().item()
    selected_reward = reward_eval[np.arange(len(actions)), actions]
    evaluation_rows: List[Dict[str, Any]] = []
    for index, meta in enumerate(eval_meta):
        evaluation_rows.append({
            **meta,
            "method": method,
            "rl_seed": int(seed),
            "selected_action": OPERATOR_ACTIONS[int(actions[index])],
            "reward": float(selected_reward[index]),
            "beneficial": bool(selected_reward[index] > 0),
            "harmful": bool(selected_reward[index] < 0),
        })
    checkpoint_path = results_root / "training" / "checkpoints" / ("%s-seed-%d.pt" % (method, seed))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "actions": list(OPERATOR_ACTIONS),
        "method": method,
        "rl_seed": int(seed),
    }, checkpoint_path)
    curve_path = results_root / "training" / "curves" / ("%s-seed-%d.csv" % (method, seed))
    evaluation_path = results_root / "training" / "evaluation" / ("%s-seed-%d.csv" % (method, seed))
    write_csv(curve_path, curves)
    write_csv(evaluation_path, evaluation_rows)
    result = {
        "method": method,
        "rl_seed": int(seed),
        "status": "complete",
        "training_steps": int(training_steps),
        "training_decision_epochs": int(decisions),
        "evaluation_decision_epochs": int(len(evaluation_rows)),
        "evaluation_mean_reward": float(selected_reward.mean()),
        "evaluation_reward_std": float(selected_reward.std()),
        "evaluation_beneficial_fraction": float((selected_reward > 0).mean()),
        "evaluation_harmful_fraction": float((selected_reward < 0).mean()),
        "evaluation_action_diversity": int(len(set(actions.tolist()))),
        "evaluation_policy_entropy": float(entropy),
        "wall_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "checkpoint": str(checkpoint_path.relative_to(results_root)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "curve_path": str(curve_path.relative_to(results_root)),
        "evaluation_path": str(evaluation_path.relative_to(results_root)),
    }
    atomic_json(results_root / "training" / "manifests" / ("%s-seed-%d.json" % (method, seed)), result)
    return result


def train_multiseed(
    repository: Path,
    results_root: Path,
    seeds: Sequence[int] = (52001, 52002, 52003, 52004, 52005),
    training_steps: int = 30000,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for method in ("ippo_kpi_only", "ippo_entropy_disagreement"):
        include_thermo = method == "ippo_entropy_disagreement"
        prepared_training = build_agent_dataset(tuple(range(51301, 51313)), include_thermo)
        prepared_evaluation = build_agent_dataset(tuple(range(52101, 52111)), include_thermo)
        for seed in seeds:
            try:
                results.append(train_seed(
                    method, int(seed), int(training_steps), results_root,
                    prepared_training=prepared_training,
                    prepared_evaluation=prepared_evaluation,
                ))
            except Exception as error:
                failures.append({
                    "method": method, "rl_seed": int(seed), "status": "failed",
                    "failure_type": type(error).__name__, "failure_reason": str(error),
                })
    write_csv(results_root / "training" / "seed_manifest.csv", results + failures)
    if failures:
        write_csv(results_root / "negative_results" / "rl_failed_seeds.csv", failures)
    by_method: Dict[str, Any] = {}
    for method in ("ippo_kpi_only", "ippo_entropy_disagreement"):
        rows = [value for value in results if value["method"] == method]
        rewards = np.asarray([value["evaluation_mean_reward"] for value in rows], dtype=float)
        by_method[method] = {
            "seeds": len(rows),
            "failed_seeds": sum(value["method"] == method for value in failures),
            "mean_reward": float(rewards.mean()) if len(rewards) else None,
            "reward_std_between_seeds": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
            "coefficient_of_variation": float(rewards.std(ddof=1) / max(abs(rewards.mean()), 1e-6)) if len(rewards) > 1 else 0.0,
            "action_diversity_minimum": min([value["evaluation_action_diversity"] for value in rows] or [0]),
        }
    gain = None
    if all(by_method[value]["mean_reward"] is not None for value in by_method):
        gain = float(by_method["ippo_entropy_disagreement"]["mean_reward"] - by_method["ippo_kpi_only"]["mean_reward"])
    report = {
        "study": "ThermoHITL v5",
        "algorithm": "independent PPO-style contextual actor-critic",
        "centralized_training": False,
        "decentralized_execution": True,
        "shared_parameters_separate_private_contexts": True,
        "training_seeds": list(map(int, seeds)),
        "training_steps_per_seed": int(training_steps),
        "methods": by_method,
        "entropy_policy_mean_gain": gain,
        "failed_seeds": len(failures),
        "source_checksum": source_checksum(repository),
        "generated_at": utc_now(),
    }
    atomic_json(results_root / "training" / "training_summary.json", report)
    return report
