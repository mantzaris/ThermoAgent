"""Transparent PPO coordination metapolicy with decentralized execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


OBSERVATION_DIM = 24
N_OPTIONS = 9


@dataclass
class PPOConfig:
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 5
    minibatch_size: int = 128
    max_grad_norm: float = 0.5


class CoordinationPolicy:
    """A shared actor/value network; only local vectors reach ``act``."""

    def __init__(self, config: Optional[PPOConfig] = None, seed: int = 0, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.config = config or PPOConfig()
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(seed)
        self.device = torch.device(device)

        class ActorCritic(nn.Module):
            def __init__(self, hidden: int) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(OBSERVATION_DIM, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                )
                self.actor = nn.Linear(hidden, N_OPTIONS)
                self.value = nn.Linear(hidden, 1)

            def forward(self, observation: Any) -> Tuple[Any, Any]:
                encoded = self.encoder(observation)
                return self.actor(encoded), self.value(encoded).squeeze(-1)

        self.model = ActorCritic(self.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        torch = self.torch
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (OBSERVATION_DIM,):
            raise ValueError("execution actor accepts exactly the 24 local features")
        mask = np.ones(N_OPTIONS, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        if mask.shape != (N_OPTIONS,) or not mask.any():
            raise ValueError("action mask must enable at least one of exactly nine options")
        with torch.no_grad():
            tensor = torch.as_tensor(vector, device=self.device).unsqueeze(0)
            logits, value = self.model(tensor)
            mask_tensor = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        return int(action.item()), float(log_probability.item()), float(value.item())

    def update(self, trajectories: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        torch = self.torch
        if not trajectories:
            raise ValueError("PPO update requires trajectories")
        observations = np.asarray([row["observation"] for row in trajectories], dtype=np.float32)
        actions = np.asarray([row["action"] for row in trajectories], dtype=np.int64)
        old_logp = np.asarray([row["log_probability"] for row in trajectories], dtype=np.float32)
        old_values = np.asarray([row["value"] for row in trajectories], dtype=np.float32)
        rewards = np.asarray([row["reward"] for row in trajectories], dtype=np.float32)
        dones = np.asarray([row.get("done", False) for row in trajectories], dtype=np.float32)
        masks = np.asarray([
            row.get("action_mask", [True] * N_OPTIONS) for row in trajectories
        ], dtype=bool)
        if masks.shape != (len(trajectories), N_OPTIONS):
            raise ValueError("trajectory action masks have invalid shape")
        if np.any(~masks[np.arange(len(actions)), actions]):
            raise ValueError("trajectory contains an action disabled by its local mask")
        advantages = np.zeros_like(rewards)
        groups: Dict[str, List[int]] = {}
        for index, row in enumerate(trajectories):
            key = str(row.get("trajectory_id", row.get("agent_id", "trajectory")))
            groups.setdefault(key, []).append(index)
        for group_indices in groups.values():
            last_advantage = 0.0
            next_value = 0.0
            for index in reversed(group_indices):
                nonterminal = 1.0 - dones[index]
                delta = rewards[index] + self.config.gamma * next_value * nonterminal - old_values[index]
                last_advantage = delta + self.config.gamma * self.config.gae_lambda * nonterminal * last_advantage
                advantages[index] = last_advantage
                next_value = old_values[index]
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(observations, device=self.device)
        action_t = torch.as_tensor(actions, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, device=self.device)
        advantage_t = torch.as_tensor(advantages, device=self.device)
        return_t = torch.as_tensor(returns, device=self.device)
        mask_t = torch.as_tensor(masks, device=self.device)
        losses: Dict[str, List[float]] = {"actor": [], "value": [], "entropy": [], "total": []}
        indices = np.arange(len(trajectories))
        for _ in range(self.config.epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                batch = indices[start : start + self.config.minibatch_size]
                logits, values = self.model(obs_t[batch])
                logits = logits.masked_fill(~mask_t[batch], -1e9)
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(action_t[batch])
                ratio = torch.exp(logp - old_logp_t[batch])
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                actor_loss = -torch.min(ratio * advantage_t[batch], clipped * advantage_t[batch]).mean()
                value_loss = 0.5 * ((values - return_t[batch]) ** 2).mean()
                entropy = distribution.entropy().mean()
                loss = actor_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses["actor"].append(float(actor_loss.item()))
                losses["value"].append(float(value_loss.item()))
                losses["entropy"].append(float(entropy.item()))
                losses["total"].append(float(loss.item()))
        return {name: float(np.mean(values)) for name, values in losses.items()}

    def behavior_clone(
        self,
        demonstrations: Sequence[Dict[str, Any]],
        epochs: int = 8,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """Initialize from local-observation scripted coordination traces."""

        if not demonstrations:
            raise ValueError("behavior cloning requires demonstrations")
        torch = self.torch
        observations = np.asarray([row["observation"] for row in demonstrations], dtype=np.float32)
        actions = np.asarray([row["action"] for row in demonstrations], dtype=np.int64)
        masks = np.asarray([
            row.get("action_mask", [True] * N_OPTIONS) for row in demonstrations
        ], dtype=bool)
        indices = np.arange(len(demonstrations))
        losses: List[float] = []
        accuracies: List[float] = []
        self.model.train()
        for _ in range(int(epochs)):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), int(batch_size)):
                batch = indices[start : start + int(batch_size)]
                obs_t = torch.as_tensor(observations[batch], device=self.device)
                action_t = torch.as_tensor(actions[batch], device=self.device)
                mask_t = torch.as_tensor(masks[batch], device=self.device)
                logits, _ = self.model(obs_t)
                logits = logits.masked_fill(~mask_t, -1e9)
                loss = torch.nn.functional.cross_entropy(logits, action_t)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))
                accuracies.append(float((logits.argmax(dim=-1) == action_t).float().mean().item()))
        self.model.eval()
        return {
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accuracies)),
            "rows": float(len(demonstrations)),
            "epochs": float(epochs),
        }

    def save(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "observation_dim": OBSERVATION_DIM,
            "n_options": N_OPTIONS,
            "metadata": metadata or {},
            "initialization_seed": self.seed,
        }
        temporary = path.with_name(path.name + ".tmp")
        self.torch.save(payload, str(temporary))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "CoordinationPolicy":
        import torch

        payload = torch.load(str(path), map_location=device)
        if payload["observation_dim"] != OBSERVATION_DIM or payload["n_options"] != N_OPTIONS:
            raise ValueError("coordination checkpoint schema mismatch")
        policy = cls(PPOConfig(**payload["config"]), seed=int(payload.get("initialization_seed", 0)), device=device)
        policy.model.load_state_dict(payload["state_dict"])
        policy.model.eval()
        return policy


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(str(path), map_location="cpu")
    return dict(payload.get("metadata", {}))
