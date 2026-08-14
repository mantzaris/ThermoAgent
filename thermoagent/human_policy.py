"""Transparent contextual-bandit policy for ThermoHITL attention decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HUMAN_ACTION_NAMES = (
    "continue_autonomy",
    "targeted_agent_coordination",
    "request_human_information",
    "request_human_recommendation",
    "request_human_approval",
    "request_conflict_resolution",
    "request_emergency_override",
)

HUMAN_FEATURE_NAMES = (
    "local_kpi_risk",
    "local_disruption_risk",
    "actionability_evidence",
    "consensus_confidence",
    "local_energy_residual",
    "energy_residual",
    "entropy_residual",
    "entropy_slope",
    "disagreement",
)


@dataclass(frozen=True)
class HumanPolicyConfig:
    input_size: int = len(HUMAN_FEATURE_NAMES)
    hidden_size: int = 48
    action_size: int = len(HUMAN_ACTION_NAMES)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 80
    gradient_clip: float = 1.0
    revision: str = "thermohitl-contextual-bandit-v1"


class _QNetwork:
    """Small wrapper that keeps torch imports optional for non-RL workflows."""

    def __init__(self, config: HumanPolicyConfig, seed: int) -> None:
        import torch

        torch.manual_seed(int(seed))
        self.torch = torch
        self.module = torch.nn.Sequential(
            torch.nn.Linear(config.input_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.action_size),
        )


class HumanAttentionPolicy:
    """Seven-action contextual bandit with decentralized execution inputs."""

    def __init__(self, config: Optional[HumanPolicyConfig] = None, seed: int = 0) -> None:
        self.config = config or HumanPolicyConfig()
        self.seed = int(seed)
        network = _QNetwork(self.config, self.seed)
        self.torch = network.torch
        self.model = network.module
        self.model.eval()

    @staticmethod
    def vector(features: Mapping[str, float]) -> np.ndarray:
        values = np.asarray([
            float(features.get(name, 0.0)) for name in HUMAN_FEATURE_NAMES
        ], dtype=np.float32)
        values[2] = np.clip(values[2], 0.0, 1.0)
        values[0:2] = np.clip(values[0:2], 0.0, 1.0)
        values[3:5] = np.clip(values[3:5], -8.0, 12.0) / 4.0
        values[5] = np.clip(values[5], -0.5, 0.5) * 2.0
        values[6] = np.clip(values[6], 0.0, 1.0)
        return values

    def q_values(self, features: Mapping[str, float]) -> np.ndarray:
        tensor = self.torch.as_tensor(self.vector(features)).unsqueeze(0)
        with self.torch.no_grad():
            values = self.model(tensor).squeeze(0)
        return values.detach().cpu().numpy().astype(float)

    def decision(self, agent_id: str, features: Mapping[str, float]) -> Tuple[float, int]:
        # ``agent_id`` is intentionally accepted but never used to access
        # another organization's state. Every invocation receives one local
        # feature mapping and returns that organization's own action.
        del agent_id
        values = self.q_values(features)
        action = int(np.argmax(values))
        ordered = np.sort(values)
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
        # The escalation controller's fixed hysteresis threshold is on an
        # evidence scale, not a probability. Human actions receive an evidence
        # score above the default gate; local actions remain below it.
        score = float(1.5 + min(2.0, max(0.0, margin))) if action >= 2 else 0.0
        return score, action

    def __call__(self, agent_id: str, features: Mapping[str, float]) -> Tuple[float, int]:
        return self.decision(agent_id, features)

    def save(self, path: Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "seed": self.seed,
            "metadata": dict(metadata or {}),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "HumanAttentionPolicy":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        policy = cls(HumanPolicyConfig(**payload["config"]), seed=int(payload["seed"]))
        policy.model.load_state_dict(payload["state_dict"])
        policy.model.eval()
        return policy


def train_contextual_bandit(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    seed: int,
    variant: str,
    config: Optional[HumanPolicyConfig] = None,
) -> List[Dict[str, Any]]:
    """Fit all-action simulator rewards using fixed-budget regression.

    During training only, the simulator supplies a seven-element reward vector
    for each context. Execution receives only the local features declared in
    :data:`HUMAN_FEATURE_NAMES`; tests enforce this distinction.
    """

    import torch

    if variant not in ("learned_no_thermodynamics", "thermohitl_rl"):
        raise ValueError("unknown human-policy variant")
    if len(rows) < 64:
        raise ValueError("contextual-bandit training requires at least 64 rows")
    policy = HumanAttentionPolicy(config=config, seed=seed)
    config = policy.config
    features = np.asarray([
        HumanAttentionPolicy.vector(row["features"]) for row in rows
    ], dtype=np.float32)
    if variant == "learned_no_thermodynamics":
        # Same model capacity; thermodynamic execution fields are exactly zero.
        features[:, 3:] = 0.0
    rewards = np.asarray([row["action_rewards"] for row in rows], dtype=np.float32)
    if rewards.shape != (len(rows), config.action_size):
        raise ValueError("every training row needs a seven-action reward vector")
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.AdamW(
        policy.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x = torch.as_tensor(features)
    y = torch.as_tensor(rewards)
    history: List[Dict[str, Any]] = []
    policy.model.train()
    for epoch in range(config.epochs):
        permutation = torch.randperm(len(rows), generator=generator)
        losses: List[float] = []
        correct = 0
        total = 0
        for start in range(0, len(rows), config.batch_size):
            index = permutation[start : start + config.batch_size]
            prediction = policy.model(x[index])
            target = y[index]
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((prediction.argmax(dim=1) == target.argmax(dim=1)).sum())
            total += len(index)
        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "best_action_accuracy": correct / max(total, 1),
            "seed": int(seed),
            "variant": variant,
        })
    policy.model.eval()
    policy.save(output, metadata={
        "variant": variant,
        "training_rows": len(rows),
        "feature_names": list(HUMAN_FEATURE_NAMES),
        "action_names": list(HUMAN_ACTION_NAMES),
        "checkpoint_selection": "final_fixed_epoch",
    })
    return history


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "seed": int(payload["seed"]),
        "config": payload["config"],
        "metadata": payload.get("metadata", {}),
    }
