"""Large-network random-sequential trajectories and pathwise irreversibility."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from thermoagent.statmech.model import ModelParameters


def simulate_stationary_pathwise(
    communication: np.ndarray,
    dependency: np.ndarray,
    parameters: ModelParameters,
    seed: int,
    burn_in_sweeps: int,
    sample_sweeps: int,
    block_sweeps: int = 20,
    private_fields: Optional[np.ndarray] = None,
    task_fields: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Estimate stationary path log-ratio under physical local-update dynamics.

    At stationarity, the mean medium log-rate ratio equals total entropy
    production because the mean boundary change of system entropy vanishes.
    The returned block values diagnose Monte Carlo uncertainty, but independent
    graph/seed trajectories remain the inferential units.
    """

    communication = np.asarray(communication, dtype=float)
    dependency = np.asarray(dependency, dtype=float)
    if communication.ndim != 2 or communication.shape[0] != communication.shape[1]:
        raise ValueError("communication must be square")
    if dependency.shape != communication.shape:
        raise ValueError("dependency shape mismatch")
    if parameters.memory_coupling != 0.0:
        raise ValueError("pathwise finite-state estimator requires zero memory coupling")
    n_agents = int(communication.shape[0])
    rng = np.random.default_rng(int(seed))
    beliefs = rng.choice(np.array([-1, 1], dtype=np.int8), n_agents)
    actions = rng.choice(np.array([-1, 1], dtype=np.int8), n_agents)
    private = np.zeros(n_agents) if private_fields is None else np.asarray(private_fields, dtype=float)
    task = np.zeros(n_agents) if task_fields is None else np.asarray(task_fields, dtype=float)
    if private.shape != (n_agents,) or task.shape != (n_agents,):
        raise ValueError("field shape mismatch")
    communication_sum = communication.dot(beliefs)
    dependency_sum = dependency.dot(actions)
    communication_recipients = [np.flatnonzero(communication[:, index]) for index in range(n_agents)]
    dependency_recipients = [np.flatnonzero(dependency[:, index]) for index in range(n_agents)]
    attempts_per_sweep = 2 * n_agents
    burn_attempts = int(burn_in_sweeps) * attempts_per_sweep
    sample_attempts = int(sample_sweeps) * attempts_per_sweep
    block_attempts = int(block_sweeps) * attempts_per_sweep
    if burn_attempts < 0 or sample_attempts <= 0 or block_attempts <= 0:
        raise ValueError("invalid trajectory lengths")

    log_ratio_sum = 0.0
    changes = 0
    magnetization_sum = 0.0
    observation_count = 0
    block_values = []
    current_block_log_ratio = 0.0
    current_block_attempts = 0
    total_attempts = burn_attempts + sample_attempts
    for attempt in range(total_attempts):
        variable = int(rng.integers(attempts_per_sweep))
        if variable < n_agents:
            index = variable
            field = (
                parameters.belief_coupling * communication_sum[index]
                + parameters.belief_action_coupling * actions[index]
                + private[index]
            )
            probability_plus = 1.0 / (
                1.0 + np.exp(-np.clip(2.0 * field / parameters.temperature, -700.0, 700.0))
            )
            old = int(beliefs[index])
            new = 1 if rng.random() < probability_plus else -1
            forward = probability_plus if new == 1 else 1.0 - probability_plus
            reverse = probability_plus if old == 1 else 1.0 - probability_plus
            if new != old:
                delta = new - old
                beliefs[index] = new
                recipients = communication_recipients[index]
                communication_sum[recipients] += communication[recipients, index] * delta
        else:
            index = variable - n_agents
            field = (
                parameters.action_coupling * dependency_sum[index]
                + parameters.belief_action_coupling * beliefs[index]
                + task[index]
            )
            probability_plus = 1.0 / (
                1.0 + np.exp(-np.clip(2.0 * field / parameters.temperature, -700.0, 700.0))
            )
            old = int(actions[index])
            new = 1 if rng.random() < probability_plus else -1
            forward = probability_plus if new == 1 else 1.0 - probability_plus
            reverse = probability_plus if old == 1 else 1.0 - probability_plus
            if new != old:
                delta = new - old
                actions[index] = new
                recipients = dependency_recipients[index]
                dependency_sum[recipients] += dependency[recipients, index] * delta
        if attempt < burn_attempts:
            continue
        ratio = float(np.log(max(forward, 1e-300) / max(reverse, 1e-300)))
        log_ratio_sum += ratio
        current_block_log_ratio += ratio
        current_block_attempts += 1
        changes += int(new != old)
        if (attempt - burn_attempts + 1) % attempts_per_sweep == 0:
            magnetization_sum += 0.5 * (float(np.mean(beliefs)) + float(np.mean(actions)))
            observation_count += 1
        if current_block_attempts == block_attempts:
            block_values.append(current_block_log_ratio / float(current_block_attempts))
            current_block_log_ratio = 0.0
            current_block_attempts = 0
    if current_block_attempts:
        block_values.append(current_block_log_ratio / float(current_block_attempts))
    per_update = log_ratio_sum / float(sample_attempts)
    metrics = {
        "n_agents": float(n_agents),
        "attempted_updates": float(sample_attempts),
        "accepted_changes": float(changes),
        "acceptance_probability": float(changes / float(sample_attempts)),
        "pathwise_irreversibility_per_update": float(per_update),
        "pathwise_irreversibility_per_agent_sweep": float(2.0 * per_update),
        "pathwise_irreversibility_per_sweep": float(attempts_per_sweep * per_update),
        "mean_signed_order": float(magnetization_sum / max(1, observation_count)),
        "block_count": float(len(block_values)),
        "block_standard_deviation": float(np.std(block_values, ddof=1)) if len(block_values) > 1 else 0.0,
    }
    return metrics, np.asarray(block_values, dtype=float)
