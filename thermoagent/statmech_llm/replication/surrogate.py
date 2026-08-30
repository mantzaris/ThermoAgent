"""Fitted kinetic surrogate and mean-field diagnostics for V13."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .observables import integrated_correlation_time
from .simulation import build_reciprocal_graph


def _logistic_fit(design: np.ndarray, target: np.ndarray, ridge: float = 1e-3) -> Tuple[np.ndarray, float]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("logistic design and target do not align")

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        linear = np.clip(x.dot(beta), -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        loss = -float(np.sum(y * np.log(probability + 1e-12) + (1.0 - y) * np.log(1.0 - probability + 1e-12)))
        loss += 0.5 * float(ridge) * float(np.dot(beta[1:], beta[1:]))
        gradient = x.T.dot(probability - y)
        gradient[1:] += float(ridge) * beta[1:]
        return loss, gradient

    result = minimize(lambda beta: objective(beta)[0], np.zeros(x.shape[1]), jac=lambda beta: objective(beta)[1], method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError("kinetic logistic fit failed")
    return np.asarray(result.x, dtype=float), float(result.fun / x.shape[0])


def fit_kinetic_surrogate(frame: pd.DataFrame) -> Dict[str, object]:
    valid = frame[frame["valid_after_repair"] == 1].copy()
    if len(valid) < 32:
        raise ValueError("too few microscopic rows for V13 surrogate")
    by_temperature: Dict[str, object] = {}
    for temperature, subset in valid.groupby("sampling_temperature", sort=True):
        belief_x = np.column_stack(
            [
                np.ones(len(subset)),
                subset["private_field"].to_numpy(float),
                subset["neighbor_field"].to_numpy(float) * subset["coupling_strength"].to_numpy(float),
                subset["current_belief"].to_numpy(float),
                subset["current_action"].to_numpy(float),
            ]
        )
        action_x = np.column_stack(
            [
                np.ones(len(subset)),
                subset["belief_after"].to_numpy(float),
                subset["current_action"].to_numpy(float),
            ]
        )
        belief_beta, belief_loss = _logistic_fit(belief_x, (subset["belief_after"].to_numpy(int) > 0).astype(float))
        action_beta, action_loss = _logistic_fit(action_x, (subset["action_after"].to_numpy(int) > 0).astype(float))
        by_temperature["%.2f" % float(temperature)] = {
            "temperature": float(temperature),
            "belief_coefficients": belief_beta.tolist(),
            "action_coefficients": action_beta.tolist(),
            "belief_log_loss": belief_loss,
            "action_log_loss": action_loss,
            "rows": int(len(subset)),
            "effective_neighbor_susceptibility_at_origin": float(0.5 * belief_beta[2]),
            "effective_private_susceptibility_at_origin": float(0.5 * belief_beta[1]),
            "belief_persistence": float(belief_beta[3]),
            "belief_action_coupling": float(belief_beta[4]),
            "action_belief_coupling": float(action_beta[1]),
            "action_persistence": float(action_beta[2]),
        }
    return {
        "belief_feature_order": ["intercept", "private_field", "weighted_neighbor_field", "previous_belief", "previous_action"],
        "action_feature_order": ["intercept", "updated_belief", "previous_action"],
        "fits_by_decoding_noise": by_temperature,
        "valid_rows": int(len(valid)),
        "interpretation": "empirical kinetic response; decoding noise is not thermodynamic temperature",
    }


def _interpolate_fit(parameters: Mapping[str, object], temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    fits = list(parameters["fits_by_decoding_noise"].values())  # type: ignore[index,union-attr]
    fits = sorted(fits, key=lambda item: float(item["temperature"]))
    levels = np.asarray([float(item["temperature"]) for item in fits])
    target = float(temperature)
    if target <= levels[0]:
        return np.asarray(fits[0]["belief_coefficients"], dtype=float), np.asarray(fits[0]["action_coefficients"], dtype=float)
    if target >= levels[-1]:
        return np.asarray(fits[-1]["belief_coefficients"], dtype=float), np.asarray(fits[-1]["action_coefficients"], dtype=float)
    upper = int(np.searchsorted(levels, target))
    lower = upper - 1
    weight = (target - levels[lower]) / (levels[upper] - levels[lower])
    belief = (1.0 - weight) * np.asarray(fits[lower]["belief_coefficients"], dtype=float) + weight * np.asarray(fits[upper]["belief_coefficients"], dtype=float)
    action = (1.0 - weight) * np.asarray(fits[lower]["action_coefficients"], dtype=float) + weight * np.asarray(fits[upper]["action_coefficients"], dtype=float)
    return belief, action


def mean_field_fixed_point(
    belief_beta: Sequence[float], action_beta: Sequence[float], coupling: float, iterations: int = 500
) -> Dict[str, float]:
    bb = np.asarray(belief_beta, dtype=float)
    ab = np.asarray(action_beta, dtype=float)
    mb = 0.0
    ma = 0.0
    for _ in range(int(iterations)):
        belief_linear = bb[0] + bb[2] * float(coupling) * mb + bb[3] * mb + bb[4] * ma
        new_b = 2.0 / (1.0 + np.exp(-np.clip(belief_linear, -35.0, 35.0))) - 1.0
        action_linear = ab[0] + ab[1] * new_b + ab[2] * ma
        new_a = 2.0 / (1.0 + np.exp(-np.clip(action_linear, -35.0, 35.0))) - 1.0
        mb, ma = 0.7 * mb + 0.3 * new_b, 0.7 * ma + 0.3 * new_a
    belief_linear = bb[0] + bb[2] * float(coupling) * mb + bb[3] * mb + bb[4] * ma
    p = 1.0 / (1.0 + np.exp(-np.clip(belief_linear, -35.0, 35.0)))
    local_slope = 2.0 * p * (1.0 - p)
    stability_index = float(local_slope * (bb[2] * float(coupling) + bb[3]))
    return {
        "mean_field_belief": float(mb),
        "mean_field_action": float(ma),
        "local_belief_stability_index": stability_index,
    }


def simulate_surrogate_grid(parameters: Mapping[str, object], settings: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sweeps = int(settings["sweeps"])
    burn = int(settings["burn_in_sweeps"])
    seeds_per_cell = int(settings["seeds_per_cell"])
    for n_agents in [int(value) for value in settings["agent_counts"]]:  # type: ignore[index]
        for topology in [str(value) for value in settings["topologies"]]:  # type: ignore[index]
            for coupling in [float(value) for value in settings["coupling_grid"]]:  # type: ignore[index]
                for temperature in [float(value) for value in settings["decision_noise_grid"]]:  # type: ignore[index]
                    belief_beta, action_beta = _interpolate_fit(parameters, temperature)
                    mean_field = mean_field_fixed_point(belief_beta, action_beta, coupling)
                    for replicate in range(seeds_per_cell):
                        seed = 13300000 + 100000 * n_agents + 10000 * (topology == "modular") + 101 * replicate + int(1000 * coupling) + int(100 * temperature)
                        graph = build_reciprocal_graph(n_agents, topology, seed + 17)
                        rng = np.random.default_rng(seed)
                        fields = np.asarray([1 if index % 2 == 0 else -1 for index in range(n_agents)], dtype=int)
                        beliefs = fields.copy()
                        actions = -beliefs.copy()
                        rng.shuffle(fields)
                        rng.shuffle(beliefs)
                        rng.shuffle(actions)
                        inboxes: List[List[int]] = [[] for _ in range(n_agents)]
                        belief_m: List[float] = []
                        action_m: List[float] = []
                        for update in range(n_agents * sweeps):
                            agent = int(rng.integers(0, n_agents))
                            neighbor = float(np.mean(inboxes[agent])) if inboxes[agent] else 0.0
                            inboxes[agent].clear()
                            linear_b = float(belief_beta.dot([1.0, fields[agent], coupling * neighbor, beliefs[agent], actions[agent]]))
                            p_b = 1.0 / (1.0 + np.exp(-np.clip(linear_b, -35.0, 35.0)))
                            beliefs[agent] = 1 if rng.random() < p_b else -1
                            linear_a = float(action_beta.dot([1.0, beliefs[agent], actions[agent]]))
                            p_a = 1.0 / (1.0 + np.exp(-np.clip(linear_a, -35.0, 35.0)))
                            actions[agent] = 1 if rng.random() < p_a else -1
                            recipients = np.flatnonzero(graph.weights[agent] > 0.0)
                            probabilities = graph.weights[agent, recipients]
                            recipient = int(rng.choice(recipients, p=probabilities / probabilities.sum()))
                            inboxes[recipient].append(int(beliefs[agent]))
                            if update >= burn * n_agents and (update + 1) % n_agents == 0:
                                belief_m.append(float(np.mean(beliefs)))
                                action_m.append(float(np.mean(actions)))
                        b = np.asarray(belief_m, dtype=float)
                        a = np.asarray(action_m, dtype=float)
                        rows.append(
                            {
                                "n_agents": n_agents,
                                "topology": topology,
                                "coupling_strength": coupling,
                                "sampling_temperature": temperature,
                                "replicate": replicate,
                                "mean_abs_belief_magnetization": float(np.mean(np.abs(b))),
                                "mean_abs_action_magnetization": float(np.mean(np.abs(a))),
                                "belief_susceptibility": float(n_agents * np.var(b, ddof=1)),
                                "belief_correlation_time_sweeps": integrated_correlation_time(b),
                                **mean_field,
                            }
                        )
    return rows
