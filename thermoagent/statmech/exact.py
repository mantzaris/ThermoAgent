"""Exact finite-state calculations for the V9 equilibrium reference."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from .model import Microstate, MultiplexModel


def decode_state(index: int, n_agents: int) -> Microstate:
    bits = ((int(index) >> np.arange(2 * n_agents)) & 1).astype(np.int8)
    spins = 2 * bits - 1
    beliefs = spins[:n_agents].copy()
    actions = spins[n_agents:].copy()
    return Microstate(beliefs, actions, beliefs.astype(float), np.zeros(n_agents))


def encode_state(state: Microstate) -> int:
    spins = np.concatenate([state.beliefs, state.actions])
    bits = (spins > 0).astype(np.uint64)
    powers = np.left_shift(np.uint64(1), np.arange(bits.size, dtype=np.uint64))
    return int(np.sum(bits * powers, dtype=np.uint64))


def enumerate_states(n_agents: int) -> Iterable[Microstate]:
    if n_agents > 12:
        raise ValueError("exact enumeration is restricted to N <= 12")
    for index in range(1 << (2 * n_agents)):
        yield decode_state(index, n_agents)


def gibbs_distribution(model: MultiplexModel) -> Tuple[np.ndarray, np.ndarray]:
    if not model.has_equilibrium_hamiltonian:
        raise ValueError("Gibbs enumeration requires the reversible model")
    energies = np.fromiter((model.energy(state) for state in enumerate_states(model.n_agents)), dtype=float)
    shifted = -(energies - np.min(energies)) / model.parameters.temperature
    weights = np.exp(shifted)
    probabilities = weights / np.sum(weights)
    return probabilities, energies


def exact_transition_matrix(model: MultiplexModel) -> np.ndarray:
    """Asynchronous heat-bath kernel, including self transitions."""

    n_states = 1 << (2 * model.n_agents)
    if n_states > 4096:
        raise ValueError("dense exact kernel is restricted to at most 4096 states")
    kernel = np.zeros((n_states, n_states), dtype=float)
    schedule_probability = 1.0 / float(2 * model.n_agents)
    for source_index in range(n_states):
        state = decode_state(source_index, model.n_agents)
        for variable in range(2 * model.n_agents):
            layer = "belief" if variable < model.n_agents else "action"
            agent = variable if layer == "belief" else variable - model.n_agents
            values = state.beliefs if layer == "belief" else state.actions
            for new_value in (-1, 1):
                probability = model.probability_value(state, layer, agent, new_value)
                destination = state.copy()
                destination_values = destination.beliefs if layer == "belief" else destination.actions
                destination_values[agent] = new_value
                kernel[source_index, encode_state(destination)] += schedule_probability * probability
    if not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-12):
        raise ArithmeticError("transition rows do not sum to one")
    return kernel


def stationary_distribution(kernel: np.ndarray) -> np.ndarray:
    n_states = kernel.shape[0]
    system = kernel.T - np.eye(n_states)
    rhs = np.zeros(n_states)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    stationary = np.linalg.solve(system, rhs)
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    return stationary


def detailed_balance_residual(probabilities: np.ndarray, kernel: np.ndarray) -> Dict[str, float]:
    flux = probabilities[:, None] * kernel
    residual = np.abs(flux - flux.T)
    return {
        "maximum": float(np.max(residual)),
        "mean": float(np.mean(residual)),
        "l1": float(np.sum(residual)),
    }


def entropy_production_rate(probabilities: np.ndarray, kernel: np.ndarray) -> float:
    """Schnakenberg stationary entropy production per Markov step."""

    forward = probabilities[:, None] * kernel
    reverse = forward.T
    mask = (forward > 0.0) & (reverse > 0.0)
    current = forward - reverse
    terms = np.zeros_like(forward)
    terms[mask] = current[mask] * np.log(forward[mask] / reverse[mask])
    return float(0.5 * np.sum(terms))


def empirical_distribution(
    model: MultiplexModel,
    seed: int,
    burn_in_steps: int,
    sample_steps: int,
    thinning: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = model.random_state(rng)
    def advance() -> None:
        variable = int(rng.integers(2 * model.n_agents))
        layer = "belief" if variable < model.n_agents else "action"
        agent = variable if layer == "belief" else variable - model.n_agents
        probability_plus = model.probability_plus(state, layer, agent)
        values = state.beliefs if layer == "belief" else state.actions
        values[agent] = 1 if rng.random() < probability_plus else -1

    for _ in range(burn_in_steps):
        advance()
    counts = np.zeros(1 << (2 * model.n_agents), dtype=np.int64)
    for sample in range(sample_steps):
        for _ in range(thinning):
            advance()
        counts[encode_state(state)] += 1
    return counts / float(counts.sum())


def distribution_distances(empirical: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    total_variation = 0.5 * np.sum(np.abs(empirical - reference))
    mask = empirical > 0.0
    kl = np.sum(empirical[mask] * np.log(empirical[mask] / np.maximum(reference[mask], 1e-300)))
    return {"total_variation": float(total_variation), "kl_empirical_reference": float(kl)}


def nonequilibrium_free_energy(probabilities: np.ndarray, energies: np.ndarray, temperature: float) -> float:
    mask = probabilities > 0.0
    entropy = -np.sum(probabilities[mask] * np.log(probabilities[mask]))
    return float(probabilities.dot(energies) - temperature * entropy)


def verify_free_energy_identity(
    probabilities: np.ndarray,
    equilibrium: np.ndarray,
    energies: np.ndarray,
    temperature: float,
) -> Dict[str, float]:
    free_energy = nonequilibrium_free_energy(probabilities, energies, temperature)
    equilibrium_free_energy = nonequilibrium_free_energy(equilibrium, energies, temperature)
    mask = probabilities > 0.0
    kl = float(np.sum(probabilities[mask] * np.log(probabilities[mask] / equilibrium[mask])))
    left = free_energy - equilibrium_free_energy
    right = temperature * kl
    return {"free_energy_gap": left, "temperature_times_kl": right, "absolute_residual": abs(left - right)}
