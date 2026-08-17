"""Efficient online Monte Carlo simulation and response observables."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .information import macrostate_entropy, mutual_information_binary
from .model import Microstate, ModelParameters, MultiplexModel, dilute_symmetric, topology_adjacency


def integrated_autocorrelation_time(values: np.ndarray, maximum_lag: Optional[int] = None) -> float:
    series = np.asarray(values, dtype=float)
    if series.size < 4 or np.var(series) <= 1e-15:
        return 0.5
    centered = series - np.mean(series)
    maximum_lag = min(series.size // 2, 200 if maximum_lag is None else maximum_lag)
    denominator = float(np.dot(centered, centered))
    tau = 0.5
    for lag in range(1, maximum_lag + 1):
        correlation = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if correlation <= 0.0:
            break
        tau += correlation
    return float(tau)


def _initial_spins(n_agents: int, rng: np.random.Generator, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "random":
        return rng.choice([-1, 1], n_agents).astype(np.int8), rng.choice([-1, 1], n_agents).astype(np.int8)
    if mode == "ordered_plus":
        return np.ones(n_agents, dtype=np.int8), np.ones(n_agents, dtype=np.int8)
    if mode == "ordered_minus":
        return -np.ones(n_agents, dtype=np.int8), -np.ones(n_agents, dtype=np.int8)
    raise ValueError("unknown initial state")


def _color_classes(adjacency: np.ndarray) -> List[np.ndarray]:
    """Independent sets for exact conditional block-Gibbs updates."""

    support = (np.asarray(adjacency) != 0.0) | (np.asarray(adjacency).T != 0.0)
    n_agents = support.shape[0]
    colors = -np.ones(n_agents, dtype=int)
    order = sorted(range(n_agents), key=lambda index: int(np.sum(support[index])), reverse=True)
    for index in order:
        forbidden = {int(colors[neighbor]) for neighbor in np.flatnonzero(support[index]) if colors[neighbor] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[index] = color
    classes = [np.flatnonzero(colors == color) for color in range(int(colors.max()) + 1)]
    for members in classes:
        if np.any(support[np.ix_(members, members)]):
            raise ArithmeticError("graph coloring produced an invalid Gibbs block")
    return classes


def _simulate_block_stationary(
    communication: np.ndarray,
    dependency: np.ndarray,
    parameters: ModelParameters,
    seed: int,
    burn_in_sweeps: int,
    sample_sweeps: int,
    sample_interval: int,
    private_fields: Optional[np.ndarray],
    task_fields: Optional[np.ndarray],
    initial_state: str,
    keep_trajectory: bool,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """Random-order chromatic Gibbs sampler with the exact Gibbs invariant law."""

    n_agents = communication.shape[0]
    rng = np.random.default_rng(seed)
    beliefs, actions = _initial_spins(n_agents, rng, initial_state)
    memory = beliefs.astype(float).copy()
    workload = np.zeros(n_agents, dtype=float)
    private = np.zeros(n_agents) if private_fields is None else np.asarray(private_fields, dtype=float).copy()
    task = np.zeros(n_agents) if task_fields is None else np.asarray(task_fields, dtype=float).copy()
    state = Microstate(beliefs, actions, memory, workload)
    model = MultiplexModel(communication, dependency, parameters, private, task)
    belief_blocks = _color_classes(communication)
    action_blocks = _color_classes(dependency)
    magnetizations: List[float] = []
    belief_magnetizations: List[float] = []
    action_magnetizations: List[float] = []
    energies: List[float] = []
    consistencies: List[float] = []
    activities: List[float] = []
    entropy_flow_samples: List[float] = []
    belief_action_samples_b: List[int] = []
    belief_action_samples_a: List[int] = []
    interval_changes = 0
    interval_log_rate = 0.0
    interval_updates = 0
    total_sweeps = burn_in_sweeps + sample_sweeps
    symmetric_model = None
    if not model.has_equilibrium_hamiltonian:
        diagnostic_parameters = ModelParameters(
            belief_coupling=parameters.belief_coupling,
            action_coupling=parameters.action_coupling,
            belief_action_coupling=parameters.belief_action_coupling,
            temperature=parameters.temperature,
        )
        symmetric_model = MultiplexModel(
            0.5 * (communication + communication.T),
            0.5 * (dependency + dependency.T),
            diagnostic_parameters,
            private,
            task,
        )
    for sweep in range(total_sweeps):
        for block_index in rng.permutation(len(belief_blocks)):
            indices = belief_blocks[int(block_index)]
            fields = (
                parameters.belief_coupling * communication[indices].dot(beliefs)
                + parameters.belief_action_coupling * actions[indices]
                + private[indices]
                + parameters.memory_coupling * memory[indices]
            )
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(2.0 * fields / parameters.temperature, -700, 700)))
            old = beliefs[indices].copy()
            new = np.where(rng.random(indices.size) < probabilities, 1, -1).astype(np.int8)
            forward = np.where(new > 0, probabilities, 1.0 - probabilities)
            reverse = np.where(old > 0, probabilities, 1.0 - probabilities)
            beliefs[indices] = new
            memory[indices] = (1.0 - parameters.memory_rate) * memory[indices] + parameters.memory_rate * new
            if sweep >= burn_in_sweeps:
                interval_changes += int(np.count_nonzero(new != old))
                interval_log_rate += float(np.sum(np.log(np.maximum(forward, 1e-300) / np.maximum(reverse, 1e-300))))
                interval_updates += int(indices.size)
        for block_index in rng.permutation(len(action_blocks)):
            indices = action_blocks[int(block_index)]
            fields = (
                parameters.action_coupling * dependency[indices].dot(actions)
                + parameters.belief_action_coupling * beliefs[indices]
                + task[indices]
                + workload[indices]
            )
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(2.0 * fields / parameters.temperature, -700, 700)))
            old = actions[indices].copy()
            new = np.where(rng.random(indices.size) < probabilities, 1, -1).astype(np.int8)
            forward = np.where(new > 0, probabilities, 1.0 - probabilities)
            reverse = np.where(old > 0, probabilities, 1.0 - probabilities)
            actions[indices] = new
            if sweep >= burn_in_sweeps:
                interval_changes += int(np.count_nonzero(new != old))
                interval_log_rate += float(np.sum(np.log(np.maximum(forward, 1e-300) / np.maximum(reverse, 1e-300))))
                interval_updates += int(indices.size)
        if sweep >= burn_in_sweeps and (sweep - burn_in_sweeps + 1) % sample_interval == 0:
            belief_magnetizations.append(float(np.mean(beliefs)))
            action_magnetizations.append(float(np.mean(actions)))
            magnetizations.append(float(0.5 * (np.mean(beliefs) + np.mean(actions))))
            energies.append((model if symmetric_model is None else symmetric_model).energy(state))
            consistencies.append(float(np.mean(beliefs * actions)))
            activities.append(interval_changes / float(max(1, interval_updates)))
            entropy_flow_samples.append(interval_log_rate / float(max(1, interval_updates)))
            belief_action_samples_b.extend(beliefs.tolist())
            belief_action_samples_a.extend(actions.tolist())
            interval_changes = 0
            interval_log_rate = 0.0
            interval_updates = 0
    m = np.asarray(magnetizations)
    e = np.asarray(energies)
    if m.size == 0:
        raise RuntimeError("no stationary samples were retained")
    second_moment = float(np.mean(m ** 2))
    fourth_moment = float(np.mean(m ** 4))
    binder = 1.0 - fourth_moment / (3.0 * second_moment ** 2) if second_moment > 1e-15 else 0.0
    tau = integrated_autocorrelation_time(m)
    information = macrostate_entropy(m)
    metrics: Dict[str, float] = {
        "n_agents": float(n_agents),
        "temperature": float(parameters.temperature),
        "belief_coupling": float(parameters.belief_coupling),
        "action_coupling": float(parameters.action_coupling),
        "belief_action_coupling": float(parameters.belief_action_coupling),
        "mean_abs_magnetization": float(np.mean(np.abs(m))),
        "mean_magnetization": float(np.mean(m)),
        "belief_magnetization": float(np.mean(belief_magnetizations)),
        "action_magnetization": float(np.mean(action_magnetizations)),
        "belief_action_consistency": float(np.mean(consistencies)),
        "energy_per_agent": float(np.mean(e) / n_agents),
        "susceptibility_per_agent": float(n_agents * np.var(m) / parameters.temperature),
        "heat_capacity_per_agent": float(np.var(e) / (n_agents * parameters.temperature ** 2)),
        "binder_cumulant": float(binder),
        "integrated_autocorrelation_time": tau,
        "activity": float(np.mean(activities)),
        "entropy_production_per_update": float(np.mean(entropy_flow_samples)),
        "belief_action_mutual_information": mutual_information_binary(
            np.asarray(belief_action_samples_b), np.asarray(belief_action_samples_a)
        ),
        "sample_count": float(m.size),
        "effective_sample_size": float(m.size / max(1.0, 2.0 * tau)),
        "communication_mean_degree": float(np.mean(np.sum(communication != 0.0, axis=1))),
        "dependency_mean_degree": float(np.mean(np.sum(dependency != 0.0, axis=1))),
        "belief_color_count": float(len(belief_blocks)),
        "action_color_count": float(len(action_blocks)),
    }
    metrics.update({"macrostate_%s" % key: value for key, value in information.items()})
    trajectory = None
    if keep_trajectory:
        trajectory = {
            "magnetization": m,
            "belief_magnetization": np.asarray(belief_magnetizations),
            "action_magnetization": np.asarray(action_magnetizations),
            "energy": e,
            "consistency": np.asarray(consistencies),
            "activity": np.asarray(activities),
            "entropy_flow": np.asarray(entropy_flow_samples),
        }
    return metrics, trajectory


def simulate_stationary(
    communication: np.ndarray,
    dependency: np.ndarray,
    parameters: ModelParameters,
    seed: int,
    burn_in_sweeps: int,
    sample_sweeps: int,
    sample_interval: int = 2,
    private_fields: Optional[np.ndarray] = None,
    task_fields: Optional[np.ndarray] = None,
    initial_state: str = "random",
    keep_trajectory: bool = False,
    update_scheme: str = "block_gibbs",
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """Run random-sequential local heat-bath dynamics.

    Independent time samples are *not* assumed.  Returned response functions are
    time-series summaries; inferential replication is across independent seeds
    and graph realizations.
    """

    communication = np.asarray(communication, dtype=float)
    dependency = np.asarray(dependency, dtype=float)
    if update_scheme == "block_gibbs":
        return _simulate_block_stationary(
            communication,
            dependency,
            parameters,
            seed,
            burn_in_sweeps,
            sample_sweeps,
            sample_interval,
            private_fields,
            task_fields,
            initial_state,
            keep_trajectory,
        )
    if update_scheme != "random_sequential":
        raise ValueError("unknown update scheme")
    n_agents = communication.shape[0]
    rng = np.random.default_rng(seed)
    beliefs, actions = _initial_spins(n_agents, rng, initial_state)
    memory = beliefs.astype(float).copy()
    workload = np.zeros(n_agents, dtype=float)
    private = np.zeros(n_agents) if private_fields is None else np.asarray(private_fields, dtype=float).copy()
    task = np.zeros(n_agents) if task_fields is None else np.asarray(task_fields, dtype=float).copy()
    state = Microstate(beliefs, actions, memory, workload)
    model = MultiplexModel(communication, dependency, parameters, private, task)

    communication_sum = communication.dot(beliefs)
    dependency_sum = dependency.dot(actions)
    communication_incoming = [np.flatnonzero(communication[:, index]) for index in range(n_agents)]
    dependency_incoming = [np.flatnonzero(dependency[:, index]) for index in range(n_agents)]
    energy = model.energy(state) if model.has_equilibrium_hamiltonian else np.nan

    magnetizations: List[float] = []
    belief_magnetizations: List[float] = []
    action_magnetizations: List[float] = []
    energies: List[float] = []
    consistencies: List[float] = []
    activities: List[float] = []
    belief_action_samples_b: List[int] = []
    belief_action_samples_a: List[int] = []
    entropy_flow_samples: List[float] = []
    changed_since_sample = 0
    log_rate_since_sample = 0.0
    total_attempts = (burn_in_sweeps + sample_sweeps) * 2 * n_agents
    burn_attempts = burn_in_sweeps * 2 * n_agents

    for attempt in range(total_attempts):
        variable = int(rng.integers(2 * n_agents))
        if variable < n_agents:
            index = variable
            field = (
                parameters.belief_coupling * communication_sum[index]
                + parameters.belief_action_coupling * actions[index]
                + private[index]
                + parameters.memory_coupling * memory[index]
            )
            p_plus = 1.0 / (1.0 + np.exp(-np.clip(2.0 * field / parameters.temperature, -700, 700)))
            old = int(beliefs[index])
            new = 1 if rng.random() < p_plus else -1
            forward = p_plus if new == 1 else 1.0 - p_plus
            reverse = p_plus if old == 1 else 1.0 - p_plus
            if new != old:
                delta = new - old
                beliefs[index] = new
                incoming = communication_incoming[index]
                communication_sum[incoming] += communication[incoming, index] * delta
                if model.has_equilibrium_hamiltonian:
                    energy -= delta * field
                changed_since_sample += int(attempt >= burn_attempts)
            memory[index] = (1.0 - parameters.memory_rate) * memory[index] + parameters.memory_rate * new
        else:
            index = variable - n_agents
            field = (
                parameters.action_coupling * dependency_sum[index]
                + parameters.belief_action_coupling * beliefs[index]
                + task[index]
                + workload[index]
            )
            p_plus = 1.0 / (1.0 + np.exp(-np.clip(2.0 * field / parameters.temperature, -700, 700)))
            old = int(actions[index])
            new = 1 if rng.random() < p_plus else -1
            forward = p_plus if new == 1 else 1.0 - p_plus
            reverse = p_plus if old == 1 else 1.0 - p_plus
            if new != old:
                delta = new - old
                actions[index] = new
                incoming = dependency_incoming[index]
                dependency_sum[incoming] += dependency[incoming, index] * delta
                if model.has_equilibrium_hamiltonian:
                    energy -= delta * field
                changed_since_sample += int(attempt >= burn_attempts)
        if attempt >= burn_attempts:
            log_rate_since_sample += float(np.log(max(forward, 1e-300) / max(reverse, 1e-300)))
        completed_sweeps = (attempt + 1) // (2 * n_agents)
        at_sample = (
            attempt >= burn_attempts
            and (attempt + 1) % (2 * n_agents * sample_interval) == 0
        )
        if at_sample:
            belief_magnetizations.append(float(np.mean(beliefs)))
            action_magnetizations.append(float(np.mean(actions)))
            magnetizations.append(float(0.5 * (np.mean(beliefs) + np.mean(actions))))
            if model.has_equilibrium_hamiltonian:
                energies.append(float(energy))
            else:
                # Symmetrized diagnostic is not called a Hamiltonian.
                symmetric_comm = 0.5 * (communication + communication.T)
                symmetric_dep = 0.5 * (dependency + dependency.T)
                diagnostic = MultiplexModel(symmetric_comm, symmetric_dep, parameters, private, task)
                energies.append(diagnostic.energy(state))
            consistencies.append(float(np.mean(beliefs * actions)))
            sample_attempts = 2 * n_agents * sample_interval
            activities.append(changed_since_sample / float(sample_attempts))
            entropy_flow_samples.append(log_rate_since_sample / float(sample_attempts))
            belief_action_samples_b.extend(beliefs.tolist())
            belief_action_samples_a.extend(actions.tolist())
            changed_since_sample = 0
            log_rate_since_sample = 0.0

    m = np.asarray(magnetizations)
    e = np.asarray(energies)
    if m.size == 0:
        raise RuntimeError("no stationary samples were retained")
    second_moment = float(np.mean(m ** 2))
    fourth_moment = float(np.mean(m ** 4))
    binder = 1.0 - fourth_moment / (3.0 * second_moment ** 2) if second_moment > 1e-15 else 0.0
    information = macrostate_entropy(m)
    metrics: Dict[str, float] = {
        "n_agents": float(n_agents),
        "temperature": float(parameters.temperature),
        "belief_coupling": float(parameters.belief_coupling),
        "action_coupling": float(parameters.action_coupling),
        "belief_action_coupling": float(parameters.belief_action_coupling),
        "mean_abs_magnetization": float(np.mean(np.abs(m))),
        "mean_magnetization": float(np.mean(m)),
        "belief_magnetization": float(np.mean(belief_magnetizations)),
        "action_magnetization": float(np.mean(action_magnetizations)),
        "belief_action_consistency": float(np.mean(consistencies)),
        "energy_per_agent": float(np.mean(e) / n_agents),
        "susceptibility_per_agent": float(n_agents * np.var(m) / parameters.temperature),
        "heat_capacity_per_agent": float(np.var(e) / (n_agents * parameters.temperature ** 2)),
        "binder_cumulant": float(binder),
        "integrated_autocorrelation_time": integrated_autocorrelation_time(m),
        "activity": float(np.mean(activities)),
        "entropy_production_per_update": float(np.mean(entropy_flow_samples)),
        "belief_action_mutual_information": mutual_information_binary(
            np.asarray(belief_action_samples_b), np.asarray(belief_action_samples_a)
        ),
        "sample_count": float(m.size),
        "effective_sample_size": float(m.size / max(1.0, 2.0 * integrated_autocorrelation_time(m))),
        "communication_mean_degree": float(np.mean(np.sum(communication != 0.0, axis=1))),
        "dependency_mean_degree": float(np.mean(np.sum(dependency != 0.0, axis=1))),
    }
    metrics.update({"macrostate_%s" % key: value for key, value in information.items()})
    trajectory = None
    if keep_trajectory:
        trajectory = {
            "magnetization": m,
            "belief_magnetization": np.asarray(belief_magnetizations),
            "action_magnetization": np.asarray(action_magnetizations),
            "energy": e,
            "consistency": np.asarray(consistencies),
            "activity": np.asarray(activities),
            "entropy_flow": np.asarray(entropy_flow_samples),
        }
    return metrics, trajectory


def make_model_layers(
    n_agents: int,
    topology: str,
    graph_seed: int,
    communication_availability: float,
    mean_degree: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    communication_base = topology_adjacency(n_agents, topology, graph_seed, mean_degree)
    communication = dilute_symmetric(communication_base, communication_availability, graph_seed + 104729)
    dependency_family = topology if topology in ("ring", "lattice") else "regular"
    dependency = topology_adjacency(n_agents, dependency_family, graph_seed + 17, mean_degree)
    return communication, dependency


def run_parameter_cell(
    n_agents: int,
    topology: str,
    graph_seed: int,
    simulation_seed: int,
    temperature: float,
    communication_availability: float,
    fragmentation: float,
    parameters: ModelParameters,
    burn_in_sweeps: int,
    sample_sweeps: int,
    sample_interval: int,
    initial_state: str = "random",
) -> Dict[str, float]:
    communication, dependency = make_model_layers(
        n_agents, topology, graph_seed, communication_availability
    )
    field_rng = np.random.default_rng(graph_seed + 200003)
    private_fields = field_rng.normal(0.0, fragmentation, n_agents)
    local_parameters = ModelParameters(**asdict(parameters))
    local_parameters.temperature = float(temperature)
    metrics, _ = simulate_stationary(
        communication,
        dependency,
        local_parameters,
        simulation_seed,
        burn_in_sweeps,
        sample_sweeps,
        sample_interval,
        private_fields=private_fields,
        initial_state=initial_state,
    )
    metrics.update(
        {
            "topology": topology,
            "graph_seed": float(graph_seed),
            "simulation_seed": float(simulation_seed),
            "communication_availability": float(communication_availability),
            "fragmentation": float(fragmentation),
            "initial_state_code": {"random": 0.0, "ordered_plus": 1.0, "ordered_minus": -1.0}[initial_state],
        }
    )
    return metrics


def directed_communication(adjacency: np.ndarray, asymmetry: float, seed: int) -> np.ndarray:
    """Break reciprocity while preserving the undirected support."""

    if not 0.0 <= asymmetry < 1.0:
        raise ValueError("asymmetry must be in [0, 1)")
    rng = np.random.default_rng(seed)
    directed = np.asarray(adjacency, dtype=float).copy()
    for left, right in zip(*np.triu_indices_from(directed, 1)):
        if directed[left, right] == 0.0:
            continue
        sign = 1.0 if rng.random() < 0.5 else -1.0
        directed[left, right] *= 1.0 + sign * asymmetry
        directed[right, left] *= 1.0 - sign * asymmetry
    return directed


def simulate_field_hysteresis(
    communication: np.ndarray,
    dependency: np.ndarray,
    parameters: ModelParameters,
    fields: List[float],
    sweeps_per_field: int,
    seed: int,
) -> List[Dict[str, float]]:
    """Quasistatic up/down field protocol retaining the agent microstate."""

    n_agents = communication.shape[0]
    rng = np.random.default_rng(seed)
    beliefs = -np.ones(n_agents, dtype=np.int8)
    actions = -np.ones(n_agents, dtype=np.int8)
    belief_blocks = _color_classes(communication)
    action_blocks = _color_classes(dependency)
    rows: List[Dict[str, float]] = []
    schedule = [("up", value) for value in fields] + [("down", value) for value in reversed(fields)]
    for branch, field_value in schedule:
        retained: List[float] = []
        for sweep in range(sweeps_per_field):
            for block_index in rng.permutation(len(belief_blocks)):
                indices = belief_blocks[int(block_index)]
                local = (
                    parameters.belief_coupling * communication[indices].dot(beliefs)
                    + parameters.belief_action_coupling * actions[indices]
                    + field_value
                )
                probability = 1.0 / (1.0 + np.exp(-np.clip(2.0 * local / parameters.temperature, -700, 700)))
                beliefs[indices] = np.where(rng.random(indices.size) < probability, 1, -1)
            for block_index in rng.permutation(len(action_blocks)):
                indices = action_blocks[int(block_index)]
                local = (
                    parameters.action_coupling * dependency[indices].dot(actions)
                    + parameters.belief_action_coupling * beliefs[indices]
                    + field_value
                )
                probability = 1.0 / (1.0 + np.exp(-np.clip(2.0 * local / parameters.temperature, -700, 700)))
                actions[indices] = np.where(rng.random(indices.size) < probability, 1, -1)
            if sweep >= sweeps_per_field // 2:
                retained.append(float(0.5 * (np.mean(beliefs) + np.mean(actions))))
        rows.append(
            {
                "branch_code": 1.0 if branch == "up" else -1.0,
                "field": float(field_value),
                "magnetization": float(np.mean(retained)),
                "magnetization_sd": float(np.std(retained, ddof=1)) if len(retained) > 1 else 0.0,
            }
        )
    return rows
