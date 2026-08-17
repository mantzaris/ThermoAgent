"""Coupled belief--action model on a multiplex network.

The equilibrium reference is a two-layer kinetic Ising model.  Each agent owns
one belief spin and one action spin.  A scheduler chooses which independent
agent is offered an update; the agent samples a local heat-bath rule.  Symmetric
static interactions yield a genuine Hamiltonian and Gibbs measure.  Directed
or stale communication uses the same local rule but generally has no global
Hamiltonian and is treated as a nonequilibrium Markov process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np


Array = np.ndarray


@dataclass
class Microstate:
    """Agent-owned binary belief/action variables and slow local state."""

    beliefs: Array
    actions: Array
    memory: Array
    workload: Array

    def copy(self) -> "Microstate":
        return Microstate(
            self.beliefs.copy(),
            self.actions.copy(),
            self.memory.copy(),
            self.workload.copy(),
        )


@dataclass
class ModelParameters:
    """Dimensionless couplings; temperature is logit decision noise."""

    belief_coupling: float = 0.45
    action_coupling: float = 0.45
    belief_action_coupling: float = 0.60
    temperature: float = 2.0
    memory_coupling: float = 0.0
    memory_rate: float = 0.08


class MultiplexModel:
    """Local heat-bath dynamics for coupled belief and action layers."""

    def __init__(
        self,
        communication: Array,
        dependency: Array,
        parameters: Optional[ModelParameters] = None,
        private_fields: Optional[Array] = None,
        task_fields: Optional[Array] = None,
    ) -> None:
        self.communication = np.asarray(communication, dtype=float)
        self.dependency = np.asarray(dependency, dtype=float)
        if self.communication.ndim != 2 or self.communication.shape[0] != self.communication.shape[1]:
            raise ValueError("communication adjacency must be square")
        if self.dependency.shape != self.communication.shape:
            raise ValueError("multiplex layers must have the same nodes")
        if np.any(np.diag(self.communication)) or np.any(np.diag(self.dependency)):
            raise ValueError("self-interactions are not allowed")
        self.n_agents = int(self.communication.shape[0])
        self.parameters = parameters or ModelParameters()
        if self.parameters.temperature <= 0:
            raise ValueError("temperature must be strictly positive")
        self.private_fields = self._field_or_zeros(private_fields)
        self.task_fields = self._field_or_zeros(task_fields)

    def _field_or_zeros(self, value: Optional[Array]) -> Array:
        if value is None:
            return np.zeros(self.n_agents, dtype=float)
        result = np.asarray(value, dtype=float)
        if result.shape != (self.n_agents,):
            raise ValueError("field shape does not match agent count")
        return result.copy()

    @property
    def has_equilibrium_hamiltonian(self) -> bool:
        return bool(
            np.allclose(self.communication, self.communication.T)
            and np.allclose(self.dependency, self.dependency.T)
            and self.parameters.memory_coupling == 0.0
        )

    def validate_state(self, state: Microstate) -> None:
        for name, values in (("beliefs", state.beliefs), ("actions", state.actions)):
            if values.shape != (self.n_agents,) or not np.all(np.isin(values, (-1, 1))):
                raise ValueError("%s must be an N-vector of -1/+1" % name)
        for name, values in (("memory", state.memory), ("workload", state.workload)):
            if values.shape != (self.n_agents,) or not np.all(np.isfinite(values)):
                raise ValueError("%s must be a finite N-vector" % name)

    def random_state(self, rng: np.random.Generator) -> Microstate:
        beliefs = rng.choice(np.array([-1, 1], dtype=np.int8), self.n_agents)
        actions = rng.choice(np.array([-1, 1], dtype=np.int8), self.n_agents)
        return Microstate(beliefs, actions, beliefs.astype(float), np.zeros(self.n_agents))

    def energy(self, state: Microstate) -> float:
        """Return the exact dimensionless Hamiltonian for the reversible limit."""

        if not self.has_equilibrium_hamiltonian:
            raise ValueError("a scalar Hamiltonian requires symmetric static interactions")
        self.validate_state(state)
        p = self.parameters
        b = state.beliefs.astype(float)
        a = state.actions.astype(float)
        return float(
            -0.5 * p.belief_coupling * b.dot(self.communication).dot(b)
            -0.5 * p.action_coupling * a.dot(self.dependency).dot(a)
            -p.belief_action_coupling * a.dot(b)
            -self.private_fields.dot(b)
            -self.task_fields.dot(a)
        )

    def local_field(self, state: Microstate, layer: str, agent_index: int) -> float:
        """Local information used by one agent's stochastic policy."""

        p = self.parameters
        i = int(agent_index)
        if layer == "belief":
            return float(
                p.belief_coupling * self.communication[i].dot(state.beliefs)
                + p.belief_action_coupling * state.actions[i]
                + self.private_fields[i]
                + p.memory_coupling * state.memory[i]
            )
        if layer == "action":
            return float(
                p.action_coupling * self.dependency[i].dot(state.actions)
                + p.belief_action_coupling * state.beliefs[i]
                + self.task_fields[i]
                + state.workload[i]
            )
        raise ValueError("layer must be 'belief' or 'action'")

    def probability_plus(self, state: Microstate, layer: str, agent_index: int) -> float:
        field = self.local_field(state, layer, agent_index)
        scaled = np.clip(2.0 * field / self.parameters.temperature, -700.0, 700.0)
        return float(1.0 / (1.0 + np.exp(-scaled)))

    def probability_value(self, state: Microstate, layer: str, agent_index: int, value: int) -> float:
        if value not in (-1, 1):
            raise ValueError("binary state must be -1 or +1")
        p_plus = self.probability_plus(state, layer, agent_index)
        return p_plus if value == 1 else 1.0 - p_plus

    def update_one(
        self,
        state: Microstate,
        rng: np.random.Generator,
        variable_index: Optional[int] = None,
        uniform_draw: Optional[float] = None,
        track_energy: bool = True,
    ) -> Dict[str, float]:
        """Apply one asynchronous local update and return transition accounting."""

        self.validate_state(state)
        variable = int(rng.integers(2 * self.n_agents) if variable_index is None else variable_index)
        if not 0 <= variable < 2 * self.n_agents:
            raise ValueError("variable index out of range")
        layer = "belief" if variable < self.n_agents else "action"
        i = variable if layer == "belief" else variable - self.n_agents
        values = state.beliefs if layer == "belief" else state.actions
        old_value = int(values[i])
        p_plus = self.probability_plus(state, layer, i)
        draw = float(rng.random() if uniform_draw is None else uniform_draw)
        new_value = 1 if draw < p_plus else -1
        forward = p_plus if new_value == 1 else 1.0 - p_plus
        old_probability = p_plus if old_value == 1 else 1.0 - p_plus
        old_energy = self.energy(state) if self.has_equilibrium_hamiltonian and track_energy else np.nan
        values[i] = new_value
        new_energy = self.energy(state) if self.has_equilibrium_hamiltonian and track_energy else np.nan
        # The local field excludes the updated variable, so the reverse heat-bath
        # probability is the old-value probability under the same conditioning.
        log_rate_ratio = float(np.log(max(forward, 1e-300) / max(old_probability, 1e-300)))
        return {
            "variable": float(variable),
            "agent": float(i),
            "layer_code": 0.0 if layer == "belief" else 1.0,
            "old_value": float(old_value),
            "new_value": float(new_value),
            "changed": float(new_value != old_value),
            "forward_probability": float(forward),
            "reverse_probability": float(old_probability),
            "log_rate_ratio": log_rate_ratio,
            "energy_change": float(new_energy - old_energy) if track_energy else np.nan,
        }

    def update_memory(self, state: Microstate) -> None:
        rate = self.parameters.memory_rate
        state.memory[:] = (1.0 - rate) * state.memory + rate * state.beliefs


def topology_adjacency(
    n_agents: int,
    family: str,
    seed: int,
    mean_degree: int = 4,
) -> Array:
    """Generate a reproducible, unweighted, loop-free topology."""

    if n_agents < 4:
        graph = nx.cycle_graph(n_agents)
    elif family == "ring":
        graph = nx.cycle_graph(n_agents)
    elif family == "regular":
        degree = min(mean_degree, n_agents - 1)
        if (degree * n_agents) % 2:
            degree -= 1
        graph = nx.random_regular_graph(max(2, degree), n_agents, seed=seed)
    elif family == "lattice":
        side = int(round(np.sqrt(n_agents)))
        if side * side != n_agents:
            raise ValueError("lattice topology requires a perfect-square agent count")
        grid = nx.grid_2d_graph(side, side, periodic=True)
        graph = nx.convert_node_labels_to_integers(grid, ordering="sorted")
    elif family == "erdos_renyi":
        probability = min(1.0, float(mean_degree) / float(n_agents - 1))
        graph = nx.erdos_renyi_graph(n_agents, probability, seed=seed)
        if not nx.is_connected(graph):
            components = [list(component) for component in nx.connected_components(graph)]
            for left, right in zip(components[:-1], components[1:]):
                graph.add_edge(left[0], right[0])
    elif family == "small_world":
        degree = min(mean_degree + mean_degree % 2, n_agents - (1 - n_agents % 2))
        graph = nx.watts_strogatz_graph(n_agents, max(2, degree), 0.15, seed=seed)
    elif family == "modular":
        sizes = [n_agents // 2, n_agents - n_agents // 2]
        graph = nx.stochastic_block_model(sizes, [[0.22, 0.025], [0.025, 0.22]], seed=seed)
        if not nx.is_connected(graph):
            components = [list(component) for component in nx.connected_components(graph)]
            for left, right in zip(components[:-1], components[1:]):
                graph.add_edge(left[0], right[0])
    else:
        raise ValueError("unknown topology family: %s" % family)
    return nx.to_numpy_array(graph, dtype=float)


def dilute_symmetric(adjacency: Array, availability: float, seed: int) -> Array:
    """Apply quenched bond dilution while retaining symmetry."""

    if not 0.0 <= availability <= 1.0:
        raise ValueError("availability must be in [0, 1]")
    adjacency = np.asarray(adjacency, dtype=float)
    rng = np.random.default_rng(seed)
    upper = np.triu(adjacency, 1)
    keep = rng.random(adjacency.shape) < availability
    diluted_upper = upper * keep
    return diluted_upper + diluted_upper.T


def mean_field_matrix(parameters: ModelParameters, communication_degree: float, dependency_degree: float) -> Array:
    """Linearization of the two order-parameter mean-field equations."""

    return np.array(
        [
            [parameters.belief_coupling * communication_degree, parameters.belief_action_coupling],
            [parameters.belief_action_coupling, parameters.action_coupling * dependency_degree],
        ],
        dtype=float,
    )


def mean_field_critical_temperature(
    parameters: ModelParameters,
    communication_degree: float,
    dependency_degree: float,
) -> float:
    """Temperature where the disordered fixed point first loses stability."""

    eigenvalues = np.linalg.eigvalsh(mean_field_matrix(parameters, communication_degree, dependency_degree))
    return float(np.max(eigenvalues))


def solve_mean_field(
    parameters: ModelParameters,
    communication_degree: float,
    dependency_degree: float,
    private_field: float = 0.0,
    task_field: float = 0.0,
    initial: Tuple[float, float] = (0.01, 0.01),
    tolerance: float = 1e-12,
    maximum_iterations: int = 10000,
) -> Tuple[float, float]:
    mb, ma = initial
    for _ in range(maximum_iterations):
        next_mb = np.tanh(
            (
                parameters.belief_coupling * communication_degree * mb
                + parameters.belief_action_coupling * ma
                + private_field
            )
            / parameters.temperature
        )
        next_ma = np.tanh(
            (
                parameters.action_coupling * dependency_degree * ma
                + parameters.belief_action_coupling * mb
                + task_field
            )
            / parameters.temperature
        )
        if max(abs(next_mb - mb), abs(next_ma - ma)) < tolerance:
            return float(next_mb), float(next_ma)
        mb, ma = 0.5 * mb + 0.5 * next_mb, 0.5 * ma + 0.5 * next_ma
    raise RuntimeError("mean-field iteration did not converge")
