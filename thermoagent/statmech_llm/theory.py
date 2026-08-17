"""Near-reciprocal response theory for the coupled belief--action model.

Conventions
-----------
``W[x, y]`` is a row-stochastic discrete-time kernel.  One Markov step is one
attempted update of one of the ``2N`` binary variables, chosen uniformly.  The
communication matrix uses ``A[i, j]`` for the influence of sender ``j`` on
recipient ``i``.  Therefore all rates below are per attempted variable update;
one agent sweep contains ``2N`` attempts.

For ``W(alpha) = W0 + alpha V + O(alpha**2)``, let ``pi0`` be the reversible
stationary law and ``r = d pi / d alpha | 0``.  Differentiating stationarity
gives

    r (I - W0) = pi0 V,       r 1 = 0.

If ``q_xy = pi0_x W0_xy = q_yx`` and

    f_xy = r_x W0_xy + pi0_x V_xy,
    j_xy = f_xy - f_yx,

then both current and affinity vanish at the reciprocal reference and begin at
first order.  Hence the physical linear term in stationary entropy production
vanishes and

    sigma(alpha) = alpha**2 C + O(alpha**3),
    C = 1/2 sum_(x,y) j_xy**2 / q_xy.

The expression is for a discrete-time Markov chain and includes the random
update scheduler.  It is not a continuous-time generator formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from thermoagent.statmech.exact import (
    decode_state,
    entropy_production_rate,
    exact_transition_matrix,
    stationary_distribution,
)
from thermoagent.statmech.model import MultiplexModel


Array = np.ndarray


@dataclass(frozen=True)
class DirectedFamily:
    """Reciprocal and antisymmetric components of one communication family."""

    symmetric: Array
    antisymmetric: Array
    orientation_seed: int

    def at(self, alpha: float) -> Array:
        if not 0.0 <= float(alpha) < 1.0:
            raise ValueError("alpha must be in [0, 1) for nonnegative weights")
        result = self.symmetric + float(alpha) * self.antisymmetric
        if np.min(result) < -1e-14:
            raise ArithmeticError("directed family produced a negative edge weight")
        result[np.abs(result) < 1e-15] = 0.0
        return result

    def diagnostics(self) -> Dict[str, float]:
        support = self.symmetric > 0.0
        directed_support = support | support.T
        nonzero = int(np.count_nonzero(directed_support))
        singular = np.linalg.svd(self.antisymmetric, compute_uv=False)
        row_divergence = np.sum(self.antisymmetric, axis=1)
        return {
            "directed_entries": float(nonzero),
            "undirected_edges": float(nonzero // 2),
            "symmetric_total_weight": float(np.sum(self.symmetric)),
            "antisymmetric_frobenius_norm": float(np.linalg.norm(self.antisymmetric)),
            "antisymmetric_spectral_norm": float(singular[0] if singular.size else 0.0),
            "orientation_divergence_rms": float(np.sqrt(np.mean(row_divergence ** 2))),
        }


@dataclass(frozen=True)
class PerturbativeEntropyProduction:
    """First stationary response and the exact second-order EPR coefficient."""

    stationary: Array
    stationary_derivative: Array
    kernel: Array
    kernel_derivative: Array
    current_derivative: Array
    coefficient_per_update: float
    belief_coefficient_per_update: float
    action_coefficient_per_update: float
    stationary_response_residual: float
    normalization_residual: float


def directed_family(adjacency: Array, orientation_seed: int) -> DirectedFamily:
    """Orient a symmetric skeleton without changing support or pair weight.

    Each undirected pair receives antisymmetric entries ``(+w, -w)`` with a
    seeded orientation.  Thus at every alpha, the two directed weights sum to
    ``2w`` and both directions remain available for ``alpha < 1``.  Global
    total edge weight and directed message opportunities are exactly fixed.
    """

    symmetric = np.asarray(adjacency, dtype=float).copy()
    if symmetric.ndim != 2 or symmetric.shape[0] != symmetric.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(symmetric, symmetric.T, atol=1e-13):
        raise ValueError("the reciprocal skeleton must be symmetric")
    if np.any(symmetric < 0.0) or np.any(np.diag(symmetric)):
        raise ValueError("the reciprocal skeleton must be nonnegative and loop-free")
    rng = np.random.default_rng(int(orientation_seed))
    antisymmetric = np.zeros_like(symmetric)
    for left, right in zip(*np.triu_indices_from(symmetric, 1)):
        weight = float(symmetric[left, right])
        if weight == 0.0:
            continue
        sign = 1.0 if rng.random() < 0.5 else -1.0
        antisymmetric[left, right] = sign * weight
        antisymmetric[right, left] = -sign * weight
    return DirectedFamily(symmetric, antisymmetric, int(orientation_seed))


def transition_kernel_derivative(
    reciprocal_model: MultiplexModel,
    communication_derivative: Array,
) -> Array:
    """Analytical derivative ``dW/dalpha`` at the reciprocal reference."""

    if not reciprocal_model.has_equilibrium_hamiltonian:
        raise ValueError("the reference model must be reciprocal")
    derivative = np.asarray(communication_derivative, dtype=float)
    if derivative.shape != reciprocal_model.communication.shape:
        raise ValueError("communication derivative shape mismatch")
    if not np.allclose(derivative, -derivative.T, atol=1e-13):
        raise ValueError("communication derivative must be antisymmetric")
    n_agents = reciprocal_model.n_agents
    n_states = 1 << (2 * n_agents)
    if n_states > 4096:
        raise ValueError("dense perturbation theory is restricted to at most 4096 states")
    result = np.zeros((n_states, n_states), dtype=float)
    schedule_probability = 1.0 / float(2 * n_agents)
    temperature = reciprocal_model.parameters.temperature
    coupling = reciprocal_model.parameters.belief_coupling
    for source_index in range(n_states):
        state = decode_state(source_index, n_agents)
        for agent in range(n_agents):
            p_plus = reciprocal_model.probability_plus(state, "belief", agent)
            field_derivative = coupling * float(derivative[agent].dot(state.beliefs))
            p_plus_derivative = (2.0 / temperature) * p_plus * (1.0 - p_plus) * field_derivative
            for new_value, probability_derivative in ((-1, -p_plus_derivative), (1, p_plus_derivative)):
                destination = state.copy()
                destination.beliefs[agent] = new_value
                from thermoagent.statmech.exact import encode_state

                result[source_index, encode_state(destination)] += (
                    schedule_probability * probability_derivative
                )
    if not np.allclose(result.sum(axis=1), 0.0, atol=2e-14):
        raise ArithmeticError("kernel derivative rows must sum to zero")
    return result


def stationary_first_order(stationary: Array, kernel: Array, kernel_derivative: Array) -> Tuple[Array, float, float]:
    """Solve the normalized first-order stationary perturbation equation."""

    stationary = np.asarray(stationary, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    derivative = np.asarray(kernel_derivative, dtype=float)
    n_states = kernel.shape[0]
    if kernel.shape != (n_states, n_states) or derivative.shape != kernel.shape:
        raise ValueError("kernel shapes are inconsistent")
    system = np.eye(n_states) - kernel.T
    rhs = derivative.T.dot(stationary)
    system[-1, :] = 1.0
    rhs[-1] = 0.0
    response = np.linalg.solve(system, rhs)
    equation_residual = response.dot(np.eye(n_states) - kernel) - stationary.dot(derivative)
    return (
        response,
        float(np.max(np.abs(equation_residual))),
        float(abs(np.sum(response))),
    )


def _layer_mask(n_states: int, n_agents: int, layer: str) -> Array:
    mask = np.zeros((n_states, n_states), dtype=bool)
    for source in range(n_states):
        for destination in range(source + 1, n_states):
            changed = source ^ destination
            if changed == 0 or changed & (changed - 1):
                continue
            variable = changed.bit_length() - 1
            belongs = variable < n_agents if layer == "belief" else variable >= n_agents
            if belongs:
                mask[source, destination] = True
                mask[destination, source] = True
    return mask


def perturbative_entropy_production(
    reciprocal_model: MultiplexModel,
    communication_derivative: Array,
) -> PerturbativeEntropyProduction:
    """Return the exact ``alpha**2`` coefficient for the finite Markov chain."""

    kernel = exact_transition_matrix(reciprocal_model)
    stationary = stationary_distribution(kernel)
    kernel_derivative = transition_kernel_derivative(reciprocal_model, communication_derivative)
    response, response_residual, normalization_residual = stationary_first_order(
        stationary, kernel, kernel_derivative
    )
    first_flux = response[:, None] * kernel + stationary[:, None] * kernel_derivative
    current_derivative = first_flux - first_flux.T
    equilibrium_flux = stationary[:, None] * kernel
    positive = equilibrium_flux > 1e-300
    contributions = np.zeros_like(kernel)
    contributions[positive] = current_derivative[positive] ** 2 / equilibrium_flux[positive]
    coefficient = 0.5 * float(np.sum(contributions))
    belief_mask = _layer_mask(kernel.shape[0], reciprocal_model.n_agents, "belief")
    action_mask = _layer_mask(kernel.shape[0], reciprocal_model.n_agents, "action")
    belief = 0.5 * float(np.sum(contributions[belief_mask]))
    action = 0.5 * float(np.sum(contributions[action_mask]))
    return PerturbativeEntropyProduction(
        stationary=stationary,
        stationary_derivative=response,
        kernel=kernel,
        kernel_derivative=kernel_derivative,
        current_derivative=current_derivative,
        coefficient_per_update=coefficient,
        belief_coefficient_per_update=belief,
        action_coefficient_per_update=action,
        stationary_response_residual=response_residual,
        normalization_residual=normalization_residual,
    )


def entropy_production_by_layer(
    probabilities: Array,
    kernel: Array,
    n_agents: int,
) -> Dict[str, float]:
    """Decompose exact stationary EPR by the layer whose variable changes."""

    probabilities = np.asarray(probabilities, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    forward = probabilities[:, None] * kernel
    reverse = forward.T
    mask = (forward > 0.0) & (reverse > 0.0)
    terms = np.zeros_like(kernel)
    terms[mask] = (forward[mask] - reverse[mask]) * np.log(forward[mask] / reverse[mask])
    belief_mask = _layer_mask(kernel.shape[0], n_agents, "belief")
    action_mask = _layer_mask(kernel.shape[0], n_agents, "action")
    return {
        "total_per_update": entropy_production_rate(probabilities, kernel),
        "belief_per_update": 0.5 * float(np.sum(terms[belief_mask])),
        "action_per_update": 0.5 * float(np.sum(terms[action_mask])),
    }


def finite_difference_kernel_derivative(
    reciprocal_model: MultiplexModel,
    communication_derivative: Array,
    epsilon: float = 1e-6,
) -> Array:
    """Independent central difference used only for derivation verification."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    plus = MultiplexModel(
        reciprocal_model.communication + epsilon * communication_derivative,
        reciprocal_model.dependency,
        reciprocal_model.parameters,
        reciprocal_model.private_fields,
        reciprocal_model.task_fields,
    )
    minus = MultiplexModel(
        reciprocal_model.communication - epsilon * communication_derivative,
        reciprocal_model.dependency,
        reciprocal_model.parameters,
        reciprocal_model.private_fields,
        reciprocal_model.task_fields,
    )
    return (exact_transition_matrix(plus) - exact_transition_matrix(minus)) / (2.0 * epsilon)


def exact_family_point(
    reciprocal_model: MultiplexModel,
    communication_derivative: Array,
    alpha: float,
) -> Dict[str, float]:
    """Exact stationary result at one point of a fixed nonreciprocal family."""

    model = MultiplexModel(
        reciprocal_model.communication + float(alpha) * communication_derivative,
        reciprocal_model.dependency,
        reciprocal_model.parameters,
        reciprocal_model.private_fields,
        reciprocal_model.task_fields,
    )
    kernel = exact_transition_matrix(model)
    stationary = stationary_distribution(kernel)
    layers = entropy_production_by_layer(stationary, kernel, model.n_agents)
    attempts_per_sweep = 2 * model.n_agents
    return {
        "alpha": float(alpha),
        **layers,
        "total_per_agent_sweep": float(layers["total_per_update"] * 2.0),
        "total_per_sweep": float(layers["total_per_update"] * attempts_per_sweep),
        "accepted_transition_probability": float(
            1.0 - np.sum(stationary * np.diag(kernel))
        ),
        "stationarity_residual": float(np.max(np.abs(stationary.dot(kernel) - stationary))),
    }
