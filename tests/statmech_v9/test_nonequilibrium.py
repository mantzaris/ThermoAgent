import numpy as np

from thermoagent.statmech.exact import entropy_production_rate, exact_transition_matrix, stationary_distribution
from thermoagent.statmech.model import ModelParameters, MultiplexModel, topology_adjacency
from thermoagent.statmech.simulate import directed_communication, simulate_stationary


def test_directed_communication_breaks_reversibility_and_produces_entropy():
    base = topology_adjacency(3, "ring", 1)
    directed = directed_communication(base, 0.65, 14)
    model = MultiplexModel(
        directed,
        base,
        ModelParameters(0.7, 0.3, 0.4, temperature=1.25),
        private_fields=np.array([0.2, -0.1, 0.0]),
    )
    assert not model.has_equilibrium_hamiltonian
    kernel = exact_transition_matrix(model)
    stationary = stationary_distribution(kernel)
    assert entropy_production_rate(stationary, kernel) > 1e-7


def test_stationary_simulation_is_reproducible():
    adjacency = topology_adjacency(16, "small_world", 9)
    parameters = ModelParameters(temperature=2.0)
    first, trajectory_first = simulate_stationary(
        adjacency, adjacency, parameters, 51, 20, 40, keep_trajectory=True
    )
    second, trajectory_second = simulate_stationary(
        adjacency, adjacency, parameters, 51, 20, 40, keep_trajectory=True
    )
    assert first == second
    assert np.array_equal(trajectory_first["magnetization"], trajectory_second["magnetization"])


def test_response_observables_are_finite_and_bounded():
    adjacency = topology_adjacency(20, "regular", 12)
    metrics, _ = simulate_stationary(
        adjacency,
        adjacency,
        ModelParameters(temperature=2.4),
        seed=18,
        burn_in_sweeps=30,
        sample_sweeps=80,
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert -2.0 / 3.0 <= metrics["binder_cumulant"] <= 2.0 / 3.0 + 1e-9
    assert 0.0 <= metrics["activity"] <= 1.0
    assert metrics["effective_sample_size"] <= metrics["sample_count"]
