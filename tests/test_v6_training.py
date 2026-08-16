import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thermoagent.v6_environment import V6PanelEnvironment
from thermoagent.v6_training import (
    METHODS, DecentralizedRolePolicies, TrajectoryController,
    assign_trajectory_rewards, delegation_mask, feature_vector,
)


@pytest.mark.parametrize("method", METHODS)
def test_training_feature_vectors_are_finite_and_method_specific(method):
    environment = V6PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 66901,
    )
    context = environment.decision_context(sorted(environment.incidents)[0], 2)
    values = feature_vector(method, context)
    assert values.ndim == 1
    assert np.isfinite(values).all()


def test_sequential_trajectory_has_multiple_epochs_and_discounted_targets():
    environment = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 66902,
    )
    context = environment.decision_context(sorted(environment.incidents)[0], 2)
    model = DecentralizedRolePolicies(len(feature_vector("ppo_kpi_only", context)))
    controller = TrajectoryController(
        model, "ppo_kpi_only", torch.device("cpu"), stochastic=False,
    )
    environment.run(controller, "test_sequential_ppo")
    assign_trajectory_rewards(controller.transitions, environment)
    assert len(controller.transitions) == len(environment.decision_steps) * environment.incident_count
    assert len({value.step for value in controller.transitions}) > 1
    assert all(np.isfinite(value.return_value) for value in controller.transitions)


def test_delegation_mask_removes_impossible_autonomous_no_action():
    environment = V6PanelEnvironment(
        "commercial", "compound", "private_fragmented", 66903,
    )
    context = environment.decision_context(sorted(environment.incidents)[0], 2)
    mask = delegation_mask(context)
    assert mask.dtype == bool
    assert mask.any()
