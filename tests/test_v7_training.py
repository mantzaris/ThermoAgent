import numpy as np
import pytest

torch = pytest.importorskip("torch")

from thermoagent.v7_training import Transition, assign_agent_grouped_gae


def test_v7_gae_never_bootstraps_across_independent_agents():
    first = Transition("ngo", "a", "x", 0, np.zeros(2), np.ones(4, dtype=bool), 0, 0.0, 0.2, reward=1.0)
    second = Transition("ngo", "b", "x", 0, np.zeros(2), np.ones(4, dtype=bool), 0, 0.0, 9.0, reward=-1.0)
    third = Transition("ngo", "a", "x", 3, np.zeros(2), np.ones(4, dtype=bool), 0, 0.0, 0.4, reward=0.5)
    assign_agent_grouped_gae([first, second, third], gamma=0.9, gae_lambda=0.8)
    expected_third = 0.5 - 0.4
    expected_first = 1.0 + 0.9 * 0.4 - 0.2 + 0.9 * 0.8 * expected_third
    assert third.advantage == pytest.approx(expected_third)
    assert first.advantage == pytest.approx(expected_first)
    assert second.advantage == pytest.approx(-10.0)
