import numpy as np
import pytest

from thermoagent.v7_entropy import (
    allocation_entropy, economic_gini, spectral_graph_entropy,
    weighted_information_state,
)
from thermoagent.v7_topology import generate_graph


def test_v7_weighted_information_state_distinguishes_uncertainty_and_disagreement():
    identical = weighted_information_state(
        [(0.90, 0.05, 0.05), (0.90, 0.05, 0.05)],
        [0.9, 0.7], [0.0, 2.0], q=1.0,
    )
    conflicting = weighted_information_state(
        [(0.98, 0.01, 0.01), (0.01, 0.98, 0.01)],
        [0.9, 0.9], [0.0, 0.0], q=1.0,
    )
    assert identical["generalized_disagreement"] == 0.0
    assert conflicting["generalized_disagreement"] > 0.5
    assert conflicting["consensus"] < identical["consensus"]


def test_v7_macrostate_diagnostics_are_numerically_stable():
    graph = generate_graph("small_world", 16, 77107)
    assert 0.0 <= spectral_graph_entropy(graph) <= 1.0
    assert allocation_entropy([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert economic_gini([1.0, 1.0, 1.0]) == 0.0
    assert economic_gini([0.0, 0.0, 3.0]) > 0.6
