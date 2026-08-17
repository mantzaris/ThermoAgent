import numpy as np
import pytest

from thermoagent.statmech.information import (
    gini_simpson,
    macrostate_entropy,
    mutual_information_binary,
    shannon_entropy,
    tsallis_entropy,
)


@pytest.mark.parametrize("q", [0.9999999, 1.0, 1.0000001])
def test_tsallis_converges_to_shannon_near_one(q):
    probabilities = np.array([0.12, 0.28, 0.60])
    assert abs(tsallis_entropy(probabilities, q) - shannon_entropy(probabilities)) < 1e-6


def test_q_two_is_normalized_gini_simpson():
    probabilities = np.array([0.05, 0.15, 0.30, 0.50])
    assert np.isclose(tsallis_entropy(probabilities, 2.0), gini_simpson(probabilities), atol=1e-14)


def test_entropy_bounds_and_extrema():
    concentrated = np.array([1.0, 0.0, 0.0, 0.0])
    uniform = np.full(4, 0.25)
    for function in (shannon_entropy, gini_simpson):
        assert function(concentrated) == pytest.approx(0.0)
        assert function(uniform) == pytest.approx(1.0)
    for q in (0.5, 1.5, 2.0, 3.0):
        assert tsallis_entropy(concentrated, q) == pytest.approx(0.0)
        assert tsallis_entropy(uniform, q) == pytest.approx(1.0)


def test_mutual_information_detects_belief_action_consistency():
    beliefs = np.array([-1, -1, 1, 1] * 20)
    assert mutual_information_binary(beliefs, beliefs) == pytest.approx(np.log(2.0))
    independent = np.tile(np.array([-1, 1, -1, 1]), 20)
    assert mutual_information_binary(beliefs, independent) == pytest.approx(0.0)


def test_macrostate_entropies_are_named_and_finite():
    result = macrostate_entropy(np.linspace(-1.0, 1.0, 101))
    assert set(result) == {"shannon", "tsallis_q_0_5", "tsallis_q_1_0", "tsallis_q_2_0", "gini_simpson"}
    assert all(np.isfinite(value) for value in result.values())
