import math

import numpy as np
import pytest

from thermoagent.v6_entropy import (
    PRESPECIFIED_Q,
    consensus_score,
    entropy_spectrum,
    generalized_disagreement,
    gini_simpson_impurity,
    graph_weighted_disagreement,
    probability_gini_concentration,
    shannon_entropy,
    temporal_information_state,
    tsallis_entropy,
    weighted_pooled_belief,
)


@pytest.mark.parametrize("q", PRESPECIFIED_Q)
def test_generalized_entropy_is_normalized(q):
    assert tsallis_entropy([1, 0, 0, 0], q) == pytest.approx(0.0)
    assert tsallis_entropy([0.25] * 4, q) == pytest.approx(1.0)


def test_tsallis_converges_to_shannon_from_both_sides():
    belief = [0.61, 0.22, 0.11, 0.06]
    expected = shannon_entropy(belief)
    assert tsallis_entropy(belief, 1.0 - 1e-6) == pytest.approx(expected, abs=2e-6)
    assert tsallis_entropy(belief, 1.0 + 1e-6) == pytest.approx(expected, abs=2e-6)


def test_q2_is_gini_simpson_impurity():
    belief = [0.55, 0.25, 0.15, 0.05]
    assert tsallis_entropy(belief, 2.0) == pytest.approx(gini_simpson_impurity(belief))


def test_gini_concentration_is_distinct_and_bounded():
    assert probability_gini_concentration([0.25] * 4) == pytest.approx(0.0)
    assert probability_gini_concentration([1, 0, 0, 0]) == pytest.approx(1.0)


def test_reliability_weighted_pooling():
    pooled = weighted_pooled_belief([[1, 0], [0, 1]], [3, 1])
    assert pooled.tolist() == pytest.approx([0.75, 0.25])


@pytest.mark.parametrize("q", PRESPECIFIED_Q)
def test_identical_beliefs_have_full_consensus(q):
    beliefs = [[0.7, 0.2, 0.1], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]]
    assert generalized_disagreement(beliefs, [1, 2, 1], q) == pytest.approx(0.0)
    assert consensus_score(beliefs, [1, 2, 1], q) == pytest.approx(1.0)


@pytest.mark.parametrize("q", PRESPECIFIED_Q)
def test_maximally_conflicting_certain_beliefs_are_bounded(q):
    beliefs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    disagreement = generalized_disagreement(beliefs, [1, 1, 1], q)
    assert disagreement == pytest.approx(1.0)
    assert consensus_score(beliefs, [1, 1, 1], q) == pytest.approx(0.0)


def test_generalized_disagreement_is_permutation_symmetric():
    beliefs = [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]]
    weights = [0.2, 0.5, 0.3]
    first = generalized_disagreement(beliefs, weights, 1.5)
    second = generalized_disagreement(beliefs[::-1], weights[::-1], 1.5)
    assert first == pytest.approx(second)


def test_graph_disagreement_uses_only_available_edges():
    beliefs = {"a": [1, 0], "b": [0, 1], "c": [1, 0]}
    full = graph_weighted_disagreement(beliefs, [("a", "b", 1), ("b", "c", 1)])
    partitioned = graph_weighted_disagreement(beliefs, [("a", "c", 1)])
    assert full > 0.9
    assert partitioned == pytest.approx(0.0)


def test_tail_emphasis_changes_entropy_spectrum():
    spectrum = entropy_spectrum([0.94, 0.04, 0.01, 0.01])
    assert spectrum["q_0_5"] > spectrum["q_3_0"]


def test_temporal_information_state():
    state = temporal_information_state([0.1, 0.2, 0.5, 0.7], threshold=0.4)
    assert state.slope == pytest.approx(0.2)
    assert state.acceleration == pytest.approx(-0.1)
    assert state.time_above_threshold == 2
    assert 0.1 < state.ewma < 0.7


@pytest.mark.parametrize("belief", ([0, 0], [-1, 2], [math.nan, 1]))
def test_invalid_beliefs_fail(belief):
    with pytest.raises(ValueError):
        shannon_entropy(belief)
