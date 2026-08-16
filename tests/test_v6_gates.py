import numpy as np

from thermoagent.v6_gates import mathematical_validation


def test_gate_mathematical_validation_is_independent_and_finite():
    result = mathematical_validation()
    assert result["normalized_bounds"]
    assert result["q_to_one_absolute_error"] <= 1e-5
    assert result["q2_gini_absolute_error"] <= 1e-12
    assert result["identical_consensus_pass"]
    assert result["conflict_pass"]
