from pathlib import Path

import numpy as np

from thermoagent.statmech_llm_v13.simulation import build_reciprocal_graph
from thermoagent.statmech_llm_v15.surrogate import (
    SHARED_RESPONSE_FEATURES,
    fit_reference_coefficients,
    simulate_kinetic_quench,
)


ROOT = Path(__file__).resolve().parents[2]


def test_surrogate_fit_uses_only_immutable_v13_microscopic_response():
    fitted = fit_reference_coefficients(ROOT)
    assert fitted["valid_rows"] > 0
    assert set(fitted["fits_by_decoding_noise"]) >= {"0.50", "0.85"}
    assert fitted["interpretation"].startswith("empirical kinetic response")


def test_kinetic_quench_is_seeded_and_restores_field_schedule():
    parameters = fit_reference_coefficients(ROOT)
    graph = build_reciprocal_graph(8, "modular", 15150031)
    first = simulate_kinetic_quench(
        graph, 15150032, 9, 0.8, 0.5, "field_reversal", [3, 3, 3], parameters
    )
    second = simulate_kinetic_quench(
        graph, 15150032, 9, 0.8, 0.5, "field_reversal", [3, 3, 3], parameters
    )
    assert len(first) == 9
    assert list(first["phase"]) == ["baseline"] * 3 + ["disruption"] * 3 + ["recovery"] * 3
    for feature in SHARED_RESPONSE_FEATURES:
        assert feature in first
        assert np.all(np.isfinite(first[feature]))
    assert first.equals(second)

