import numpy as np

from thermoagent.doet_monitoring import communication_group, disruption_group, stateful


def test_monitoring_strata_are_deterministic() -> None:
    assert communication_group("reliable-moderate-p0.5-o0.5") == "connected"
    assert communication_group("intermittent-compound-p1.0-o1.0") == "degraded"
    assert communication_group("partition-correlated-p0.8-o0.8") == "partition"
    assert disruption_group("nominal_control") == "nominal"
    assert disruption_group("private_mixed_correlated") == "correlated"
    assert disruption_group("unseen_compound") == "compound"
    assert disruption_group("factor_private_mixed") == "isolated"


def test_cusum_is_stateful_nonnegative_and_resets_per_call() -> None:
    values = np.asarray([0.0, 0.0, 2.0, 2.0, -2.0])
    first = stateful(values, "cusum")
    second = stateful(values, "cusum")
    assert np.all(first >= 0.0)
    assert first[3] > first[2]
    assert np.allclose(first, second)


def test_page_hinkley_responds_to_a_level_shift() -> None:
    values = np.asarray([0.0] * 5 + [3.0] * 5)
    score = stateful(values, "page_hinkley")
    assert score[-1] > score[4]
