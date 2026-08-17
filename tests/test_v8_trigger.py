import pytest

from thermoagent.v8_trigger import LocalBeliefScheduler, TriggerConfig


def _mark(scheduler, belief, step=0, confidence=0.9, kpi=0.2):
    scheduler.mark_transmitted(
        sender="agent", asset="asset", belief=belief,
        confidence=confidence, local_kpi=kpi, step=step,
    )


def test_v8_generalized_trigger_detects_equal_shannon_entropy_mode_change():
    first = (0.90, 0.05, 0.03, 0.02)
    switched = (0.05, 0.90, 0.03, 0.02)
    shannon = LocalBeliefScheduler(TriggerConfig(method="v7_shannon_change"))
    generalized = LocalBeliefScheduler(TriggerConfig(
        method="generalized_information", tau_on=0.05, tau_off=0.01,
        weights={"js": 1.0, "spectrum": 0.0, "confidence": 0.0, "age": 0.0},
    ))
    _mark(shannon, first)
    _mark(generalized, first)
    # A quiet observation crosses the off threshold and re-arms the
    # generalized Schmitt trigger after the initial reference send.
    generalized.evaluate(
        sender="agent", asset="asset", belief=first,
        confidence=0.9, local_kpi=0.2, step=1,
    )
    v7 = shannon.evaluate(
        sender="agent", asset="asset", belief=switched,
        confidence=0.9, local_kpi=0.2, step=3,
    )
    v8 = generalized.evaluate(
        sender="agent", asset="asset", belief=switched,
        confidence=0.9, local_kpi=0.2, step=3,
    )
    assert v7.shannon_drift == pytest.approx(0.0, abs=1e-12)
    assert not v7.transmit
    assert v8.js_drift > 0.3
    assert v8.transmit


def test_v8_maximum_silence_forces_refresh_but_cooldown_blocks_bursts():
    scheduler = LocalBeliefScheduler(TriggerConfig(
        method="generalized_information", tau_on=0.05, tau_off=0.03,
        cooldown_steps=3, maximum_silence_steps=6,
    ))
    first = (0.8, 0.1, 0.1)
    changed = (0.3, 0.6, 0.1)
    _mark(scheduler, first, step=0)
    scheduler.evaluate(
        sender="agent", asset="asset", belief=first,
        confidence=0.9, local_kpi=0.2, step=1,
    )
    early = scheduler.evaluate(
        sender="agent", asset="asset", belief=changed,
        confidence=0.9, local_kpi=0.2, step=2,
    )
    assert not early.transmit
    assert early.reason == "cooldown"
    stale = scheduler.evaluate(
        sender="agent", asset="asset", belief=first,
        confidence=0.9, local_kpi=0.2, step=6,
    )
    assert stale.transmit
    assert stale.reason == "maximum_silence"


def test_v8_hysteresis_rearms_only_below_tau_off():
    scheduler = LocalBeliefScheduler(TriggerConfig(
        method="generalized_information", tau_on=0.05, tau_off=0.01,
        cooldown_steps=0,
        weights={"js": 1.0, "spectrum": 0.0, "confidence": 0.0, "age": 0.0},
    ))
    first = (0.9, 0.05, 0.05)
    changed = (0.05, 0.9, 0.05)
    _mark(scheduler, first)
    scheduler.evaluate(
        sender="agent", asset="asset", belief=first,
        confidence=0.9, local_kpi=0.2, step=1,
    )
    decision = scheduler.evaluate(
        sender="agent", asset="asset", belief=changed,
        confidence=0.9, local_kpi=0.2, step=3,
    )
    assert decision.transmit
    scheduler.mark_transmitted(
        sender="agent", asset="asset", belief=changed,
        confidence=0.9, local_kpi=0.2, step=3,
    )
    assert not scheduler.references[("agent", "asset")].armed
    steady = scheduler.evaluate(
        sender="agent", asset="asset", belief=changed,
        confidence=0.9, local_kpi=0.2, step=4,
    )
    assert not steady.transmit
    assert scheduler.references[("agent", "asset")].armed


def test_v8_hysteresis_suppresses_mid_band_but_allows_fresh_high_excursion():
    scheduler = LocalBeliefScheduler(TriggerConfig(
        method="generalized_information", tau_on=0.05, tau_off=0.01,
        cooldown_steps=0, maximum_silence_steps=20,
        weights={"js": 1.0, "spectrum": 0.0, "confidence": 0.0, "age": 0.0},
    ))
    reference = (0.90, 0.05, 0.05)
    first_change = (0.05, 0.90, 0.05)
    _mark(scheduler, reference)
    # The initial helper models a successful send, so first re-arm at the
    # transmitted reference before producing a fresh threshold crossing.
    scheduler.evaluate(
        sender="agent", asset="asset", belief=reference,
        confidence=0.9, local_kpi=0.2, step=1,
    )
    assert scheduler.evaluate(
        sender="agent", asset="asset", belief=first_change,
        confidence=0.9, local_kpi=0.2, step=2,
    ).transmit
    scheduler.mark_transmitted(
        sender="agent", asset="asset", belief=first_change,
        confidence=0.9, local_kpi=0.2, step=2,
    )
    # A mid-band innovation is suppressed while the latch is active.
    middle = (0.20, 0.75, 0.05)
    assert not scheduler.evaluate(
        sender="agent", asset="asset", belief=middle,
        confidence=0.9, local_kpi=0.2, step=3,
    ).transmit
    assert not scheduler.references[("agent", "asset")].armed
    # A genuinely fresh excursion above tau_on is not hidden by the latch.
    continuation = scheduler.evaluate(
        sender="agent", asset="asset", belief=reference,
        confidence=0.9, local_kpi=0.2, step=4,
    )
    assert continuation.transmit
    assert continuation.reason == "generalized_information_continuation"


def test_v8_hysteresis_release_excludes_monotone_age_component():
    scheduler = LocalBeliefScheduler(TriggerConfig(
        method="generalized_information", tau_on=0.12, tau_off=0.04,
        cooldown_steps=0, maximum_silence_steps=30,
        weights={"js": 0.45, "spectrum": 0.25, "confidence": 0.15, "age": 0.15},
    ))
    belief = (0.70, 0.20, 0.10)
    _mark(scheduler, belief, step=0)
    decision = scheduler.evaluate(
        sender="agent", asset="asset", belief=belief,
        confidence=0.9, local_kpi=0.2, step=10,
    )
    assert decision.age_fraction > 0.0
    assert decision.hysteresis_release_score == pytest.approx(0.0)
    assert scheduler.references[("agent", "asset")].armed


def test_v8_partition_recovery_forces_current_reference():
    scheduler = LocalBeliefScheduler(TriggerConfig(method="generalized_information"))
    _mark(scheduler, (0.7, 0.2, 0.1), step=0)
    decision = scheduler.evaluate(
        sender="agent", asset="asset", belief=(0.7, 0.2, 0.1),
        confidence=0.9, local_kpi=0.2, step=2, partition_healed=True,
    )
    assert decision.transmit
    assert decision.reason == "partition_recovery"


def test_v8_none_never_sends_initial_reference_and_random_is_reproducible():
    none = LocalBeliefScheduler(TriggerConfig(method="none"))
    assert not none.evaluate(
        sender="a", asset="x", belief=(0.5, 0.5), confidence=1.0,
        local_kpi=0.0, step=0,
    ).transmit
    config = TriggerConfig(method="matched_random", random_probability=0.5)
    first = LocalBeliefScheduler(config, deterministic_seed=19)
    second = LocalBeliefScheduler(config, deterministic_seed=19)
    for value in (first, second):
        _mark(value, (0.5, 0.5))
    assert first.evaluate(
        sender="agent", asset="asset", belief=(0.5, 0.5), confidence=0.9,
        local_kpi=0.2, step=3,
    ) == second.evaluate(
        sender="agent", asset="asset", belief=(0.5, 0.5), confidence=0.9,
        local_kpi=0.2, step=3,
    )
