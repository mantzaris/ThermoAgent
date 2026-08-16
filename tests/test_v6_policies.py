import numpy as np
import pandas as pd
import pytest

from thermoagent.v6_analysis import (
    DIRECT_RISK_METHODS, FittedRiskController, contexts_to_risk_frame,
    crossfit_conformal_risk, direct_risk_scores, holm_adjust,
    make_risk_pipeline, select_at_coverage,
)
from thermoagent.v6_environment import V6PanelEnvironment
from thermoagent.v6_policies import SelectiveController, risk_score


def _contexts():
    environment = V6PanelEnvironment("humanitarian", "compound", "private_fragmented", 60201)
    environment.deliver_observations(2)
    for incident_id in sorted(environment.incidents):
        environment.exchange_sketches(incident_id, 2)
    return [environment.decision_context(value, 2) for value in sorted(environment.incidents)]


@pytest.mark.parametrize(
    "method",
    [
        "fixed_severity", "kpi_confidence", "action_value_margin",
        "calibrated_max_probability", "predictive_action_entropy",
        "ensemble_variance_proxy", "conformal_margin_proxy", "local_shannon",
        "pooled_shannon", "jensen_shannon", "gini_simpson", "tsallis_q_0_5",
        "tsallis_q_1_5", "tsallis_q_2", "tsallis_q_3",
        "jensen_tsallis_q_0_5", "jensen_tsallis_q_1_5",
        "jensen_tsallis_q_2", "jensen_tsallis_q_3", "graph_disagreement",
        "combined_generalized_entropic", "random_abstention",
    ],
)
def test_deployable_risk_scores_are_finite(method):
    for context in _contexts():
        assert float("-inf") < risk_score(method, context) < float("inf")


def test_coverage_matching_is_exact_per_decision_window():
    contexts = _contexts()
    controller = SelectiveController(
        "combined_generalized_entropic", 0.5, 1,
        escalation_risk_threshold=0.0,
    )
    decisions = controller(contexts, 2)
    assert sum(value == "execute_autonomously" for value in decisions.values()) == 2
    assert sum(value == "escalate_operator" for value in decisions.values()) == 1


def test_escalation_requires_the_frozen_risk_threshold():
    contexts = _contexts()
    never = SelectiveController(
        "combined_generalized_entropic", 0.5, 1,
        escalation_risk_threshold=1.1,
    )(contexts, 2)
    forced_eligible = SelectiveController(
        "combined_generalized_entropic", 0.5, 1,
        escalation_risk_threshold=0.0,
    )(contexts, 2)
    assert "escalate_operator" not in never.values()
    assert list(forced_eligible.values()).count("escalate_operator") == 1


def test_oracle_cannot_be_instantiated_as_deployable_controller():
    with pytest.raises(ValueError):
        SelectiveController("oracle_risk", 0.5)


def test_fitted_controller_uses_deployable_schema_and_exact_epoch_coverage():
    environment = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 66801,
    )
    step = 2
    environment.deliver_observations(step)
    for incident_id in sorted(environment.incidents):
        environment.exchange_sketches(incident_id, step)
    contexts = [
        environment.decision_context(incident_id, step)
        for incident_id in sorted(environment.incidents)
    ]
    frame = contexts_to_risk_frame(contexts, "compound")
    training = pd.concat([
        frame.assign(harmful_label=0), frame.assign(harmful_label=1),
    ])
    features = ("visible_severity", "action_probability", "js_disagreement")
    model = make_risk_pipeline(features, 0.1)
    model.fit(training, training["harmful_label"])
    decisions = FittedRiskController(
        model, "compound", 0.5, 1, escalation_risk_threshold=0.0,
    )(contexts, step)
    assert list(decisions.values()).count("execute_autonomously") == 2
    assert list(decisions.values()).count("escalate_operator") == 1
    assert list(decisions.values()).count("abstain") == 1


def test_nominal_pre_disruption_contexts_do_not_force_escalation():
    environment = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 66802,
    )
    contexts = [
        environment.decision_context(incident_id, 0)
        for incident_id in sorted(environment.incidents)
    ]
    decisions = SelectiveController(
        "combined_generalized_entropic", 0.5, 1,
        escalation_risk_threshold=0.0,
    )(contexts, 0)
    assert all(context.proposal.action == "no_action" for context in contexts)
    assert set(decisions.values()) == {"abstain"}


def test_holm_adjustment_is_monotone_in_sorted_order_and_familywise():
    raw = np.asarray([0.04, 0.01, 0.03])
    adjusted = holm_adjust(raw)
    assert np.all(adjusted >= raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert adjusted[1] == pytest.approx(0.03)


def test_direct_risk_baselines_are_finite_and_oracle_is_explicit():
    frame = contexts_to_risk_frame(_contexts(), "compound")
    frame["cluster_id"] = "panel"
    frame["step"] = 2
    frame["incident_id"] = ["incident-%d" % value for value in range(len(frame))]
    for method in DIRECT_RISK_METHODS:
        scores = direct_risk_scores(frame, method)
        assert scores.shape == (len(frame),)
        assert np.isfinite(scores).all()
    assert np.array_equal(
        direct_risk_scores(frame, "oracle_risk_upper_bound"),
        frame.harmful_label.to_numpy(dtype=float),
    )


def test_conformal_risk_control_uses_group_isolated_calibration():
    frames = []
    for family in range(5):
        for repeat in range(8):
            value = contexts_to_risk_frame(_contexts(), "compound")
            value["cluster_id"] = "cluster-%d-%d" % (family, repeat)
            value["environment_seed"] = 70000 + family
            value["topology_family"] = "topology-%d" % family
            value["scenario_family"] = "scenario-%d" % family
            value["split_family"] = "split-%d" % family
            value["harmful_label"] = [0, 1, 0, 1]
            value["evaluator_harmful_if_executed"] = [False, True, False, True]
            frames.append(value)
    frame = pd.concat(frames, ignore_index=True)
    scores, folds = crossfit_conformal_risk(frame)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert len(folds) == 5
    assert all(value["conformal_level"] == 0.90 for value in folds)


def test_static_coverage_is_matched_inside_each_decision_window():
    rows = []
    for step in (2, 4):
        for incident in range(4):
            rows.append({
                "cluster_id": "panel", "application": "humanitarian",
                "regime": "compound", "information_condition": "private_fragmented",
                "environment_seed": 1, "topology_family": "t",
                "scenario_family": "s", "split_family": "f",
                "step": step, "incident_id": "i%d" % incident,
                "harmful_label": int(incident == 0),
                "evaluator_causal_utility_if_executed": 1.0,
            })
    frame = pd.DataFrame(rows)
    # If selection were episode-wide, the four best scores could all come
    # from step 2. Per-window matching must select two from each epoch.
    scores = np.asarray([0, 1, 2, 3, 10, 11, 12, 13], dtype=float)
    selected = select_at_coverage(frame, scores, .5).iloc[0]
    assert selected.selected_actions == 4
    assert selected.action_coverage == 0.5
    assert selected.harmful_actions == 2
