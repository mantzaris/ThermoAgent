from thermoagent.statmech_llm_v11.qualification import (
    expected_decision_requests,
    prompt_pair_fingerprint,
    qualification_conditions,
)
from thermoagent.statmech_llm_v11.core import IndependentEvidenceAgent
from thermoagent.statmech_llm_v11.qualification import _packet, _prompt_with_delivery_mode
from thermoagent.statmech_llm_v11.baselines import ScriptedBayesianProvider


def test_engineering_design_size_and_counterbalanced_conditions():
    settings = {
        "evidence_reliabilities": [0.55, 0.70, 0.85],
        "private_signal_reliability": 0.65,
        "prompt_paraphrases": 2,
        "inference_replicates": 1,
    }
    assert expected_decision_requests(settings, False) == 128
    conditions = qualification_conditions("route_viability", settings["evidence_reliabilities"], False)
    assert {item.expected_direction for item in conditions} == {-1, 0, 1}


def test_frozen_design_exactly_counterbalances_order_domain_and_private_signal():
    settings = {
        "evidence_reliabilities": [0.55, 0.65, 0.75, 0.85],
        "private_signal_reliability": 0.65,
        "prompt_paraphrases": 3,
        "inference_replicates": 2,
    }
    from thermoagent.statmech_llm_v11.qualification import _stage_design

    design = _stage_design(settings, True)
    cells = {(row["domain"], row["paraphrase"], row["option_order_right_first"], row["private_observation"], row["replicate"]) for row in design}
    assert len(design) == 864
    assert len(cells) == 48
    assert sum(row["option_order_right_first"] for row in design) == len(design) // 2


def test_paired_prompts_differ_only_in_treatment_projection():
    private = _packet("private", "right", 0.65, 1.0, "route_viability")
    base = IndependentEvidenceAgent(4, "role", private)
    treated = base.clone()
    treated.receive(_packet("peer", "left", 0.75, 1.0, "route_viability"))
    first = _prompt_with_delivery_mode(base, "qualification_unanchored", ("left", "right"), 1, 0, "none")
    second = _prompt_with_delivery_mode(treated, "qualification_unanchored", ("left", "right"), 1, 0, "one_way")
    assert first != second
    assert prompt_pair_fingerprint(first) == prompt_pair_fingerprint(second)


def test_scripted_bayesian_reference_is_monotone_in_reliability():
    probabilities = []
    for reliability in (0.55, 0.65, 0.75, 0.85):
        private = _packet("private", "left", 0.55, 1.0, "route_viability")
        agent = IndependentEvidenceAgent(4, "role", private)
        agent.receive(_packet("peer", "right", reliability, 1.0, "route_viability"))
        prompt = _prompt_with_delivery_mode(agent, "qualification_unanchored", ("left", "right"), 0, 0, "one_way")
        result = ScriptedBayesianProvider().decide(prompt, 1)
        probabilities.append(float(result.payload["probability_right"]))
    assert probabilities == sorted(probabilities)
