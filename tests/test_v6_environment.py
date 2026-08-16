from copy import deepcopy

import numpy as np
import pytest

from thermoagent.agents import PrivacyViolation
from thermoagent.v6_agents import V6ToolRegistry
from thermoagent.v6_environment import V6PanelEnvironment
from thermoagent.v6_policies import NeverActController, SelectiveController
from thermoagent.v6_types import INCIDENT_MODES, OPERATIONAL_ACTIONS, V6ToolCall


def test_private_observations_and_memories_are_separate():
    environment = V6PanelEnvironment("humanitarian", "compound", "private_fragmented", 60101)
    incident_id = sorted(environment.incidents)[0]
    first, second, _ = environment.incident_agents[incident_id]
    first_observation = environment.agents[first].vault.observation(first, incident_id)
    second_observation = environment.agents[second].vault.observation(second, incident_id)
    assert first_observation.private_evidence != second_observation.private_evidence
    assert environment.agents[first].vault is not environment.agents[second].vault
    with pytest.raises(PrivacyViolation):
        environment.agents[first].vault.observation(second, incident_id)


def test_agents_negotiate_and_keep_separate_commitment_ledgers():
    environment = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 60111,
    )
    outcomes = [
        value.payload["status"] for value in environment.ledger.events
        if value.kind == "commitment" and "status" in value.payload
    ]
    assert len(outcomes) == environment.incident_count
    assert set(outcomes).issubset({"accepted", "rejected"})
    incident_id = sorted(environment.incidents)[0]
    first, second, _ = environment.incident_agents[incident_id]
    assert environment.agents[first].commitments is not environment.agents[second].commitments
    key = next(iter(environment.agents[first].commitments))
    environment.agents[first].commitments[key].status = "withdrawn"
    assert environment.agents[second].commitments[key].status != "withdrawn"


def test_deployable_context_excludes_true_state_and_future_tape():
    environment = V6PanelEnvironment("utility_restoration", "telemetry_integrity", "private_fragmented", 60102)
    context = environment.decision_context(sorted(environment.incidents)[0], 2).deployable()
    serialized = repr(context).lower()
    assert "true_mode" not in serialized
    assert "stochastic_tape" not in serialized
    assert "correct_action" not in serialized
    assert "counterfactual" not in serialized
    assert np.isfinite(context["operational_energy"])
    assert np.isfinite(context["free_energy_diagnostic"])
    assert 0.25 <= context["effective_temperature"] <= 1.0
    assert "consensus_slope" in context
    evaluator_events = [
        value for value in environment.ledger.events
        if value.kind == "v6_consensus_state"
    ]
    assert evaluator_events
    assert all(value.private_to == "evaluator" for value in evaluator_events)
    assert not any(
        value.kind == "v6_consensus_state"
        for value in environment.ledger.visible_to(next(iter(environment.agents)))
    )


def test_role_action_mask_blocks_disallowed_actions():
    environment = V6PanelEnvironment("commercial", "compound", "private_fragmented", 60103)
    agent = next(iter(environment.agents.values()))
    mask = agent.action_mask(environment.registry)
    assert mask.shape == (len(OPERATIONAL_ACTIONS),)
    assert mask.any() and not mask.all()
    disallowed = next(action for action, allowed in zip(OPERATIONAL_ACTIONS, mask) if not allowed)
    call = V6ToolCall(disallowed, agent.identity.incident_scope[0])
    valid, code, _ = environment.registry.validate(agent.identity, call)
    assert not valid
    assert code == "action_not_permitted"


def test_partition_blocks_cross_component_delivery():
    environment = V6PanelEnvironment("utility_restoration", "partition", "private_fragmented", 60104)
    incident_id = sorted(environment.incidents)[0]
    agent_ids = environment.incident_agents[incident_id]
    blocked = [
        (first, second) for first in agent_ids for second in agent_ids
        if first != second and not environment._edge_available(first, second, 3)
    ]
    assert blocked
    environment.exchange_sketches(incident_id, 3)
    for first, second in blocked:
        assert first not in environment.sketch_cache[second]


def test_real_conservation_detects_deliberate_fault():
    environment = V6PanelEnvironment("humanitarian", "compound", "private_fragmented", 60105)
    assert environment.conservation_report()["feasible"]
    environment.inject_conservation_fault_for_test("emergency_units", 0.25)
    report = environment.conservation_report()
    assert not report["feasible"]
    assert report["maximum_residual"] == pytest.approx(0.25)


def test_matched_stochastic_tape_and_dynamic_policy_divergence():
    first = V6PanelEnvironment("utility_restoration", "compound", "private_fragmented", 60106)
    second = V6PanelEnvironment("utility_restoration", "compound", "private_fragmented", 60106)
    assert first.stochastic_tape_digest == second.stochastic_tape_digest
    always = first.run(SelectiveController("always_act", 1.0, 0), "always_act")
    never = second.run(NeverActController(), "never_act")
    assert always["event_ledger_digest"] != never["event_ledger_digest"]
    assert always["accepted_typed_actions"] > never["accepted_typed_actions"]
    assert always["service_loss"] != never["service_loss"]


def test_information_conditions_share_exogenous_scenario_and_tape():
    private = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 66123,
    )
    public = V6PanelEnvironment(
        "humanitarian", "compound", "public_shared", 66123,
    )
    assert private.stochastic_tape_digest == public.stochastic_tape_digest
    for incident_id in sorted(private.incidents):
        first = private.incidents[incident_id]
        second = public.incidents[incident_id]
        assert first.true_mode == second.true_mode
        assert first.correct_action == second.correct_action
        assert first.severity == second.severity
        assert first.priority == second.priority
        assert first.fragmentation == second.fragmentation


def test_dynamic_actions_propagate_and_conserve_resources():
    environment = V6PanelEnvironment("humanitarian", "correlated", "private_fragmented", 60107)
    summary = environment.run(SelectiveController("action_value_margin", 0.5, 1), "margin")
    assert summary["accepted_typed_actions"] > 0
    assert len(environment.action_records) > 0
    assert any(record["completed_step"] >= record["scheduled_step"] for record in environment.action_records)
    assert summary["maximum_conservation_residual"] <= 1e-9
    assert summary["conservation_feasible"]
    assert summary["autonomous_completed_actions"] > 0
    assert summary["autonomous_harmful_actions"] <= summary["autonomous_completed_actions"]
    assert summary["operator_harmful_actions"] <= summary["operator_completed_actions"]
    assert summary["net_causal_utility"] == pytest.approx(
        sum(record["causal_effect"] for record in environment.action_records)
    )


def test_matched_counterfactual_branch_restores_full_state_and_rng_tape():
    environment = V6PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 66124,
    )
    step = 2
    environment._advance_service(step)
    environment.deliver_observations(step)
    for incident_id in sorted(environment.incidents):
        environment.exchange_sketches(incident_id, step)
    context = environment.decision_context(sorted(environment.incidents)[0], step)
    result = environment.evaluator_counterfactual_branch(context.proposal, step)
    assert result["stochastic_tape_digest_action"] == result["stochastic_tape_digest_no_action"]
    assert np.isfinite(result["loss_reduction"])
    assert any(
        value.kind == "v6_counterfactual_branch" and value.private_to == "evaluator"
        for value in environment.ledger.events
    )


def test_sketch_accounting_includes_bytes_and_messages():
    none = V6PanelEnvironment("humanitarian", "compound", "private_fragmented", 60108, "none")
    always = V6PanelEnvironment("humanitarian", "compound", "private_fragmented", 60108, "always_on")
    none_summary = none.run(NeverActController(), "never")
    always_summary = always.run(NeverActController(), "never")
    assert none_summary["sketch_messages"] == 0
    assert none_summary["sketch_bytes"] == 0
    assert always_summary["sketch_messages"] > 0
    assert always_summary["sketch_bytes"] > 0
    assert always_summary["total_messages"] >= always_summary["sketch_messages"]


def test_operator_budget_and_queue_are_bounded():
    environment = V6PanelEnvironment("utility_restoration", "ood", "private_fragmented", 60109)
    summary = environment.run(
        SelectiveController(
            "jensen_shannon", 0.0, 4,
            escalation_risk_threshold=0.0,
        ),
        "all_escalate",
    )
    assert summary["escalations"] <= 4
    assert summary["operator_minutes"] > 0.0
    assert summary["maximum_queue_length"] <= 4


def test_stale_sketches_reduce_weight_and_increase_consensus_residual():
    environment = V6PanelEnvironment(
        "humanitarian", "compound", "private_fragmented", 60110,
        "always_on",
    )
    incident_id = sorted(environment.incidents)[0]
    recipient = environment.incident_agents[incident_id][0]
    environment.exchange_sketches(incident_id, 2)
    fresh = environment.information_state(incident_id, recipient, 2)
    stale = environment.information_state(incident_id, recipient, 10)
    assert stale["consensus_residual"] > fresh["consensus_residual"]
    assert np.isfinite(stale["pooled_uncertainty"])


def test_integrity_loss_can_create_confident_disagreement_without_label_leakage():
    environment = V6PanelEnvironment(
        "utility_restoration", "telemetry_integrity",
        "private_fragmented", 60112,
    )
    environment.deliver_observations(2)
    assert any(
        len({
            int(np.argmax(environment.agents[agent].private_beliefs[incident]))
            for agent in agents
        }) > 1
        for incident, agents in environment.incident_agents.items()
    )
    for incident_id in sorted(environment.incidents):
        context = environment.decision_context(incident_id, 2).deployable()
        assert "true_mode" not in repr(context)
        assert "correct_action" not in repr(context)


def test_pre_disruption_beliefs_do_not_reveal_future_incident_mode():
    environment = V6PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 60114,
    )
    assert all(
        int(np.argmax(environment.agents[agent].private_beliefs[incident]))
        == INCIDENT_MODES.index("nominal")
        for incident, agents in environment.incident_agents.items()
        for agent in agents
    )
    summary = environment.run(
        SelectiveController(
            "combined_generalized_entropic", 0.5, 1,
            escalation_risk_threshold=0.0,
        ),
        "timing_test",
    )
    assert summary["pre_disruption_escalations"] == 0


def test_nominal_panels_have_no_hidden_disruption_or_service_creation():
    environment = V6PanelEnvironment(
        "humanitarian", "nominal", "private_fragmented", 60115,
    )
    assert {value.true_mode for value in environment.incidents.values()} == {"nominal"}
    summary = environment.run(NeverActController(), "nominal_control")
    assert summary["service_loss"] < 0.25
    assert summary["conservation_feasible"]


def test_candidate_harm_uses_matched_dynamic_branch_not_static_label():
    environment = V6PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 60113,
    )
    step = 2
    environment._advance_service(step)
    environment.deliver_observations(step)
    for incident_id in sorted(environment.incidents):
        environment.exchange_sketches(incident_id, step)
    context = environment.decision_context(sorted(environment.incidents)[0], step)
    environment.record_candidate(context)
    row = environment.candidate_records[-1]
    assert "evaluator_causal_utility_if_executed" in row
    assert "evaluator_immediate_effect_if_executed" in row
    assert row["evaluator_harmful_if_executed"] == (
        row["evaluator_causal_utility_if_executed"] < -1e-9
    )
    assert any(
        value.kind == "v6_counterfactual_branch" and value.private_to == "evaluator"
        for value in environment.ledger.events
    )
