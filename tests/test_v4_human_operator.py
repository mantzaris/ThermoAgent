"""Engineering, privacy, and causal-mechanism tests for ThermoHITL v4."""

from __future__ import annotations

from dataclasses import replace

import pytest

from thermoagent.agents import PrivacyViolation
from thermoagent.types import PlanOutput
from thermoagent.v4_environment import FragmentedOversightEnvironment
from thermoagent.v4_operator import (
    OPERATOR_PROFILES_V4,
    SimulatedOperatorV4,
)
from thermoagent.v4_qwen import _agent_and_action, _validate
from thermoagent.v4_runner import V4EpisodeConfig, V4EpisodeRunner
from thermoagent.v4_types import (
    AttentionRequestV4,
    InformationCondition,
    OperatorInterventionV4,
    OperatorViewCondition,
    V4Application,
    V4Method,
    jensen_shannon_disagreement,
    normalized_entropy,
    validate_operator_view_v4,
)


def _environment(
    application: str = V4Application.UTILITY.value,
    regime: str = "compound",
    information: str = InformationCondition.PRIVATE_FRAGMENTED.value,
    communication: bool = True,
    seed: int = 24001,
) -> FragmentedOversightEnvironment:
    environment = FragmentedOversightEnvironment(
        application=application,
        regime=regime,
        information_condition=information,
        seed=seed,
        horizon=20,
        disruption_step=6,
        communication_enabled=communication,
    )
    environment.deliver_observations()
    return environment


def _advance_to_post_disruption(environment: FragmentedOversightEnvironment):
    features = None
    for _ in range(8):
        environment.step()
        environment.deliver_observations()
        features = environment.exchange_sketches(gossip_rounds=3)
    assert features is not None
    return features


@pytest.mark.parametrize("application", [value.value for value in V4Application])
def test_v4_all_applications_conserve_resources(application: str) -> None:
    result = V4EpisodeRunner(V4EpisodeConfig(
        application=application,
        regime="compound",
        information_condition="private_fragmented",
        method=V4Method.THERMOHITL_RULE.value,
        environment_seed=24001,
        operator_seed=124001,
        counterfactual_probes=True,
        stage="unit",
    )).run()
    assert result.status == "complete"
    assert result.metrics["maximum_conservation_residual"] <= 1e-9
    assert result.metrics["conservation_feasible"] is True


def test_v4_private_vault_rejects_peer_access() -> None:
    environment = _environment()
    first, second = list(environment.agents)[:2]
    with pytest.raises(PrivacyViolation):
        environment.agents[first].vault.observation(second)


def test_v4_agents_have_separate_context_and_utility() -> None:
    environment = _environment()
    first, second = list(environment.agents.values())[:2]
    assert first.vault is not second.vault
    assert first.inbox is not second.inbox
    assert first.identity.agent_id != second.identity.agent_id
    assert first.utility != second.utility


def test_v4_private_telemetry_never_enters_public_sketch() -> None:
    environment = _environment()
    sketch = next(iter(environment.agents.values())).coarse_sketch()
    assert "private_cost" not in sketch
    assert "resource_required" not in sketch
    assert "true_mode" not in sketch
    assert "raw_telemetry" not in sketch


def test_v4_normal_operator_view_has_no_oracle_or_future_state() -> None:
    environment = _environment()
    features = _advance_to_post_disruption(environment)
    incident_id = next(iter(features))
    agent = next(
        value for value in environment.agents.values()
        if value.vault.observation(value.agent_id).incident_id == incident_id
    )
    request = AttentionRequestV4(
        "request", incident_id, agent.agent_id, environment.step_index,
        "test", "authorize_verification", 0.5, 0.2, 0.3, 8.0, 1.0,
        features[incident_id].consensus_confidence, 2,
    )
    operator = SimulatedOperatorV4(OPERATOR_PROFILES_V4["high_accuracy_bounded"], 5, 2)
    view = operator.build_view(
        environment, request, features[incident_id],
        OperatorViewCondition.COMPLETE_THERMO,
    )
    validate_operator_view_v4(view)
    rendered = str(view.as_dict())
    for forbidden in (
        "true_mode", "resource_required", "future_disruptions",
        "counterfactual_loss", "rng_state", "evaluator_global_state",
        "private_agent_state", "oracle_state",
    ):
        assert forbidden not in rendered


def test_v4_oracle_view_is_explicitly_labeled() -> None:
    environment = _environment()
    features = _advance_to_post_disruption(environment)
    incident_id = next(iter(features))
    agent = next(
        value for value in environment.agents.values()
        if value.vault.observation(value.agent_id).incident_id == incident_id
    )
    request = AttentionRequestV4(
        "request", incident_id, agent.agent_id, environment.step_index,
        "test", "authorize_verification", 0.5, 0.2, 0.3, 8.0, 1.0,
        features[incident_id].consensus_confidence, 2,
    )
    operator = SimulatedOperatorV4(OPERATOR_PROFILES_V4["oracle"], 5, 2)
    view = operator.build_view(environment, request, features[incident_id], OperatorViewCondition.ORACLE)
    assert view.oracle is True
    assert "oracle_state" in view.features
    validate_operator_view_v4(view)


def test_v4_operator_cannot_exceed_attention_budget() -> None:
    result = V4EpisodeRunner(V4EpisodeConfig(
        application="utility_restoration", regime="compound",
        information_condition="private_fragmented",
        method="thermohitl_v4_rule", environment_seed=24001,
        operator_seed=124001, operator_budget=1,
        counterfactual_probes=False, stage="unit",
    )).run()
    assert result.metrics["operator_interventions"] <= 1


def test_v4_no_pre_disruption_trigger_artifact() -> None:
    result = V4EpisodeRunner(V4EpisodeConfig(
        application="utility_restoration", regime="compound",
        information_condition="private_fragmented",
        method="thermohitl_v4_rule", environment_seed=24001,
        operator_seed=124001, counterfactual_probes=False, stage="unit",
    )).run()
    assert result.metrics["pre_disruption_false_activation"] is False
    assert result.metrics["first_request_step"] >= 6


def test_v4_counterfactual_branch_restores_common_rng_and_changes_service() -> None:
    result = V4EpisodeRunner(V4EpisodeConfig(
        application="utility_restoration", regime="compound",
        information_condition="private_fragmented",
        method="thermohitl_v4_rule", environment_seed=24001,
        operator_seed=124001, counterfactual_probes=True, stage="unit",
    )).run()
    assert result.counterfactuals
    assert all(row["common_randomness_verified"] for row in result.counterfactuals)
    assert any(row["primary_outcome_changed"] for row in result.counterfactuals)
    assert any(row["reached_demand_or_critical_service"] for row in result.counterfactuals)


def test_v4_late_verification_replans_only_after_wrong_action() -> None:
    result = V4EpisodeRunner(V4EpisodeConfig(
        application="utility_restoration", regime="compound",
        information_condition="private_fragmented",
        method="thermohitl_v4_rule", environment_seed=24003,
        operator_seed=124003, counterfactual_probes=False, stage="unit",
    )).run()
    revisions = [event for event in result.ledger.events if event.kind == "plan_revision"]
    for event in revisions:
        assert event.payload["reason"] == "bounded_operator_evidence_after_failed_material_action"
        assert event.payload["result"]["ok"] is True


def test_v4_duplicate_crew_assignment_is_rejected() -> None:
    environment = _environment()
    _advance_to_post_disruption(environment)
    agent = next(value for value in environment.agents.values() if value.identity.role == "crew_dispatch")
    incident = agent.vault.observation(agent.agent_id).incident_id
    plan = PlanOutput(
        "dispatch", "dispatch_field_crew",
        {"crew_id": "crew_1", "target_zone": incident, "skill": "electrical"},
        "test", 1.0,
    )
    first = environment.validate_and_execute_plan(agent.agent_id, plan)
    second = environment.validate_and_execute_plan(agent.agent_id, plan)
    assert first.ok is True
    assert second.ok is False
    assert second.code == "crew_already_assigned"


def test_v4_only_abstract_cyber_disruptions_are_logged() -> None:
    environment = _environment()
    _advance_to_post_disruption(environment)
    events = [event for event in environment.ledger.events if event.kind == "disruption"]
    assert events
    assert all(event.payload["abstract_only"] is True for event in events)
    rendered = str([event.payload for event in events]).lower()
    assert "credential" not in rendered
    assert "malware" not in rendered
    assert "exploit" not in rendered


def test_v4_entropy_and_js_are_bounded() -> None:
    assert normalized_entropy((1.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert normalized_entropy((1.0, 1.0, 1.0)) == pytest.approx(1.0)
    disagreement = jensen_shannon_disagreement(((0.9, 0.05, 0.05), (0.05, 0.05, 0.9)))
    assert 0.0 < disagreement <= 1.0


def test_v4_fragmentation_changes_disagreement_not_global_severity() -> None:
    private = _environment(information="private_fragmented")
    public = _environment(information="globally_public")
    private_features = _advance_to_post_disruption(private)
    public_features = _advance_to_post_disruption(public)
    incident = "hospital_zone"
    assert private.incidents[incident].service_deficit == pytest.approx(
        public.incidents[incident].service_deficit
    )
    assert private_features[incident].belief_disagreement > public_features[incident].belief_disagreement


def test_v4_operator_intervention_is_bounded_and_logged() -> None:
    environment = _environment()
    _advance_to_post_disruption(environment)
    agent = next(iter(environment.agents))
    invalid = OperatorInterventionV4(
        "i1", "hospital_zone", environment.step_index, "unbounded_control",
        agent, {}, True, 1, 1.0, "test",
    )
    assert environment.apply_operator_intervention(invalid).ok is False
    valid = replace(invalid, intervention_id="i2", action="authorize_verification", mandatory=False)
    assert environment.apply_operator_intervention(valid).ok is True
    assert any(event.kind == "operator_action" for event in environment.ledger.events)


def test_v4_material_action_enters_then_reaches_physical_stage() -> None:
    environment = _environment(regime="isolated_physical", seed=24001)
    _advance_to_post_disruption(environment)
    result = environment.automated_response("hospital_zone", coordinated=True)
    assert result.ok is True
    for _ in range(3):
        environment.step()
    assert environment.metric_counters["material_actions_next_stage"] >= 1
    assert environment.metric_counters["material_actions_reached_service"] >= 1
    assert environment.conservation_report()["feasible"] is True


@pytest.mark.parametrize("application", [value.value for value in V4Application])
def test_v4_real_qwen_qualification_affordance_is_typed(application: str) -> None:
    environment = _environment(application=application, information="globally_public")
    _advance_to_post_disruption(environment)
    agent_id, tool, arguments = _agent_and_action(environment, 0)
    raw = __import__("json").dumps({
        "plan_summary": "Execute bounded response.",
        "tool": tool,
        "arguments": arguments,
        "justification": "Authorized private plan.",
        "confidence": 0.9,
    })
    plan, error = _validate(environment, agent_id, tool, arguments, raw)
    assert error == ""
    assert plan is not None
    assert environment.registry.validate(
        environment.agents[agent_id].identity.role, plan
    ).ok
