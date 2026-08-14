from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from thermoagent.human_environment import (
    HumanOperatorToolRegistry,
    HumanOversightEnvironment,
    HumanScenarioConfig,
)
from thermoagent.environment import SOURCE_ROLES
from thermoagent.human_operator import (
    AssistanceKind,
    AssistanceRequest,
    AttentionAllocator,
    AutonomyLevel,
    DistributedThermodynamicMonitor,
    EnergyWeights,
    EscalationConfig,
    HumanMethod,
    IndependentEscalationController,
    LocalThermodynamicState,
    OPERATOR_PROFILES,
    OperatorIntervention,
    OperatorView,
    OperatorViewCondition,
    SimulatedOperator,
    ThermodynamicCalibration,
    build_operator_view,
    canonical_payload_sha256,
    jensen_shannon_divergence,
    validate_operator_view,
)
from thermoagent.human_runner import HumanOperatorEpisodeRunner
from thermoagent.planners import PlannerRequest, TransformersPlanner
from thermoagent.tools import ToolRegistry


def _config(application="commercial", **values):
    base = dict(
        application=application,
        seed=9101,
        horizon=14,
        n_agents=8,
        topology="human_v3_development",
        disruption="moderate",
        decision_interval=2,
        communication_budget=60,
        operator_seed=17,
    )
    base.update(values)
    return HumanScenarioConfig(**base)


def _state(agent_id="retailer_08", role="retailer", step=5, **values):
    base = dict(
        agent_id=agent_id,
        role=role,
        step=step,
        energy=0.7,
        local_energy_residual=2.2,
        distributed_energy=0.65,
        energy_residual=2.0,
        flow_entropy=0.4,
        belief_entropy=0.7,
        distributed_entropy=0.75,
        entropy_residual=2.1,
        entropy_slope=0.12,
        entropy_acceleration=0.03,
        disagreement=0.18,
        consensus_confidence=0.75,
        local_disruption_risk=0.75,
        local_kpi_risk=0.72,
        actionability_evidence=1.0,
        temperature=0.4,
        free_energy=0.35,
        free_energy_residual=1.4,
        components={
            "backlog": 0.8,
            "unmet": 0.7,
            "congestion": 0.5,
            "lateness": 0.4,
            "commitment": 0.2,
            "safety": 0.6,
        },
        macrostate=7,
        sketch_contributors=3,
    )
    base.update(values)
    return LocalThermodynamicState(**base)


def _request(**values):
    base = dict(
        incident_id="HA000001",
        requesting_agent="retailer_08",
        application="commercial",
        step=5,
        assistance_kind=AssistanceKind.APPROVAL.value,
        reason="severity",
        severity=0.8,
        entropy_anomaly=2.1,
        disagreement=0.18,
        consensus_confidence=0.75,
        local_kpi_risk=0.72,
        expected_loss_without=8.0,
        expected_loss_with=4.0,
        expected_benefit=4.0,
        prediction_uncertainty=0.2,
        estimated_operator_minutes=8.0,
        priority_score=2.0,
        predicted_steps_until_collapse=2,
        suggested_intervention="authorize_emergency_route",
        intervention_arguments={
            "source": "supplier_01",
            "target": "retailer_08",
            "duration": 5,
        },
        requested_autonomy_level=int(AutonomyLevel.HUMAN_APPROVAL),
    )
    base.update(values)
    return AssistanceRequest(**base)


@lru_cache(maxsize=2)
def _fitted_nominal_calibration(application):
    rows = []
    for seed in (1, 2):
        env = HumanOversightEnvironment(_config(
            application=application, seed=seed, horizon=12,
            disruption="nominal",
        ))
        monitor = DistributedThermodynamicMonitor(env.agent_ids, gossip_rounds=3)
        for _ in range(12):
            env.transition()
            env.deliver_observations()
            update = monitor.update(
                env, [env.active_communication_edges()] * 3
            )
            for state in update.local.values():
                rows.append({
                    "role": state.role,
                    "energy": state.distributed_energy,
                    "distributed_entropy": state.distributed_entropy,
                    "flow_entropy": state.flow_entropy,
                    "belief_entropy": state.belief_entropy,
                    "free_energy": state.free_energy,
                })
            env.advance()
    return ThermodynamicCalibration.fit(rows)


def test_energy_weights_validate_and_normalize():
    weights = EnergyWeights().normalized()
    assert weights.shape == (6,)
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        EnergyWeights(backlog=-1.0).normalized()


def test_jensen_shannon_is_bounded_symmetric_and_zero_for_equal_inputs():
    left = [0.9, 0.1]
    right = [0.1, 0.9]
    assert jensen_shannon_divergence(left, left) == pytest.approx(0.0)
    assert jensen_shannon_divergence(left, right) == pytest.approx(
        jensen_shannon_divergence(right, left)
    )
    assert 0.0 <= jensen_shannon_divergence(left, right) <= 1.0


def test_nominal_calibration_requires_development_data_and_uses_robust_scale():
    with pytest.raises(ValueError, match="20"):
        ThermodynamicCalibration.fit([])
    rows = []
    for index in range(30):
        rows.append({
            "role": "retailer" if index % 2 else "supplier",
            "energy": 0.2 + 0.001 * index,
            "distributed_entropy": 0.5 + 0.001 * index,
            "flow_entropy": 0.1,
            "belief_entropy": 0.3,
            "free_energy": 0.05,
        })
    calibration = ThermodynamicCalibration.fit(rows)
    assert calibration.energy_scale >= 0.025
    assert set(calibration.by_role) == {"retailer", "supplier"}


def test_distributed_monitor_uses_private_vault_and_link_local_sketches():
    env = HumanOversightEnvironment(_config(horizon=8))
    env.transition()
    env.deliver_observations()
    monitor = DistributedThermodynamicMonitor(env.agent_ids, gossip_rounds=1)
    connected = monitor.update(env, [env.active_communication_edges()])
    isolated = monitor.update(env, [set()])
    assert set(connected.local) == set(env.agent_ids)
    assert connected.evaluator.sketch_messages > 0
    assert isolated.evaluator.sketch_messages == 0
    assert any(state.consensus_confidence == 0.0 for state in isolated.local.values())
    assert all(0.0 <= state.distributed_entropy <= 1.0 for state in connected.local.values())


def test_independent_escalation_state_and_hysteresis():
    config = EscalationConfig(tau_on=0.5, tau_off=0.1, minimum_dwell=2, cooldown=0)
    controller = IndependentEscalationController(["a", "b"], config)
    high = _state(agent_id="a", energy_residual=3.0)
    request, _, activated, _ = controller.should_request(
        "a", HumanMethod.THERMOHITL_RULE, high, 0.0, np.random.RandomState(1)
    )
    assert request and activated
    assert not controller.states["b"].active
    low = _state(agent_id="a", step=6, local_energy_residual=-1.0, energy_residual=-1.0, entropy_residual=0.0, entropy_slope=0.0, disagreement=0.0, local_disruption_risk=0.0)
    repeated, _, _, _ = controller.should_request(
        "a", HumanMethod.THERMOHITL_RULE, low, 0.0, np.random.RandomState(1)
    )
    assert not repeated
    assert controller.states["a"].active
    low.step = 7
    _, _, _, deactivated = controller.should_request("a", HumanMethod.THERMOHITL_RULE, low, 0.0, np.random.RandomState(1))
    assert deactivated


@pytest.mark.parametrize(
    "condition,expected,absent",
    [
        (OperatorViewCondition.LOCAL_KPI, "local_kpi_risk", "distributed_entropy"),
        (OperatorViewCondition.ENTROPY_ONLY, "distributed_entropy", "distributed_energy"),
        (OperatorViewCondition.ENERGY_ONLY, "distributed_energy", "distributed_entropy"),
        (OperatorViewCondition.THERMODYNAMIC, "free_energy_diagnostic", "agent_disagreement"),
        (OperatorViewCondition.THERMODYNAMIC_DISAGREEMENT, "agent_disagreement", "raw_private_state"),
    ],
)
def test_operator_view_conditions_enforce_exact_feature_boundary(condition, expected, absent):
    view = build_operator_view(
        _request(), _state(), condition,
        {"workload": 0.1, "queue_length": 0},
        {"nodes": [], "physical_edges": [], "communication_edges": []},
    )
    assert expected in view.payload["features"]
    assert absent not in view.payload["features"]
    validate_operator_view(view)


def test_normal_operator_view_rejects_private_or_evaluator_leak():
    payload = {"features": {"private_cost": 1.2}}
    view = OperatorView(
        "thermohitl-operator-view-v1",
        OperatorViewCondition.LOCAL_KPI.value,
        1,
        "x",
        payload,
        canonical_payload_sha256(payload),
    )
    with pytest.raises(PermissionError, match="forbidden"):
        validate_operator_view(view)


def test_operator_view_hash_detects_mutation():
    view = build_operator_view(
        _request(), _state(), OperatorViewCondition.LOCAL_KPI,
        {"workload": 0.0}, {"nodes": []},
    )
    view.payload["features"]["local_kpi_risk"] = 999.0
    with pytest.raises(ValueError, match="hash"):
        validate_operator_view(view)


def test_oracle_view_is_explicitly_privileged():
    view = build_operator_view(
        _request(), _state(), OperatorViewCondition.EVALUATOR_ORACLE,
        {"workload": 0.0}, {"nodes": []},
        oracle_payload={"raw_private_state": {"allowed": "oracle only"}, "true_disruption_label": "moderate"},
    )
    validate_operator_view(view)
    assert view.payload["provenance"]["information_boundary"] == "evaluator_global_oracle"


def test_attention_allocator_prioritizes_benefit_per_operator_minute():
    first = _request(incident_id="a", expected_benefit=2.0, estimated_operator_minutes=8.0)
    second = _request(incident_id="b", expected_benefit=3.0, estimated_operator_minutes=6.0)
    views = [
        build_operator_view(request, _state(), OperatorViewCondition.LOCAL_KPI, {}, {"nodes": []})
        for request in (first, second)
    ]
    ranked = AttentionAllocator("benefit_per_minute").rank(list(zip((first, second), views)))
    assert ranked[0][0].incident_id == "b"


def test_simulated_operator_enforces_slots_latency_workload_and_recovery():
    profile = replace(OPERATOR_PROFILES["high_accuracy_bounded"], slots=1, base_latency_steps=1, service_minutes=5.0)
    operator = SimulatedOperator(profile, AttentionAllocator("fcfs"), seed=3)
    request = _request(estimated_operator_minutes=5.0)
    view = build_operator_view(request, _state(), OperatorViewCondition.LOCAL_KPI, operator.workload_snapshot(), {"nodes": []})
    assert operator.enqueue(request, view)
    assert operator.step(5) == []
    assert len(operator.active) == 1
    workload = operator.workload
    assert operator.step(6) == []
    completed = operator.step(7)
    assert len(completed) == 1
    assert operator.workload < workload
    assert operator.operator_minutes == 5.0


def test_human_tool_registry_rejects_unknown_fields_and_bad_duration():
    registry = HumanOperatorToolRegistry()
    bad = registry.validate("authorize_emergency_route", {
        "source": "a", "target": "b", "duration": 99,
    })
    assert not bad.ok and bad.code == "above_maximum"
    unknown = registry.validate("teleport_material", {})
    assert not unknown.ok and unknown.code == "unknown_human_tool"


def test_v3_conservative_private_capacity_is_never_above_executable_state():
    env = HumanOversightEnvironment(_config())
    for agent_id in env.agent_ids:
        observation = env.private_observation(agent_id)
        assert observation.inventory <= env.states[agent_id].inventory + 1e-12
        assert observation.capacity <= env.states[agent_id].capacity + 1e-12


def test_emergency_resource_is_a_conserved_exogenous_inflow():
    env = HumanOversightEnvironment(_config())
    source = next(agent_id for agent_id in env.agent_ids if env.agents[agent_id].identity.role in SOURCE_ROLES)
    before_total = env.total_material()
    intervention = OperatorIntervention(
        "HI1", "HA1", source, 0, "approve_emergency_resource",
        "approve_emergency_resource", {"recipient": source, "quantity": 5.0},
        False, "view", 1.0, 5.0,
    )
    result = env.execute_human_intervention(intervention)
    assert result.ok
    assert env.total_material() == pytest.approx(before_total + 5.0)
    assert env.conservation_error() == pytest.approx(0.0)


def test_authorized_route_changes_feasibility_but_agent_executes_material_action():
    env = HumanOversightEnvironment(_config(horizon=12))
    env.transition()
    env.deliver_observations()
    source, target = sorted(env.initial_physical_edges)[0]
    env.physical_edges.discard((source, target))
    env.closed_physical_edges.add((source, target))
    quantity = min(1.0, env.states[source].inventory, env.states[source].capacity)
    denied = env.execute_tool(source, "schedule_shipment", {
        "target": target, "quantity": quantity, "arrival_step": 2,
    })
    assert denied.code == "no_route"
    intervention = OperatorIntervention(
        "HI2", "HA2", target, 0, "approve", "authorize_emergency_route",
        {"source": source, "target": target, "duration": 4}, False,
        "view", 1.0, 5.0,
    )
    assert env.execute_human_intervention(intervention).ok
    accepted, _ = env.directive_response(source)
    assert accepted in (True, False)  # private utility retains refusal authority
    result = env.execute_tool(source, "schedule_shipment", {
        "target": target, "quantity": quantity, "arrival_step": 2,
    })
    assert result.ok


def test_mandatory_override_and_return_control_record_autonomy_transitions():
    env = HumanOversightEnvironment(_config())
    source, target = sorted(env.initial_physical_edges)[0]
    override = OperatorIntervention(
        "HI3", "HA3", target, 0, "initiate_temporary_override",
        "temporary_emergency_override",
        {"source": source, "target": target, "quantity": 2.0, "duration": 3},
        True, "view", 2.0, 8.0,
    )
    assert env.execute_human_intervention(override).ok
    accepted, directive = env.directive_response(source)
    assert accepted and directive["mandatory"]
    assert env.autonomy_levels[source] == int(AutonomyLevel.EMERGENCY_OVERRIDE)
    returned = OperatorIntervention(
        "HI4", "HA3", source, 0, "return_control", "return_control",
        {"agent_id": source}, False, "view", 0.0, 1.0,
    )
    assert env.execute_human_intervention(returned).ok
    assert env.autonomy_levels[source] == int(AutonomyLevel.QUIET_DECENTRALIZED)


def test_counterfactual_snapshot_clones_all_state_and_rng_without_aliasing():
    env = HumanOversightEnvironment(_config())
    env.transition()
    env.deliver_observations()
    clone, digests = env.counterfactual_snapshot()
    assert clone.state_digest() == env.state_digest()
    assert set(digests) == {"initialization", "exogenous", "observation", "communication"}
    first = env.agent_ids[0]
    clone.states[first].inventory += 1.0
    assert clone.states[first].inventory != env.states[first].inventory


@pytest.mark.parametrize("application", ["commercial", "humanitarian"])
def test_mock_thermohitl_episode_executes_complete_material_chain(application):
    runner = HumanOperatorEpisodeRunner(
        _config(application=application, horizon=16),
        HumanMethod.THERMOHITL_RULE.value,
        thermodynamic_calibration=_fitted_nominal_calibration(application),
        enable_counterfactual_probes=True,
    )
    result = runner.run("chain-%s" % application)
    assert result.completion_status == "complete"
    assert result.metrics["operator_requests"] > 0
    assert result.metrics["operator_interventions"] > 0
    assert result.metrics["material_actions_accepted"] > 0
    assert result.metrics["material_actions_reached_demand"] > 0
    assert result.metrics["counterfactual_interventions"] > 0
    assert any(row["primary_outcome_changed"] for row in result.counterfactuals)
    assert abs(result.metrics["conservation_error"]) < 1e-8


def test_human_episode_is_deterministic_with_mock_planner():
    outputs = []
    digests = []
    for _ in range(2):
        runner = HumanOperatorEpisodeRunner(
            _config(horizon=10), HumanMethod.THERMOHITL_RULE.value,
            enable_counterfactual_probes=False,
        )
        outputs.append(runner.run("deterministic").metrics)
        digests.append(runner.env.ledger.digest())
    assert outputs[0] == outputs[1]
    assert digests[0] == digests[1]


def test_no_human_baseline_cannot_enqueue_or_apply_operator_action():
    runner = HumanOperatorEpisodeRunner(
        _config(horizon=8), HumanMethod.AUTONOMOUS_NO_HUMAN.value,
        enable_counterfactual_probes=False,
    )
    result = runner.run("no-human")
    assert result.metrics["operator_requests"] == 0
    assert result.metrics["operator_interventions"] == 0
    assert not any(event.kind == "operator_action" for event in runner.env.ledger.events)


class _ScriptedTransformersPlanner(TransformersPlanner):
    def __init__(self, batches):
        self.registry = ToolRegistry()
        self.batches = list(batches)
        self.max_input_tokens = 256
        self.max_new_tokens = 64

    def _prompt(self, request):
        return "private-agent-prompt-" + request.agent_id

    def _generate_prompts(self, prompts):
        values = self.batches.pop(0)
        assert len(values) == len(prompts)
        return values, [10] * len(values), [5] * len(values), 0.2


def _repair_request():
    return PlannerRequest(
        agent_id="supplier_01",
        role="supplier",
        application="commercial",
        option=0,
        context={
            "observation": {"step": 2, "delay": 0.0},
            "utility": {},
            "identity": {"agent_id": "supplier_01", "role": "supplier"},
            "messages": [],
            "commitments": [],
            "material_action_guidance": {},
        },
        candidate_agents=[],
    )


def test_transformers_planner_allows_exactly_one_validated_repair():
    repaired = (
        '{"plan_summary":"pause","tool":"no_op","arguments":{},'
        '"justification":"bounded repair","confidence":0.9}'
    )
    planner = _ScriptedTransformersPlanner([["not json"], [repaired]])
    response = planner.plan_batch([_repair_request()])[0]
    assert response.repair_attempted
    assert response.first_pass_valid is False
    assert response.valid_json
    assert response.recovery == "single_repair"
    assert response.output.tool == "no_op"
    assert response.prompt_tokens == 20
    assert response.generated_tokens == 10
    assert not planner.batches


def test_transformers_planner_stops_after_failed_single_repair():
    planner = _ScriptedTransformersPlanner([
        ["not json"], ["still not json"],
    ])
    response = planner.plan_batch([_repair_request()])[0]
    assert response.repair_attempted
    assert response.first_pass_valid is False
    assert not response.valid_json
    assert response.recovery == "safe_no_op_after_single_repair"
    assert response.output.tool == "no_op"
    assert not planner.batches


def test_qwen_repair_is_encoded_as_a_new_user_turn():
    original = "<|im_start|>system\nrules<|im_end|>\n<|im_start|>assistant\n"
    repaired = TransformersPlanner._repair_prompt(
        original,
        '{"tool":"invented"}',
        "typed_schema:unknown_tool",
    )
    assert '{"tool":"invented"}<|im_end|>\n<|im_start|>user\n' in repaired
    assert "exact_allowed_tool_names" in repaired
    assert repaired.endswith("<|im_start|>assistant\n")


def test_actionability_threshold_is_role_local_and_not_a_central_alarm():
    controller = IndependentEscalationController(
        ["supplier_01", "supplier_02"],
        EscalationConfig(
            tau_on=1.5,
            tau_off=0.6,
            actionable_tau_on=1.1,
            minimum_dwell=1,
            cooldown=0,
        ),
    )
    actionable = _state(
        agent_id="supplier_01",
        role="supplier",
        local_energy_residual=4.0,
        energy_residual=0.0,
        entropy_residual=0.0,
        entropy_slope=0.0,
        disagreement=0.0,
        local_disruption_risk=0.0,
        local_kpi_risk=0.0,
        actionability_evidence=1.0,
    )
    request, _, activated, _ = controller.should_request(
        "supplier_01", HumanMethod.THERMOHITL_RULE,
        actionable, 0.0, np.random.RandomState(1),
    )
    assert request and activated
    assert not controller.states["supplier_02"].active
    unactionable = replace(
        actionable,
        agent_id="supplier_02",
        actionability_evidence=0.0,
    )
    request, _, activated, _ = controller.should_request(
        "supplier_02", HumanMethod.THERMOHITL_RULE,
        unactionable, 0.0, np.random.RandomState(1),
    )
    assert not request and not activated
