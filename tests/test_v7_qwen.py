import json

from thermoagent.v7_experiments import make_environment
from thermoagent.v7_qwen import (
    authorized_payload, formal_specifications, validate_qwen_decision,
)


def _environment():
    environment = make_environment(
        "humanitarian", "small", "high", "high", "medium", "modular", 778001,
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    return environment


def test_v7_qwen_payload_contains_only_authorized_agent_view():
    environment = _environment()
    agent_id = sorted(environment.agents)[0]
    asset = environment.agents[agent_id].identity.asset_scope[0]
    payload = authorized_payload(environment, agent_id, asset)
    serialized = json.dumps(payload).lower()
    for prohibited in (
        "true_mode", "future", "counterfactual", "stochastic_tape",
        "evaluator", "correct_action",
    ):
        assert prohibited not in serialized


def test_v7_qwen_validator_enforces_four_separate_action_fields():
    environment = _environment()
    agent_id = sorted(environment.agents)[0]
    agent = environment.agents[agent_id]
    asset = agent.identity.asset_scope[0]
    observation = agent.vault.observation(agent_id, asset)
    action = observation.feasible_physical_actions[0]
    raw = json.dumps({
        "proposed_operational_action": action,
        "target_asset_or_location": asset,
        "quantity_or_capacity": 1.0,
        "information_action": "request_peer_evidence",
        "communication_action": "send_targeted_summary",
        "delegation_action": "escalate_operator",
        "confidence": 0.72,
        "reason_code": "conflicting_private_evidence",
        "compact_plan_summary": "Request evidence and escalate a bounded proposal.",
    })
    decision, error, _ = validate_qwen_decision(environment, agent_id, raw)
    assert error == ""
    assert decision is not None
    assert decision.proposal.proposed_operational_action != decision.information_action


def test_v7_qwen_validator_rejects_out_of_scope_target_and_invalid_action():
    environment = _environment()
    agent_id = sorted(environment.agents)[0]
    outside = next(
        asset for asset in environment.node_role
        if asset not in environment.agents[agent_id].identity.asset_scope
    )
    raw = json.dumps({
        "proposed_operational_action": "verify_observation",
        "target_asset_or_location": outside,
        "quantity_or_capacity": 1.0,
        "information_action": "no_information_action",
        "communication_action": "no_communication_action",
        "delegation_action": "execute_autonomously",
        "confidence": 0.8,
        "reason_code": "bad",
        "compact_plan_summary": "bad",
    })
    decision, error, _ = validate_qwen_decision(environment, agent_id, raw)
    assert decision is None
    assert error == "target_outside_private_scope"


def test_v7_formal_qwen_design_has_twenty_episodes_per_primary_application():
    values = formal_specifications()
    assert len(values) == 40
    assert sum(value[0] == "humanitarian" for value in values) == 20
    assert sum(value[0] == "utility_restoration" for value in values) == 20
    assert {value[7] for value in values} == {"private_fragmented", "public_shared"}
