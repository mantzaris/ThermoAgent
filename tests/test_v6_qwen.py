import json

from thermoagent.v6_environment import V6PanelEnvironment
from thermoagent.v6_qwen import _proposal, _validate, summarize_qwen_decisions


def test_qwen_validator_enforces_role_scope_and_delegation():
    environment = V6PanelEnvironment(
        "utility_restoration", "compound", "private_fragmented", 66890,
    )
    agent_id = sorted(environment.agents)[0]
    agent = environment.agents[agent_id]
    incident_id = agent.identity.incident_scope[0]
    action = environment.registry.allowed_actions(agent.identity.role)[0]
    raw = json.dumps({
        "action": action,
        "incident_id": incident_id,
        "quantity": 1.0,
        "reason_code": "private_evidence",
        "plan_summary": "bounded action",
        "confidence": 0.7,
        "delegation": "execute_autonomously",
    })
    call, delegation, confidence, error, _ = _validate(environment, agent_id, raw)
    assert call is not None and delegation == "execute_autonomously"
    assert confidence == 0.7 and error == ""
    invalid = raw.replace("execute_autonomously", "central_oracle")
    assert _validate(environment, agent_id, invalid)[0] is None


def test_qwen_proposal_uses_only_private_belief_and_typed_call():
    environment = V6PanelEnvironment(
        "humanitarian", "partition", "private_fragmented", 66891,
    )
    agent_id = sorted(environment.agents)[0]
    agent = environment.agents[agent_id]
    incident_id = agent.identity.incident_scope[0]
    action = next(
        value for value in environment.registry.allowed_actions(agent.identity.role)
        if value not in ("no_action", "defer")
    )
    from thermoagent.v6_types import V6ToolCall
    proposal = _proposal(
        environment, agent_id,
        V6ToolCall(action, incident_id, 1.0, reason_code="test"), 0.6,
    )
    assert proposal.agent_id == agent_id
    assert proposal.incident_id == incident_id
    assert 0.0 <= proposal.action_probability <= 1.0


def test_qwen_harm_accounting_uses_executed_physical_actions_only():
    rows = [
        {"accepted_physical_action": True, "harmful": True, "beneficial": False, "causal_effect": -0.2},
        {"accepted_physical_action": True, "harmful": False, "beneficial": True, "causal_effect": 0.4},
        {"accepted_physical_action": False, "harmful": True, "beneficial": False, "causal_effect": -9.0},
    ]
    result = summarize_qwen_decisions(rows)
    assert result["physical_action_acceptance"] == 2 / 3
    assert result["harmful_action_rate_among_physical"] == 0.5
    assert result["mean_causal_effect_among_physical"] == 0.1
    assert result["harmful_physical_actions"] == 1
