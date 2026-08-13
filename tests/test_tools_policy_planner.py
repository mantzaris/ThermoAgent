import json

import numpy as np
import pytest

from thermoagent.planners import (
    PlannerRequest,
    coerce_plan,
    extract_json_object,
    request_affordances,
    validate_request_plan,
)
from thermoagent.policy import CoordinationPolicy, OBSERVATION_DIM
from thermoagent.tools import ToolRegistry
from thermoagent.types import PlanOutput


def test_tool_types_bounds_and_role_permissions():
    registry = ToolRegistry()
    plan = PlanOutput("ship", "schedule_shipment", {"target": "r", "quantity": 2.0, "arrival_step": 3}, "valid")
    assert registry.validate("supplier", plan).ok
    assert registry.validate("retailer", plan).code == "permission_denied"
    bad = PlanOutput("ship", "schedule_shipment", {"target": "r", "quantity": -1.0, "arrival_step": 3}, "bad")
    assert registry.validate("supplier", bad).code == "below_minimum"
    extra = PlanOutput("ship", "schedule_shipment", {"target": "r", "quantity": 1.0, "arrival_step": 3, "secret": True}, "bad")
    assert registry.validate("supplier", extra).code == "unknown_field"


def test_structured_output_extraction_and_recovery_inputs():
    raw = "prefix ```json\n{\"plan_summary\":\"x\",\"tool\":\"no_op\",\"arguments\":{},\"justification\":\"y\"}\n``` suffix"
    value = extract_json_object(raw)
    assert value is not None
    plan = coerce_plan(value)
    assert plan.tool == "no_op"
    assert extract_json_object("not json") is None
    with pytest.raises(ValueError):
        coerce_plan({"tool": "no_op"})


def test_private_offer_affordance_uses_only_local_utility_and_commitment():
    context = {
        "utility": {"reservation_price": 1.0},
        "commitments": [{
            "commitment_id": "C_PRIVATE", "proposer": "seller", "partner": "buyer",
            "quantity": 4.0, "unit_price": 4.0, "due_step": 4, "status": "proposed",
        }],
        "messages": [],
    }
    request = PlannerRequest("buyer", "retailer", "commercial", 4, context, [])
    tools, guidance = request_affordances(request)
    assert guidance["private_offer_rule"]["required_tool"] == "reject_offer"
    assert "accept_offer" in tools
    inconsistent = PlanOutput("accept", "accept_offer", {"commitment_id": "C_PRIVATE"}, "mistake")
    assert validate_request_plan(request, inconsistent).code == "private_utility_constraint"
    consistent = PlanOutput("reject", "reject_offer", {"commitment_id": "C_PRIVATE", "reason": "too expensive"}, "private utility")
    assert validate_request_plan(request, consistent) is None


def test_resource_owner_uses_private_cost_floor_for_buyer_counter():
    context = {
        "utility": {"reservation_price": 9.0},
        "observation": {"private_cost": 3.0},
        "commitments": [{
            "commitment_id": "C_COUNTER", "proposer": "buyer", "partner": "seller",
            "resource_owner": "seller", "resource_recipient": "buyer",
            "quantity": 4.0, "unit_price": 2.5, "due_step": 4, "status": "proposed",
        }],
        "messages": [],
    }
    request = PlannerRequest("seller", "supplier", "commercial", 4, context, [])
    _, guidance = request_affordances(request)
    rule = guidance["private_offer_rule"]
    assert rule["decision_side"] == "resource_owner"
    assert rule["required_tool"] == "counter_offer"
    too_low = PlanOutput(
        "counter", "counter_offer",
        {"commitment_id": "C_COUNTER", "quantity": 4.0, "unit_price": 2.9, "due_step": 5},
        "mistake",
    )
    assert validate_request_plan(request, too_low).code == "private_counter_constraint"
    viable = PlanOutput(
        "counter", "counter_offer",
        {"commitment_id": "C_COUNTER", "quantity": 4.0, "unit_price": 3.0, "due_step": 5},
        "cost floor",
    )
    assert validate_request_plan(request, viable) is None


def test_negotiation_terminates_after_two_incompatible_counter_rounds():
    context = {
        "utility": {"reservation_price": 1.0},
        "observation": {"private_cost": 0.8},
        "commitments": [{
            "commitment_id": "C_FINAL", "proposer": "seller", "partner": "buyer",
            "resource_owner": "seller", "resource_recipient": "buyer",
            "quantity": 2.0, "unit_price": 1.2, "due_step": 5,
            "status": "proposed", "negotiation_round": 2,
        }],
        "messages": [],
    }
    request = PlannerRequest("buyer", "retailer", "commercial", 4, context, [])
    assert request_affordances(request)[1]["private_offer_rule"]["required_tool"] == "reject_offer"


def test_coalition_affordances_do_not_allow_invented_coalition_ids():
    context = {
        "utility": {"reservation_price": 1.0}, "commitments": [],
        "messages": [], "observation": {"step": 7},
    }
    request = PlannerRequest(
        "warehouse", "warehouse", "commercial", 5, context,
        [
            {"agent_id": "warehouse", "role": "warehouse"},
            {"agent_id": "supplier", "role": "supplier"},
            {"agent_id": "carrier", "role": "carrier"},
        ],
    )
    tools, guidance = request_affordances(request)
    assert tools == {"propose_coalition"}
    state = guidance["coalition_state"]
    assert "do not invent" in state["instruction"]
    assert state["minimum_expires_step"] == 9
    assert state["recommended_expires_step"] == 11
    assert state["exact_argument_names"] == ["members", "purpose", "expires_step"]
    assert state["proposer_already_member"] == "warehouse"
    assert state["eligible_invitee_ids"] == ["carrier", "supplier"]
    assert "never include your own" in state["instruction"]


def test_nonproposer_member_has_explicit_withdrawal_affordance():
    context = {
        "utility": {"reservation_price": 1.0}, "commitments": [],
        "messages": [], "observation": {"step": 7},
        "coalitions": {
            "K_PRIVATE": {
                "status": "member", "expires_step": 11,
                "proposer": "supplier", "purpose": "temporary recovery",
            }
        },
    }
    request = PlannerRequest("warehouse", "warehouse", "commercial", 5, context, [])
    tools, guidance = request_affordances(request)
    assert tools == {"withdraw_coalition"}
    assert guidance["coalition_state"]["coalition_id"] == "K_PRIVATE"
    wrong = PlanOutput(
        "withdraw", "withdraw_coalition",
        {"coalition_id": "K_INVENTED", "reason": "local choice"}, "independent",
    )
    assert validate_request_plan(request, wrong).code == "private_coalition_mismatch"


def test_route_execution_probe_uses_local_plan_not_coalition_reallocation():
    context = {
        "utility": {"reservation_price": 1.0},
        "commitments": [],
        "messages": [{"kind": "need", "sender": "retailer", "payload": {"quantity": 3.0}}],
        "observation": {"step": 1, "inventory": 10.0, "capacity": 5.0},
    }
    local_request = PlannerRequest("supplier", "supplier", "commercial", 0, context, [])
    local_tools, local_guidance = request_affordances(local_request)
    assert "schedule_shipment" in local_tools
    assert "coalition_state" not in local_guidance

    reallocation_request = PlannerRequest("supplier", "supplier", "commercial", 6, context, [])
    reallocation_tools, reallocation_guidance = request_affordances(reallocation_request)
    assert reallocation_tools == {"propose_coalition"}
    assert "coalition_state" in reallocation_guidance


def test_mock_central_coordinator_uses_only_coarse_reported_parties():
    from thermoagent.planners import MockPlanner

    context = {
        "observation": {"step": 2}, "messages": [], "commitments": [],
    }
    candidates = [
        {"agent_id": "source", "role": "supplier", "shared_operational_state": {"inventory": "high", "backlog": "low"}},
        {"agent_id": "demand", "role": "retailer", "shared_operational_state": {"inventory": "low", "backlog": "high"}},
    ]
    plan = MockPlanner().plan(PlannerRequest("coordinator", "coordinator", "commercial", 7, context, candidates))
    assert plan.tool == "central_dispatch"
    assert plan.arguments["source"] == "source" and plan.arguments["target"] == "demand"


def test_execution_actor_rejects_global_critic_features():
    policy = CoordinationPolicy(seed=2)
    local = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    action, _, _ = policy.act(local, deterministic=True)
    assert 0 <= action < 9
    with pytest.raises(ValueError):
        policy.act(np.zeros(OBSERVATION_DIM + 5, dtype=np.float32))
    mask = np.zeros(9, dtype=bool)
    mask[8] = True
    assert policy.act(local, deterministic=True, action_mask=mask)[0] == 8
    with pytest.raises(ValueError):
        policy.act(local, action_mask=np.zeros(9, dtype=bool))


def test_ppo_updates_and_checkpoint_roundtrip(tmp_path):
    policy = CoordinationPolicy(seed=3)
    rows = []
    for index in range(32):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[index % OBSERVATION_DIM] = 1.0
        action, logp, value = policy.act(obs)
        rows.append({"observation": obs, "action": action, "log_probability": logp, "value": value, "reward": float(index % 3 - 1), "done": index in (15, 31), "trajectory_id": "a" if index < 16 else "b"})
    losses = policy.update(rows)
    assert all(np.isfinite(value) for value in losses.values())
    path = tmp_path / "policy.pt"
    policy.save(path, {"training_seed": 3})
    restored = CoordinationPolicy.load(path)
    assert policy.act(np.zeros(OBSERVATION_DIM), deterministic=True)[0] == restored.act(np.zeros(OBSERVATION_DIM), deterministic=True)[0]


def test_behavior_cloning_uses_only_local_observation_and_valid_mask():
    policy = CoordinationPolicy(seed=9)
    rows = []
    for index in range(32):
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[index % OBSERVATION_DIM] = 1.0
        rows.append({
            "observation": observation,
            "action": 8,
            "action_mask": [False] * 8 + [True],
        })
    summary = policy.behavior_clone(rows, epochs=2, batch_size=16)
    assert summary["rows"] == 32
    assert policy.act(np.zeros(OBSERVATION_DIM), deterministic=True, action_mask=np.asarray(rows[0]["action_mask"]))[0] == 8
