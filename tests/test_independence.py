import pytest

from thermoagent.agents import PrivacyViolation
from thermoagent.environment import LogisticsEnvironment, ScenarioConfig
from thermoagent.events import EventLedger
from thermoagent.planners import MockPlanner, PlannerRequest
from thermoagent.types import Commitment, EntropySummary, Message


def make_env():
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=17, n_agents=8))
    env.transition()
    env.deliver_observations()
    return env


def test_agent_cannot_inspect_another_private_observation_or_memory():
    env = make_env()
    first, second = env.agent_ids[:2]
    with pytest.raises(PrivacyViolation):
        env.agents[second].vault.observation(first)
    with pytest.raises(PrivacyViolation):
        env.agents[second].vault.working_memory(first)


def test_information_crosses_only_explicit_messages_or_public_events():
    env = make_env()
    first, second = env.agent_ids[:2]
    private_events = [e for e in env.ledger.events if e.kind == "observation_delivery" and e.private_to == first]
    assert private_events
    assert all(e not in env.ledger.visible_to(second) for e in private_events)
    message = Message("test", first, second, "summary", {"level": "high"}, 0, 0)
    env.agents[second].deliver_message(message)
    assert env.agents[second].inbox[-1].sender == first


def test_environment_logs_explicit_message_delivery_event():
    env = make_env()
    first, second = env.agent_ids[:2]
    assert env._send(first, second, "information_request", {"topic": "capacity"}).ok
    env.advance()
    env._deliver_messages()
    deliveries = [event for event in env.ledger.events if event.kind == "message_delivery"]
    assert deliveries and deliveries[-1].payload["recipient"] == second


def test_sender_outbox_retains_even_a_dropped_explicit_attempt():
    env = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=170, n_agents=8,
        communication="partition",
    ))
    first, second = env.agent_ids[0], env.agent_ids[-1]
    # Remove the link so delivery probability is deterministically zero.
    env.communication_edges.discard(tuple(sorted((first, second))))
    result = env._send(first, second, "information_request", {"topic": "capacity"})
    assert result.data["dropped"]
    assert env.agents[first].outbox[-1].recipient == second
    assert not env.agents[second].inbox


def test_information_regime_controls_only_explicit_public_summary():
    private = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=17, n_agents=8, private_information=1.0))
    shared = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=17, n_agents=8, private_information=0.0))
    assert all("shared_operational_state" not in row for row in private.public_identities())
    assert all("shared_operational_state" in row for row in shared.public_identities())
    assert all("marginal_cost" in row["shared_operational_state"] for row in shared.public_identities())
    with pytest.raises(PrivacyViolation):
        shared.agents[shared.agent_ids[1]].vault.observation(shared.agent_ids[0])


def test_shared_operational_information_is_an_explicit_public_signal_event():
    from thermoagent.runner import EpisodeRunner

    runner = EpisodeRunner(
        ScenarioConfig(
            application="commercial", seed=171, horizon=2, n_agents=8,
            private_information=0.0, decision_interval=1,
        ),
        "scripted_independent",
    )
    runner.run()
    signals = [
        event for event in runner.env.ledger.events
        if event.kind == "public_signal"
    ]
    assert signals
    assert all(event.private_to is None for event in signals)
    assert all(
        "shared_operational_state" in report
        for event in signals for report in event.payload["reports"]
    )


def test_identical_objective_regime_really_has_identical_utility_weights():
    env = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=18, n_agents=8, objective_misalignment=0.0
    ))
    utilities = [agent.utility.vector() + [agent.utility.reservation_price] for agent in env.agents.values()]
    assert all(values == utilities[0] for values in utilities[1:])


def test_offer_enters_recipient_ledger_only_after_explicit_delivery():
    env = make_env()
    proposer = next(a for a in env.agent_ids if env.agents[a].identity.role == "supplier")
    recipient = next(a for a in env.agent_ids if env.agents[a].identity.role == "retailer")
    result = env.execute_tool(proposer, "submit_offer", {
        "target": recipient, "quantity": 2.0,
        "unit_price": 1.0, "due_step": env.step_index + 3,
    })
    commitment_id = result.data["commitment_id"]
    assert commitment_id in env.agents[proposer].commitments
    assert commitment_id not in env.agents[recipient].commitments
    env.advance()
    env._deliver_messages()
    assert commitment_id in env.agents[recipient].commitments


def test_removing_agent_changes_negotiation_graph_and_targets():
    env = make_env()
    planner = MockPlanner()
    demand_id = next(a for a in env.agent_ids if env.agents[a].identity.role == "retailer")
    agent = env.agents[demand_id]
    context = agent.retrieve_context(0, env.ledger)
    candidates = env.public_identities()
    first = planner.plan(PlannerRequest(demand_id, "retailer", "commercial", 1, context, candidates))
    removed = first.arguments["target"]
    candidates_without = [row for row in candidates if row["agent_id"] != removed]
    second = planner.plan(PlannerRequest(demand_id, "retailer", "commercial", 1, context, candidates_without))
    assert first.arguments["target"] != second.arguments["target"]


def test_different_private_utilities_drive_different_offer_responses():
    env = make_env()
    planner = MockPlanner()
    ids = env.agent_ids[:2]
    commitment = Commitment("C_TEST", "external", ids[0], 5.0, 1.2, 4)
    for agent_id in ids:
        env.agents[agent_id].commitments["C_TEST"] = Commitment("C_TEST", "external", agent_id, 5.0, 1.2, 4)
    env.agents[ids[0]].utility.reservation_price = 1.5
    env.agents[ids[1]].utility.reservation_price = 0.6
    plans = []
    for agent_id in ids:
        agent = env.agents[agent_id]
        context = agent.retrieve_context(0, env.ledger)
        plans.append(planner.plan(PlannerRequest(agent_id, agent.identity.role, "commercial", 4, context, env.public_identities())))
    assert plans[0].tool == "accept_offer"
    assert plans[1].tool == "reject_offer"


def test_agent_can_reject_counter_and_revise_after_failure():
    env = make_env()
    recipient = env.agent_ids[1]
    proposer = env.agent_ids[0]
    commitment = Commitment("C_TEST", proposer, recipient, 5.0, 1.0, 4)
    env.commitments[commitment.commitment_id] = commitment
    for owner in (proposer, recipient):
        env.agents[owner].commitments[commitment.commitment_id] = Commitment(**commitment.__dict__)
    rejected = env.execute_tool(recipient, "reject_offer", {"commitment_id": "C_TEST", "reason": "private objective conflict"})
    assert rejected.ok
    second = Commitment("C_TEST2", proposer, recipient, 5.0, 1.0, 4)
    env.commitments[second.commitment_id] = second
    for owner in (proposer, recipient):
        env.agents[owner].commitments[second.commitment_id] = Commitment(**second.__dict__)
    countered = env.execute_tool(recipient, "counter_offer", {"commitment_id": "C_TEST2", "quantity": 4.0, "unit_price": 0.8, "due_step": 5})
    assert countered.ok
    agent = env.agents[recipient]
    agent.reflect(0, "invalid route", False, "no_route")
    agent.reflect(1, "request alternative", True, "sent")
    assert agent.last_tool_ok
    assert agent.vault.working_memory(recipient)["needs_revision"] is False


def test_counteroffer_preserves_resource_direction_and_can_be_fulfilled():
    env = make_env()
    seller = next(a for a in env.agent_ids if env.agents[a].identity.role == "supplier")
    buyer = next(a for a in env.agent_ids if env.agents[a].identity.role == "retailer")
    offered = env.execute_tool(seller, "submit_offer", {
        "target": buyer, "quantity": 1.0, "unit_price": 2.0,
        "due_step": env.step_index + 5,
    })
    original_id = offered.data["commitment_id"]
    env.advance()
    env._deliver_messages()
    countered = env.execute_tool(buyer, "counter_offer", {
        "commitment_id": original_id, "quantity": 1.0, "unit_price": 1.8,
        "due_step": env.step_index + 4,
    })
    counter_id = countered.data["commitment_id"]
    env.advance()
    env._deliver_messages()
    counter = env.commitments[counter_id]
    assert counter.proposer == buyer and counter.partner == seller
    assert counter.resource_owner == seller and counter.resource_recipient == buyer
    assert counter.parent_commitment_id == original_id
    assert counter.negotiation_round == 1
    accepted = env.execute_tool(seller, "accept_offer", {"commitment_id": counter_id})
    assert accepted.ok
    shipped = env.execute_tool(seller, "schedule_shipment", {
        "target": buyer, "quantity": 1.0, "arrival_step": env.step_index + 1,
    })
    assert shipped.ok
    assert env.commitments[counter_id].status == "in_transit"


def test_individually_rational_agreement_requires_both_private_parties():
    env = make_env()
    seller = next(a for a in env.agent_ids if env.agents[a].identity.role == "supplier")
    buyer = next(a for a in env.agent_ids if env.agents[a].identity.role == "retailer")
    env.states[seller].private_cost = 2.0
    env.agents[buyer].utility.reservation_price = 3.0
    one_sided = Commitment(
        "C_ONE_SIDED", seller, buyer, 1.0, 1.0, env.step_index + 3,
        resource_owner=seller, resource_recipient=buyer,
    )
    env.commitments[one_sided.commitment_id] = one_sided
    env.agents[buyer].commitments[one_sided.commitment_id] = Commitment(**one_sided.__dict__)
    accepted = env.execute_tool(buyer, "accept_offer", {
        "commitment_id": one_sided.commitment_id,
    })
    assert accepted.ok
    assert env.individually_rational_acceptances == 0

    env.states[seller].private_cost = 0.8
    two_sided = Commitment(
        "C_TWO_SIDED", seller, buyer, 1.0, 1.0, env.step_index + 3,
        resource_owner=seller, resource_recipient=buyer,
    )
    env.commitments[two_sided.commitment_id] = two_sided
    env.agents[buyer].commitments[two_sided.commitment_id] = Commitment(**two_sided.__dict__)
    accepted = env.execute_tool(buyer, "accept_offer", {
        "commitment_id": two_sided.commitment_id,
    })
    assert accepted.ok
    assert env.individually_rational_acceptances == 1


def test_joined_temporary_coalition_grants_bounded_recovery_route():
    env = make_env()
    seller = next(a for a in env.agent_ids if env.agents[a].identity.role == "supplier")
    buyer = next(a for a in env.agent_ids if env.agents[a].identity.role == "retailer")
    env.physical_edges.discard((seller, buyer))
    failed = env.execute_tool(seller, "schedule_shipment", {
        "target": buyer, "quantity": 1.0, "arrival_step": env.step_index + 1,
    })
    assert failed.code == "no_route"
    proposal = env.execute_tool(seller, "propose_coalition", {
        "members": [buyer], "purpose": "temporary recovery route",
        "expires_step": env.step_index + 4,
    })
    coalition_id = proposal.data["coalition_id"]
    joined = env.execute_tool(buyer, "join_coalition", {"coalition_id": coalition_id})
    assert joined.ok
    assert env.agents[buyer].coalition_ledger[coalition_id]["status"] == "member"
    recovered = env.execute_tool(seller, "schedule_shipment", {
        "target": buyer, "quantity": 1.0, "arrival_step": env.step_index + 1,
    })
    assert recovered.ok
    withdrawn = env.execute_tool(buyer, "withdraw_coalition", {
        "coalition_id": coalition_id, "reason": "local objective changed",
    })
    assert withdrawn.ok
    assert env.agents[buyer].coalition_ledger[coalition_id]["status"] == "withdrawn"
    assert not env._coalition_route_available(seller, buyer)
    env.step_index = env.coalitions[coalition_id].expires_step + 1
    assert not env._coalition_route_available(seller, buyer)


def test_coalition_validation_does_not_silently_rewrite_members_or_join_expired_contracts():
    env = make_env()
    proposer, invitee = env.agent_ids[:2]
    invalid = env.execute_tool(proposer, "propose_coalition", {
        "members": [invitee, "unknown_agent"],
        "purpose": "recovery", "expires_step": env.step_index + 4,
    })
    assert invalid.code == "invalid_member"
    assert not env.coalitions
    proposed = env.execute_tool(proposer, "propose_coalition", {
        "members": [invitee], "purpose": "recovery",
        "expires_step": env.step_index + 2,
    })
    coalition_id = proposed.data["coalition_id"]
    assert not env.formed_coalitions()
    env.step_index += 3
    expired = env.execute_tool(invitee, "join_coalition", {
        "coalition_id": coalition_id,
    })
    assert expired.code == "coalition_expired"
    assert not env.formed_coalitions()


def test_independent_recurrent_and_rng_state_objects():
    env = make_env()
    first, second = [env.agents[a] for a in env.agent_ids[:2]]
    assert first.policy_state is not second.policy_state
    draw_before = second.rng.rand()
    first.rng.rand(100)
    # Recreating the environment proves the second stream depends only on its seed.
    env2 = make_env()
    second2 = env2.agents[env2.agent_ids[1]]
    assert draw_before == second2.rng.rand()
