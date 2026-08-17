from pathlib import Path

import numpy as np

from thermoagent.v8_training import (
    V8RoleIPPOPolicy,
    V8Transition,
    grouped_terminal_gae,
    ppo_update,
)


def _transition(agent: str, step: int, reward: float) -> V8Transition:
    value = V8Transition(
        agent_id=agent,
        role="test_role",
        asset="asset",
        step=step,
        observation=np.linspace(0.0, 1.0, 33),
        mask=np.asarray([True, True, True, True]),
        action=step % 4,
        old_log_probability=float(np.log(0.25)),
        value=0.1,
        proposal_is_physical=True,
        reward=reward,
    )
    return value


def test_grouped_gae_respects_agent_and_terminal_boundaries():
    values = [
        _transition("a", 0, 0.2),
        _transition("b", 0, -0.1),
        _transition("a", 4, 0.5),
        _transition("b", 4, 0.3),
    ]
    grouped_terminal_gae(values, gamma=0.9, gae_lambda=0.8)
    assert values[2].terminal
    assert values[3].terminal
    assert not values[0].terminal
    assert not values[1].terminal
    # The final return in each independent trajectory cannot bootstrap from
    # the other agent's critic value.
    assert np.isclose(values[2].return_value, values[2].reward)
    assert np.isclose(values[3].return_value, values[3].reward)


def test_numpy_ippo_update_is_finite_and_checkpoint_roundtrips(tmp_path: Path):
    policy = V8RoleIPPOPolicy(seed=8801, stochastic=True)
    values = [_transition("a", step, 0.1 * (step - 1)) for step in range(4)]
    grouped_terminal_gae(values)
    before = policy._parameters("test_role").actor_weights.copy()
    diagnostics = ppo_update(policy, values, epochs=2)
    after = policy._parameters("test_role").actor_weights
    assert np.isfinite(list(diagnostics.values())).all()
    assert not np.array_equal(before, after)
    path = tmp_path / "policy.json.gz"
    first = policy.save(path, {"purpose": "unit_test"})
    loaded = V8RoleIPPOPolicy.load(path)
    assert first["sha256"] == loaded.checkpoint_digest
    assert np.allclose(
        policy._parameters("test_role").actor_weights,
        loaded._parameters("test_role").actor_weights,
    )


def test_masked_actions_receive_zero_probability():
    from thermoagent.v8_training import _softmax

    probabilities = _softmax(
        np.asarray([2.0, 3.0, 10.0, -1.0]),
        np.asarray([True, True, False, False]),
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[2] == 0.0
    assert probabilities[3] == 0.0


def test_deployable_features_ignore_evaluator_only_fields():
    from thermoagent.v7_experiments import make_environment
    from thermoagent.v8_training import deployable_feature_vector

    environment = make_environment(
        "humanitarian", "small", "medium", "high", "medium",
        "small_world", 889801, "private_fragmented", sketch_policy="none",
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    agent = environment.agents[sorted(environment.agents)[0]]
    asset = sorted(agent.private_beliefs)[0]
    proposal = agent.propose(asset).proposal
    base = {
        "distributed_pooled_belief": agent.private_beliefs[asset],
        "contributors": 1, "scoped_agents": 2, "maximum_age": 0,
        "distributed_disagreement": 0.1,
    }
    first = deployable_feature_vector(
        agent=agent, asset=asset, proposal=proposal,
        distributed_estimate={**base, "evaluator_global_disagreement": 0.0}, step=0,
    )
    second = deployable_feature_vector(
        agent=agent, asset=asset, proposal=proposal,
        distributed_estimate={**base, "evaluator_global_disagreement": 999.0}, step=0,
    )
    assert np.array_equal(first, second)


def test_same_frozen_policy_and_stochastic_tape_are_used_across_schedulers(tmp_path: Path):
    from thermoagent.v8_experiments import run_v8_episode
    from thermoagent.v8_trigger import TriggerConfig

    checkpoint = tmp_path / "frozen-policy.json.gz"
    source = V8RoleIPPOPolicy(seed=8899, stochastic=False)
    source.save(checkpoint, {"purpose": "matched_scheduler_unit_test"})
    policies = [
        V8RoleIPPOPolicy.load(checkpoint, stochastic=False),
        V8RoleIPPOPolicy.load(checkpoint, stochastic=False),
    ]
    common = {
        "application": "utility_restoration",
        "complexity": "small",
        "coupling": "high",
        "fragmentation": "high",
        "network_disruption": "medium",
        "topology_family": "grid",
        "environment_seed": 889902,
        "information_condition": "private_fragmented",
        "encoding": "uint8_simplex",
        "maximum_hops": 1,
        "results_root": None,
        "resume": False,
        "ledger_scope": "dynamic_delta",
    }
    always = run_v8_episode(
        **common, action_policy=policies[0],
        trigger_config=TriggerConfig(method="always_on"), stage="unit_always",
    )["summary"]
    none = run_v8_episode(
        **common, action_policy=policies[1],
        trigger_config=TriggerConfig(method="none"), stage="unit_none",
    )["summary"]
    assert always["stochastic_tape_digest"] == none["stochastic_tape_digest"]
    assert always["action_policy_id"] == none["action_policy_id"]


def test_v8_training_reserves_modular_topology_for_evaluation():
    from thermoagent.v8_training import _training_panel

    panels = [_training_panel(index, 88201) for index in range(18)]
    assert all(value["topology_family"] != "modular" for value in panels)
    assert {value["topology_family"] for value in panels if value["application"] == "humanitarian"} == {
        "random_geometric", "small_world",
    }
    assert {value["topology_family"] for value in panels if value["application"] == "utility_restoration"} == {
        "grid", "scale_free",
    }
