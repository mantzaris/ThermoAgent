import numpy as np

from thermoagent.v7_experiments import make_environment
from thermoagent.v7_learned_controller import FittedRiskController, context_row


class _SeverityModel:
    def predict_proba(self, frame):
        values = np.asarray(frame["severity"], dtype=float)
        return np.column_stack([1.0 - values, values])


def test_v7_fitted_controller_executes_exact_matched_epoch_coverage():
    environment = make_environment(
        "humanitarian", "small", "high", "high", "medium", "modular", 77123,
    )
    environment.advance_domain(0)
    environment.deliver_private_observations(0)
    contexts = []
    for agent in environment.agents.values():
        asset = sorted(agent.identity.asset_scope)[0]
        contexts.append(environment.risk_context(agent.propose(asset), 0))
    controller = FittedRiskController(
        _SeverityModel(), "crossfit_test", "modular", autonomous_coverage=0.5,
        operator_slots_per_epoch=0,
    )
    decisions = controller(contexts, 0)
    actionable = [value for value in contexts if value.proposal.is_physical]
    executed = sum(
        decisions["%s|%s" % (
            value.proposal.agent_id, value.proposal.target_asset_or_location,
        )] == "execute_autonomously"
        for value in actionable
    )
    assert executed == round(0.5 * len(actionable))
    row = context_row(contexts[0], "modular")
    assert "evaluator" not in repr(row)
    assert row["topology_family"] == "modular"
