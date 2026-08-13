from pathlib import Path

import thermoagent.smoke as smoke_module
from thermoagent.planners import MockPlanner


def test_stage1_harness_requires_an_independent_coalition_join(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setattr(
        smoke_module,
        "TransformersPlanner",
        lambda *args, **kwargs: MockPlanner(),
    )
    result = smoke_module.agentic_smoke(tmp_path, "mock", "mock-revision")
    checks = result["checks"]
    assert checks["coalition_proposal_observed"]
    assert checks["independent_coalition_join_observed"]
    join_rows = [
        row for row in result["negotiation_and_revision"]
        if row["tool_result"]["code"] == "coalition_joined"
    ]
    assert len(join_rows) == 1
    assert "independent response" in join_rows[0]["label"]
