from pathlib import Path

import pytest

from thermoagent.v7_protocol import development_manifest, freeze_protocol, sealed_manifest


def test_v7_development_manifest_has_frozen_response_surface_and_unique_panels():
    rows = development_manifest()
    assert len(rows) == 100
    assert len({row["panel_id"] for row in rows}) == 100
    for application in ("humanitarian", "utility_restoration"):
        subset = [row for row in rows if row["application"] == application]
        assert len(subset) == 50
        high = [
            row for row in subset
            if row["complexity"] == "medium"
            and row["coupling"] == "high"
            and row["fragmentation"] == "high"
            and row["information_condition"] == "private_fragmented"
        ]
        assert len(high) == 12


def test_v7_validation_and_holdout_manifests_use_distinct_seed_namespaces():
    validation = sealed_manifest("validation", 797000)
    holdout = sealed_manifest("holdout", 807000)
    assert len(validation) == 32
    assert len(holdout) == 40
    assert {row["environment_seed"] for row in validation}.isdisjoint(
        {row["environment_seed"] for row in holdout}
    )


def test_v7_freeze_refuses_without_passing_feasibility_gates(tmp_path):
    with pytest.raises(RuntimeError, match="feasibility gates"):
        freeze_protocol(tmp_path, tmp_path / "results")
