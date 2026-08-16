from thermoagent.v6_protocol import HOLDOUT_REGIMES, VALIDATION_REGIMES, _stage_rows


def test_sealed_stage_manifests_use_disjoint_namespaces_and_full_applications():
    validation = _stage_rows("validation", (67101, 67102), VALIDATION_REGIMES)
    holdout = _stage_rows("holdout", (68101, 68102), HOLDOUT_REGIMES)
    assert {value["environment_seed"] for value in validation}.isdisjoint(
        {value["environment_seed"] for value in holdout}
    )
    assert {value["application"] for value in validation} == {
        "commercial", "humanitarian", "utility_restoration",
    }
    assert all(value["status"] == "sealed_not_run" for value in holdout)
