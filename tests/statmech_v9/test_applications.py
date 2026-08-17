import numpy as np

from thermoagent.statmech.driven import run_application_mapping, workload_conservation_residual


def test_application_mappings_are_distinct_and_reproducible():
    humanitarian, _ = run_application_mapping("humanitarian", 20, 11, horizon=45)
    utility, _ = run_application_mapping("utility", 20, 11, horizon=45)
    humanitarian_again, _ = run_application_mapping("humanitarian", 20, 11, horizon=45)
    assert humanitarian == humanitarian_again
    assert humanitarian != utility
    assert humanitarian[-1]["application_code"] == 0.0
    assert utility[-1]["application_code"] == 1.0


def test_driven_outputs_are_finite_and_show_drive():
    for application in ("humanitarian", "utility"):
        rows, _ = run_application_mapping(application, 20, 17, horizon=50)
        assert any(row["drive_active"] == 1.0 for row in rows)
        assert any(row["partition_active"] == 1.0 for row in rows)
        assert max(row["workload_density"] for row in rows) > 0.0
        assert all(np.isfinite(value) for row in rows for value in row.values())


def test_application_workload_and_resources_are_conserved_independently():
    for application in ("humanitarian", "utility"):
        _, accounting = run_application_mapping(application, 20, 29, horizon=50)
        assert abs(float(accounting["workload_residual"][0])) < 1e-10
        assert abs(float(accounting["resource_residual"][0])) < 1e-10


def test_deliberate_conservation_corruption_is_detected():
    residual = workload_conservation_residual(0.0, 10.0, 2.0, 4.0, 7.5)
    assert abs(residual) > 0.1
