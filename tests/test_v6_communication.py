from pathlib import Path

from thermoagent.v6_communication import analyze_sketch_stage
from thermoagent.v6_experiments import aggregate_stage, run_matrix


def test_sketch_analysis_counts_thermodynamic_traffic_and_matches_panels(tmp_path):
    root = tmp_path / "results"
    run_matrix(
        Path.cwd(), root, "development_sketch_reference",
        ("humanitarian",), ("compound",), ("private_fragmented",),
        (66101, 66102, 66103, 66104, 66105), ("never_act",), (0.5,),
        ("event_triggered", "always_on"), 0,
    )
    report = analyze_sketch_stage(
        root / "development" / "sketch_reference",
        root / "development" / "communication",
    )
    policies = {value["sketch_policy"] for value in report["costs"]}
    assert policies == {"event_triggered", "always_on"}
    event = next(value for value in report["costs"] if value["sketch_policy"] == "event_triggered")
    always = next(value for value in report["costs"] if value["sketch_policy"] == "always_on")
    assert event["sketch_messages_mean"] < always["sketch_messages_mean"]
    assert report["safety"][0]["matched_panels"] == 5
