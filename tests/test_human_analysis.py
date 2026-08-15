import json
from pathlib import Path

import pandas as pd
import pytest

from thermoagent.human_analysis import _paired_bootstrap, build_index
from thermoagent.human_cli import _guarded_stage, _real_profile
from thermoagent.human_figures import _blocked


def test_hierarchical_input_bootstrap_is_fixed_seed_deterministic():
    frame = pd.DataFrame([
        {
            "application": application,
            "scenario": f"human-v3-{regime}-{seed}",
            "environment_seed": seed,
            "method": method,
            "primary_outcome": value,
        }
        for application in ("commercial", "humanitarian")
        for regime in ("moderate", "correlated", "compound")
        for seed in (1, 2)
        for method, value in (("reference", 10.0 + seed), ("treatment", 9.0 + seed))
    ])
    first = _paired_bootstrap(frame, "reference", "treatment", "comparison")
    second = _paired_bootstrap(frame, "reference", "treatment", "comparison")
    assert first == second
    assert len(first) == 8
    assert all(row["bootstrap_replicates"] == 10_000 for row in first)


def test_fail_closed_stage_guard_rejects_incomplete_gates(tmp_path: Path):
    gate_path = tmp_path / "development" / "gate_status.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps({"holdout_unlocked": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="holdout is locked"):
        _guarded_stage(tmp_path, "holdout")


def test_real_llm_projection_remains_below_cap_and_records_zero_large_stages(tmp_path: Path):
    record = _real_profile(tmp_path, episodes=4)
    assert record["projected_single_gpu_hours"] < record["cap_single_gpu_hours"]
    assert record["training_validation_holdout_hours"] == 0.0
    stored = json.loads(
        (tmp_path / "reproducibility" / "v3_real_llm_prelaunch_projection.json").read_text()
    )
    assert stored["planned_episodes"] == 4


def test_blocked_figure_is_explicit_non_result_and_indexed(tmp_path: Path):
    name = _blocked("training_seed_curves", tmp_path, "Training", "Gate 5 failed")
    assert name == "reproducibility/not_run_figures/training_seed_curves.pdf"
    pdf = tmp_path / name
    assert pdf.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "figures" / "pdf" / "training_seed_curves.pdf").exists()
    index = build_index(tmp_path)
    indexed = pd.read_csv(index)
    assert any(indexed.artifact_path.str.endswith(name))
