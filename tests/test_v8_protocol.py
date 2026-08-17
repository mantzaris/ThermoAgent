from collections import Counter

import hashlib
import json
from pathlib import Path
from typing import Tuple

import pytest

from thermoagent.v8_protocol import _panels, close_v8_development_no_go


def test_validation_and_holdout_panels_are_disjoint_and_balanced():
    validation = _panels("validation", 30)
    holdout = _panels("holdout", 40)
    assert len(validation) == 60
    assert len(holdout) == 80
    assert not (
        {value["environment_seed"] for value in validation}
        & {value["environment_seed"] for value in holdout}
    )
    assert Counter(value["application"] for value in validation) == {
        "humanitarian": 30, "utility_restoration": 30,
    }
    assert Counter(value["application"] for value in holdout) == {
        "humanitarian": 40, "utility_restoration": 40,
    }


def test_validation_uses_structurally_held_out_modular_family():
    validation = _panels("validation", 30)
    assert {value["topology_family"] for value in validation} == {"modular"}
    for application in ("humanitarian", "utility_restoration"):
        values = [value for value in validation if value["application"] == application]
        assert {value["complexity"] for value in values} == {"small", "medium", "large"}
        assert {value["fragmentation"] for value in values} == {"low", "medium", "high"}


def _no_go_repository(tmp_path: Path) -> Tuple[Path, Path]:
    repository = tmp_path / "repository"
    results = repository / "results" / "entropy_triggered_belief_monitoring_v8"
    for relative in (
        "configs/v8_hysteresis_repair_pilot.yaml",
        "configs/v8_hysteresis_repair_pilot_v2.yaml",
        "configs/v8_hysteresis_repair_pilot_v3.yaml",
        "notes/97_v8_hysteresis_repair_pilot_rule.md",
        "notes/98_v8_replacement_formal_development_rule.md",
        "notes/99_v8_hysteresis_repair_pilot_iteration_2.md",
        "notes/100_v8_hysteresis_state_machine_repair.md",
        "notes/101_v8_pilot_no_go_disposition.md",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("prospective\n", encoding="utf-8")
    for stage in ("training", "validation", "holdout"):
        path = results / stage / "NOT_RUN.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not run\n", encoding="utf-8")
    stop = {
        "formal_development_unlocked": False,
        "stop_reason": "fixed pilot gate failed",
    }
    path = results / "negative_results" / "v8_stop_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stop) + "\n", encoding="utf-8")
    table = results / "tables" / "trigger_feasibility.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("candidate,eligible\ngeneralized_011_u8,false\n", encoding="utf-8")
    return repository, results


def test_no_go_closure_seals_provenance_without_formal_freeze(tmp_path: Path):
    repository, results = _no_go_repository(tmp_path)
    manifest = close_v8_development_no_go(repository, results)
    protocol = results / "protocol" / "v8_development_protocol_no_go.json"
    assert manifest["development_protocol_sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()
    assert not (results / "protocol" / "v8_frozen_protocol.json").exists()
    assert not manifest["validation_manifest_created"]
    assert not manifest["holdout_manifest_created"]


def test_no_go_closure_refuses_locked_outcome_data(tmp_path: Path):
    repository, results = _no_go_repository(tmp_path)
    path = results / "validation" / "episode_summary.csv"
    path.write_text("outcome\n1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked V8 stage"):
        close_v8_development_no_go(repository, results)
