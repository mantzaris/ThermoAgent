import json
from pathlib import Path

import pytest

from thermoagent.statmech_llm.discovery.core import decision_schema
from thermoagent.statmech_llm.provider import (
    MODEL_SPECS,
    TransformersStatmechProvider,
    schema_checksum,
)
from thermoagent.statmech_llm.experiment import (
    _assert_next_panel_within_compute_budget,
    _record_pilot_failure,
)
from thermoagent.statmech_llm.workflow import artifact_root, execution_source_checksum, sha256_json


ROOT = Path(__file__).resolve().parents[3]
def test_cross_model_study_uses_semantic_namespace():
    assert (ROOT / "thermoagent/statmech_llm").is_dir()
    assert (ROOT / "configs/statmech_llm/cross_model/protocol.yaml").is_file()


def test_model_families_and_revisions_are_exactly_pinned():
    assert MODEL_SPECS["qwen"].revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert MODEL_SPECS["granite"].identifier == "ibm-granite/granite-3.3-8b-instruct"
    assert MODEL_SPECS["granite"].revision == "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
    assert len(schema_checksum()) == 64
    assert schema_checksum() == sha256_json(decision_schema())


def test_raw_provider_artifacts_cannot_be_written_inside_repository(tmp_path):
    with pytest.raises(ValueError):
        TransformersStatmechProvider(
            MODEL_SPECS["qwen"], ROOT / "results/JSTAT/stages/cross_model/raw", ROOT
        )
    provider = TransformersStatmechProvider(MODEL_SPECS["qwen"], tmp_path / "external", ROOT)
    assert provider.artifact_root == (tmp_path / "external").resolve()


def test_parser_requires_the_typed_scientific_decision_schema():
    valid = {
        "belief_choice": "amber",
        "action_choice": "cobalt",
        "confidence": 0.6,
        "commitment_status": "provisional",
        "memory_state": "stable",
        "outgoing_signal": "amber",
        "tool_action": "no_action",
        "reason_code": "persistence",
    }
    assert TransformersStatmechProvider._parse(json.dumps(valid)) == valid
    with pytest.raises((ValueError, TypeError)):
        TransformersStatmechProvider._parse('{"belief_choice":"amber"}')


def test_source_checksum_is_deterministic_and_excludes_frozen_protocol():
    first = execution_source_checksum(ROOT)
    second = execution_source_checksum(ROOT)
    assert first == second
    assert len(first) == 64


def test_source_checksum_excludes_root_level_bytecode(tmp_path):
    source = tmp_path / "thermoagent/statmech_llm/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "configs/statmech_llm/cross_model/settings.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("version: test\n", encoding="utf-8")
    tests = tmp_path / "tests/statmech_llm/cross_model/test_model.py"
    tests.parent.mkdir(parents=True)
    tests.write_text("def test_value(): pass\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run-formal-experiment.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    first = execution_source_checksum(tmp_path)
    (source.parent / "model.pyc").write_bytes(b"ignored bytecode")
    assert execution_source_checksum(tmp_path) == first
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert execution_source_checksum(tmp_path) != first


def test_external_artifact_root_rejects_repository(monkeypatch):
    monkeypatch.setenv("THERMOAGENT_ARTIFACT_ROOT", str(ROOT / "raw"))
    with pytest.raises(ValueError):
        artifact_root()


def test_engineering_failure_accounting_is_external_and_contains_no_prompt(monkeypatch, tmp_path):
    external = tmp_path / "external"
    monkeypatch.setenv("THERMOAGENT_ARTIFACT_ROOT", str(external))
    provider = TransformersStatmechProvider(
        MODEL_SPECS["granite"], external / "raw/pilot/granite", ROOT
    )
    _record_pilot_failure(
        "granite",
        provider,
        ValueError("hf_transfer optional backend unavailable"),
    )
    path = external / "pilot/granite_failures.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value[0]["classification"] == "missing_optional_huggingface_transfer_backend"
    assert value[0]["decision_requests"] == 0
    assert "prompt" not in value[0]


def test_atomic_panel_budget_check_uses_pilot_projection(tmp_path):
    protocol = {
        "compute": {
            "maximum_total_decisions": 100,
            "maximum_prompt_tokens": 1000,
            "hard_generation_gpu_hours": 1.0,
        },
        "engineering_pilot_results": {
            "qwen": {"mean_prompt_tokens": 11.0, "mean_latency_seconds_per_decision": 1.0}
        },
    }
    panel = {"model_key": "qwen", "n_agents": 10, "sweeps": 10}
    with pytest.raises(RuntimeError, match="prompt-token ceiling"):
        _assert_next_panel_within_compute_budget(tmp_path, panel, protocol)
