import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from thermoagent.statmech_llm.agents import DecentralizedLLMNetwork, FunctionalProvider, make_agents
from thermoagent.statmech_llm.llm_experiments import (
    _dynamic_condition_rows,
    _kernel_rows,
    _message_pilot_design,
    _pilot_design,
    empirical_kernel_from_rows,
)
from thermoagent.statmech_llm.qwen import MODEL_ID, MODEL_REVISION, QwenStructuredProvider, prompt_schema_manifest


def valid_payload(side="plan_right"):
    return {
        "belief_choice": side,
        "belief_confidence": 0.7,
        "action_choice": side,
        "commitment_status": "retain",
        "outgoing_signal": "support_right" if side == "plan_right" else "support_left",
        "outgoing_message": "bounded local signal",
        "tool_action": "commit_plan_right" if side == "plan_right" else "commit_plan_left",
        "reason_code": "private_evidence",
    }


def test_pinned_model_and_prompt_schema_checksums_are_stable():
    manifest = prompt_schema_manifest(("one", "two"))
    assert manifest["model_id"] == MODEL_ID == "Qwen/Qwen2.5-7B-Instruct"
    assert manifest["model_revision"] == MODEL_REVISION
    assert manifest["schema_sha256"] == prompt_schema_manifest(("different",))["schema_sha256"]
    assert manifest["prompt_template_sha256"][0] == hashlib.sha256(b"one").hexdigest()


def test_pilot_crosses_every_evidence_field_with_both_option_orders():
    design = _pilot_design(120)
    cells = {}
    for field, order, _paraphrase, _replicate in design:
        cells[(field, order)] = cells.get((field, order), 0) + 1
    assert len(cells) == 12
    assert set(cells.values()) == {10}
    for field in {-1.0, -0.55, -0.15, 0.15, 0.55, 1.0}:
        assert cells[(field, ("plan_left", "plan_right"))] == 10
        assert cells[(field, ("plan_right", "plan_left"))] == 10


def test_message_pilot_is_balanced_across_applications_paraphrases_and_orders():
    design = _message_pilot_design(24)
    assert len(design) == 24
    cells = {}
    for application, paraphrase, order, _replicate in design:
        cells[(application, paraphrase, order)] = cells.get((application, paraphrase, order), 0) + 1
    assert len(cells) == 12
    assert set(cells.values()) == {2}


def test_qwen_provider_rejects_repository_raw_artifact_path(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ValueError):
        QwenStructuredProvider(repository / "raw", repository)


def test_qwen_provider_continues_external_call_indices_on_resume(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "call_00000007_prior.json").write_text("{}\n", encoding="utf-8")
    provider = QwenStructuredProvider(external, repository)
    assert provider._call_index == 7


def test_directed_influence_metadata_reaches_only_the_recipient():
    communication = np.zeros((3, 3))
    communication[1, 0] = 1.6
    communication[2, 0] = 0.4
    network = DecentralizedLLMNetwork(make_agents(3, 70), communication)
    network.offered_update(
        0,
        FunctionalProvider(lambda prompt, seed: valid_payload()),
        1,
        None,
        ("plan_left", "plan_right"),
        0,
    )
    assert network.private_agent_for_test(1)._inbox[0].influence_weight == 1.6
    assert network.private_agent_for_test(2)._inbox[0].influence_weight == 0.4


def test_empirical_controlled_kernel_is_stochastic_and_switches_are_agent_selected():
    rows = []
    n_agents = 2
    for state in range(16):
        for variable in range(4):
            rows.append(
                {
                    "alpha": 0.2,
                    "state_index": state,
                    "destination_state": state ^ (1 << variable),
                }
            )
    kernel = empirical_kernel_from_rows(rows, n_agents, 0.2, pseudocount=0.5)
    assert np.all(kernel >= 0.0)
    assert np.allclose(kernel.sum(axis=1), 1.0)
    assert np.count_nonzero(kernel[0]) == 5  # four one-variable switches plus retained state


def test_formal_inference_seeds_are_matched_across_nonreciprocity_conditions():
    provider = FunctionalProvider(lambda prompt, seed: valid_payload())
    kernel_settings = {
        "prompt_paraphrases": 3,
        "small_kernel": {
            "n_agents": 2,
            "alphas": [0.0, 0.2],
            "repeats_per_state_variable_condition": 1,
        },
    }
    kernel_rows = _kernel_rows(provider, kernel_settings)
    grouped = {}
    for row in kernel_rows:
        grouped.setdefault((row["state_index"], row["variable"], row["replicate"]), set()).add(row["seed"])
    assert grouped and all(len(seeds) == 1 for seeds in grouped.values())

    dynamic_settings = {
        "prompt_paraphrases": 3,
        "dynamic_networks": {
            "agent_counts": [4],
            "independent_panels": 1,
            "alphas": [0.0, 0.2],
            "turns_per_panel": 3,
        },
    }
    reciprocal = _dynamic_condition_rows(provider, dynamic_settings, 0, 0, 0)
    directed = _dynamic_condition_rows(provider, dynamic_settings, 0, 0, 1)
    assert [row["seed"] for row in reciprocal] == [row["seed"] for row in directed]
