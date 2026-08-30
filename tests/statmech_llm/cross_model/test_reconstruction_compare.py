import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def _comparison_module():
    path = ROOT / "scripts/compare-reconstruction.py"
    specification = importlib.util.spec_from_file_location("v15_reconstruction_compare", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_reconstruction_comparison_excludes_only_declared_nonscientific_column():
    module = _comparison_module()
    reference = pd.DataFrame(
        {
            "panel_id": ["b", "a"],
            "effect": [1.0, 2.0],
            "latency_seconds": [100.0, 200.0],
        }
    )
    reconstructed = pd.DataFrame(
        {
            "panel_id": ["a", "b"],
            "effect": [2.0, 1.0],
            "latency_seconds": [8.0, 9.0],
        }
    )
    matched = module.compare_frames(
        reference, reconstructed, ("latency_seconds",), 1.0e-12
    )
    assert matched["status"] == "matched"
    reconstructed.loc[0, "effect"] = 3.0
    mismatch = module.compare_frames(
        reference, reconstructed, ("latency_seconds",), 1.0e-12
    )
    assert mismatch["status"] == "value_mismatch"
    assert mismatch["mismatch_cells_preview"][0]["column"] == "effect"


def test_reconstruction_comparison_treats_json_numbers_as_numbers():
    module = _comparison_module()
    reference = pd.DataFrame(
        {
            "hypothesis": ["H1"],
            "cluster_values": ['{"cluster": 42.26333930099211}'],
        }
    )
    reconstructed = pd.DataFrame(
        {
            "hypothesis": ["H1"],
            "cluster_values": ['{"cluster": 42.2633393009927}'],
        }
    )
    matched = module.compare_frames(reference, reconstructed, (), 1.0e-12)
    assert matched["status"] == "matched"
    reconstructed.loc[0, "cluster_values"] = '{"cluster": 42.26334}'
    mismatch = module.compare_frames(reference, reconstructed, (), 1.0e-12)
    assert mismatch["status"] == "value_mismatch"


def test_reconstruction_comparison_allows_only_machine_scale_relative_roundoff():
    module = _comparison_module()
    reference = pd.DataFrame(
        {"panel_id": ["panel"], "macrostate_distance": [150.1168274773606]}
    )
    reconstructed = pd.DataFrame(
        {"panel_id": ["panel"], "macrostate_distance": [150.1168274773622]}
    )
    matched = module.compare_frames(reference, reconstructed, (), 1.0e-12)
    assert matched["status"] == "matched"
    reconstructed.loc[0, "macrostate_distance"] += 1.0e-8
    mismatch = module.compare_frames(reference, reconstructed, (), 1.0e-12)
    assert mismatch["status"] == "value_mismatch"


def test_reconstruction_primary_json_keeps_tokens_and_effects_but_omits_latency():
    module = _comparison_module()
    model = {
        "status": "complete",
        "model_key": "qwen",
        "model_id": "model",
        "model_revision": "revision",
        "protocol_sha256": "protocol",
        "execution_source_sha256": "source",
        "planned_trajectories": 1,
        "completed_trajectories": 1,
        "planned_decisions": 2,
        "observed_decision_rows": 2,
        "model_calls": 2,
        "prompt_tokens": 20,
        "generated_tokens": 4,
        "invalid_after_repair": 0,
        "invalid_after_repair_fraction": 0.0,
        "latency_seconds": 99.0,
    }
    payload = {
        "confirmatory_dispositions": {"H1": {"estimate": 1.0, "supported": True}},
        "formal_trajectories": 1,
        "independent_clusters_per_model": 1,
        "model_keys": ["qwen"],
        "cluster_seed_audit_passed": True,
        "privacy_mutations": 0,
        "memory_control_audit": {"future_information_violations": 0},
        "nonfinite_primary_features": 0,
        "formal_completion": {
            "status": "complete",
            "dynamic_trajectories": 1,
            "observed_decision_rows": 2,
            "model_calls": 2,
            "prompt_tokens": 20,
            "generated_tokens": 4,
            "protocol_sha256": "protocol",
            "execution_source_sha256": "source",
            "generation_gpu_hours": 9.0,
            "models": [model],
        },
        "analysis_wall_seconds": 20.0,
    }
    reconstructed = module._selected_primary_science(payload)
    changed_latency = module._selected_primary_science(
        {
            **payload,
            "formal_completion": {
                **payload["formal_completion"],
                "generation_gpu_hours": 17.0,
                "models": [{**model, "latency_seconds": 42.0}],
            },
        }
    )
    assert module._compare_json_values(reconstructed, changed_latency, 1.0e-12) == []
    changed_tokens = {
        **changed_latency,
        "formal_completion": {
            **changed_latency["formal_completion"],
            "prompt_tokens": 21,
        },
    }
    mismatch = module._compare_json_values(reconstructed, changed_tokens, 1.0e-12)
    assert mismatch[0]["location"] == "$.formal_completion.prompt_tokens"


def test_reconstruction_primary_json_aligns_optional_post_audit_fields():
    module = _comparison_module()
    model = {
        "status": "complete",
        "model_key": "qwen",
        "model_id": "model",
        "model_revision": "revision",
        "protocol_sha256": "protocol",
        "execution_source_sha256": "source",
        "planned_trajectories": 1,
        "completed_trajectories": 1,
        "planned_decisions": 2,
        "observed_decision_rows": 2,
        "model_calls": 2,
        "prompt_tokens": 20,
        "generated_tokens": 4,
        "invalid_after_repair": 0,
        "invalid_after_repair_fraction": 0.0,
    }
    common = {
        "confirmatory_dispositions": {"H1": {"estimate": 1.0}},
        "formal_trajectories": 1,
        "independent_clusters_per_model": 1,
        "model_keys": ["qwen"],
        "privacy_mutations": 0,
        "nonfinite_primary_features": 0,
        "formal_completion": {
            "status": "complete",
            "dynamic_trajectories": 1,
            "observed_decision_rows": 2,
            "model_calls": 2,
            "prompt_tokens": 20,
            "generated_tokens": 4,
            "protocol_sha256": "protocol",
            "execution_source_sha256": "source",
            "models": [model],
        },
    }
    historical = dict(common)
    extended = {
        **common,
        "cluster_seed_audit_passed": True,
        "memory_control_audit": {"future_information_violations": 0},
    }
    left, right = module._aligned_primary_science(historical, extended)
    assert left == right

    reference_extended = {**extended, "cluster_seed_audit_passed": False}
    left, right = module._aligned_primary_science(reference_extended, extended)
    mismatch = module._compare_json_values(left, right, 1.0e-12)
    assert mismatch[0]["location"] == "$.cluster_seed_audit_passed"
