import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from thermoagent.statmech_llm_v12.provider import KineticIsingProvider
from thermoagent.statmech_llm_v15.analysis import analyze_formal
from thermoagent.statmech_llm_v15.experiment import formal_panel_design, graph_for_panel
from thermoagent.statmech_llm_v15.simulation import run_v15_trajectory
from thermoagent.statmech_llm_v15.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_small_cpu_formal_analysis_is_complete_and_leakage_free(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("THERMO_V15_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("THERMO_V15_ANALYSIS_WORKERS", "1")
    shutil.copytree(ROOT / "configs/statmech_v15", repository / "configs/statmech_v15")
    protocol = load_yaml(repository / "configs/statmech_v15/protocol_template.yaml")
    protocol["network"]["n_agents"] = 8
    protocol["network"]["clusters_per_model"] = 3
    protocol["trajectory"]["sweeps"] = 9
    protocol["trajectory"]["periods_sweeps"] = [3, 3, 3]
    protocol["analysis"]["information_null"]["replicates_per_window"] = 2
    protocol["analysis"]["irreversibility"]["time_shuffle_replicates_per_panel"] = 2
    protocol["analysis"]["recovery"]["early_sweeps"] = [7, 7]
    protocol["analysis"]["recovery"]["late_sweeps"] = [9, 9]
    protocol["provenance"]["execution_source_sha256"] = "0" * 64
    frozen = repository / "configs/statmech_v15/protocol_frozen.yaml"
    frozen.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    panel_root = artifacts / "formal/panels"
    panel_root.mkdir(parents=True)
    for panel in formal_panel_design(protocol):
        rows = run_v15_trajectory(
            KineticIsingProvider(),
            graph_for_panel(panel),
            int(panel["panel_seed"]),
            int(panel["sweeps"]),
            str(panel["condition"]),
            float(panel["coupling_strength"]),
            float(panel["sampling_temperature"]),
            panel["periods_sweeps"],
            metadata={
                "model_key": panel["model_key"],
                "model_id": panel["model_id"],
                "model_revision": panel["model_revision"],
                "cluster_id": panel["cluster_id"],
                "panel_id": panel["panel_id"],
                "protocol_sha256": "test",
                "execution_source_sha256": "0" * 64,
            },
            control_seed=int(panel["control_seed"]),
        )
        pd.DataFrame(rows).to_csv(panel_root / (str(panel["panel_id"]) + ".csv"), index=False)
    completion = {
        "status": "complete",
        "dynamic_trajectories": 24,
        "observed_decision_rows": 24 * 8 * 9,
        "model_calls": 24 * 8 * 9,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "generation_gpu_hours": 0.0,
        "execution_source_sha256": "0" * 64,
        "protocol_sha256": "test",
    }
    (artifacts / "formal/completion.json").write_text(json.dumps(completion), encoding="utf-8")
    primary = analyze_formal(repository)
    assert primary["formal_trajectories"] == 24
    assert primary["independent_clusters_per_model"] == 3
    assert primary["privacy_mutations"] == 0
    diagnostics = pd.read_csv(
        repository / "results/collective_agent_statmech_v15/tables/nominal_distance_diagnostics.csv"
    )
    assert set(diagnostics["held_out_cluster_excluded"]) == {True}
    for row in diagnostics.itertuples():
        assert str(row.held_out_cluster) not in json.loads(row.training_clusters)
    effects = pd.read_csv(
        repository / "results/collective_agent_statmech_v15/tables/hypothesis_effects.csv"
    )
    assert set(effects["hypothesis"]) == {"H1", "H2", "H3", "H4"}

