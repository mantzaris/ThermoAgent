import json
import shutil
from pathlib import Path

import pandas as pd

from thermoagent.statmech_llm_v12.provider import KineticIsingProvider
from thermoagent.statmech_llm_v14.analysis import analyze_formal
from thermoagent.statmech_llm_v14.experiment import formal_panel_design, graph_for_panel, panel_seed
from thermoagent.statmech_llm_v14.figures import generate_figures
from thermoagent.statmech_llm_v14.reporting import build_results
from thermoagent.statmech_llm_v14.simulation import run_v14_trajectory
from thermoagent.statmech_llm_v14.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_cpu_surrogate_end_to_end_analysis_and_all_candidate_figures(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("THERMO_V14_ARTIFACT_ROOT", str(artifacts))
    for relative in (
        "configs/statmech_v14",
        "results/llm_agent_statmech_v12/statistics",
        "results/llm_agent_statmech_v12/tables",
        "results/collective_agent_statmech_v13/tables",
    ):
        shutil.copytree(ROOT / relative, repository / relative)
    protocol = load_yaml(repository / "configs/statmech_v14/protocol_template.yaml")
    protocol["status"] = "cpu_test_fixture"
    protocol["provenance"]["execution_source_sha256"] = "0" * 64
    protocol["engineering_pilot"] = {
        "provider_environment": {
            "accounting": {"latency_seconds": 0.0, "model_calls": 0, "prompt_tokens": 0, "generated_tokens": 0}
        }
    }
    frozen = repository / "configs/statmech_v14/protocol_frozen.yaml"
    import yaml

    frozen.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    panel_root = artifacts / "formal/panels"
    panel_root.mkdir(parents=True)
    for panel in formal_panel_design(protocol):
        rows = run_v14_trajectory(
            KineticIsingProvider(),
            graph_for_panel(panel),
            panel_seed(panel),
            int(panel["sweeps"]),
            float(panel["coupling_strength"]),
            float(panel["sampling_temperature"]),
            str(panel["disruption"]),
            panel["periods_sweeps"],
            metadata={key: panel[key] for key in ("family", "subset", "cluster_id", "panel_id", "burn_in_sweeps")},
        )
        pd.DataFrame(rows).to_csv(panel_root / (str(panel["panel_id"]) + ".csv"), index=False)
    completion = {
        "status": "complete",
        "dynamic_trajectories": 24,
        "observed_decision_rows": 17280,
        "planned_decisions": 17280,
        "model_calls": 17280,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "generation_latency_seconds": 0.0,
        "generation_gpu_hours": 0.0,
    }
    (artifacts / "formal/completion.json").write_text(json.dumps(completion), encoding="utf-8")
    primary = analyze_formal(repository)
    assert primary["formal_trajectories"] == 24
    figures = generate_figures(repository)
    assert figures["figure_count"] == 26
    assert figures["pdf_count"] == 26
    report = build_results(repository)
    assert report["repository_files"] > 26
    assert (repository / "results/collective_agent_statmech_v14/README.md").exists()
