import json
import shutil
from pathlib import Path

import pandas as pd

from thermoagent.statmech_llm.discovery.provider import KineticIsingProvider
from thermoagent.statmech_llm.corrected_quench.analysis import analyze_formal
from thermoagent.statmech_llm.corrected_quench.experiment import formal_panel_design, graph_for_panel, panel_seed
from thermoagent.statmech_llm.corrected_quench.simulation import run_corrected_quench_trajectory
from thermoagent.statmech_llm.corrected_quench.workflow import load_yaml


ROOT = Path(__file__).resolve().parents[3]


def test_cpu_surrogate_end_to_end_analysis(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT", str(artifacts))
    for relative in (
        "configs/statmech_llm/corrected_quench",
        "results/JSTAT/stages/discovery/statistics",
        "results/JSTAT/stages/discovery/tables",
        "results/JSTAT/stages/replication/tables",
    ):
        shutil.copytree(ROOT / relative, repository / relative)
    protocol = load_yaml(repository / "configs/statmech_llm/corrected_quench/protocol_template.yaml")
    protocol["status"] = "cpu_test_fixture"
    protocol["provenance"]["execution_source_sha256"] = "0" * 64
    protocol["engineering_pilot"] = {
        "provider_environment": {
            "accounting": {"latency_seconds": 0.0, "model_calls": 0, "prompt_tokens": 0, "generated_tokens": 0}
        }
    }
    frozen = repository / "configs/statmech_llm/corrected_quench/protocol.yaml"
    import yaml

    frozen.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    audit_path = repository / "configs/statmech_llm/corrected_quench/scientific_audit.yaml"
    audit = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    audit["permutation_analysis"]["replicates"] = 4
    audit["information_bias_audit"]["null_replicates_per_window"] = 3
    audit_path.write_text(yaml.safe_dump(audit, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("THERMOAGENT_CORRECTED_QUENCH_PERMUTATION_WORKERS", "1")
    panel_root = artifacts / "formal/panels"
    panel_root.mkdir(parents=True)
    for panel in formal_panel_design(protocol):
        rows = run_corrected_quench_trajectory(
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
    assert primary["all_rolling_window_rows"] == 3 * primary["macrostate_rows"]
    assert not primary["confirmatory_dispositions"]["H3"]["inferential_support"]
    robustness = pd.read_csv(
        repository / "results/JSTAT/stages/corrected_quench/tables/macrostate_distance_robustness.csv"
    )
    assert set(robustness["rolling_window_sweeps"]) == {3, 5, 7}
    assert set(
        robustness.loc[robustness["deleted_observable"] != "none", "deleted_observable"]
    ) == set(primary_feature for primary_feature in (
        "belief_magnetization",
        "action_magnetization",
        "belief_action_overlap",
        "reference_energy_per_agent",
        "energy_variance",
        "configuration_entropy",
        "entropy_rate",
        "total_correlation",
        "pairwise_mutual_information",
        "edge_mutual_information",
        "belief_susceptibility",
        "spatial_belief_correlation",
        "belief_disagreement",
    ))
