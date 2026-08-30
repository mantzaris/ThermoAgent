import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from thermoagent.statmech_llm.discovery.provider import KineticIsingProvider
from thermoagent.statmech_llm.analysis import analyze_formal
from thermoagent.statmech_llm.experiment import formal_panel_design, graph_for_panel
from thermoagent.statmech_llm.reporting import (
    _manifest,
    _pdf_fonts_embedded,
    _repository_files,
    record_manual_pdf_qa,
)
from thermoagent.statmech_llm.simulation import run_study_trajectory
from thermoagent.statmech_llm.workflow import load_yaml, sha256_file


ROOT = Path(__file__).resolve().parents[3]


def test_repository_inventory_excludes_latex_intermediates(tmp_path):
    paper = tmp_path / "paper/JSTAT"
    paper.mkdir(parents=True)
    (paper / "main.tex").write_text("paper source", encoding="utf-8")
    (paper / "main.pdf").write_bytes(b"paper")
    (paper / "main.bbl").write_text("generated bibliography", encoding="utf-8")
    (paper / "main.aux").write_text("generated auxiliary", encoding="utf-8")
    names = {path.name for path in _repository_files(tmp_path)}
    assert {"main.tex", "main.pdf"} <= names
    assert "main.bbl" not in names
    assert "main.aux" not in names


def test_repository_inventory_includes_pinned_runpod_requirements(tmp_path):
    requirements = tmp_path / "requirements-runpod.txt"
    requirements.write_text("transformers==4.55.4\n", encoding="utf-8")
    paths = {path.relative_to(tmp_path).as_posix() for path in _repository_files(tmp_path)}
    assert "requirements-runpod.txt" in paths


def test_final_verifier_runs_publication_and_pdf_checks():
    script = (ROOT / "scripts/verify-results.sh").read_text(encoding="utf-8")
    assert "verify-jstat-paper-assets.sh" in script
    assets = (ROOT / "scripts/verify-jstat-paper-assets.sh").read_text(
        encoding="utf-8"
    )
    assert "validate_publication" in assets


def test_verifier_reports_post_analysis_source_difference_without_claiming_equality():
    source = (ROOT / "thermoagent/statmech_llm/reporting.py").read_text(
        encoding="utf-8"
    )
    assert '"analysis_source_recorded": analysis_source_is_sha256' in source
    assert (
        '"analysis_source_matches_current_tree": recorded_analysis_source'
        in source
    )
    status_expression = '"status": "passed" if all(checks.values()) else "failed"'
    assert status_expression in source


def test_pdf_font_parser_reads_embedding_not_subset_or_unicode_columns():
    header = (
        "name type encoding emb sub uni object ID\n"
        "------------------------------------------\n"
    )
    embedded_not_subset = "ABCDEE+Font Type 1 Builtin yes no no 10 0\n"
    assert _pdf_fonts_embedded(header + embedded_not_subset)
    not_embedded = "Font TrueType WinAnsi no no yes 11 0\n"
    assert not _pdf_fonts_embedded(header + not_embedded)


def test_repository_manifest_excludes_only_self_referential_outputs(tmp_path):
    result = tmp_path / "results/JSTAT/stages/cross_model"
    reproducibility = result / "reproducibility"
    reproducibility.mkdir(parents=True)
    (reproducibility / "pdf_qa.csv").write_text("opens\nTrue\n", encoding="utf-8")
    (reproducibility / "pdf_qa_summary.json").write_text("{}\n", encoding="utf-8")
    (reproducibility / "verification.json").write_text("{}\n", encoding="utf-8")
    (result / "INDEX.csv").write_text("relative_path,bytes,sha256\n", encoding="utf-8")
    paths = set(_manifest(tmp_path)["relative_path"])
    assert "results/JSTAT/stages/cross_model/reproducibility/pdf_qa.csv" in paths
    assert (
        "results/JSTAT/stages/cross_model/reproducibility/pdf_qa_summary.json"
        in paths
    )
    assert "results/JSTAT/stages/cross_model/reproducibility/verification.json" not in paths
    assert "results/JSTAT/stages/cross_model/INDEX.csv" not in paths


def test_manual_pdf_qa_rechecks_digest_and_records_review(tmp_path):
    result = tmp_path / "results/JSTAT/stages/cross_model"
    reproducibility = result / "reproducibility"
    figure = result / "figures/pdf/figure.pdf"
    reproducibility.mkdir(parents=True)
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"fixed vector fixture")
    pd.DataFrame(
        [
            {
                "relative_path": figure.relative_to(tmp_path).as_posix(),
                "pages": 1,
                "opens": True,
                "fonts_embedded": True,
                "text_extractable": True,
                "rendered_pages": 1,
                "render_dpi": 300,
                "manual_visual_status": "pending",
                "sha256": sha256_file(figure),
            }
        ]
    ).to_csv(reproducibility / "pdf_qa.csv", index=False)
    (reproducibility / "pdf_qa_summary.json").write_text(
        json.dumps({"automated_passed": True, "manual_visual_status": "pending"}),
        encoding="utf-8",
    )
    summary = record_manual_pdf_qa(tmp_path, "passed", "fixture inspected")
    assert summary["manual_visual_status"] == "passed"
    recorded = pd.read_csv(reproducibility / "pdf_qa.csv")
    assert set(recorded["manual_visual_status"]) == {"passed"}
    assert set(recorded["manual_review_notes"]) == {"fixture inspected"}
    assert (result / "INDEX.csv").is_file()


@pytest.mark.skipif(
    not all(shutil.which(name) for name in ("latexmk", "pdffonts", "pdftotext")),
    reason="LaTeX and Poppler are required for manuscript QA",
)
def test_manuscript_compiles_in_disposable_tree_with_embedded_fonts(tmp_path):
    paper = tmp_path / "JSTAT"
    shutil.copytree(ROOT / "paper/JSTAT", paper)
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ],
        cwd=paper,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pdf = paper / "main.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert subprocess.check_output(["pdftotext", str(pdf), "-"], text=True).strip()
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    assert _pdf_fonts_embedded(fonts)


def test_small_cpu_formal_analysis_is_complete_and_leakage_free(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("THERMOAGENT_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("THERMOAGENT_ANALYSIS_WORKERS", "1")
    shutil.copytree(ROOT / "configs/statmech_llm/cross_model", repository / "configs/statmech_llm/cross_model")
    protocol = load_yaml(repository / "configs/statmech_llm/cross_model/protocol_template.yaml")
    protocol["network"]["n_agents"] = 8
    protocol["network"]["clusters_per_model"] = 3
    protocol["trajectory"]["sweeps"] = 9
    protocol["trajectory"]["periods_sweeps"] = [3, 3, 3]
    protocol["analysis"]["information_null"]["replicates_per_window"] = 2
    protocol["analysis"]["irreversibility"]["time_shuffle_replicates_per_panel"] = 2
    protocol["analysis"]["recovery"]["early_sweeps"] = [7, 7]
    protocol["analysis"]["recovery"]["late_sweeps"] = [9, 9]
    protocol["provenance"]["execution_source_sha256"] = "0" * 64
    frozen = repository / "configs/statmech_llm/cross_model/protocol.yaml"
    frozen.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    panel_root = artifacts / "formal/panels"
    panel_root.mkdir(parents=True)
    for panel in formal_panel_design(protocol):
        rows = run_study_trajectory(
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
    primary = analyze_formal(repository, allow_synthetic_raw_records=True)
    assert primary["formal_trajectories"] == 24
    assert primary["independent_clusters_per_model"] == 3
    assert primary["privacy_mutations"] == 0
    diagnostics = pd.read_csv(
        repository / "results/JSTAT/stages/cross_model/tables/nominal_distance_diagnostics.csv"
    )
    assert set(diagnostics["held_out_cluster_excluded"]) == {True}
    for row in diagnostics.itertuples():
        assert str(row.held_out_cluster) not in json.loads(row.training_clusters)
    effects = pd.read_csv(
        repository / "results/JSTAT/stages/cross_model/tables/hypothesis_effects.csv"
    )
    assert set(effects["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    extension_tables = (
        "connected_correlation_profiles.csv",
        "connected_correlation_matrix_means.csv",
        "autocorrelation_curves.csv",
        "integrated_autocorrelation.csv",
        "binder_cumulants.csv",
        "binder_cumulant_sensitivity.csv",
        "binder_cumulant_pooling_sensitivity.csv",
        "magnetization_distributions.csv",
        "collective_extension_contrasts.csv",
        "raw_generation_accounting.csv",
    )
    for name in extension_tables:
        path = repository / "results/JSTAT/stages/cross_model/tables" / name
        assert path.is_file()
        assert len(pd.read_csv(path)) > 0
    contrasts = pd.read_csv(
        repository
        / "results/JSTAT/stages/cross_model/tables/collective_extension_contrasts.csv"
    )
    assert len(contrasts) == 10
    assert set(contrasts["role"]) == {"secondary_descriptive_extension"}
    binder_sensitivity = pd.read_csv(
        repository
        / "results/JSTAT/stages/cross_model/tables/binder_cumulant_sensitivity.csv"
    )
    assert len(binder_sensitivity) == 24 * 3 * 3
    assert set(binder_sensitivity["temporal_window"]) == {
        "full_phase",
        "early_half",
        "late_half",
    }
    binder_pooling = pd.read_csv(
        repository
        / "results/JSTAT/stages/cross_model/tables/binder_cumulant_pooling_sensitivity.csv"
    )
    assert len(binder_pooling) == 2 * 4 * 3 * 3 * 2
    assert set(binder_pooling["pooling_rule"]) == {
        "mean_of_cluster_cumulants",
        "pooled_moments_across_clusters",
    }
    control_audit = pd.read_csv(
        repository
        / "results/JSTAT/stages/cross_model/tables/memory_control_panel_audit.csv"
    )
    assert len(control_audit) == 24
    assert set(control_audit["all_entries_reconstructed"]) == {True}
    assert int(control_audit["future_information_violations"].sum()) == 0
    control_balance = pd.read_csv(
        repository
        / "results/JSTAT/stages/cross_model/tables/memory_control_balance_audit.csv"
    )
    assert len(control_balance) == 6
    assert set(control_balance["both_controls_fully_reconstructed"]) == {True}
    assert primary["memory_control_audit"]["panels_fully_reconstructed"] == 24
    seed_audit = pd.read_csv(
        repository / "results/JSTAT/stages/cross_model/tables/cluster_seed_audit.csv"
    )
    assert len(seed_audit) == 6
    assert seed_audit["model_seed_namespaces_disjoint"].all()
