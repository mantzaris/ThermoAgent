"""Compact reports, manifests, PDF QA, and corrected-quench integrity checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from .workflow import (
    PARENT_COMMIT,
    artifact_root,
    atomic_bytes,
    atomic_csv,
    atomic_json,
    execution_source_checksum,
    load_yaml,
    sha256_file,
    tree_digest,
    utc_now,
)


def _v14_files(repository: Path) -> List[Path]:
    roots = [
        repository / "configs/statmech_llm/corrected_quench",
        repository / "thermoagent/statmech_llm/corrected_quench",
        repository / "tests/statmech_llm/corrected_quench",
        repository / "results/JSTAT/stages/corrected_quench",
        repository / "paper/JSTAT",
    ]
    latex_intermediates = {
        ".aux",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".synctex.gz",
    }
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(path.name.endswith(suffix) for suffix in latex_intermediates)
    ]
    files += list((repository / "scripts").glob("*v14*"))
    files += list((repository / "notes").glob("v14_*.md"))
    if (repository / ".gitignore").exists():
        files.append(repository / ".gitignore")
    return sorted(set(path for path in files if path.is_file()))


def _manifest(repository: Path) -> List[Dict[str, object]]:
    rows = []
    for path in _v14_files(repository):
        if path.name in ("repository_manifest.csv", "INDEX.csv"):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _effect(frame: pd.DataFrame, hypothesis: str) -> pd.Series:
    selected = frame[frame["hypothesis"] == hypothesis]
    if len(selected) != 1:
        raise RuntimeError("expected one row for %s" % hypothesis)
    return selected.iloc[0]


def _format_effect(row: pd.Series, digits: int = 3) -> str:
    return f"{row.estimate:.{digits}f} (95% CI {row.ci_low:.{digits}f} to {row.ci_high:.{digits}f})"


def build_results(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/JSTAT/stages/corrected_quench"
    protocol_path = repository / "configs/statmech_llm/corrected_quench/protocol.yaml"
    protocol = load_yaml(protocol_path)
    primary = json.loads((result / "statistics/primary_results.json").read_text(encoding="utf-8"))
    effects = pd.read_csv(result / "tables/hypothesis_effects.csv")
    memory = pd.read_csv(result / "tables/memory_discovery_replication.csv")
    recovery = pd.read_csv(result / "tables/quench_recovery.csv")
    folds = pd.read_csv(result / "tables/representation_cv.csv")
    permutations = pd.read_csv(result / "tables/representation_permutation_summary.csv")
    information = pd.read_csv(result / "tables/information_estimator_contrast_summary.csv")
    panels = pd.read_csv(result / "tables/panel_statistics.csv")
    completion = primary["formal_accounting"]
    pilot = protocol["engineering_pilot"]
    h2, h3, h4 = (_effect(effects, key) for key in ("H2", "H3", "H4"))
    v12 = memory[memory["study"] == "V12_discovery"].iloc[0]
    v13 = memory[memory["study"] == "V13_replication"].iloc[0]
    accuracies = folds.groupby("representation")["balanced_accuracy"].mean().to_dict()
    field = recovery[recovery["disruption"] == "field_reversal"]
    nominal = recovery[recovery["disruption"] == "nominal"]
    pilot_accounting = pilot["provider_environment"]["accounting"]
    formal_hours = float(completion.get("all_formal_generation_gpu_hours_including_invalidated", completion["generation_gpu_hours"]))
    pilot_hours = float(pilot_accounting["latency_seconds"]) / 3600.0
    total_hours = formal_hours + pilot_hours
    cost_low, cost_high = 0.34 * total_hours, 0.69 * total_hours
    dispositions = primary["confirmatory_dispositions"]
    h3_disposition = dispositions["H3"]
    if bool(h3_disposition.get("inferential_support", h3_disposition.get("supported", False))):
        raise RuntimeError("structurally invalid V14 H3 cannot be inferentially supported")
    recovery_times = field["recovery_time_sweeps"].to_numpy(float)
    recovery_count = int(pd.Series(recovery_times).notna().sum())
    exact_recovery_time = (
        float(recovery_times[0])
        if recovery_count == len(recovery_times)
        and len(set(float(value) for value in recovery_times)) == 1
        else float("nan")
    )
    fixed_recovery = float(field["fixed_early_minus_late_recovery_distance"].mean())
    final_residual = float(field["final_five_sweep_mean_distance"].mean())
    permutation_lookup = {
        str(row.metric): row for row in permutations.itertuples(index=False)
    }
    full_permutation = permutation_lookup["full_statmech_balanced_accuracy"]
    increment_permutation = permutation_lookup[
        "full_minus_order_only_balanced_accuracy"
    ]
    adjusted_information = information[
        (information["window_sweeps"] == int(protocol["analysis"]["primary_window_sweeps"]))
        & (information["metric"] == "total_correlation_bias_adjusted")
    ].iloc[0]
    protocol_destination = result / "protocol/protocol_frozen_v1.0.yaml"
    atomic_bytes(protocol_path.read_bytes(), protocol_destination)
    readme = f"""# V14 scientific audit: memory and quench response in LLM-agent networks

## Supported scientific scope

V14 uses statistical-mechanical observables as a reduced language for state-separated, locally informed Qwen-agent trajectories. Each agent instance owns its belief, typed action, confidence, commitment, workload, private field, inbox, outbox, and local context; a random-sequential scheduler only offers updates and transports the model-selected packet. Reference energy is an effective symmetric-layer observable. Reversal divergence is coarse-grained temporal asymmetry, not exact thermodynamic entropy production. Decoding temperature is not physical temperature.

V12 is the immutable discovery study and V13 the immutable prospective replication. Their memory effects remain separate: V12 {_format_effect(v12, 5)} and V13 {_format_effect(v13, 5)} nats per attempted update. The synthesis is descriptive and does not retroactively pool their protocols.

## Frozen V14 experiment

- Protocol: `{protocol['protocol']}`; SHA-256 `{sha256_file(protocol_path)}`.
- Execution source: `{protocol['provenance']['execution_source_sha256']}`.
- Parent V13 commit: `{PARENT_COMMIT}`.
- Model: `{protocol['model']['identifier']}` at revision `{protocol['model']['revision']}`; `{protocol['model']['quantization']}`, sampling temperature {protocol['model']['inference_sampling_temperature']}, top-p {protocol['model']['top_p']}, maximum {protocol['model']['maximum_new_tokens']} output tokens, no chain-of-thought request.
- Design: six new independent graph/environment clusters, four matched conditions, `N=16`, modular reciprocal delivery, coupling `J=0.8`, 45 sweeps (15 baseline, 15 perturbation, 15 restoration).
- Conditions: nominal, private-field reversal, inter-community partition, and 50% sender-preassigned categorical message corruption.
- Independent unit: complete matched graph/environment trajectory cluster. Agents, messages, tokens, and time steps are not independent replicates.

The complete frozen experiment ran {completion['dynamic_trajectories']} trajectories and {completion['observed_decision_rows']:,} analyzed decisions. It used {completion['model_calls']:,} formal model calls, {completion['prompt_tokens']:,} formal prompt tokens, and {completion['generated_tokens']:,} generated tokens. Including the engineering pilot and any retained formal attempts, generation used {total_hours:.3f} metered GPU-hours. The approximate incremental RTX 4090 cost range is USD {cost_low:.2f}–{cost_high:.2f}. Raw decisions and full trajectories remain external at `{artifact_root()}`.

## Confirmatory results and versioned scientific correction

- H2, field-reversal maximum departure minus matched nominal: {_format_effect(h2)} regularized macrostate-distance units; exact one-sided sign-flip `p={h2.exact_one_sided_sign_flip_p:.5f}`, Holm `p={h2.holm_adjusted_p:.5f}`. {('Supported.' if dispositions['H2']['supported'] else 'Not supported.')}
- H3 historical frozen number, early counter-quench peak minus final-five-sweep distance: {_format_effect(h3)} distance units. Its archived raw `p={h3.exact_one_sided_sign_flip_p:.5f}` and Holm `p={h3.holm_adjusted_p:.5f}` are retained only for provenance. Because this estimand is structurally nonnegative for almost every nonconstant trajectory, its directional sign-flip test is invalid and H3 has **no inferential support**. The complete trajectories are nevertheless consistent with finite-time return: {recovery_count}/{len(field)} field-reversal panels re-enter the leave-one-cluster-out training-nominal threshold{(' after exactly %.0f sweeps' % exact_recovery_time) if pd.notna(exact_recovery_time) else ''}, their mean final-five distance is {final_residual:.3f}, and the non-tautological fixed early-five-minus-late-five descriptive change is {fixed_recovery:.3f} distance units.
- H4, full-minus-order-only leave-one-cluster-out balanced accuracy: {_format_effect(h4)}; exact sign-flip `p={h4.exact_one_sided_sign_flip_p:.5f}`, Holm `p={h4.holm_adjusted_p:.5f}`. {('Supported.' if dispositions['H4']['supported'] else 'Not supported.')}

Across the new clusters, mean field-reversal maximum post-quench distance was {field.maximum_post_quench_distance.mean():.3f}, versus {nominal.maximum_post_quench_distance.mean():.3f} for nominal evolution. This magnitude belongs to the frozen training-standardized shrinkage metric; it is not a universal physical scale. Mean LOCO balanced accuracy was {accuracies.get('order_only', float('nan')):.3f} for order only, {accuracies.get('simple_uncertainty', float('nan')):.3f} for simple uncertainty, and {accuracies.get('full_statmech', float('nan')):.3f} for the full statistical-mechanics representation. The prespecified 10,000-replicate cluster-preserving permutation audit gives `p={float(full_permutation.upper_tail_empirical_p):.5f}` for absolute full-representation accuracy and `p={float(increment_permutation.upper_tail_empirical_p):.5f}` for its increment over order only; every fold refits imputation, standardization, and the classifier using training clusters alone.

The delayed prespecified sensitivity audit recomputes three-, five-, and seven-sweep macrostates and nominal fits separately, deletes every observable and observable family in turn, and retains raw as well as marginal-preserving-null-adjusted dependence estimates. At the primary five-sweep window, the mean field-minus-nominal adjusted-total-correlation contrast is {float(adjusted_information.estimate):.3f} nats (95% cluster-bootstrap interval {float(adjusted_information.ci_low):.3f} to {float(adjusted_information.ci_high):.3f}). These delayed analyses are sensitivity evidence, not a new prospective experiment.

## Interpretation and boundaries

The analysis jointly reports magnetization, belief-action alignment, uncertainty, configuration entropy, entropy rate, total correlation, mutual information, effective energy, energy fluctuations, susceptibility, correlations, pathwise irreversibility, macrostate distance, recovery, and route asymmetry. No single entropy is assigned a universal good/bad meaning. The full representation is evaluated with transparent multinomial logistic regression and leave-one-cluster-out preprocessing; no test cluster contributes to standardization, imputation, covariance fitting, or regularization.

V12's degree- and traffic-matched nonreciprocity boundary remains negative and is not reopened. V13's coupling/noise directions and four-cluster classifier were preliminary or unsupported and are not relabeled as V14 confirmation. No thermodynamic-limit phase transition, physical free energy, universal LLM behavior, controller benefit, application benefit, field validity, or human evidence is claimed.

## Reproduction order

```bash
scripts/run-tests.sh tests/statmech_llm/corrected_quench
THERMOAGENT_ENABLE_LLM=1 scripts/run-formal-experiment.sh corrected-quench
scripts/replay-results.sh corrected-quench
scripts/analyze-results.sh corrected-quench
scripts/generate-figures.sh
scripts/build-jstat-paper.sh
scripts/verify-results.sh
```
"""
    atomic_bytes(readme.encode("utf-8"), result / "README.md")
    summary = f"""# Paper summary and V14 scientific audit

V14 studies collective memory and controlled quench response in networks of state-separated, locally informed Qwen-agent instances. The microscopic process preserves separate contexts, private state, typed authority, explicit message delivery, and random-sequential local updates. Statistical-mechanical observables form a finite-size reduced state rather than a claim of literal thermodynamics.

The immutable V12 discovery memory effect was {_format_effect(v12, 5)} and the V13 prospective replication was {_format_effect(v13, 5)} nats/update. V14 adds six new matched quench clusters. Field reversal changed maximum post-quench macrostate departure relative to nominal by {_format_effect(h2)}. All {recovery_count} field-reversal paths re-entered their cluster-excluded training-nominal threshold{(' after exactly %.0f sweeps' % exact_recovery_time) if pd.notna(exact_recovery_time) else ''}; their mean fixed early-five-minus-late-five change was {fixed_recovery:.3f}. The historical H3 peak-minus-final number, {_format_effect(h3)}, is retained but its directional test is invalid because the estimand is structurally nonnegative. The full representation's LOCO balanced-accuracy increment over order-only features was {_format_effect(h4)}, with cluster-preserving permutation `p={float(increment_permutation.upper_tail_empirical_p):.5f}`.

The contribution is the integrated augmented-state-to-observable-projection formulation, bias-aware temporal-asymmetry analysis, field quench/counter-quench paths, leakage-free nominal-manifold robustness audit, finite-sample dependence audit, and transparent representation ablation. Results are finite-size and model-specific; reference energy and reversal divergence remain effective coarse-grained observables.
"""
    atomic_bytes(summary.encode("utf-8"), result / "PAPER_SUMMARY.md")
    claims = [
        ("Persistent memory increases bias-adjusted pathwise irreversibility", "replicated across V12 discovery and V13 prospective replication", "V12/V13 separate aggregate estimates", "No new V14 memory trajectories"),
        ("Field reversal causes larger macrostate departure than nominal", "supported" if dispositions["H2"]["supported"] else "not supported", "V14 H2, six matched clusters", "Metric-specific finite-size distance"),
        ("Frozen H3 directionally establishes relaxation", "not inferentially supported", "historical number and p-values retained; structurally nonnegative estimand", "Invalid directional sign test"),
        ("Restoration trajectories return toward the restored nominal regime", "consistent trajectory evidence", "%d/%d paths cross cluster-excluded training-nominal thresholds; fixed early-minus-late change %.3f" % (recovery_count, len(field), fixed_recovery), "Descriptive finite-time recovery, not equilibrium relaxation proof"),
        ("Full stat.-mech. representation adds held-out discrimination", "supported" if dispositions["H4"]["supported"] else "not supported", "V14 H4 LOCO comparison", "Small transparent classifier, one model"),
        ("Representation result survives cluster-preserving label permutation", "supported" if float(increment_permutation.upper_tail_empirical_p) < 0.05 else "not supported", "10,000 full-pipeline permutations; empirical p %.5f" % float(increment_permutation.upper_tail_empirical_p), "Delayed completion of a prespecified analysis"),
        ("Reference energy is physical energy", "prohibited", "not tested", "Effective symmetric-layer coordinate only"),
        ("Pathwise irreversibility is exact entropy production", "prohibited", "observable process is coarse-grained", "Use temporal-asymmetry terminology"),
        ("A thermodynamic-limit phase transition exists", "unsupported", "N=16 V14; V12/V13 finite sizes", "Finite-size regime language only"),
        ("Statistical mechanics improves agent performance", "not tested", "no performance endpoint", "Characterization study"),
    ]
    lines = ["# Claims matrix", "", "| Claim | Disposition | Evidence | Boundary |", "|---|---|---|---|"]
    lines += ["| %s | %s | %s | %s |" % row for row in claims]
    atomic_bytes(("\n".join(lines) + "\n").encode("utf-8"), result / "CLAIMS_MATRIX.md")
    macros = rf"""% Generated from V14 aggregate tables; do not edit manually.
\newcommand{{\VFourteenTrajectories}}{{{int(completion['dynamic_trajectories'])}}}
\newcommand{{\VFourteenDecisions}}{{{int(completion['observed_decision_rows']):,}}}
\newcommand{{\VFourteenClusters}}{{6}}
\newcommand{{\VFourteenGPUHours}}{{{total_hours:.2f}}}
\newcommand{{\VFourteenHtwo}}{{{h2.estimate:.3f}}}
\newcommand{{\VFourteenHtwoCI}}{{{h2.ci_low:.3f} to {h2.ci_high:.3f}}}
\newcommand{{\VFourteenHthree}}{{{h3.estimate:.3f}}}
\newcommand{{\VFourteenHthreeCI}}{{{h3.ci_low:.3f} to {h3.ci_high:.3f}}}
\newcommand{{\VFourteenHthreeValidDirectionalTest}}{{false}}
\newcommand{{\VFourteenRecoveryPanels}}{{{recovery_count}/{len(field)}}}
\newcommand{{\VFourteenRecoverySweeps}}{{{exact_recovery_time:.0f}}}
\newcommand{{\VFourteenFixedRecovery}}{{{fixed_recovery:.3f}}}
\newcommand{{\VFourteenHfour}}{{{h4.estimate:.3f}}}
\newcommand{{\VFourteenHfourCI}}{{{h4.ci_low:.3f} to {h4.ci_high:.3f}}}
\newcommand{{\VTwelveMemory}}{{{v12.estimate:.5f}}}
\newcommand{{\VThirteenMemory}}{{{v13.estimate:.5f}}}
\newcommand{{\FullAccuracy}}{{{accuracies.get('full_statmech', float('nan')):.3f}}}
\newcommand{{\OrderAccuracy}}{{{accuracies.get('order_only', float('nan')):.3f}}}
\newcommand{{\FullPermutationP}}{{{float(full_permutation.upper_tail_empirical_p):.5f}}}
\newcommand{{\IncrementPermutationP}}{{{float(increment_permutation.upper_tail_empirical_p):.5f}}}
"""
    paper = repository / "paper/JSTAT"
    paper.mkdir(parents=True, exist_ok=True)
    atomic_bytes(macros.encode("utf-8"), paper / "results_macros.tex")
    external = {
        "generated_at": utc_now(),
        "artifact_root": str(artifact_root()),
        "raw_artifacts_committed": False,
        "tree": tree_digest(artifact_root()),
        "formal": tree_digest(artifact_root() / "formal"),
        "raw_formal": tree_digest(artifact_root() / "raw/formal"),
        "pilot": tree_digest(artifact_root() / "pilot"),
    }
    atomic_json(external, result / "reproducibility/external_artifact_manifest.json")
    compute = {
        "generated_at": utc_now(),
        "formal": completion,
        "pilot_accounting": pilot_accounting,
        "total_metered_generation_gpu_hours": total_hours,
        "estimated_cost_usd_range": [cost_low, cost_high],
        "cpu_analysis_seconds": primary["analysis_cpu_seconds"],
        "model": protocol["model"],
    }
    atomic_json(compute, result / "reproducibility/compute_accounting.json")
    manifest = _manifest(repository)
    atomic_csv(manifest, result / "reproducibility/repository_manifest.csv")
    atomic_csv(manifest, result / "INDEX.csv")
    return {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "execution_source_sha256": protocol["provenance"]["execution_source_sha256"],
        "repository_files": len(manifest),
        "repository_bytes": int(sum(int(row["bytes"]) for row in manifest)),
        "external_artifact_tree_sha256": external["tree"]["tree_sha256"],
    }


def validate_pdfs(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/JSTAT/stages/corrected_quench"
    pdfs = sorted((result / "figures/pdf").glob("*.pdf"))
    manuscript = repository / "paper/JSTAT/main.pdf"
    supplement = repository / "paper/JSTAT/supplement.pdf"
    pdfs += [path for path in (manuscript, supplement) if path.exists()]
    if not pdfs:
        raise RuntimeError("no V14 PDFs found")
    qa_root = artifact_root() / "pdf_qa/rendered_300dpi"
    qa_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in pdfs:
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        fonts = subprocess.check_output(["pdffonts", str(path)], text=True)
        extracted = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        prefix = qa_root / path.stem
        subprocess.run(["pdftoppm", "-png", "-r", "300", str(path), str(prefix)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")))
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        embedded = bool(font_lines and all("yes" in line.lower().split() for line in font_lines))
        rendered = sorted(qa_root.glob(path.stem + "-*.png"))
        rows.append(
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "pages": pages,
                "opens": True,
                "fonts_embedded": embedded,
                "text_extractable": bool(extracted.strip()),
                "rendered_pages": len(rendered),
                "render_dpi": 300,
                "manual_visual_status": "pending",
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(rows, result / "reproducibility/pdf_qa.csv")
    summary = {
        "generated_at": utc_now(),
        "pdf_count": len(pdfs),
        "pages": int(sum(row["pages"] for row in rows)),
        "automated_passed": bool(all(row["opens"] and row["fonts_embedded"] and row["text_extractable"] and row["pages"] == row["rendered_pages"] for row in rows)),
        "manual_visual_status": "pending",
        "render_root_external": str(qa_root),
    }
    atomic_json(summary, result / "reproducibility/pdf_qa_summary.json")
    return summary


def verify_package(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/JSTAT/stages/corrected_quench"
    protocol = load_yaml(repository / "configs/statmech_llm/corrected_quench/protocol.yaml")
    files = _v14_files(repository)
    forbidden_suffixes = {".jsonl", ".safetensors", ".pt", ".bin", ".npy", ".npz", ".tar", ".zip", ".png"}
    oversized = [path for path in files if path.stat().st_size > 10 * 1024 * 1024]
    forbidden = [path for path in files if path.suffix.lower() in forbidden_suffixes]
    crlf = [path for path in files if path.suffix.lower() in {".py", ".yaml", ".md", ".sh", ".tex", ".bib", ".csv", ".json"} and b"\r\n" in path.read_bytes()]
    index = pd.read_csv(result / "INDEX.csv")
    missing = [row.relative_path for row in index.itertuples() if not (repository / row.relative_path).exists()]
    checksum_mismatch = [row.relative_path for row in index.itertuples() if (repository / row.relative_path).exists() and sha256_file(repository / row.relative_path) != row.sha256]
    replay_path = artifact_root() / "reproducibility/replay_summary.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {"status": "missing"}
    completion_path = artifact_root() / "formal/completion.json"
    completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.exists()
        else {}
    )
    correction_path = result / "corrections/v14_scientific_audit_v1_1/correction_record.json"
    correction = (
        json.loads(correction_path.read_text(encoding="utf-8"))
        if correction_path.exists()
        else {}
    )
    package_bytes = int(sum(path.stat().st_size for path in files))
    current_source = execution_source_checksum(repository)
    checks = {
        "formal_execution_source_matches_freeze": completion.get("execution_source_sha256")
        == protocol["provenance"]["execution_source_sha256"],
        "formal_protocol_matches_freeze": completion.get("protocol_sha256")
        == sha256_file(repository / "configs/statmech_llm/corrected_quench/protocol.yaml"),
        "audit_source_change_versioned": bool(correction)
        and correction.get("audit_execution_source_sha256") == current_source
        and correction.get("raw_outcomes_changed") is False,
        "invalid_H3_not_supported": not bool(
            json.loads(
                (result / "statistics/primary_results.json").read_text(encoding="utf-8")
            )["confirmatory_dispositions"]["H3"].get("inferential_support", True)
        ),
        "replay_passed": replay.get("status") == "passed",
        "no_oversized_files": not oversized,
        "no_forbidden_artifacts": not forbidden,
        "no_crlf": not crlf,
        "index_complete": not missing,
        "index_checksums_match": not checksum_mismatch,
        "package_below_25_mib": package_bytes < 25 * 1024 * 1024,
        "privacy_passed": json.loads((result / "statistics/primary_results.json").read_text())["privacy_mutations"] == 0,
    }
    summary = {
        "generated_at": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "package_files": len(files),
        "package_bytes": package_bytes,
        "largest_files": [
            {"relative_path": path.relative_to(repository).as_posix(), "bytes": path.stat().st_size}
            for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:20]
        ],
        "missing_indexed_files": missing,
        "checksum_mismatches": checksum_mismatch,
        "oversized_files": [str(path) for path in oversized],
        "forbidden_files": [str(path) for path in forbidden],
        "replay": replay,
    }
    atomic_json(summary, result / "reproducibility/verification.json")
    return summary


__all__ = ["build_results", "validate_pdfs", "verify_package"]
