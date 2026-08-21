"""Compact reports, manifests, PDF QA, and integrity checks for V14."""

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
        repository / "configs/statmech_v14",
        repository / "thermoagent/statmech_llm_v14",
        repository / "tests/statmech_v14",
        repository / "results/collective_agent_statmech_v14",
        repository / "paper/jstat_v14",
    ]
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
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
    result = repository / "results/collective_agent_statmech_v14"
    protocol_path = repository / "configs/statmech_v14/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    primary = json.loads((result / "statistics/primary_results.json").read_text(encoding="utf-8"))
    effects = pd.read_csv(result / "tables/hypothesis_effects.csv")
    memory = pd.read_csv(result / "tables/memory_discovery_replication.csv")
    recovery = pd.read_csv(result / "tables/quench_recovery.csv")
    folds = pd.read_csv(result / "tables/representation_cv.csv")
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
    protocol_destination = result / "protocol/protocol_frozen_v1.0.yaml"
    atomic_bytes(protocol_path.read_bytes(), protocol_destination)
    readme = f"""# V14: memory and quench response in decentralized LLM-agent networks

## Supported scientific scope

V14 uses statistical-mechanical observables as a reduced language for actual independent Qwen-agent trajectories. Each agent owns its belief, typed action, confidence, commitment, workload, private field, inbox, outbox, and local context; a random-sequential scheduler only offers updates and transports the model-selected packet. Reference energy is an effective symmetric-layer observable. Reversal divergence is coarse-grained temporal asymmetry, not exact thermodynamic entropy production. Decoding temperature is not physical temperature.

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

## Confirmatory results

- H2, field-reversal maximum departure minus matched nominal: {_format_effect(h2)} regularized macrostate-distance units; exact one-sided sign-flip `p={h2.exact_one_sided_sign_flip_p:.5f}`, Holm `p={h2.holm_adjusted_p:.5f}`. {('Supported.' if dispositions['H2']['supported'] else 'Not supported.')}
- H3, early counter-quench peak minus final-five-sweep distance: {_format_effect(h3)} distance units; exact sign-flip `p={h3.exact_one_sided_sign_flip_p:.5f}`, Holm `p={h3.holm_adjusted_p:.5f}`. {('Supported.' if dispositions['H3']['supported'] else 'Not supported.')}
- H4, full-minus-order-only leave-one-cluster-out balanced accuracy: {_format_effect(h4)}; exact sign-flip `p={h4.exact_one_sided_sign_flip_p:.5f}`, Holm `p={h4.holm_adjusted_p:.5f}`. {('Supported.' if dispositions['H4']['supported'] else 'Not supported.')}

Across the new clusters, mean field-reversal maximum post-quench distance was {field.maximum_post_quench_distance.mean():.3f}, versus {nominal.maximum_post_quench_distance.mean():.3f} for nominal evolution. This magnitude belongs to the frozen training-standardized shrinkage metric; it is not a universal physical scale. Mean LOCO balanced accuracy was {accuracies.get('order_only', float('nan')):.3f} for order only, {accuracies.get('simple_uncertainty', float('nan')):.3f} for simple uncertainty, and {accuracies.get('full_statmech', float('nan')):.3f} for the full statistical-mechanics representation.

## Interpretation and boundaries

The analysis jointly reports magnetization, belief-action alignment, uncertainty, configuration entropy, entropy rate, total correlation, mutual information, effective energy, energy fluctuations, susceptibility, correlations, pathwise irreversibility, macrostate distance, recovery, and route asymmetry. No single entropy is assigned a universal good/bad meaning. The full representation is evaluated with transparent multinomial logistic regression and leave-one-cluster-out preprocessing; no test cluster contributes to standardization, imputation, covariance fitting, or regularization.

V12's degree- and traffic-matched nonreciprocity boundary remains negative and is not reopened. V13's coupling/noise directions and four-cluster classifier were preliminary or unsupported and are not relabeled as V14 confirmation. No thermodynamic-limit phase transition, physical free energy, universal LLM behavior, controller benefit, application benefit, field validity, or human evidence is claimed.

## Reproduction order

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/run-statmech-v14-tests.sh
THERMO_V14_ENABLE_QWEN=1 THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/run-statmech-v14-pilot.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/freeze-statmech-v14-protocol.sh
THERMO_V14_ENABLE_QWEN=1 THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/run-statmech-v14-formal.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/replay-statmech-v14.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/analyze-statmech-v14.sh
scripts/generate-statmech-v14-figures.sh
scripts/build-statmech-v14-results.sh
scripts/build-statmech-v14-paper.sh
scripts/verify-statmech-v14.sh
```
"""
    atomic_bytes(readme.encode("utf-8"), result / "README.md")
    summary = f"""# Paper summary

V14 studies collective memory and controlled quench response in networks of actual independent Qwen agents. The microscopic process preserves separate contexts, private state, typed authority, explicit message delivery, and random-sequential local updates. Statistical-mechanical observables form a finite-size reduced state rather than a claim of literal thermodynamics.

The immutable V12 discovery memory effect was {_format_effect(v12, 5)} and the V13 prospective replication was {_format_effect(v13, 5)} nats/update. V14 adds six new matched quench clusters. Field reversal changed maximum post-quench macrostate departure relative to nominal by {_format_effect(h2)} and subsequent restoration changed early-to-late distance by {_format_effect(h3)}. The full representation's LOCO balanced-accuracy increment over order-only features was {_format_effect(h4)}.

The contribution is the integrated microstate-to-macrostate formulation, bias-aware temporal-asymmetry analysis, field quench/counter-quench paths, nominal-manifold robustness audit, and transparent representation ablation. Results are finite-size and model-specific; reference energy and reversal divergence remain effective coarse-grained observables.
"""
    atomic_bytes(summary.encode("utf-8"), result / "PAPER_SUMMARY.md")
    claims = [
        ("Persistent memory increases bias-adjusted pathwise irreversibility", "replicated across V12 discovery and V13 prospective replication", "V12/V13 separate aggregate estimates", "No new V14 memory trajectories"),
        ("Field reversal causes larger macrostate departure than nominal", "supported" if dispositions["H2"]["supported"] else "not supported", "V14 H2, six matched clusters", "Metric-specific finite-size distance"),
        ("Restoration produces measurable relaxation", "supported" if dispositions["H3"]["supported"] else "not supported", "V14 H3, six field-reversal trajectories", "Not equilibrium relaxation proof"),
        ("Full stat.-mech. representation adds held-out discrimination", "supported" if dispositions["H4"]["supported"] else "not supported", "V14 H4 LOCO comparison", "Small transparent classifier, one model"),
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
\newcommand{{\VFourteenHfour}}{{{h4.estimate:.3f}}}
\newcommand{{\VFourteenHfourCI}}{{{h4.ci_low:.3f} to {h4.ci_high:.3f}}}
\newcommand{{\VTwelveMemory}}{{{v12.estimate:.5f}}}
\newcommand{{\VThirteenMemory}}{{{v13.estimate:.5f}}}
\newcommand{{\FullAccuracy}}{{{accuracies.get('full_statmech', float('nan')):.3f}}}
\newcommand{{\OrderAccuracy}}{{{accuracies.get('order_only', float('nan')):.3f}}}
"""
    paper = repository / "paper/jstat_v14"
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
    result = repository / "results/collective_agent_statmech_v14"
    pdfs = sorted((result / "figures/pdf").glob("*.pdf"))
    manuscript = repository / "paper/jstat_v14/main.pdf"
    supplement = repository / "paper/jstat_v14/supplement.pdf"
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
    result = repository / "results/collective_agent_statmech_v14"
    protocol = load_yaml(repository / "configs/statmech_v14/protocol_frozen.yaml")
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
    package_bytes = int(sum(path.stat().st_size for path in files))
    current_source = execution_source_checksum(repository)
    checks = {
        "source_matches_freeze": current_source == protocol["provenance"]["execution_source_sha256"],
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
