"""Compact paper-facing reports, PDF QA, and V13 integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from .workflow import (
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


PARENT = "457f6d635b60292623c8d97aa3b0c60d8d0aac4e"


def _effect_row(frame: pd.DataFrame, hypothesis: str, metric: str) -> pd.Series:
    selected = frame[(frame["hypothesis"] == hypothesis) & (frame["metric"] == metric)]
    if len(selected) != 1:
        raise RuntimeError("expected one effect row for %s %s" % (hypothesis, metric))
    return selected.iloc[0]


def _fmt(row: pd.Series, digits: int = 4) -> str:
    return ("%.*f (95%% CI %.*f to %.*f)" % (digits, row["estimate"], digits, row["ci_low"], digits, row["ci_high"]))


def _repository_files(repository: Path) -> List[Path]:
    roots = [
        repository / "configs/statmech_v13",
        repository / "thermoagent/statmech_llm_v13",
        repository / "tests/statmech_v13",
        repository / "results/collective_agent_statmech_v13",
        repository / "paper/jstat_v13",
    ]
    files = [path for root in roots if root.exists() for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    files += [path for path in (repository / "scripts").glob("*v13*") if path.is_file()]
    files += [path for path in (repository / "notes").glob("v13_*.md") if path.is_file()]
    return sorted(set(files))


def _manifest(repository: Path) -> List[Dict[str, object]]:
    output = []
    for path in _repository_files(repository):
        if path.name == "repository_manifest.csv":
            continue
        output.append(
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return output


def _write_prose(repository: Path, protocol: Mapping[str, object], primary: Mapping[str, object]) -> None:
    root = repository / "results/collective_agent_statmech_v13"
    effects = pd.read_csv(root / "tables/hypothesis_effects.csv")
    panels = pd.read_csv(root / "tables/panel_statistics.csv")
    recovery = pd.read_csv(root / "tables/disruption_recovery.csv")
    cv = pd.read_csv(root / "tables/representation_cv.csv")
    completion = primary["formal_accounting"]
    h1_m = _effect_row(effects, "H1", "mean_abs_belief_magnetization")
    h1_s = _effect_row(effects, "H1", "belief_susceptibility")
    h1_t = _effect_row(effects, "H1", "belief_integrated_autocorrelation_time_updates")
    h2_m = _effect_row(effects, "H2", "mean_abs_belief_magnetization")
    h2_s = _effect_row(effects, "H2", "belief_susceptibility")
    h2_t = _effect_row(effects, "H2", "belief_integrated_autocorrelation_time_updates")
    h3 = _effect_row(effects, "H3", "adjusted_block_irreversibility_nats_per_update")
    h4 = _effect_row(effects, "H4", "maximum_macrostate_departure_disrupted_minus_nominal")
    h5 = _effect_row(effects, "H5", "full_representation_accuracy_minus_chance")
    h6 = _effect_row(effects, "H6", "full_accuracy_minus_strongest_reduced")
    full_accuracy = float(cv[cv["representation"] == "full_statmech"]["accuracy"].mean())
    simple_accuracy = float(cv[cv["representation"] == "simple"]["accuracy"].mean())
    order_accuracy = float(cv[cv["representation"] == "order_only"]["accuracy"].mean())
    memory = panels[panels["subset"] == "memory_confirmation"]
    external = artifact_root()
    formal_gpu_hours = float(completion.get("all_formal_generation_gpu_hours_including_invalidated", completion["generation_gpu_hours"]))
    pilot_accounting = protocol["engineering_pilot"]["provider_environment"]["accounting"]  # type: ignore[index]
    pilot_gpu_hours = float(pilot_accounting["latency_seconds"]) / 3600.0
    gpu_hours = formal_gpu_hours + pilot_gpu_hours
    cost_low, cost_high = 0.34 * gpu_hours, 0.69 * gpu_hours
    dispositions = {**primary["primary_confirmatory"], **primary["disruption_hypotheses"], "H7": primary["H7_surrogate"]}
    statements = {
        key: ("%s was supported under its frozen criterion." % key if bool(dispositions[key]["supported"]) else "%s was not supported under its frozen criterion." % key)
        for key in ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
    }
    readme = f"""# V13: collective statistical mechanics of decentralized LLM agents

## Supported scope

V13 prospectively tests whether statistical-mechanical observables provide a compact reduced description of actual independent LLM-agent networks. The binary beliefs and actions are Qwen choices, reference energy is an effective symmetric-layer observable, decoding temperature is a decision-noise control, and adjusted block reversal divergence is coarse-grained pathwise irreversibility. The study does not claim physical heat, exact LLM entropy production, a thermodynamic-limit transition, controller superiority, application benefit, or human evidence.

V12 is an immutable discovery study. V13 does not reopen its negative nonreciprocity endpoint. Directed communication alone remains a documented boundary: it did not reliably raise irreversibility under the V12 degree- and traffic-matched conditions.

## Model, agents, and update

- Model: `{protocol['model']['identifier']}`, revision `{protocol['model']['revision']}`.
- Runtime: {protocol['model']['quantization']}; top-p {protocol['model']['top_p']}; at most {protocol['model']['maximum_new_tokens']} generated tokens; no chain-of-thought request.
- Each agent separately owns belief, action, confidence, commitment, bounded memory, private field, workload, inbox, outbox, context, and typed authority.
- The scheduler uses a random permutation within each sweep, offers one local update, transports the model-selected packet, and never selects a scientific state or action. One sweep is `N` attempted updates.
- The graph/environment trajectory cluster is the independent inferential unit.

## Frozen experiment

Protocol `{protocol['protocol']}`; SHA-256 `{sha256_file(repository / 'configs/statmech_v13/protocol_frozen_v1.2.yaml')}`; execution-source SHA-256 `{protocol['provenance']['execution_source_sha256']}`. Amendment 01 was made before any network panel existed because four paired clusters made the frozen sign-flip threshold mathematically unattainable; all 163 interrupted microscopic records were retained. Amendment 02 clarified before restart that the 18M-token ceiling includes pilot and interrupted calls; no scientific design changed. Work Package A uses `N={{8,16}}`, modular primary and ring replication graphs, coupling `{{0.35,0.80}}`, decoding noise `{{0.50,0.85}}`, and Markovized agents. Work Package B pairs Markovized and bounded-memory agents on six `N=16` modular field-quench clusters. Work Package C pairs nominal, field-reversal, inter-community partition, and 50% categorical packet-corruption trajectories on four `N=16` modular clusters.

Formal execution completed {completion['observed_decision_rows']:,}/{completion['planned_decisions']:,} analyzed decisions in {completion['dynamic_trajectories']} graph trajectories plus the microscopic grid. Including the retained pre-amendment records and pilot, the study used {int(completion['all_formal_model_calls_including_invalidated']) + int(pilot_accounting['model_calls']):,} calls, {int(completion['all_formal_prompt_tokens_including_invalidated']) + int(pilot_accounting['prompt_tokens']):,} prompt tokens, and {int(completion['all_formal_generated_tokens_including_invalidated']) + int(pilot_accounting['generated_tokens']):,} generated tokens. Total metered generation was {gpu_hours:.3f} GPU-hours; estimated incremental cost is USD {cost_low:.2f}--{cost_high:.2f}. Raw model records and transitions remain external at `{external}`.

## Confirmatory V12-to-V13 effects

- H1 coupling: order {_fmt(h1_m)}, susceptibility {_fmt(h1_s)}, and integrated correlation time {_fmt(h1_t)}.
- H2 decoding noise: order {_fmt(h2_m)}, susceptibility {_fmt(h2_s)}, and integrated correlation time {_fmt(h2_t)}.
- H3 bounded memory: adjusted pathwise irreversibility {_fmt(h3, 5)} nats per attempted update.

The H1--H3 family uses paired graph-cluster bootstrap intervals, intersection-union directional tests for the multi-endpoint H1/H2 claims, and Holm correction across the three frozen hypotheses. V12 discovery and V13 confirmation estimates are stored separately in `tables/v12_discovery_effects.csv` and `tables/hypothesis_effects.csv`.

## Disruptions and reduced representation

Controlled disruptions changed maximum distance from the leave-cluster-out nominal manifold by {_fmt(h4)} on average relative to the matched undisturbed trajectory. Four-class disruption separation by the full statistical-mechanical representation achieved mean leave-cluster-out accuracy {full_accuracy:.3f}, versus {simple_accuracy:.3f} for simple aggregates and {order_accuracy:.3f} for order-only features. H5 accuracy above chance was {_fmt(h5)}; H6 full-minus-strongest-reduced accuracy was {_fmt(h6)}.

Energy--entropy portraits and the fixed reduced vector trace baseline, quench, and recovery, but this is disruption response—not early warning. High entropy is not assigned a universal good/bad meaning. The fitted kinetic surrogate is explanatory and is never substituted for direct LLM evidence.

## What ran and what did not

Ran: a 192-decision engineering pilot limited to estimability/runtime, the complete frozen formal grid, deterministic content-addressed replay, CPU surrogate map, all frozen analyses, 22 candidate vector figures, and manuscript/PDF QA. Did not run: a second LLM, new nonreciprocity search, application-performance trial, human study, validation/holdout reuse, thermodynamic-limit scaling claim, or outcome-dependent rerun.

## Files and reproduction

- Operative protocol: `protocol/protocol_frozen_v1.2.yaml`; invalidated pre-outcome freezes: `protocol/protocol_frozen_v1.0_invalidated.yaml` and `protocol/protocol_frozen_v1.1_invalidated.yaml`
- Primary results: `statistics/primary_results.json`
- Panel and trajectory tables: `tables/`
- Figures and exact sources: `figures/pdf/`, `figures/source_data/`, and `figures/figure_catalog.csv`
- Replay, checksums, compute, and PDF QA: `reproducibility/`
- Manuscript: `../../paper/jstat_v13/main.tex` and `main.pdf`

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python THERMO_V13_ARTIFACT_ROOT=/workspace/ThermoAgent-v13-artifacts scripts/run-statmech-v13-tests.sh
THERMO_V13_ENABLE_QWEN=1 scripts/run-statmech-v13-pilot.sh
scripts/freeze-statmech-v13-protocol.sh
THERMO_V13_ENABLE_QWEN=1 scripts/run-statmech-v13-formal.sh
scripts/replay-statmech-v13.sh
scripts/analyze-statmech-v13.sh
scripts/generate-statmech-v13-figures.sh
scripts/build-statmech-v13-results.sh
scripts/build-statmech-v13-paper.sh
scripts/verify-statmech-v13.sh
```
"""
    atomic_bytes(readme.encode("utf-8"), root / "README.md")
    summary = f"""# Paper summary

V13 asks whether a statistical-mechanical reduced state connects measured local LLM response to collective order, fluctuations, relaxation, memory, and quench recovery. It prospectively confirms or refutes the V12 discovery directions using new graph trajectories; no V12 trajectory enters a V13 confidence interval.

The local process consists of independent Qwen agents with private fields and explicit delivered messages. Coupling and decoding-noise effects are {_fmt(h1_m)} and {_fmt(h2_m)} for order, with parallel susceptibility and correlation-time estimates in `tables/hypothesis_effects.csv`. Bounded memory changes coarse-grained path irreversibility by {_fmt(h3, 5)} nats/update. Controlled field, topology, and message quenches trace distinct energy--entropy--order paths quantified by leave-cluster-out nominal-manifold distance. The full reduced representation has cross-validated four-class accuracy {full_accuracy:.3f}; its incremental value over reduced representations is {_fmt(h6)}.

The central interpretation is finite-size collective characterization. Reference energy is effective, decoding noise is not temperature, reversal divergence is not exact total entropy production, and two direct sizes establish no thermodynamic-limit phase transition. V12's failure to find a reliable nonreciprocity effect remains unchanged.
"""
    atomic_bytes(summary.encode("utf-8"), root / "PAPER_SUMMARY.md")
    disposition = {**primary["primary_confirmatory"], **primary["disruption_hypotheses"], "H7": primary["H7_surrogate"]}
    lines = ["# Claims matrix", "", "| Claim | Status | Evidence | Boundary |", "|---|---|---|---|"]
    descriptions = {
        "H1": "Coupling increases finite-size order, susceptibility, and persistence",
        "H2": "Decoding noise decreases finite-size order, susceptibility, and persistence",
        "H3": "Bounded memory increases coarse-grained path irreversibility",
        "H4": "Quenches depart from the nominal statistical-mechanical manifold",
        "H5": "Frozen macrostates distinguish disruption trajectories",
        "H6": "The full statistical-mechanical vector adds regime information",
        "H7": "The fitted kinetic surrogate captures principal trend directions",
    }
    for key in descriptions:
        status = "supported" if bool(disposition[key]["supported"]) else "not supported"
        lines.append(f"| {descriptions[key]} | {status} | `tables/hypothesis_effects.csv` / `statistics/primary_results.json` | Finite-size, one model, trajectory clusters |")
    lines += [
        "", "## Prohibited extensions", "",
        "- No claim of exact thermodynamic entropy production or physical energy/temperature.",
        "- No thermodynamic-limit phase transition, universal exponent, or model universality.",
        "- No positive nonreciprocity claim; the immutable V12 boundary result remains negative.",
        "- No controller, application, operational, human, or field benefit was tested.",
    ]
    atomic_bytes(("\n".join(lines) + "\n").encode("utf-8"), root / "CLAIMS_MATRIX.md")
    macros = rf"""% Generated from V13 aggregate tables; do not edit manually.
\newcommand{{\VThirteenDecisions}}{{{int(completion['observed_decision_rows']):,}}}
\newcommand{{\VThirteenTrajectories}}{{{int(completion['dynamic_trajectories'])}}}
\newcommand{{\VThirteenGPUHours}}{{{gpu_hours:.2f}}}
\newcommand{{\CouplingOrderEffect}}{{{h1_m['estimate']:.4f}}}
\newcommand{{\CouplingOrderCI}}{{{h1_m['ci_low']:.4f} to {h1_m['ci_high']:.4f}}}
\newcommand{{\NoiseOrderEffect}}{{{h2_m['estimate']:.4f}}}
\newcommand{{\NoiseOrderCI}}{{{h2_m['ci_low']:.4f} to {h2_m['ci_high']:.4f}}}
\newcommand{{\MemoryIrreversibilityEffect}}{{{h3['estimate']:.5f}}}
\newcommand{{\MemoryIrreversibilityCI}}{{{h3['ci_low']:.5f} to {h3['ci_high']:.5f}}}
\newcommand{{\DisruptionDistanceEffect}}{{{h4['estimate']:.3f}}}
\newcommand{{\DisruptionDistanceCI}}{{{h4['ci_low']:.3f} to {h4['ci_high']:.3f}}}
\newcommand{{\FullRepresentationAccuracy}}{{{full_accuracy:.3f}}}
\newcommand{{\FullRepresentationIncrement}}{{{h6['estimate']:.3f}}}
\newcommand{{\FullRepresentationIncrementCI}}{{{h6['ci_low']:.3f} to {h6['ci_high']:.3f}}}
\newcommand{{\HOneStatement}}{{{statements['H1']}}}
\newcommand{{\HTwoStatement}}{{{statements['H2']}}}
\newcommand{{\HThreeStatement}}{{{statements['H3']}}}
\newcommand{{\HFourStatement}}{{{statements['H4']}}}
\newcommand{{\HFiveStatement}}{{{statements['H5']}}}
\newcommand{{\HSixStatement}}{{{statements['H6']}}}
\newcommand{{\HSevenStatement}}{{{statements['H7']}}}
"""
    paper = repository / "paper/jstat_v13"
    paper.mkdir(parents=True, exist_ok=True)
    atomic_bytes(macros.encode("utf-8"), paper / "results_macros.tex")


def build_results(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    root = repository / "results/collective_agent_statmech_v13"
    protocol_path = repository / "configs/statmech_v13/protocol_frozen_v1.2.yaml"
    protocol = load_yaml(protocol_path)
    primary = json.loads((root / "statistics/primary_results.json").read_text(encoding="utf-8"))
    replay_path = artifact_root() / "reproducibility/replay_summary.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {"status": "not_run"}
    root.joinpath("protocol").mkdir(parents=True, exist_ok=True)
    atomic_bytes(protocol_path.read_bytes(), root / "protocol/protocol_frozen_v1.2.yaml")
    invalidated = repository / "configs/statmech_v13/protocol_frozen.yaml"
    if invalidated.exists():
        atomic_bytes(invalidated.read_bytes(), root / "protocol/protocol_frozen_v1.0_invalidated.yaml")
    invalidated_v11 = repository / "configs/statmech_v13/protocol_frozen_v1.1.yaml"
    if invalidated_v11.exists():
        atomic_bytes(invalidated_v11.read_bytes(), root / "protocol/protocol_frozen_v1.1_invalidated.yaml")
    root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    atomic_json(primary["formal_accounting"], root / "logs/formal_completion.json")
    _write_prose(repository, protocol, primary)
    reproducibility = root / "reproducibility"
    reproducibility.mkdir(parents=True, exist_ok=True)
    external = {
        "artifact_root": str(artifact_root()),
        "pilot": tree_digest(artifact_root() / "pilot"),
        "formal": tree_digest(artifact_root() / "formal"),
        "raw": tree_digest(artifact_root() / "raw"),
        "analysis": tree_digest(artifact_root() / "analysis"),
    }
    completion = primary["formal_accounting"]
    pilot_accounting = protocol["engineering_pilot"]["provider_environment"]["accounting"]  # type: ignore[index]
    total_gpu_hours = float(completion["all_formal_generation_gpu_hours_including_invalidated"]) + float(pilot_accounting["latency_seconds"]) / 3600.0
    compute = {
        "analyzed_formal_decision_rows": completion["observed_decision_rows"],
        "formal_attempted_decisions_including_invalidated": completion["all_formal_attempted_decisions_including_invalidated"],
        "pilot_decisions": pilot_accounting["decision_requests"],
        "total_model_calls": int(completion["all_formal_model_calls_including_invalidated"]) + int(pilot_accounting["model_calls"]),
        "total_prompt_tokens": int(completion["all_formal_prompt_tokens_including_invalidated"]) + int(pilot_accounting["prompt_tokens"]),
        "total_generated_tokens": int(completion["all_formal_generated_tokens_including_invalidated"]) + int(pilot_accounting["generated_tokens"]),
        "total_generation_gpu_hours": total_gpu_hours,
        "estimated_cost_usd_low": 0.34 * total_gpu_hours,
        "estimated_cost_usd_high": 0.69 * total_gpu_hours,
        "analysis_cpu_seconds": primary["analysis_cpu_seconds"],
        "analysis_wall_seconds": primary["analysis_wall_seconds"],
    }
    atomic_json(external, reproducibility / "external_artifact_summary.json")
    atomic_json(compute, reproducibility / "compute_accounting.json")
    atomic_json(replay, reproducibility / "replay_summary.json")
    summary = {
        "generated_at": utc_now(),
        "parent_commit": PARENT,
        "protocol_version": protocol["protocol"],
        "protocol_sha256": sha256_file(protocol_path),
        "execution_source_sha256": protocol["provenance"]["execution_source_sha256"],
        "model": protocol["model"],
        "formal": completion,
        "primary": {key: primary[key] for key in ("primary_confirmatory", "disruption_hypotheses", "H7_surrogate")},
        "replay": replay,
        "external_artifacts": external,
        "compute": compute,
    }
    atomic_json(summary, reproducibility / "summary.json")
    manifest = _manifest(repository)
    atomic_csv(manifest, reproducibility / "repository_manifest.csv")
    return {"status": "built", "repository_files": len(manifest), "protocol_sha256": summary["protocol_sha256"]}


def _run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def validate_pdfs(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    root = repository / "results/collective_agent_statmech_v13"
    paths = sorted((root / "figures/pdf").glob("*.pdf"))
    manuscript = repository / "paper/jstat_v13/main.pdf"
    if manuscript.exists():
        paths.append(manuscript)
    qa_root = artifact_root() / "reproducibility/qa_png"
    qa_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for path in paths:
        info = _run(["pdfinfo", str(path)])
        fonts = _run(["pdffonts", str(path)])
        text_path = qa_root / (path.stem + ".txt")
        extract = _run(["pdftotext", str(path), str(text_path)])
        prefix = qa_root / path.stem
        render = _run(["pdftoppm", "-png", "-r", "300", str(path), str(prefix)])
        font_lines = [line for line in fonts.stdout.splitlines()[2:] if line.strip()]
        embedded = bool(font_lines) and all(" yes " in (" " + line.lower() + " ") for line in font_lines)
        extracted_bytes = text_path.stat().st_size if text_path.exists() else 0
        renders = sorted(qa_root.glob(path.stem + "-*.png"))
        pages = 0
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                pages = int(line.split(":", 1)[1].strip())
        rows.append(
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "opens": int(info.returncode == 0),
                "page_count": pages,
                "fonts_embedded": int(embedded),
                "text_extractable": int(extract.returncode == 0 and extracted_bytes > 0),
                "render_300_dpi": int(render.returncode == 0 and len(renders) == pages),
                "rendered_pages": len(renders),
                "manual_original_resolution": "pending",
                "clipping_or_overlap": "pending",
                "reviewer": "pending",
            }
        )
    if not rows:
        raise RuntimeError("no V13 PDFs exist for QA")
    atomic_csv(rows, root / "reproducibility/pdf_qa.csv")
    return {
        "pdf_count": len(rows),
        "page_count": int(sum(int(row["page_count"]) for row in rows)),
        "automated_pass": bool(all(row["opens"] and row["fonts_embedded"] and row["text_extractable"] and row["render_300_dpi"] for row in rows)),
        "manual_status": "pending",
        "render_root": str(qa_root),
    }


def verify_package(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    root = repository / "results/collective_agent_statmech_v13"
    protocol = load_yaml(repository / "configs/statmech_v13/protocol_frozen_v1.2.yaml")
    completion = json.loads((artifact_root() / "formal/completion.json").read_text(encoding="utf-8"))
    replay = json.loads((artifact_root() / "reproducibility/replay_summary.json").read_text(encoding="utf-8"))
    qa = pd.read_csv(root / "reproducibility/pdf_qa.csv")
    files = _repository_files(repository)
    forbidden_suffixes = {".jsonl", ".safetensors", ".pt", ".bin", ".npy", ".npz", ".tar", ".zip"}
    secret_parts = ("jupyter" + "_token=", "api" + "_key=", "begin openssh" + " private key", "hf" + "_token=")
    secret_hits = []
    for path in files:
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".tex", ".bib", ".sh", ".json", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(token in text for token in secret_parts):
                secret_hits.append(path.relative_to(repository).as_posix())
    diff = _run(["git", "diff", "--name-only", PARENT, "--"])
    changed = diff.stdout.splitlines()
    frozen_changes = [path for path in changed if any(token in path for token in ("statmech_v12", "llm_agent_statmech_v12", "jstat_v12", "notes/v12_"))]
    staged = _run(["git", "diff", "--cached", "--name-only"])
    source_files = list((root / "figures/source_data").glob("*.csv"))
    figures = list((root / "figures/pdf").glob("*.pdf"))
    checks = {
        "formal_complete": completion["status"] == "complete" and int(completion["observed_decision_rows"]) == int(protocol["compute"]["expected_formal_decisions"]),  # type: ignore[index]
        "source_frozen": execution_source_checksum(repository) == str(protocol["provenance"]["execution_source_sha256"]),  # type: ignore[index]
        "replay": replay["status"] == "passed" and int(replay["rows_checked"]) == int(completion["observed_decision_rows"]),
        "privacy": json.loads((root / "statistics/primary_results.json").read_text(encoding="utf-8"))["privacy_mutations"] == 0,
        "v12_immutable": not frozen_changes,
        "index_empty": staged.returncode == 0 and not staged.stdout.strip(),
        "pdf_automation": bool((qa[["opens", "fonts_embedded", "text_extractable", "render_300_dpi"]] == 1).all().all()),
        "pdf_manual": bool((qa["manual_original_resolution"] == "passed").all() and (qa["clipping_or_overlap"] == "none").all()),
        "figure_sources": len(figures) == 22 and len(source_files) >= 22,
        "repository_size": sum(path.stat().st_size for path in files) < 30 * 1024 * 1024,
        "individual_file_size": not [path for path in files if path.stat().st_size > 10 * 1024 * 1024],
        "forbidden_artifacts": not [path for path in files if path.suffix.lower() in forbidden_suffixes or "__pycache__" in path.parts],
        "secrets": not secret_hits,
    }
    result = {
        "generated_at": utc_now(),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "repository_facing_bytes": int(sum(path.stat().st_size for path in files)),
        "repository_facing_files": len(files),
        "largest_files": sorted(
            ({"path": path.relative_to(repository).as_posix(), "bytes": path.stat().st_size} for path in files),
            key=lambda item: int(item["bytes"]), reverse=True,
        )[:20],
        "frozen_parent_changes": frozen_changes,
        "staged_files": staged.stdout.splitlines(),
        "secret_hits": secret_hits,
    }
    atomic_json(result, root / "reproducibility/verification.json")
    if not result["passed"]:
        raise RuntimeError("V13 package verification failed: %s" % [key for key, value in checks.items() if not value])
    return result
