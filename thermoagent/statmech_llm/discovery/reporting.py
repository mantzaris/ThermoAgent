"""Compact publication package, PDF QA, and discovery-stage integrity checks."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from .workflow import (
    artifact_root,
    atomic_csv,
    atomic_json,
    execution_source_checksum,
    external_manifest,
    load_yaml,
    repository_root,
    sha256_file,
    utc_now,
)


ANALYSIS_TABLES = (
    "panel_statistics.csv",
    "agent_statistics.csv",
    "microscopic_models.csv",
    "cluster_effects.csv",
    "nonreciprocity_dose_response.csv",
    "orientation_replication.csv",
    "collective_factor_effects.csv",
    "control_effects.csv",
    "memory_effects.csv",
    "quadratic_models.csv",
    "probability_currents.csv",
    "hysteresis.csv",
    "controls.csv",
    "memory.csv",
    "relaxation.csv",
    "fitted_surrogate.csv",
)


def _results(repository: Path) -> Path:
    return Path(repository) / "results/JSTAT/stages/discovery"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _format_effect(value: Mapping[str, object]) -> str:
    return "{estimate:.4g} (95% CI {ci_low:.4g}, {ci_high:.4g}; n={n:d} clusters)".format(
        estimate=float(value["estimate"]),
        ci_low=float(value["ci_low"]),
        ci_high=float(value["ci_high"]),
        n=int(float(value["independent_clusters"])),
    )


def _claim_disposition(primary: Mapping[str, object]) -> Dict[str, Tuple[str, str]]:
    h1 = primary["H1_individual_neighbor_response"]  # type: ignore[index]
    effects = primary["paired_effects"]  # type: ignore[index]
    dose = primary["nonreciprocity_dose_response"]  # type: ignore[index]
    orientation = primary["orientation_replication"]  # type: ignore[index]
    quadratic = primary["quadratic_models"]  # type: ignore[index]
    h1_pass = float(h1["ci_low"]) > 0.0
    h2_small = effects["small_network:adjusted_block_kl_nats_per_update"]
    h2_large = effects["collective_network:adjusted_block_kl_nats_per_update"]
    h2_pass = float(h2_small["ci_low"]) > 0.0 and float(h2_large["ci_low"]) > 0.0
    h3_pass = all(float(dose[key]["ci_low"]) > 0.0 for key in ("small_network", "collective_network"))
    quadratic_rows = [row for row in quadratic if row["model"] == "quadratic"]
    h4_positive = all(float(row["quadratic_ci_low"]) > 0.0 for row in quadratic_rows)
    h4_predictive = all(
        float(row["leave_cluster_out_rmse"])
        <= min(
            float(other["leave_cluster_out_rmse"])
            for other in quadratic
            if other["family"] == row["family"]
        )
        for row in quadratic_rows
    )
    factor = primary["collective_factor_effects"]  # type: ignore[index]
    h5_pass = any(float(value["holm_adjusted_pvalue"]) < 0.05 for value in factor.values())
    h6_pass = bool(orientation) and all(float(value["ci_low"]) > 0.0 for value in orientation.values())
    return {
        "H1": ("supported" if h1_pass else "not supported", "paired latent choice response to neighbor field"),
        "H2": ("supported" if h2_pass else "not supported", "strong directed-minus-reciprocal path irreversibility"),
        "H3": ("supported" if h3_pass else "not supported", "positive cluster-level nonreciprocity dose response"),
        "H4": (
            "supported" if h4_positive and h4_predictive else ("mixed" if h4_positive else "not supported"),
            "positive quadratic component and held-cluster predictive comparison",
        ),
        "H5": ("supported" if h5_pass else "not supported", "Holm-adjusted collective response family"),
        "H6": ("supported" if h6_pass else "not supported", "orientation and size replication"),
    }


def _tree_summary(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not root.exists():
        return rows
    for stage in sorted(path for path in root.iterdir() if path.is_dir()):
        digest = hashlib.sha256()
        count = 0
        size = 0
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            relative = path.relative_to(stage).as_posix()
            file_digest = sha256_file(path)
            digest.update(relative.encode("utf-8") + b"\0" + file_digest.encode("ascii") + b"\0")
            count += 1
            size += int(path.stat().st_size)
        rows.append(
            {
                "external_stage": stage.name,
                "file_count": count,
                "total_bytes": size,
                "tree_sha256": digest.hexdigest(),
            }
        )
    return rows


def _compute_accounting(root: Path, completion: Mapping[str, object]) -> Dict[str, object]:
    pilot_summaries: List[Mapping[str, object]] = []
    for path in sorted((root / "pilot").rglob("summary.json")):
        pilot_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    pilot = {
        "decision_requests": 0,
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "latency_seconds": 0.0,
        "valid_pilot_summaries": len(pilot_summaries),
    }
    for summary in pilot_summaries:
        accounting = summary.get("provider_environment", {}).get("accounting", {})  # type: ignore[union-attr]
        for key in ("decision_requests", "model_calls", "prompt_tokens", "generated_tokens"):
            pilot[key] += int(accounting.get(key, 0))  # type: ignore[union-attr]
        pilot["latency_seconds"] += float(accounting.get("latency_seconds", 0.0))  # type: ignore[union-attr]
    formal = completion["tokens_and_latency"]  # type: ignore[index]
    total = {
        "decision_requests": int(pilot["decision_requests"]) + int(formal["decision_requests"]),
        "model_calls": int(pilot["model_calls"]) + int(formal["model_calls"]),
        "prompt_tokens": int(pilot["prompt_tokens"]) + int(formal["prompt_tokens"]),
        "generated_tokens": int(pilot["generated_tokens"]) + int(formal["generated_tokens"]),
        "generation_latency_seconds": float(pilot["latency_seconds"]) + float(formal["latency_seconds"]),
    }
    total["metered_generation_gpu_hours"] = float(total["generation_latency_seconds"] / 3600.0)
    # Historical engineering attempts predate model-load timing.  Therefore
    # this exact call-level total is a lower bound on occupied GPU wall time.
    total["gpu_time_scope"] = "exact summed generation latency; excludes unmetered model-load and shell overhead"
    total["estimated_cost_usd_range_from_generation_time"] = [
        float(total["metered_generation_gpu_hours"] * 0.34),
        float(total["metered_generation_gpu_hours"] * 0.69),
    ]
    return {"pilot": pilot, "formal": formal, "total": total}


def _repository_manifest(repository: Path) -> List[Dict[str, object]]:
    roots = [_results(repository), repository / "paper/JSTAT"]
    rows: List[Dict[str, object]] = []
    manifest_path = _results(repository) / "reproducibility/repository_manifest.csv"
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path == manifest_path or path.name == "verification.json" or path.suffix in (
                ".aux", ".log", ".out", ".fls", ".fdb_latexmk"
            ):
                continue
            rows.append(
                {
                    "relative_path": path.relative_to(repository).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def build_results(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    root = artifact_root()
    analysis_root = root / "analysis"
    primary = json.loads((analysis_root / "primary_results.json").read_text(encoding="utf-8"))
    completion = json.loads((root / "formal/completion.json").read_text(encoding="utf-8"))
    compute = _compute_accounting(root, completion)
    compute["analysis"] = {
        "cpu_seconds": float(primary.get("analysis_cpu_seconds", 0.0)),
        "wall_seconds": float(primary.get("analysis_wall_seconds", 0.0)),
    }
    protocol_path = repository / "configs/statmech_llm/discovery/protocol.yaml"
    protocol = load_yaml(protocol_path)
    results = _results(repository)
    for directory in ("protocol", "statistics", "tables", "reproducibility", "logs", "figures/pdf", "figures/source_data"):
        (results / directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol_path, results / "protocol/protocol_frozen.yaml")
    shutil.copyfile(analysis_root / "primary_results.json", results / "statistics/primary_results.json")
    shutil.copyfile(
        analysis_root / "fitted_surrogate_parameters.json",
        results / "statistics/fitted_surrogate_parameters.json",
    )
    for name in ANALYSIS_TABLES:
        shutil.copyfile(analysis_root / name, results / "tables" / name)
    pilot_attempt = str(protocol["pilot_estimability"]["attempt_id"])  # type: ignore[index]
    pilot_path = root / "pilot" / pilot_attempt / "summary.json"
    shutil.copyfile(pilot_path, results / "logs/engineering_pilot_summary.json")
    shutil.copyfile(root / "formal/completion.json", results / "logs/formal_completion.json")
    replay_path = root / "reproducibility/replay_summary.json"
    if not replay_path.exists():
        raise RuntimeError("formal deterministic replay has not completed")
    shutil.copyfile(replay_path, results / "reproducibility/replay_summary.json")
    claims = _claim_disposition(primary)
    h1 = primary["H1_individual_neighbor_response"]
    strong_small = primary["paired_effects"]["small_network:adjusted_block_kl_nats_per_update"]
    strong_large = primary["paired_effects"]["collective_network:adjusted_block_kl_nats_per_update"]
    readme = f"""# V12: collective dynamics and entropy production in decentralized LLM-agent networks

## Disposition

This is a complete, frozen, development-independent formal experiment using actual decentralized Qwen agents. It does not alter V1--V11. Its primary state is the realized categorical belief--action projection; confidence, commitment, bounded memory, inbox, outbox, private field, and workload remain explicit observables. One sweep is exactly N attempted random-sequential agent updates. The graph trajectory, not a token or time step, is the independent unit.

The projected Markov entropy-production statistic is reported in nats per attempted update (and multiplied by N for nats per sweep). Whenever first-order closure is inadequate, the primary empirical quantity is the bias-adjusted block time-reversal KL, described only as coarse-grained path irreversibility or a lower bound—not exact total thermodynamic entropy production.

## Main numerical results

- H1 individual neighbor response: {_format_effect(h1)}.
- Strong-alpha small-network excess path irreversibility: {_format_effect(strong_small)}.
- Strong-alpha collective-network excess path irreversibility: {_format_effect(strong_large)}.
- Decisions: {int(completion['observed_decision_rows']):,}; dynamic panels: {int(completion['dynamic_panel_count']):,}; planned decisions: {int(completion['planned_decisions']):,}.
- All V12 engineering and formal calls: {int(compute['total']['model_calls']):,}; prompt tokens: {int(compute['total']['prompt_tokens']):,}; generated tokens: {int(compute['total']['generated_tokens']):,}; exact summed generation latency: {float(compute['total']['metered_generation_gpu_hours']):.3f} GPU-hours (model-load overhead excluded).
- Messages: {int(primary['total_messages']):,}; complete binary wire bytes: {int(primary['total_wire_bytes']):,}; privacy mutations: {int(primary['privacy_mutations'])}.

## Frozen design

- Model: `{protocol['model']['identifier']}` at revision `{protocol['model']['revision']}`, NF4 with BF16 computation.
- Sizes: N=3 and 4 for transition-current studies; N=8 and 16 for collective studies.
- Topologies: fixed-degree ring and fixed-degree modular graphs.
- Nonreciprocity: alpha in {protocol['network']['nonreciprocity_levels']}, with paired forward/transposed orientations and unchanged support, weighted degree, payload, and opportunity count.
- Coupling/noise: a 2x2 collective factorial over {protocol['collective_network']['coupling_strengths']} and {protocol['collective_network']['inference_sampling_temperatures']}.
- Memory: matched Markovized and bounded-persistent-memory panels.
- Controls: no message, reciprocal, directed, orientation reversal, content/time/sender permutation, natural-language placebo, V10 heat-bath reference, and fitted kinetic-Ising surrogate.

## Hypotheses

""" + "\n".join(f"- {key}: **{status}** — {basis}." for key, (status, basis) in claims.items()) + f"""

## Limits and prohibited claims

The model is one pinned open-weight instruction model, not a universal LLM population. Decoding temperature is a control parameter, not physical temperature. The reference energy is an effective equilibrium-reference observable. No thermodynamic-limit phase transition, universal exponent, Bayesian rationality, controller superiority, operational application benefit, or real-human effect is claimed. Short large-network trajectories motivate careful finite-size and mixing qualifications.

## Reproduction order

```bash
scripts/run-tests.sh tests/statmech_llm/discovery
THERMOAGENT_ENABLE_LLM=1 scripts/run-formal-experiment.sh discovery
scripts/replay-results.sh discovery
scripts/analyze-results.sh discovery
scripts/generate-figures.sh
scripts/build-jstat-paper.sh
scripts/verify-results.sh
```

Raw model interactions and panel trajectories are external at `{protocol['execution']['external_artifact_root']}`. Compact tree checksums are in `reproducibility/external_artifact_trees.csv`.
"""
    _write_text(results / "README.md", readme)
    summary = f"""# Paper summary

V12 realizes a random-sequential stochastic network with genuinely independent LLM agents. Each model call chooses a local belief, action, commitment, bounded memory state, outgoing signal, and typed tool action from only its private field, current local state, and delivered inbox. A divergence-free antisymmetric circulation changes communication reciprocity without changing support, weighted degree, message opportunity count, or binary packet schema.

The study compares exact V10 heat-bath theory, a fitted kinetic-Ising surrogate, and direct Qwen trajectories. H1 was {claims['H1'][0]}; the strong nonreciprocity comparison was {claims['H2'][0]}; the dose response was {claims['H3'][0]}; quadratic compatibility was {claims['H4'][0]}. All negative and mixed findings remain in the package.

The defensible contribution is conditional on the recorded Markov-adequacy diagnostics. Exact equilibrium and entropy-production claims belong only to the V10 heat-bath reference. V12's empirical LLM currents are projected categorical-state diagnostics, and its block reversal divergence is a coarse-grained pathwise irreversibility measure.
"""
    _write_text(results / "PAPER_SUMMARY.md", summary)
    matrix = "# Claims matrix\n\n| Claim | Disposition | Evidentiary basis | Prohibited extension |\n|---|---|---|---|\n"
    for key, (status, basis) in claims.items():
        matrix += f"| {key} | {status} | {basis} | No universality, exact hidden-state entropy production, or thermodynamic-limit claim |\n"
    matrix += "| Agent independence | supported engineering fact | privacy/scheduler tests and zero recorded peer-private mutations | Does not imply organizational or human autonomy |\n"
    matrix += "| Reference energy | defined effective observable | symmetric empirical influence layer | Not literal physical energy |\n"
    _write_text(results / "CLAIMS_MATRIX.md", matrix)
    external_rows = _tree_summary(root)
    atomic_csv(external_rows, results / "reproducibility/external_artifact_trees.csv")
    reproducibility = {
        "generated_at": utc_now(),
        "parent_v11_commit": protocol["provenance"]["parent_commit"],
        "protocol_sha256": sha256_file(protocol_path),
        "execution_source_sha256": execution_source_checksum(repository),
        "frozen_execution_source_sha256": protocol["provenance"]["execution_source_sha256"],
        "model": completion["provider_environment"],
        "decision_rows": completion["observed_decision_rows"],
        "raw_artifact_root": str(root),
        "external_stage_tree_count": len(external_rows),
        "claims": {key: value[0] for key, value in claims.items()},
    }
    atomic_json(reproducibility, results / "reproducibility/summary.json")
    atomic_json(compute, results / "reproducibility/compute_accounting.json")
    final_status = "# V12 final status\n\n" + "\n".join(
        [
            f"- Protocol: `{sha256_file(protocol_path)}`.",
            f"- Source: `{execution_source_checksum(repository)}`.",
            f"- Formal rows: {int(completion['observed_decision_rows']):,}.",
            f"- Claims: " + ", ".join(f"{key} {value[0]}" for key, value in claims.items()) + ".",
            "- V1--V11 remain immutable; all raw V12 model interactions are external.",
            "- This working tree remains uncommitted and unpushed for human review.",
        ]
    )
    _write_text(repository / "notes/v12_final_status.md", final_status)
    manifest = _repository_manifest(repository)
    atomic_csv(manifest, results / "reproducibility/repository_manifest.csv")
    return {
        "result_files_indexed": len(manifest),
        "external_stage_trees": len(external_rows),
        "claims": {key: value[0] for key, value in claims.items()},
        "protocol_sha256": sha256_file(protocol_path),
    }


def _command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(arguments), check=False, text=True, capture_output=True)


def validate_pdfs(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result_root = _results(repository)
    paths = sorted((result_root / "figures/pdf").glob("*.pdf"))
    manuscript = repository / "paper/JSTAT/main.pdf"
    if manuscript.exists():
        paths.append(manuscript)
    qa_root = artifact_root() / "pdf_qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    prior_path = result_root / "reproducibility/pdf_qa.csv"
    prior_manual: Dict[str, bool] = {}
    if prior_path.exists():
        for row in pd.read_csv(prior_path).to_dict(orient="records"):
            prior_manual[str(row["relative_path"])] = bool(row.get("manual_original_resolution_passed", False))
    for path in paths:
        relative = path.relative_to(repository).as_posix()
        info = _command(["pdfinfo", str(path)])
        fonts = _command(["pdffonts", str(path)])
        text_output = qa_root / (path.stem + ".txt")
        text_result = _command(["pdftotext", str(path), str(text_output)])
        render_prefix = qa_root / path.stem
        render = _command(["pdftoppm", "-png", "-r", "300", str(path), str(render_prefix)])
        font_lines = [line for line in fonts.stdout.splitlines()[2:] if line.strip()]
        embedded = bool(font_lines) and all(" yes " in (" " + line.lower() + " ") for line in font_lines)
        extracted = text_output.read_text(encoding="utf-8", errors="ignore") if text_output.exists() else ""
        renders = sorted(qa_root.glob(path.stem + "-*.png"))
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "opens": info.returncode == 0,
                "page_count": len(renders),
                "fonts_embedded": embedded,
                "text_extractable": text_result.returncode == 0 and bool(extracted.strip()),
                "render_300dpi": render.returncode == 0 and bool(renders),
                "manual_original_resolution_passed": prior_manual.get(relative, False),
                "qa_render_directory": str(qa_root),
            }
        )
    atomic_csv(records, result_root / "reproducibility/pdf_qa.csv")
    return {
        "pdf_count": len(records),
        "automated_pass": bool(records) and all(
            bool(row["opens"] and row["fonts_embedded"] and row["text_extractable"] and row["render_300dpi"])
            for row in records
        ),
        "manual_pass_count": sum(bool(row["manual_original_resolution_passed"]) for row in records),
        "render_root": str(qa_root),
    }


def verify_package(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    results = _results(repository)
    protocol_path = repository / "configs/statmech_llm/discovery/protocol.yaml"
    protocol = load_yaml(protocol_path)
    completion = json.loads((artifact_root() / "formal/completion.json").read_text(encoding="utf-8"))
    replay = json.loads(
        (artifact_root() / "reproducibility/replay_summary.json").read_text(encoding="utf-8")
    )
    primary = json.loads((results / "statistics/primary_results.json").read_text(encoding="utf-8"))
    qa = pd.read_csv(results / "reproducibility/pdf_qa.csv")
    indexed = pd.read_csv(results / "reproducibility/repository_manifest.csv")
    checksum_failures = []
    for row in indexed.itertuples():
        path = repository / row.relative_path
        if not path.exists() or sha256_file(path) != row.sha256:
            checksum_failures.append(str(row.relative_path))
    new_roots = [
        repository / "thermoagent/statmech_llm/discovery",
        repository / "configs/statmech_llm/discovery",
        repository / "tests/statmech_llm/discovery",
        results,
        repository / "paper/JSTAT",
    ]
    files = [path for root in new_roots if root.exists() for path in root.rglob("*") if path.is_file()]
    size = int(sum(path.stat().st_size for path in files))
    oversized = [path.relative_to(repository).as_posix() for path in files if path.stat().st_size > 10 * 1024 * 1024]
    forbidden = [
        path.relative_to(repository).as_posix()
        for path in files
        if path.suffix.lower() in (".jsonl", ".safetensors", ".pt", ".bin", ".tar", ".zip", ".npy", ".npz")
    ]
    staged = _command(["git", "diff", "--cached", "--name-only"])
    checks = {
        "formal_complete": completion.get("status") == "complete",
        "row_accounting": int(completion["observed_decision_rows"]) == int(completion["planned_decisions"]),
        "source_frozen": execution_source_checksum(repository) == protocol["provenance"]["execution_source_sha256"],
        "privacy": int(primary["privacy_mutations"]) == 0,
        "deterministic_replay": replay.get("status") == "passed"
        and int(replay.get("rows_checked", -1)) == int(completion["observed_decision_rows"]),
        "checksums": not checksum_failures,
        "pdf_automated": bool(qa[["opens", "fonts_embedded", "text_extractable", "render_300dpi"]].all().all()),
        "pdf_manual": bool(qa["manual_original_resolution_passed"].all()),
        "repository_size": size < 30 * 1024 * 1024,
        "individual_file_size": not oversized,
        "forbidden_artifacts": not forbidden,
        "empty_git_index": staged.returncode != 0 or not staged.stdout.strip(),
    }
    output = {
        "verified_at": utc_now(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "repository_facing_v12_bytes": size,
        "checksum_failures": checksum_failures,
        "oversized_files": oversized,
        "forbidden_files": forbidden,
    }
    atomic_json(output, results / "reproducibility/verification.json")
    return output
