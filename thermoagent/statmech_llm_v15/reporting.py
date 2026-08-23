"""Publication summaries, inventories, PDF QA, and scientific integrity checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from .provider import MODEL_SPECS, schema_checksum
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


LATEX_INTERMEDIATES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def _repository_files(repository: Path) -> List[Path]:
    roots = (
        repository / "configs/statmech_v15",
        repository / "thermoagent/statmech_llm_v15",
        repository / "tests/statmech_v15",
        repository / "results/collective_agent_statmech_v15",
        repository / "paper/jstat_v15",
    )
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(path.name.endswith(suffix) for suffix in LATEX_INTERMEDIATES)
    ]
    files.extend((repository / "scripts").glob("*statmech-v15*"))
    files.extend((repository / "notes").glob("v15_*.md"))
    files.extend(
        (
            repository / "configs/statmech_v14/scientific_audit_v1.1.yaml",
            repository / "thermoagent/statmech_llm_v14/analysis.py",
            repository / "thermoagent/statmech_llm_v14/figures.py",
            repository / "thermoagent/statmech_llm_v14/observables.py",
            repository / "thermoagent/statmech_llm_v14/reporting.py",
            repository / "tests/statmech_v14/test_analysis.py",
            repository / "tests/statmech_v14/test_end_to_end_cpu.py",
            repository / "tests/statmech_v14/test_observables.py",
        )
    )
    if (repository / ".gitignore").exists():
        files.append(repository / ".gitignore")
    return sorted(set(path for path in files if path.is_file()))


def _manifest(repository: Path) -> pd.DataFrame:
    rows = []
    for path in _repository_files(repository):
        if path.name in ("INDEX.csv", "repository_manifest.csv"):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def _effect(effects: pd.DataFrame, hypothesis: str) -> pd.Series:
    selected = effects[effects["hypothesis"] == hypothesis]
    if len(selected) != 1:
        raise RuntimeError("expected one V15 effect for %s" % hypothesis)
    return selected.iloc[0]


def _format(row: pd.Series, digits: int = 4) -> str:
    return "%.*f (95%% CI %.*f to %.*f)" % (
        digits,
        float(row.estimate),
        digits,
        float(row.ci_low),
        digits,
        float(row.ci_high),
    )


def _pilot_accounting() -> Dict[str, object]:
    values = []
    qualified_failures: List[Dict[str, object]] = []
    for model in ("qwen", "granite"):
        path = artifact_root() / "pilot" / (model + "_summary.json")
        values.append(json.loads(path.read_text(encoding="utf-8")))
        failure_path = artifact_root() / "pilot" / (model + "_failures.json")
        if failure_path.exists():
            failures = json.loads(failure_path.read_text(encoding="utf-8"))
            qualified_failures.extend(failures if isinstance(failures, list) else [])
    rejected: List[Dict[str, object]] = []
    rejected_summary = artifact_root() / "pilot/mistral_summary.json"
    if rejected_summary.exists():
        summary = json.loads(rejected_summary.read_text(encoding="utf-8"))
        rejected.append(
            {
                "model_key": "mistral",
                "classification": "engineering_rejected_before_freeze",
                "decision_requests": int(summary.get("decision_requests", 0)),
                "model_calls": int(
                    summary.get("provider_environment", {}).get("accounting", {}).get(
                        "model_calls", 0
                    )
                ),
                "prompt_tokens": int(
                    summary.get("provider_environment", {}).get("accounting", {}).get(
                        "prompt_tokens", 0
                    )
                ),
                "generated_tokens": int(
                    summary.get("provider_environment", {}).get("accounting", {}).get(
                        "generated_tokens", 0
                    )
                ),
                "latency_seconds": float(
                    summary.get("provider_environment", {}).get("accounting", {}).get(
                        "latency_seconds", 0.0
                    )
                ),
                "valid_after_repair_fraction": float(
                    summary.get("valid_after_repair_fraction", 0.0)
                ),
                "scientific_contrasts_inspected": False,
            }
        )
    rejected_failures = artifact_root() / "pilot/mistral_failures.json"
    if rejected_failures.exists():
        failures = json.loads(rejected_failures.read_text(encoding="utf-8"))
        for failure in failures if isinstance(failures, list) else []:
            rejected.append(dict(failure))
    rejected_totals = {
        "decisions": int(sum(int(value.get("decision_requests", 0)) for value in rejected)),
        "calls": int(sum(int(value.get("model_calls", 0)) for value in rejected)),
        "prompt_tokens": int(sum(int(value.get("prompt_tokens", 0)) for value in rejected)),
        "generated_tokens": int(
            sum(int(value.get("generated_tokens", 0)) for value in rejected)
        ),
        "latency_seconds": float(
            sum(float(value.get("latency_seconds", 0.0)) for value in rejected)
        ),
    }
    qualified_failure_totals = {
        "decisions": int(
            sum(int(value.get("decision_requests", 0)) for value in qualified_failures)
        ),
        "calls": int(sum(int(value.get("model_calls", 0)) for value in qualified_failures)),
        "prompt_tokens": int(
            sum(int(value.get("prompt_tokens", 0)) for value in qualified_failures)
        ),
        "generated_tokens": int(
            sum(int(value.get("generated_tokens", 0)) for value in qualified_failures)
        ),
        "latency_seconds": float(
            sum(float(value.get("latency_seconds", 0.0)) for value in qualified_failures)
        ),
    }
    return {
        "models": values,
        "decisions": int(sum(int(value["decision_requests"]) for value in values)),
        "calls": int(
            sum(
                int(value["provider_environment"]["accounting"]["model_calls"])
                for value in values
            )
        ),
        "prompt_tokens": int(
            sum(
                int(value["provider_environment"]["accounting"]["prompt_tokens"])
                for value in values
            )
        ),
        "generated_tokens": int(
            sum(
                int(value["provider_environment"]["accounting"]["generated_tokens"])
                for value in values
            )
        ),
        "latency_seconds": float(
            sum(
                float(value["provider_environment"]["accounting"]["latency_seconds"])
                for value in values
            )
        ),
        "rejected_attempts": rejected,
        "rejected_totals": rejected_totals,
        "qualified_model_infrastructure_failures": qualified_failures,
        "qualified_model_failure_totals": qualified_failure_totals,
    }


def build_results(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    primary = json.loads((result / "statistics/primary_results.json").read_text(encoding="utf-8"))
    effects = pd.read_csv(result / "tables/hypothesis_effects.csv")
    panels = pd.read_csv(result / "tables/panel_statistics.csv")
    quench = pd.read_csv(result / "tables/quench_recovery.csv")
    prompt_balance = pd.read_csv(result / "tables/memory_prompt_balance.csv")
    v14_primary = json.loads(
        (
            repository
            / "results/collective_agent_statmech_v14/statistics/primary_results.json"
        ).read_text(encoding="utf-8")
    )
    completion = primary["formal_completion"]
    pilot = _pilot_accounting()
    h1, h2, h3, h4 = (_effect(effects, key) for key in ("H1", "H2", "H3", "H4"))
    disposition = primary["confirmatory_dispositions"]
    formal_hours = float(completion["generation_gpu_hours"])
    pilot_hours = float(pilot["latency_seconds"]) / 3600.0
    rejected_hours = float(pilot["rejected_totals"]["latency_seconds"]) / 3600.0
    failure_hours = float(pilot["qualified_model_failure_totals"]["latency_seconds"]) / 3600.0
    total_hours = formal_hours + pilot_hours + rejected_hours + failure_hours
    cost = (0.34 * total_hours, 0.69 * total_hours)
    qwen_memory = panels[panels["model_key"] == "qwen"].groupby("condition")[
        "adjusted_pathwise_irreversibility_nats_per_update"
    ].mean()
    granite_memory = panels[panels["model_key"] == "granite"].groupby("condition")[
        "adjusted_pathwise_irreversibility_nats_per_update"
    ].mean()
    h3_v14 = v14_primary["confirmatory_dispositions"]["H3"]
    readme = f"""# V15: cross-model memory controls and field-quench replication

## Scientific scope

V15 treats state-separated, locally informed LLM-agent instances as an interacting finite stochastic system. Every persistent identity has its own local belief, action, confidence, commitment, bounded memory, workload, inbox, outbox, private field, and typed authority. The random-sequential scheduler chooses only update opportunities and packet delivery; model-generated structured responses determine the scientific state changes.

The complete augmented simulator state $\\Xi_t$ includes all private agent state, graph and delivery state, the quench phase, and the specified randomness source. The recorded belief-action projection $Y_t=\\phi(\\Xi_t)$ and rolling collective representation $Z_t=\\psi(Y_{{t-w+1:t}})$ need not be Markov. Genuine memory can therefore act as a hidden slow coordinate when omitted from the projection. Effective reference energy is not literal physical energy, decoding temperature is not physical temperature, and bias-adjusted path-reversal divergence is coarse-grained temporal asymmetry rather than exact thermodynamic entropy production.

## Prospective design

- Frozen protocol: `{protocol['protocol']}`; SHA-256 `{sha256_file(protocol_path)}`.
- Frozen execution source: `{protocol['provenance']['execution_source_sha256']}`.
- Parent V14 commit: `{PARENT_COMMIT}`.
- Models: Qwen `{MODEL_SPECS['qwen'].revision}` and Granite `{MODEL_SPECS['granite'].revision}`.
- Inference: NF4 double quantization, BF16 computation, decoding temperature 0.5, top-p 0.9, maximum 96 generated tokens, and one bounded greedy structured-output repair.
- Design: six independent graph/environment clusters per model, `N=16`, reciprocal modular graph, `J=0.8`, 45 sweeps (15 baseline, 15 field reversal or nominal continuation, 15 restoration).
- Matched arms: nominal Markovized, field-reversal Markovized, field-reversal genuine persistent memory, and field-reversal deterministic scrambled-history placebo.
- Independent unit: complete graph/environment trajectory cluster. Agents, updates, messages, windows, calls, and tokens are not independent replicates.

The formal study ran {int(completion['dynamic_trajectories'])} trajectories and {int(completion['observed_decision_rows']):,} attempted decisions. Formal generation used {int(completion['model_calls']):,} calls, {int(completion['prompt_tokens']):,} prompt tokens, {int(completion['generated_tokens']):,} generated tokens, and {formal_hours:.3f} metered GPU-hours. Successful Qwen/Granite engineering pilots added {pilot['decisions']} decisions and {pilot_hours:.3f} GPU-hours. Their retained infrastructure failures added {pilot['qualified_model_failure_totals']['decisions']} decision requests and {failure_hours:.3f} GPU-hours. The retained, engineering-rejected Mistral attempts used {pilot['rejected_totals']['decisions']} decision requests, {pilot['rejected_totals']['calls']} model calls, and {rejected_hours:.3f} GPU-hours; no network contrast was computed from them. Total measured generation was {total_hours:.3f} hours, with an approximate RTX 4090 cost range of USD {cost[0]:.2f}-{cost[1]:.2f}.

## Frozen hypotheses

- H1 (Granite field quench versus nominal): {_format(h1, 3)} distance units; exact sign-flip `p={float(h1.exact_one_sided_sign_flip_p):.5f}`, allocated alpha 0.02. **{('Supported' if disposition['H1']['supported'] else 'Not supported')}**.
- H2 (persistent minus Markovized path divergence, pooled across model-stratified pairs): {_format(h2, 5)} nats/update; Holm `p={float(h2.multiplicity_adjusted_p):.5f}` within the alpha-0.03 family. **{('Supported' if disposition['H2']['supported'] else 'Not supported')}**.
- H3 (persistent minus scrambled-history path divergence): {_format(h3, 5)} nats/update; Holm `p={float(h3.multiplicity_adjusted_p):.5f}`. **{('Supported' if disposition['H3']['supported'] else 'Not supported')}**.
- H4 (fixed recovery sweeps 31-35 minus 41-45): {_format(h4, 3)} distance units; Holm `p={float(h4.multiplicity_adjusted_p):.5f}`. **{('Supported' if disposition['H4']['supported'] else 'Not supported')}**.

The exact direction and model-specific heterogeneity are retained in `tables/hypothesis_effects.csv` and `tables/panel_statistics.csv`; the README does not reinterpret null or adverse signs. Qwen mean adjusted divergence was {float(qwen_memory['field_markovized']):.5f}, {float(qwen_memory['field_persistent']):.5f}, and {float(qwen_memory['field_scrambled']):.5f} nats/update for Markovized, persistent, and scrambled arms. The corresponding Granite means were {float(granite_memory['field_markovized']):.5f}, {float(granite_memory['field_persistent']):.5f}, and {float(granite_memory['field_scrambled']):.5f}. Mean persistent-minus-scrambled prompt length was {float(prompt_balance['persistent_minus_scrambled_mean_prompt_tokens'].mean()):.2f} tokens.

## V14 scientific correction

No frozen V14 decision or trajectory was altered. The versioned V14 audit now fits recovery thresholds using training clusters only, completes the frozen 10,000-replicate cluster-preserving permutation analysis, recomputes three-, five-, and seven-sweep nominal geometries, deletes individual observables, and audits finite-sample dependence bias. The historical H3 maximum-minus-final estimand, interval, raw p-value, and Holm value remain archived, but its structurally nonnegative sign makes the directional test invalid. Its machine-readable disposition is `inferential_support: false`; recovery language uses threshold re-entry, final residual, the complete path, and fixed early-versus-late descriptive changes.

## Supported boundaries

Results are finite-size and model-specific. Neither model is a human participant. No field validity, application benefit, controller advantage, performance superiority, thermodynamic-limit phase transition, physical free energy, or exact LLM entropy production is claimed. Persistent history and scrambled history are prompt-format controls; they do not make the projected binary process fully observed. Negative adjusted information quantities are retained rather than truncated.

## Reproduction order

```bash
PYTHON_BIN=/workspace/ThermoAgent/.venv/bin/python scripts/run-statmech-v15-tests.sh
THERMO_V14_ARTIFACT_ROOT=/workspace/ThermoAgent-v14-artifacts scripts/audit-statmech-v14.sh
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-pilot.sh qwen
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-pilot.sh granite
scripts/freeze-statmech-v15-protocol.sh
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-formal.sh qwen
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-formal.sh granite
scripts/replay-statmech-v15.sh
scripts/analyze-statmech-v15.sh
scripts/run-statmech-v15-surrogate.sh
scripts/generate-statmech-v15-figures.sh
scripts/build-statmech-v15-results.sh
scripts/build-statmech-v15-paper.sh
scripts/verify-statmech-v15.sh
```

Raw prompts, completions, and trajectory tables are external at `{artifact_root()}`. Compact aggregate tables, checksums, vector figures, and manuscript sources are repository-facing.
"""
    atomic_bytes(readme.encode("utf-8"), result / "README.md")
    summary = f"""# Paper summary

V15 prospectively tests memory and field-quench dynamics in state-separated Qwen and Granite agent networks. Six matched graph/environment clusters per model contribute four 45-sweep trajectories each. H1 is {_format(h1, 3)} distance units; H2 is {_format(h2, 5)} nats/update; H3 is {_format(h3, 5)} nats/update; and the non-tautological fixed-window recovery H4 is {_format(h4, 3)} distance units. Formal dispositions are H1={bool(disposition['H1']['supported'])}, H2={bool(disposition['H2']['supported'])}, H3={bool(disposition['H3']['supported'])}, and H4={bool(disposition['H4']['supported'])}.

The study separates complete augmented state, observable microscopic projection, and rolling macrostate. It compares genuine persistent history with both a Markovized state and a deterministically generated own-agent, past-only, format-matched scrambled-history placebo. V14's threshold leakage and invalid H3 inference are corrected without changing a frozen trajectory. The strongest interpretation must follow the signs and intervals above; path divergence remains coarse-grained temporal asymmetry, not exact thermodynamic entropy production.
"""
    atomic_bytes(summary.encode("utf-8"), result / "PAPER_SUMMARY.md")
    claims = []
    for row in (h1, h2, h3, h4):
        claims.append(
            {
                "claim": str(row.estimand),
                "disposition": "supported" if bool(row.supported) else "not supported",
                "evidence": _format(row, 5),
                "boundary": "%d complete graph/environment clusters; %s" % (int(row.independent_clusters), str(row.unit)),
            }
        )
    claims.extend(
        [
            {"claim": "V14 historical H3 sign test establishes relaxation", "disposition": "invalidated as directional evidence", "evidence": "structurally nonnegative peak-minus-final estimand", "boundary": "historical values archived; trajectory evidence retained"},
            {"claim": "Effective reference energy is physical energy", "disposition": "prohibited", "evidence": "not tested", "boundary": "symmetric reference coordinate only"},
            {"claim": "LLM path divergence is exact entropy production", "disposition": "prohibited", "evidence": "projected dynamics may be non-Markov", "boundary": "coarse-grained temporal asymmetry"},
            {"claim": "Result is universal across LLMs", "disposition": "unsupported", "evidence": "two pinned 7B model families", "boundary": "cross-family replication is not universality"},
            {"claim": "Statistical mechanics improves agent performance", "disposition": "not tested", "evidence": "no task-performance endpoint", "boundary": "characterization study"},
        ]
    )
    claim_frame = pd.DataFrame(claims)
    lines = ["# Claims matrix", "", "| Claim | Disposition | Evidence | Boundary |", "|---|---|---|---|"]
    lines.extend("| %s | %s | %s | %s |" % tuple(row) for row in claim_frame.itertuples(index=False, name=None))
    atomic_bytes(("\n".join(lines) + "\n").encode("utf-8"), result / "CLAIMS_MATRIX.md")
    macros = rf"""% Generated from V15 aggregate results; do not edit manually.
\newcommand{{\V15Trajectories}}{{{int(completion['dynamic_trajectories'])}}}
\newcommand{{\V15Decisions}}{{{int(completion['observed_decision_rows']):,}}}
\newcommand{{\V15ClustersPerModel}}{{{int(primary['independent_clusters_per_model'])}}}
\newcommand{{\V15GPUHours}}{{{total_hours:.2f}}}
\newcommand{{\V15HOne}}{{{float(h1.estimate):.3f}}}
\newcommand{{\V15HOneCI}}{{{float(h1.ci_low):.3f} to {float(h1.ci_high):.3f}}}
\newcommand{{\V15HTwo}}{{{float(h2.estimate):.5f}}}
\newcommand{{\V15HTwoCI}}{{{float(h2.ci_low):.5f} to {float(h2.ci_high):.5f}}}
\newcommand{{\V15HThree}}{{{float(h3.estimate):.5f}}}
\newcommand{{\V15HThreeCI}}{{{float(h3.ci_low):.5f} to {float(h3.ci_high):.5f}}}
\newcommand{{\V15HFour}}{{{float(h4.estimate):.3f}}}
\newcommand{{\V15HFourCI}}{{{float(h4.ci_low):.3f} to {float(h4.ci_high):.3f}}}
\newcommand{{\V15HOneDisposition}}{{{'supported' if disposition['H1']['supported'] else 'not supported'}}}
\newcommand{{\V15HTwoDisposition}}{{{'supported' if disposition['H2']['supported'] else 'not supported'}}}
\newcommand{{\V15HThreeDisposition}}{{{'supported' if disposition['H3']['supported'] else 'not supported'}}}
\newcommand{{\V15HFourDisposition}}{{{'supported' if disposition['H4']['supported'] else 'not supported'}}}
"""
    paper = repository / "paper/jstat_v15"
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
        "v14_raw_outcomes_modified": False,
    }
    atomic_json(external, result / "reproducibility/external_artifact_manifest.json")
    accounting = {
        "generated_at": utc_now(),
        "formal": completion,
        "pilot": pilot,
        "total_metered_generation_gpu_hours": total_hours,
        "estimated_cost_usd_range": list(cost),
        "analysis_cpu_seconds": primary["analysis_cpu_seconds"],
        "model_specs": {
            key: {"identifier": value.identifier, "revision": value.revision}
            for key, value in MODEL_SPECS.items()
        },
    }
    atomic_json(accounting, result / "reproducibility/compute_accounting.json")
    manifest = _manifest(repository)
    atomic_csv(manifest, result / "reproducibility/repository_manifest.csv")
    atomic_csv(manifest, result / "INDEX.csv")
    return {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "execution_source_sha256": protocol["provenance"]["execution_source_sha256"],
        "schema_sha256": schema_checksum(),
        "repository_files": len(manifest),
        "repository_bytes": int(manifest["bytes"].sum()),
        "external_tree_sha256": external["tree"]["tree_sha256"],
    }


def validate_pdfs(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    pdfs = sorted((result / "figures/pdf").glob("*.pdf"))
    pdfs.extend(
        path
        for path in (
            repository / "paper/jstat_v15/main.pdf",
            repository / "paper/jstat_v15/supplement.pdf",
        )
        if path.exists()
    )
    if not pdfs:
        raise RuntimeError("no V15 PDFs found")
    render_root = artifact_root() / "pdf_qa/rendered_300dpi"
    render_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for path in pdfs:
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        fonts = subprocess.check_output(["pdffonts", str(path)], text=True)
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        prefix = render_root / (path.parent.name + "_" + path.stem)
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", str(path), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pages = int(
            next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:"))
        )
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        embedded = bool(font_lines) and all(
            len(line.split()) >= 6 and line.split()[5].lower() == "yes"
            for line in font_lines
        )
        rendered = sorted(render_root.glob(prefix.name + "-*.png"))
        rows.append(
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "pages": pages,
                "opens": True,
                "fonts_embedded": embedded,
                "text_extractable": bool(text.strip()),
                "rendered_pages": len(rendered),
                "render_dpi": 300,
                "manual_visual_status": "pending",
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(rows, result / "reproducibility/pdf_qa.csv")
    summary = {
        "generated_at": utc_now(),
        "pdf_count": len(rows),
        "pages": int(sum(int(row["pages"]) for row in rows)),
        "automated_passed": all(
            bool(row["opens"])
            and bool(row["fonts_embedded"])
            and bool(row["text_extractable"])
            and int(row["pages"]) == int(row["rendered_pages"])
            for row in rows
        ),
        "manual_visual_status": "pending",
        "render_root_external": str(render_root),
    }
    atomic_json(summary, result / "reproducibility/pdf_qa_summary.json")
    return summary


def verify_package(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
    protocol = load_yaml(protocol_path)
    files = _repository_files(repository)
    forbidden_suffixes = {
        ".jsonl",
        ".safetensors",
        ".pt",
        ".bin",
        ".npy",
        ".npz",
        ".tar",
        ".zip",
        ".png",
    }
    oversized = [path for path in files if path.stat().st_size > 10 * 1024 * 1024]
    forbidden = [path for path in files if path.suffix.lower() in forbidden_suffixes]
    crlf = [
        path
        for path in files
        if path.suffix.lower() in {".py", ".yaml", ".md", ".sh", ".tex", ".bib", ".csv", ".json"}
        and b"\r\n" in path.read_bytes()
    ]
    index = pd.read_csv(result / "INDEX.csv")
    missing = [row.relative_path for row in index.itertuples() if not (repository / row.relative_path).exists()]
    mismatches = [
        row.relative_path
        for row in index.itertuples()
        if (repository / row.relative_path).exists()
        and sha256_file(repository / row.relative_path) != row.sha256
    ]
    completion = json.loads((artifact_root() / "formal/completion.json").read_text(encoding="utf-8"))
    replay_path = artifact_root() / "reproducibility/replay_summary.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {"status": "missing"}
    primary = json.loads((result / "statistics/primary_results.json").read_text(encoding="utf-8"))
    v14_primary = json.loads(
        (repository / "results/collective_agent_statmech_v14/statistics/primary_results.json").read_text(encoding="utf-8")
    )
    package_bytes = int(sum(path.stat().st_size for path in files))
    checks = {
        "execution_source_matches_freeze": execution_source_checksum(repository)
        == protocol["provenance"]["execution_source_sha256"],
        "completion_source_matches_freeze": completion["execution_source_sha256"]
        == protocol["provenance"]["execution_source_sha256"],
        "completion_protocol_matches_freeze": completion["protocol_sha256"]
        == sha256_file(protocol_path),
        "schema_matches_freeze": protocol["provenance"]["schema_sha256"] == schema_checksum(),
        "replay_passed": replay.get("status") == "passed",
        "privacy_passed": int(primary["privacy_mutations"]) == 0,
        "V14_invalid_H3_not_supported": not bool(
            v14_primary["confirmatory_dispositions"]["H3"].get("inferential_support", True)
        ),
        "no_oversized_files": not oversized,
        "no_forbidden_artifacts": not forbidden,
        "no_crlf": not crlf,
        "index_complete": not missing,
        "index_checksums_match": not mismatches,
        "package_below_25_mib": package_bytes < 25 * 1024 * 1024,
    }
    summary = {
        "generated_at": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "package_files": len(files),
        "package_bytes": package_bytes,
        "largest_files": [
            {
                "relative_path": path.relative_to(repository).as_posix(),
                "bytes": int(path.stat().st_size),
            }
            for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:20]
        ],
        "missing_indexed_files": missing,
        "checksum_mismatches": mismatches,
        "oversized_files": [path.relative_to(repository).as_posix() for path in oversized],
        "forbidden_files": [path.relative_to(repository).as_posix() for path in forbidden],
        "replay": replay,
    }
    atomic_json(summary, result / "reproducibility/verification.json")
    return summary


__all__ = ["build_results", "validate_pdfs", "verify_package"]
