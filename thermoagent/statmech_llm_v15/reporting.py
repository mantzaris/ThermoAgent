"""Publication summaries, inventories, PDF QA, and scientific integrity checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
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
    ".bbl",
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
        repository / "paper/JSTAT",
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
    files.extend(
        path
        for path in (
            repository / "scripts/build-jstat-paper.sh",
            repository / "scripts/verify-jstat-paper-assets.sh",
        )
        if path.exists()
    )
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
    if (repository / "requirements-runpod.txt").exists():
        files.append(repository / "requirements-runpod.txt")
    return sorted(set(path for path in files if path.is_file()))


def _manifest(repository: Path) -> pd.DataFrame:
    rows = []
    for path in _repository_files(repository):
        # INDEX and repository_manifest are the manifest itself.  The final
        # verification attestation is written only after its checks complete,
        # so indexing it would create a self-referential checksum cycle.
        if path.name in ("INDEX.csv", "repository_manifest.csv", "verification.json"):
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


def _extension_effect(
    frame: pd.DataFrame, model: str, observable: str, contrast: str
) -> pd.Series:
    selected = frame[
        (frame["model_key"] == model)
        & (frame["observable"] == observable)
        & (frame["contrast"] == contrast)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "expected one extension effect for %s/%s/%s"
            % (model, observable, contrast)
        )
    return selected.iloc[0]


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


def _unrecorded_infrastructure_accounting() -> Dict[str, object]:
    """Summarize generated calls lost before an atomic raw record was written.

    Content-addressed call records account exactly for retained and orphaned
    generations.  A storage failure can occur after model generation but before
    the atomic record is durable; its token and latency totals are then
    unknowable.  Such calls remain separate from measured totals so the latter
    are reported explicitly as lower bounds rather than silently treated as
    zero-cost attempts.
    """

    invalidated = artifact_root() / "invalidated"
    records: List[Dict[str, object]] = []
    for path in sorted(invalidated.rglob("accounting.json")) if invalidated.exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        explicitly_unrecorded = (
            payload.get("accounting_scope") == "unrecorded_infrastructure_attempt"
        )
        legacy_quota_record = (
            payload.get("classification")
            == "external_artifact_disk_quota_exceeded_during_atomic_raw_record"
            and payload.get("generated_call_tokens_and_latency")
            == "unavailable_because_atomic_record_was_not_written"
            and not bool(payload.get("scientific_panel_completed", False))
        )
        if not (explicitly_unrecorded or legacy_quota_record):
            continue
        model_calls = int(payload.get("model_calls", 1))
        records.append(
            {
                "relative_path": path.relative_to(artifact_root()).as_posix(),
                "classification": str(payload.get("classification", "unknown")),
                "model_key": str(payload.get("model_key", "unknown")),
                "panel_id": str(payload.get("panel_id", "unknown")),
                "decision_requests": int(payload.get("decision_requests", 1)),
                "model_calls": model_calls,
                "prompt_tokens": payload.get("prompt_tokens"),
                "generated_tokens": payload.get("generated_tokens"),
                "latency_seconds": payload.get("latency_seconds"),
            }
        )
    return {
        "records": records,
        "decision_requests": int(sum(int(row["decision_requests"]) for row in records)),
        "model_calls": int(sum(int(row["model_calls"]) for row in records)),
        "calls_with_unknown_prompt_tokens": int(
            sum(int(row["model_calls"]) for row in records if row["prompt_tokens"] is None)
        ),
        "calls_with_unknown_generated_tokens": int(
            sum(int(row["model_calls"]) for row in records if row["generated_tokens"] is None)
        ),
        "calls_with_unknown_latency": int(
            sum(int(row["model_calls"]) for row in records if row["latency_seconds"] is None)
        ),
        "measured_generation_accounting_is_lower_bound": bool(records),
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
    raw_accounting = pd.read_csv(result / "tables/raw_generation_accounting.csv")
    stratified = pd.read_csv(result / "tables/hypothesis_model_stratified.csv")
    extension_contrasts = pd.read_csv(
        result / "tables/collective_extension_contrasts.csv"
    )
    extension_path = repository / "configs/statmech_v15/collective_extension.yaml"
    extension = load_yaml(extension_path)
    v14_primary = json.loads(
        (
            repository
            / "results/collective_agent_statmech_v14/statistics/primary_results.json"
        ).read_text(encoding="utf-8")
    )
    completion = primary["formal_completion"]
    pilot = _pilot_accounting()
    unrecorded = _unrecorded_infrastructure_accounting()
    historical_accounting_path = (
        artifact_root()
        / "reproducibility/committed_reference_result/reproducibility/compute_accounting.json"
    )
    historical_accounting_source = (
        historical_accounting_path
        if historical_accounting_path.exists()
        else result / "reproducibility/compute_accounting.json"
    )
    historical_accounting = (
        json.loads(historical_accounting_source.read_text(encoding="utf-8"))
        if historical_accounting_source.exists()
        else {}
    )
    historical_rejected = (
        historical_accounting.get("pilot", {}).get("rejected_totals", {})
        if isinstance(historical_accounting, dict)
        else {}
    )
    historical_total_hours = float(
        historical_accounting.get("total_metered_generation_gpu_hours", 0.0)
    )
    reconstruction_comparison_path = (
        artifact_root() / "reproducibility/reconstructed_vs_committed.json"
    )
    reconstruction_comparison = (
        json.loads(reconstruction_comparison_path.read_text(encoding="utf-8"))
        if reconstruction_comparison_path.exists()
        else {"status": "missing"}
    )
    h1, h2, h3, h4 = (_effect(effects, key) for key in ("H1", "H2", "H3", "H4"))
    disposition = primary["confirmatory_dispositions"]
    formal_hours = float(completion["generation_gpu_hours"])
    pilot_hours = float(pilot["latency_seconds"]) / 3600.0
    rejected_hours = float(pilot["rejected_totals"]["latency_seconds"]) / 3600.0
    failure_hours = float(pilot["qualified_model_failure_totals"]["latency_seconds"]) / 3600.0
    orphan_accounting = raw_accounting[
        raw_accounting["accounting_scope"] == "orphan_interrupted_attempt"
    ]
    orphan_decisions = int(orphan_accounting["decision_requests"].sum())
    orphan_calls = int(orphan_accounting["model_calls"].sum())
    orphan_prompt_tokens = int(orphan_accounting["prompt_tokens"].sum())
    orphan_generated_tokens = int(orphan_accounting["generated_tokens"].sum())
    orphan_hours = float(orphan_accounting["latency_seconds"].sum()) / 3600.0
    total_hours = (
        formal_hours + pilot_hours + rejected_hours + failure_hours + orphan_hours
    )
    cost = (0.34 * total_hours, 0.69 * total_hours)
    if unrecorded["measured_generation_accounting_is_lower_bound"]:
        unrecorded_sentence = (
            "The incident audit counts %d additional post-generation, pre-record "
            "infrastructure model call; its prompt tokens, generated tokens, and "
            "latency were not durably recorded, so measured token and GPU-hour "
            "totals are lower bounds."
            % int(unrecorded["model_calls"])
        )
        total_phrase = "at least %.3f" % total_hours
        cost_phrase = "at least USD %.2f-%.2f" % (cost[0], cost[1])
    else:
        unrecorded_sentence = (
            "No post-generation infrastructure call lacked a durable accounting "
            "record."
        )
        total_phrase = "%.3f" % total_hours
        cost_phrase = "USD %.2f-%.2f" % (cost[0], cost[1])
    qwen_memory = panels[panels["model_key"] == "qwen"].groupby("condition")[
        "adjusted_pathwise_irreversibility_nats_per_update"
    ].mean()
    granite_memory = panels[panels["model_key"] == "granite"].groupby("condition")[
        "adjusted_pathwise_irreversibility_nats_per_update"
    ].mean()
    qwen_h3_stratum = stratified[
        (stratified["model_key"] == "qwen")
        & (stratified["contrast"] == "persistent_minus_scrambled")
    ].iloc[0]
    granite_h3_stratum = stratified[
        (stratified["model_key"] == "granite")
        & (stratified["contrast"] == "persistent_minus_scrambled")
    ].iloc[0]
    h3_v14 = v14_primary["confirmatory_dispositions"]["H3"]
    extension_lines = []
    for row in extension_contrasts.itertuples():
        extension_lines.append(
            "| %s | %s | %.5f | %.5f to %.5f | %d |"
            % (
                str(row.model_key).title(),
                str(row.contrast).replace("_", " "),
                float(row.estimate),
                float(row.ci_low),
                float(row.ci_high),
                int(row.independent_clusters),
            )
        )
    extension_table = "\n".join(extension_lines)
    extension_macros = {
        "QwenCorrQuench": _extension_effect(
            extension_contrasts,
            "qwen",
            "connected_graph_correlation",
            "disruption_minus_baseline_at_graph_distance_1",
        ),
        "GraniteCorrQuench": _extension_effect(
            extension_contrasts,
            "granite",
            "connected_graph_correlation",
            "disruption_minus_baseline_at_graph_distance_1",
        ),
        "QwenTauMemory": _extension_effect(
            extension_contrasts,
            "qwen",
            "truncated_integrated_autocorrelation_time",
            "persistent_minus_markovized_during_recovery",
        ),
        "GraniteTauMemory": _extension_effect(
            extension_contrasts,
            "granite",
            "truncated_integrated_autocorrelation_time",
            "persistent_minus_markovized_during_recovery",
        ),
        "QwenBinderQuench": _extension_effect(
            extension_contrasts,
            "qwen",
            "binder_cumulant",
            "disruption_minus_baseline_field_markovized",
        ),
        "GraniteBinderQuench": _extension_effect(
            extension_contrasts,
            "granite",
            "binder_cumulant",
            "disruption_minus_baseline_field_markovized",
        ),
    }
    extension_macro_lines = []
    for name, row in extension_macros.items():
        extension_macro_lines.extend(
            [
                r"\newcommand{\V15%s}{%.5f}" % (name, float(row.estimate)),
                r"\newcommand{\V15%sCI}{%.5f to %.5f}"
                % (name, float(row.ci_low), float(row.ci_high)),
            ]
        )
    extension_macro_text = "\n".join(extension_macro_lines)
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

Each arm constructs a fresh agent-network object from its frozen panel seeds.
The loaded model weights and tokenizer are shared read-only for throughput, but
no conversational history, key/value cache, mutable agent object, or unseeded
scientific RNG state crosses an arm boundary. Every generation receives its
frozen per-decision seed; provider accounting and raw-record indices have no
causal input to the prompt or transition law.

The formal study ran {int(completion['dynamic_trajectories'])} trajectories and {int(completion['observed_decision_rows']):,} attempted decisions. Formal generation used {int(completion['model_calls']):,} calls, {int(completion['prompt_tokens']):,} prompt tokens, {int(completion['generated_tokens']):,} generated tokens, and {formal_hours:.3f} metered GPU-hours. The content-addressed raw-record audit additionally found {orphan_decisions} interrupted-panel decision records ({orphan_calls} calls, {orphan_prompt_tokens} prompt tokens, {orphan_generated_tokens} generated tokens, {orphan_hours:.3f} GPU-hours) that do not enter a completed trajectory. {unrecorded_sentence} Successful Qwen/Granite engineering pilots added {pilot['decisions']} decisions and {pilot_hours:.3f} GPU-hours. Their retained infrastructure failures added {pilot['qualified_model_failure_totals']['decisions']} decision requests and {failure_hours:.3f} GPU-hours. Any rejected-model attempts made during this fresh reconstruction used {pilot['rejected_totals']['decisions']} decision requests, {pilot['rejected_totals']['calls']} model calls, and {rejected_hours:.3f} GPU-hours; no network contrast was computed from them. Total fresh measured generation was {total_phrase} hours, with an approximate measured RTX 4090 cost range of {cost_phrase}.

The fresh reconstruction does not rerun the engineering-rejected Mistral
pilot. Its original pre-freeze boundary is retained from the committed
reference accounting (historical requests:
{int(historical_rejected.get('decisions', 0))}; historical calls:
{int(historical_rejected.get('calls', 0))}) and is not added to the fresh
reconstruction compute total. The original sealed execution used
{historical_total_hours:.3f} measured generation GPU-hours in total; the fresh
reconstruction accounted for {total_phrase} measured generation GPU-hours. The
runtime difference is reported as an environment-dependent reproducibility
cost, not a scientific effect.

The original external raw tree was unavailable after the Pod replacement.
Fresh records are replayed at decision resolution, and the frozen reconstructed
package is compared with the committed aggregate reference before extended
reporting is authorized. The machine-readable comparison status is
`{reconstruction_comparison.get('status', 'missing')}`. This verifies the
declared aggregate science and accounting scope; it cannot establish digest
identity with deleted historical call files.

## Frozen hypotheses

- H1 (Granite field quench versus nominal): {_format(h1, 3)} distance units; exact sign-flip `p={float(h1.exact_one_sided_sign_flip_p):.5f}`, allocated alpha 0.02. **{('Supported' if disposition['H1']['supported'] else 'Not supported')}**.
- H2 (persistent minus Markovized path divergence, pooled across model-stratified pairs): {_format(h2, 5)} nats/update; Holm `p={float(h2.multiplicity_adjusted_p):.5f}` within the alpha-0.03 family. **{('Supported' if disposition['H2']['supported'] else 'Not supported')}**.
- H3 (persistent minus scrambled-history path divergence): {_format(h3, 5)} nats/update; Holm `p={float(h3.multiplicity_adjusted_p):.5f}`. **{('Supported' if disposition['H3']['supported'] else 'Not supported')}**.
- H4 (fixed recovery sweeps 31-35 minus 41-45): {_format(h4, 3)} distance units; Holm `p={float(h4.multiplicity_adjusted_p):.5f}`. **{('Supported' if disposition['H4']['supported'] else 'Not supported')}**.

The exact direction and model-specific heterogeneity are retained in `tables/hypothesis_effects.csv` and `tables/panel_statistics.csv`; the README does not reinterpret null or adverse signs. Qwen mean adjusted divergence was {float(qwen_memory['field_markovized']):.5f}, {float(qwen_memory['field_persistent']):.5f}, and {float(qwen_memory['field_scrambled']):.5f} nats/update for Markovized, persistent, and scrambled arms. The corresponding Granite means were {float(granite_memory['field_markovized']):.5f}, {float(granite_memory['field_persistent']):.5f}, and {float(granite_memory['field_scrambled']):.5f}. Mean persistent-minus-scrambled prompt length was {float(prompt_balance['persistent_minus_scrambled_mean_prompt_tokens'].mean()):.2f} tokens.

`tables/hypothesis_model_stratified.csv` is a descriptive sensitivity, not a
replacement confirmatory analysis. It makes explicit that Qwen's
persistent-minus-scrambled mean is {float(qwen_h3_stratum.estimate):.5f} nats/update with {int(qwen_h3_stratum.positive_clusters)}
of six positive clusters and an unadjusted within-model exact sign-flip
`p={float(qwen_h3_stratum.exact_one_sided_sign_flip_p):.5f}`, whereas Granite's mean is {float(granite_h3_stratum.estimate):.5f} with {int(granite_h3_stratum.positive_clusters)} of six
positive clusters. The pooled H3 result is therefore not independent
confirmation within both families. Arm-level adjusted estimates may be
negative because the raw block divergence can lie below its shuffled floor;
positive paired contrasts do not establish positive absolute entropy
production.

The primary floor still uses the frozen 500 time permutations.  The extended
`tables/irreversibility_sensitivity.csv` retains the empirical permutation-null
interval, its standard deviation, and the Monte Carlo standard error and
normal-approximation interval for the floor mean.  These audit columns
reproduce the frozen raw divergence, mean floor, and adjusted value exactly;
they do not redefine H2 or H3.

## Secondary collective-observable extension

The frozen protocol and H1-H4 are unchanged. A versioned descriptive extension
(`{extension['version']}`, SHA-256 `{sha256_file(extension_path)}`) computes,
within each complete trajectory and phase, connected belief correlation by
actual shortest-path distance, magnetization autocorrelation with a primary
two-sweep lag truncation, and the Binder cumulant. One- and three-sweep lag
truncations are sensitivities. Binder window and pooling sensitivities compare
full versus early/late half-phases and cluster-first versus moment-pooled
estimates. Pair, node, and update counts are not used as
replicates. Undefined zero-variance or zero-second-moment cases remain missing.

| Model | Descriptive contrast | Estimate | 95% cluster-bootstrap interval | Independent clusters |
|---|---|---:|---:|---:|
{extension_table}

These are finite-window descriptive contrasts. Connected correlation exposes
spatial organization beyond mean order; truncated autocorrelation summarizes
persistence; Binder $U_4$ summarizes order-parameter shape. They do not imply
a correlation length, critical slowing down, a Binder crossing, or a phase
transition.

## V14 scientific correction

No frozen V14 decision or trajectory was altered. The versioned V14 audit now fits recovery thresholds using training clusters only, completes the frozen 10,000-replicate cluster-preserving permutation analysis, recomputes three-, five-, and seven-sweep nominal geometries, deletes individual observables, and audits finite-sample dependence bias. The historical H3 maximum-minus-final estimand, interval, raw p-value, and Holm value remain archived, but its structurally nonnegative sign makes the directional test invalid. Its machine-readable disposition is `inferential_support: false`; recovery language uses threshold re-entry, final residual, the complete path, and fixed early-versus-late descriptive changes.

## Supported boundaries

Results are finite-size and model-specific. Neither model is a human participant. No field validity, application benefit, controller advantage, performance superiority, thermodynamic-limit phase transition, physical free energy, or exact LLM entropy production is claimed. Persistent history and scrambled history are prompt-format controls; they do not make the projected binary process fully observed. Negative adjusted information quantities are retained rather than truncated.

## Reproduction order

```bash
export THERMO_V15_ARTIFACT_ROOT=/workspace/ThermoAgent-v15-reconstruction-b309f0ab
scripts/setup-statmech-v15-runpod.sh
.venv/bin/python scripts/prefetch-statmech-v15-models.py
PYTHON_BIN=.venv/bin/python scripts/run-statmech-v15-tests.sh
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-reconstruction-pilot.sh qwen
THERMO_V15_ENABLE_LLM=1 scripts/run-statmech-v15-reconstruction-pilot.sh granite
# Run the next commands against a clean checkout at the committed V15
# reference through audited reconstruction wrappers.  The 50-hour value is
# the explicitly authorized operational reconstruction guard; it does not
# alter the frozen protocol's scientific design or its historical 25-hour cap.
THERMO_V15_ENABLE_LLM=1 THERMO_V15_AUTHORIZED_GPU_HOURS=50 scripts/run-statmech-v15-reconstruction-formal.sh qwen
THERMO_V15_ENABLE_LLM=1 THERMO_V15_AUTHORIZED_GPU_HOURS=50 scripts/run-statmech-v15-reconstruction-formal.sh granite
scripts/run-statmech-v15-reconstruction-analysis.sh
scripts/run-statmech-v15-surrogate.sh
scripts/generate-statmech-v15-figures.sh
scripts/build-statmech-v15-results.sh
scripts/build-jstat-paper.sh
scripts/verify-statmech-v15.sh
```

Raw prompts, completions, and trajectory tables are external at `{artifact_root()}`. Compact aggregate tables, checksums, vector figures, and manuscript sources are repository-facing.
"""
    atomic_bytes(readme.encode("utf-8"), result / "README.md")
    summary = f"""# Paper summary

V15 prospectively tests memory and field-quench dynamics in state-separated Qwen and Granite agent networks. Six matched graph/environment clusters per model contribute four 45-sweep trajectories each. H1 is {_format(h1, 3)} distance units; H2 is {_format(h2, 5)} nats/update; H3 is {_format(h3, 5)} nats/update; and the non-tautological fixed-window recovery H4 is {_format(h4, 3)} distance units. Formal dispositions are H1={bool(disposition['H1']['supported'])}, H2={bool(disposition['H2']['supported'])}, H3={bool(disposition['H3']['supported'])}, and H4={bool(disposition['H4']['supported'])}.

The study separates complete augmented state, observable microscopic projection, and rolling macrostate. It compares genuine persistent history with both a Markovized state and a deterministically generated own-agent, past-only, format-matched scrambled-history placebo. V14's threshold leakage and invalid H3 inference are corrected without changing a frozen trajectory. The strongest interpretation must follow the signs and intervals above; path divergence remains coarse-grained temporal asymmetry, not exact thermodynamic entropy production.

The post-reconstruction descriptive extension adds connected graph-distance
correlation, a fixed-truncation integrated autocorrelation time, and the Binder
cumulant. These quantities are computed per graph trajectory before pooling
and provide spatial, temporal, and distribution-shape views that mean
magnetization alone cannot supply. They remain finite-size diagnostics rather
than evidence of a phase transition.
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
            {"claim": "H3 replicated separately in both model families", "disposition": "unsupported", "evidence": "Qwen 4/6 positive, unadjusted exact p=0.21875; Granite 6/6 positive", "boundary": "the frozen pooled estimand uses 12 model-by-cluster units"},
            {"claim": "Connected correlation, truncated persistence, and Binder shape are finite-system descriptors", "disposition": "supported as secondary descriptive analysis", "evidence": "trajectory-first estimates and cluster intervals", "boundary": "N=16, one reciprocal modular topology, three 15-sweep phases"},
            {"claim": "The Binder statistic or autocorrelation establishes a phase transition", "disposition": "prohibited", "evidence": "no direct size crossing or stationary scaling", "boundary": "finite-size shape and persistence only"},
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
\newcommand{{\V15OriginalGPUHours}}{{{historical_total_hours:.2f}}}
\newcommand{{\V15ReconstructionGPUHours}}{{{total_hours:.2f}}}
\newcommand{{\V15OrphanFormalDecisions}}{{{orphan_decisions}}}
\newcommand{{\V15OrphanFormalGPUHours}}{{{orphan_hours:.3f}}}
\newcommand{{\V15UnrecordedInfrastructureCalls}}{{{int(unrecorded['model_calls'])}}}
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
{extension_macro_text}
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
        "v14_raw_outcomes_modified": False,
    }
    atomic_json(external, result / "reproducibility/external_artifact_manifest.json")
    accounting = {
        "generated_at": utc_now(),
        "formal": completion,
        "formal_raw_generation_audit": primary[
            "raw_generation_accounting_audit"
        ],
        "orphan_interrupted_formal_attempts": {
            "decisions": orphan_decisions,
            "calls": orphan_calls,
            "prompt_tokens": orphan_prompt_tokens,
            "generated_tokens": orphan_generated_tokens,
            "latency_seconds": float(orphan_hours * 3600.0),
            "generation_gpu_hours": orphan_hours,
        },
        "unrecorded_infrastructure_attempts": unrecorded,
        "pilot": pilot,
        "historical_reference_rejected_mistral": historical_rejected,
        "historical_reference_accounting_sha256": sha256_file(
            historical_accounting_source
        )
        if historical_accounting_source.exists()
        else None,
        "historical_reference_total_metered_generation_gpu_hours": historical_total_hours,
        "total_metered_generation_gpu_hours": total_hours,
        "total_metered_generation_gpu_hours_is_lower_bound": bool(
            unrecorded["measured_generation_accounting_is_lower_bound"]
        ),
        "prompt_and_generated_token_totals_are_lower_bounds": bool(
            unrecorded["measured_generation_accounting_is_lower_bound"]
        ),
        "estimated_cost_usd_range": list(cost),
        "analysis_cpu_seconds": primary["analysis_cpu_seconds"],
        "model_specs": {
            key: {"identifier": value.identifier, "revision": value.revision}
            for key, value in MODEL_SPECS.items()
        },
    }
    atomic_json(accounting, result / "reproducibility/compute_accounting.json")
    external_records = {
        "reconstruction_identity.json": artifact_root()
        / "reproducibility/reconstruction_identity.json",
        "environment_reconstruction.json": artifact_root()
        / "reproducibility/environment_reconstruction.json",
        "model_snapshot_verification.json": artifact_root()
        / "reproducibility/model_snapshot_verification.json",
        "reconstruction_source_compatibility.json": artifact_root()
        / "reproducibility/reconstruction_source_compatibility.json",
        "reconstructed_vs_committed.json": artifact_root()
        / "reproducibility/reconstructed_vs_committed.json",
        "replay_summary.json": artifact_root() / "reproducibility/replay_summary.json",
    }
    for name, source in external_records.items():
        if source.is_file():
            atomic_bytes(
                source.read_bytes(), result / "reproducibility" / name
            )
    manifest = _manifest(repository)
    atomic_csv(manifest, result / "reproducibility/repository_manifest.csv")
    atomic_csv(manifest, result / "INDEX.csv")
    return {
        "generated_at": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "collective_extension_sha256": sha256_file(extension_path),
        "execution_source_sha256": protocol["provenance"]["execution_source_sha256"],
        "schema_sha256": schema_checksum(),
        "repository_files": len(manifest),
        "repository_bytes": int(manifest["bytes"].sum()),
        "external_tree_sha256": external["tree"]["tree_sha256"],
    }


def _pdf_fonts_embedded(output: str) -> bool:
    """Parse Poppler ``pdffonts`` output using its stable trailing columns.

    Font type names have a variable number of whitespace-separated tokens, so
    a fixed left-hand index can accidentally read the subset or Unicode flag.
    The final five fields are always ``emb sub uni object-ID generation``.
    """

    rows = [line.split() for line in output.splitlines()[2:] if line.strip()]
    return bool(rows) and all(len(row) >= 7 and row[-5].lower() == "yes" for row in rows)


def validate_pdfs(repository: Path) -> Dict[str, object]:
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    pdfs = sorted((result / "figures/pdf").glob("*.pdf"))
    pdfs.extend(
        path
        for path in (
            repository / "paper/JSTAT/main.pdf",
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
        # Remove only this PDF's prior external QA pages. Otherwise a shorter
        # deterministic rebuild can inherit stale numbered renderings and fail
        # the page-count check despite a valid current PDF.
        for prior_render in render_root.glob(prefix.name + "-*.png"):
            prior_render.unlink()
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", str(path), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pages = int(
            next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:"))
        )
        embedded = _pdf_fonts_embedded(fonts)
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
    # PDF QA is intentionally run after the paper and figures are rebuilt.
    # Refresh both copies of the repository manifest here so the hashes record
    # the newly written QA table and summary rather than their prior versions.
    manifest = _manifest(repository)
    atomic_csv(manifest, result / "reproducibility/repository_manifest.csv")
    atomic_csv(manifest, result / "INDEX.csv")
    return summary


def record_manual_pdf_qa(
    repository: Path,
    status: str,
    notes: str,
) -> Dict[str, object]:
    """Record a completed visual review without weakening automated checks.

    The reviewer invokes this only after inspecting the externally rendered
    300-DPI pages and the original vector PDFs.  Every PDF digest is rechecked
    first so a later rebuild cannot inherit an earlier manual disposition.
    """

    if status not in {"passed", "failed"}:
        raise ValueError("manual PDF QA status must be passed or failed")
    repository = Path(repository).resolve()
    result = repository / "results/collective_agent_statmech_v15"
    table_path = result / "reproducibility/pdf_qa.csv"
    summary_path = result / "reproducibility/pdf_qa_summary.json"
    frame = pd.read_csv(table_path)
    required = {
        "relative_path",
        "opens",
        "fonts_embedded",
        "text_extractable",
        "rendered_pages",
        "pages",
        "sha256",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise RuntimeError("automated PDF QA table is incomplete")
    for row in frame.itertuples(index=False):
        path = repository / str(row.relative_path)
        if not path.is_file() or sha256_file(path) != str(row.sha256):
            raise RuntimeError("PDF changed after automated rendering: %s" % path)
    automated = bool(
        frame[["opens", "fonts_embedded", "text_extractable"]]
        .astype(bool)
        .to_numpy()
        .all()
        and np.array_equal(
            frame["pages"].to_numpy(int), frame["rendered_pages"].to_numpy(int)
        )
    )
    if status == "passed" and not automated:
        raise RuntimeError("manual pass cannot override failed automated PDF QA")
    reviewed_at = utc_now()
    frame["manual_visual_status"] = status
    frame["manual_reviewed_at"] = reviewed_at
    frame["manual_review_scope"] = "original_vector_and_external_300_dpi_render"
    frame["manual_review_notes"] = str(notes)
    atomic_csv(frame, table_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "manual_visual_status": status,
            "manual_reviewed_at": reviewed_at,
            "manual_review_scope": "original_vector_and_external_300_dpi_render",
            "manual_review_notes": str(notes),
        }
    )
    atomic_json(summary, summary_path)
    manifest = _manifest(repository)
    atomic_csv(manifest, result / "reproducibility/repository_manifest.csv")
    atomic_csv(manifest, result / "INDEX.csv")
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
    expected_index = _manifest(repository)
    indexed_paths = set(index["relative_path"].astype(str))
    expected_paths = set(expected_index["relative_path"].astype(str))
    missing = [row.relative_path for row in index.itertuples() if not (repository / row.relative_path).exists()]
    unindexed = sorted(expected_paths.difference(indexed_paths))
    unexpected_index_entries = sorted(indexed_paths.difference(expected_paths))
    mismatches = [
        row.relative_path
        for row in index.itertuples()
        if (repository / row.relative_path).exists()
        and sha256_file(repository / row.relative_path) != row.sha256
    ]
    completion = json.loads((artifact_root() / "formal/completion.json").read_text(encoding="utf-8"))
    reconstruction_path = (
        artifact_root() / "reproducibility/reconstruction_source_compatibility.json"
    )
    reconstruction = (
        json.loads(reconstruction_path.read_text(encoding="utf-8"))
        if reconstruction_path.exists()
        else {"status": "missing"}
    )
    replay_path = artifact_root() / "reproducibility/replay_summary.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {"status": "missing"}
    comparison_path = artifact_root() / "reproducibility/reconstructed_vs_committed.json"
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.exists()
        else {"status": "missing"}
    )
    environment_path = artifact_root() / "reproducibility/environment_reconstruction.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.exists()
        else {}
    )
    snapshot_path = artifact_root() / "reproducibility/model_snapshot_verification.json"
    snapshots = (
        json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot_path.exists()
        else {}
    )
    identity_path = artifact_root() / "reproducibility/reconstruction_identity.json"
    identity = (
        json.loads(identity_path.read_text(encoding="utf-8"))
        if identity_path.exists()
        else {}
    )
    primary = json.loads((result / "statistics/primary_results.json").read_text(encoding="utf-8"))
    pdf_qa = pd.read_csv(result / "reproducibility/pdf_qa.csv")
    pdf_qa_summary = json.loads(
        (result / "reproducibility/pdf_qa_summary.json").read_text(encoding="utf-8")
    )
    v14_primary = json.loads(
        (repository / "results/collective_agent_statmech_v14/statistics/primary_results.json").read_text(encoding="utf-8")
    )
    package_bytes = int(sum(path.stat().st_size for path in files))
    recorded_analysis_source = str(primary.get("analysis_source_sha256", ""))
    current_review_source = execution_source_checksum(repository)
    try:
        analysis_source_is_sha256 = (
            len(recorded_analysis_source) == 64
            and int(recorded_analysis_source, 16) >= 0
        )
    except ValueError:
        analysis_source_is_sha256 = False
    checks = {
        "frozen_execution_used_audited_clean_checkout": bool(
            reconstruction.get("commit")
            == "b309f0ab76cb24377de5872eebc811582af1f43f"
            and reconstruction.get("audited_legacy_execution_source_sha256")
            == protocol["provenance"]["execution_source_sha256"]
            and reconstruction.get("clean_semantic_source_sha256")
            == "f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2"
            and reconstruction.get("scientific_source_or_protocol_modified") is False
        ),
        # The analysis record binds the reconstructed numerical analysis to
        # the source tree that actually produced it.  Later, audited figure,
        # reporting, test, or documentation fixes need not pretend that this
        # historical hash equals the review tree.  Exact replay and the
        # reconstructed-versus-committed comparison remain required gates.
        "analysis_source_recorded": analysis_source_is_sha256,
        "collective_extension_matches_analysis": primary.get(
            "collective_extension_sha256"
        )
        == sha256_file(repository / "configs/statmech_v15/collective_extension.yaml"),
        "completion_source_matches_freeze": completion["execution_source_sha256"]
        == protocol["provenance"]["execution_source_sha256"],
        "completion_protocol_matches_freeze": completion["protocol_sha256"]
        == sha256_file(protocol_path),
        "schema_matches_freeze": protocol["provenance"]["schema_sha256"] == schema_checksum(),
        "replay_passed": replay.get("status") == "passed",
        "replay_complete": replay.get("units_checked") == 48
        and replay.get("rows_checked") == 34560
        and replay.get("units_with_mismatches") == 0,
        "reconstruction_matches_committed_science": comparison.get("status")
        == "passed",
        "reconstructed_environment_exact": bool(
            environment.get("python") == "3.12.3"
            and environment.get("packages")
            == {
                "PyMuPDF": "1.28.2",
                "PyYAML": "6.0.2",
                "accelerate": "1.10.1",
                "bitsandbytes": "0.47.0",
                "huggingface-hub": "0.34.4",
                "joblib": "1.5.3",
                "matplotlib": "3.10.5",
                "networkx": "3.5",
                "numpy": "2.1.2",
                "pandas": "2.3.1",
                "protobuf": "5.29.5",
                "pydantic": "2.11.7",
                "pytest": "8.4.1",
                "safetensors": "0.6.2",
                "scikit-learn": "1.5.2",
                "scipy": "1.16.1",
                "sentencepiece": "0.2.0",
                "torch": "2.8.0+cu128",
                "threadpoolctl": "3.6.0",
                "tokenizers": "0.21.4",
                "transformers": "4.55.4",
            }
            and environment.get("torch_cuda") == "12.8"
            and environment.get("cuda_available") is True
        ),
        "reconstruction_identity_exact": bool(
            identity.get("reconstruction_label") == "fresh-v15-b309f0ab"
            and identity.get("repository_commit")
            == "b309f0ab76cb24377de5872eebc811582af1f43f"
            and identity.get("protocol_sha256") == sha256_file(protocol_path)
            and identity.get("qwen_revision") == MODEL_SPECS["qwen"].revision
            and identity.get("granite_revision") == MODEL_SPECS["granite"].revision
            and environment.get("reconstruction_identity") == identity
        ),
        "model_snapshots_exact": bool(
            {
                (row.get("model_key"), row.get("resolved_revision"))
                for row in snapshots.get("models", [])
            }
            == {
                ("qwen", MODEL_SPECS["qwen"].revision),
                ("granite", MODEL_SPECS["granite"].revision),
            }
        ),
        "cluster_seed_audit_passed": primary.get("cluster_seed_audit_passed")
        is True,
        "memory_control_reconstruction_passed": bool(
            primary.get("memory_control_audit", {}).get(
                "panels_fully_reconstructed"
            )
            == 48
            and primary.get("memory_control_audit", {}).get(
                "future_information_violations"
            )
            == 0
            and primary.get("memory_control_audit", {}).get(
                "donor_agent_state_used"
            )
            is False
            and primary.get("memory_control_audit", {}).get(
                "peer_private_state_used"
            )
            is False
        ),
        "raw_generation_accounting_passed": bool(
            primary.get("raw_generation_accounting_audit", {}).get("status")
            == "passed"
            and primary.get("raw_generation_accounting_audit", {}).get(
                "referenced_records"
            )
            == 34560
            and primary.get("raw_generation_accounting_audit", {}).get(
                "missing_referenced_records"
            )
            == 0
        ),
        "privacy_passed": int(primary["privacy_mutations"]) == 0,
        "automated_pdf_qa_passed": bool(pdf_qa_summary.get("automated_passed")),
        "manual_pdf_qa_passed": bool(
            pdf_qa_summary.get("manual_visual_status") == "passed"
            and set(pdf_qa["manual_visual_status"].astype(str)) == {"passed"}
        ),
        "V14_invalid_H3_not_supported": not bool(
            v14_primary["confirmatory_dispositions"]["H3"].get("inferential_support", True)
        ),
        "no_oversized_files": not oversized,
        "no_forbidden_artifacts": not forbidden,
        "no_crlf": not crlf,
        "index_complete": not missing and not unindexed and not unexpected_index_entries,
        "index_checksums_match": not mismatches,
        "package_below_25_mib": package_bytes < 25 * 1024 * 1024,
    }
    summary = {
        "generated_at": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "informational_checks": {
            "analysis_source_matches_current_tree": recorded_analysis_source
            == current_review_source,
            "recorded_analysis_source_sha256": recorded_analysis_source,
            "current_review_source_sha256": current_review_source,
            "interpretation": (
                "A false equality is expected after documented post-analysis "
                "presentation, verification, or reporting changes. It is "
                "reported, not relabeled as equality; replay and aggregate "
                "comparison remain mandatory scientific checks."
            ),
        },
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
        "unindexed_repository_files": unindexed,
        "unexpected_index_entries": unexpected_index_entries,
        "checksum_mismatches": mismatches,
        "oversized_files": [path.relative_to(repository).as_posix() for path in oversized],
        "forbidden_files": [path.relative_to(repository).as_posix() for path in forbidden],
        "replay": replay,
        "reconstruction_comparison": comparison,
    }
    atomic_json(summary, result / "reproducibility/verification.json")
    return summary


__all__ = [
    "build_results",
    "record_manual_pdf_qa",
    "validate_pdfs",
    "verify_package",
]
