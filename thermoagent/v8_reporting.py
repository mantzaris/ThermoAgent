"""Build the V8 publication-facing package from stored evidence only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from xml.etree import ElementTree

import pandas as pd

from .v5_experiments import atomic_json, write_csv


STAGES = (
    "pilots", "pilots_v2", "pilots_v3", "routing_repair_pilot",
    "hysteresis_repair_pilot", "hysteresis_repair_pilot_v2",
    "hysteresis_repair_pilot_v3",
    "development", "development_final", "development_agent",
    "seed_stability", "validation", "holdout", "ablations",
)
PRIMARY_METRICS = (
    "message_reduction", "wire_byte_reduction",
    "fully_counted_byte_reduction", "fully_counted_message_reduction",
    "log_sketch_message_ratio", "log_wire_byte_ratio",
    "primary_error_increase",
    "primary_pointwise_p95_increase", "detection_delay_increase",
    "primary_error_advantage_vs_comparator", "relative_service_degradation",
    "absolute_service_loss_difference", "causal_utility_degradation",
    "harmful_action_rate_degradation", "harmful_action_count_difference",
    "reward_degradation",
)


def _json(path: Path, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _frame(results_root: Path, stage: str) -> pd.DataFrame:
    path = results_root / stage / "episode_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _test_count(path: Path) -> Optional[Dict[str, int]]:
    if not path.exists():
        return None
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        field: sum(int(float(suite.attrib.get(field, 0))) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _highest_primary_stage(results_root: Path) -> Optional[str]:
    for stage in ("holdout", "validation", "development_agent"):
        if (results_root / stage / "primary_gate_results.json").exists():
            return stage
    return None


def _stage_tables(results_root: Path) -> Dict[str, Any]:
    stage_rows: List[Dict[str, Any]] = []
    communication_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        frame = _frame(results_root, stage)
        if frame.empty:
            continue
        started = pd.to_datetime(frame.get("started_at"), utc=True, errors="coerce")
        completed = pd.to_datetime(frame.get("completed_at"), utc=True, errors="coerce")
        elapsed = (completed - started).dt.total_seconds().clip(lower=0)
        stage_rows.append({
            "stage": stage,
            "episodes": len(frame),
            "independent_panels": int(frame[["application", "environment_seed"]].drop_duplicates().shape[0]),
            "applications": ";".join(sorted(frame.application.astype(str).unique())),
            "action_policy_ids": ";".join(sorted(frame.action_policy_id.astype(str).unique())),
            "episode_cpu_hours_sum": float(elapsed.sum() / 3600.0),
            "gpu_hours": 0.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "incremental_cloud_cost_usd": 0.0,
        })
        for keys, values in frame.groupby(["application", "scheduler"], sort=True):
            # The first two retained pilots predate the explicit primary-error
            # alias. Their estimator target was the same normalized integrated
            # error, so use that versioned source column rather than rewriting
            # frozen pilot summaries.
            primary_error_column = (
                "primary_distributed_state_error"
                if "primary_distributed_state_error" in values.columns
                else "normalized_time_integrated_estimation_error"
            )
            actions = (
                values.autonomous_beneficial_actions
                + values.autonomous_neutral_actions
                + values.autonomous_harmful_actions
            ).clip(lower=1)
            communication_rows.append({
                "stage": stage, "application": keys[0], "scheduler": keys[1],
                "episodes": len(values),
                "independent_panels": int(values.environment_seed.nunique()),
                "attempted_sketch_messages": int(values.attempted_sketch_messages.sum()),
                "transmitted_sketch_messages": int(values.transmitted_sketch_messages.sum()),
                "delivered_sketch_messages": int(values.delivered_sketch_messages.sum()),
                "dropped_sketch_messages": int(values.dropped_sketch_messages.sum()),
                "forwarded_sketch_messages": int(values.forwarded_sketch_messages.sum()),
                "sketch_on_wire_bytes": int(values.sketch_on_wire_bytes.sum()),
                "operational_messages": int(values.operational_messages.sum()),
                "operational_bytes": int(values.operational_bytes.sum()),
                "fully_counted_messages": int(values.fully_counted_messages.sum()),
                "fully_counted_bytes": int(values.fully_counted_bytes.sum()),
                "mean_trigger_activation_rate": float(values.trigger_activation_rate.mean()),
                "mean_trigger_compute_seconds": (
                    float(values.trigger_compute_seconds.mean())
                    if "trigger_compute_seconds" in values else "not_recorded"
                ),
                "mean_trigger_compute_microseconds_per_evaluation": (
                    float(values.trigger_compute_microseconds_per_evaluation.mean())
                    if "trigger_compute_microseconds_per_evaluation" in values else "not_recorded"
                ),
                "mean_primary_distributed_state_error": float(
                    values[primary_error_column].mean()
                ),
            })
            outcome_rows.append({
                "stage": stage, "application": keys[0], "scheduler": keys[1],
                "episodes": len(values),
                "beneficial_actions": int(values.autonomous_beneficial_actions.sum()),
                "neutral_actions": int(values.autonomous_neutral_actions.sum()),
                "harmful_actions": int(values.autonomous_harmful_actions.sum()),
                "harmful_action_rate": float(values.autonomous_harmful_actions.sum() / actions.sum()),
                "accepted_physical_actions": int(values.accepted_physical_actions_v8.sum()),
                "mean_service_loss": float(values.service_loss.mean()),
                "mean_net_causal_utility": float(values.net_causal_utility.mean()),
                "mean_normalized_reward": float(values.normalized_autonomous_reward.mean()),
                "operator_escalation_requests": int(values.operator_escalation_requests.sum()),
            })
        for keys, values in frame.groupby(
            ["application", "complexity", "topology_family"], sort=True,
        ):
            coverage_rows.append({
                "stage": stage, "application": keys[0], "complexity": keys[1],
                "topology_family": keys[2], "episodes": len(values),
                "independent_panels": int(values.environment_seed.nunique()),
                "minimum_agents": int(values.agent_count.min()),
                "maximum_agents": int(values.agent_count.max()),
                "minimum_horizon": int(values.horizon.min()),
                "maximum_horizon": int(values.horizon.max()),
            })
    write_csv(results_root / "tables" / "stage_compute_accounting.csv", stage_rows)
    write_csv(results_root / "tables" / "communication_accounting.csv", communication_rows)
    write_csv(results_root / "tables" / "autonomous_outcomes.csv", outcome_rows)
    write_csv(results_root / "tables" / "environment_coverage.csv", coverage_rows)
    return {
        "stage_rows": stage_rows,
        "communication_rows": communication_rows,
        "outcome_rows": outcome_rows,
        "coverage_rows": coverage_rows,
    }


def _gate_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    development = _json(results_root / "development_final" / "combined_progression_gates.json")
    if development:
        for gate, value in development.get("gates", {}).items():
            rows.append({
                "stage": "development", "gate": gate,
                "status": "pass" if bool(value) else "fail",
                "evidence": "development_final/combined_progression_gates.json",
            })
    for stage in ("development_agent", "validation", "holdout"):
        report = _json(results_root / stage / "primary_gate_results.json")
        if not report:
            continue
        for application, gates in report.get("application_gates", {}).items():
            for gate, value in gates.items():
                rows.append({
                    "stage": stage, "application": application, "gate": gate,
                    "status": "pass" if bool(value) else "fail",
                    "evidence": "%s/primary_gate_results.json" % stage,
                })
        rows.append({
            "stage": stage, "application": "both", "gate": "progression_pass",
            "status": "pass" if report.get("progression_pass") else "fail",
            "evidence": "%s/primary_gate_results.json" % stage,
        })
    stability = _json(results_root / "seed_stability" / "seed_stability_gates.json")
    for gate, value in stability.get("gates", {}).items():
        rows.append({
            "stage": "seed_stability", "gate": gate,
            "status": "pass" if bool(value) else "fail",
            "evidence": "seed_stability/seed_stability_gates.json",
        })
    write_csv(results_root / "tables" / "gate_outcomes.csv", rows)
    return rows


def _primary_rows(results_root: Path, stage: Optional[str]) -> List[Dict[str, Any]]:
    if stage is None:
        return []
    path = results_root / stage / "primary_bootstrap_intervals.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame.metric.isin(PRIMARY_METRICS)].copy()
    rows = frame.to_dict("records")
    write_csv(results_root / "tables" / "primary_effects.csv", rows)
    return rows


def _hypotheses(results_root: Path, stage: Optional[str]) -> List[Dict[str, Any]]:
    report = _json(results_root / stage / "primary_gate_results.json") if stage else {}
    level = stage or "not_evaluated_with_frozen_agents"
    rows = [
        {
            "hypothesis": "H1 communication-efficient estimation",
            "status": "supported_%s" % level if report.get("H1_communication_efficient_estimation_pass") else "unsupported_or_not_tested",
            "evidence": "%s/primary_gate_results.json" % stage if stage else "development_final/development_selection.json",
        },
        {
            "hypothesis": "H2 entropy-specific matched-byte superiority",
            "status": "supported_%s" % level if report.get("H2_entropy_specific_extension_pass") else "unsupported_or_not_tested",
            "evidence": "%s/primary_gate_results.json" % stage if stage else "development_final/development_selection.json",
        },
        {
            "hypothesis": "H3 frozen-agent downstream retention",
            "status": "supported_%s" % level if report.get("H3_downstream_policy_retention_pass") else "unsupported_or_not_tested",
            "evidence": "%s/primary_gate_results.json" % stage if stage else "training/training_summary.json",
        },
    ]
    write_csv(results_root / "tables" / "hypothesis_outcomes.csv", rows)
    return rows


def _interval_text(rows: Sequence[Mapping[str, Any]], application: str, metric: str) -> str:
    matches = [
        value for value in rows
        if value.get("application") == application and value.get("metric") == metric
    ]
    if len(matches) != 1:
        return "not evaluated"
    value = matches[0]
    return "%.4f [%.4f, %.4f]" % (
        float(value["mean"]), float(value["ci_low"]), float(value["ci_high"]),
    )


def _claims_matrix(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# V8 claims-to-evidence matrix", "",
        "| Claim | Status | Evidence | Prohibited extension |",
        "|---|---|---|---|",
    ]
    for row in rows:
        extension = (
            "No generalized-entropy superiority unless H2 passes in both applications"
            if str(row["hypothesis"]).startswith("H2")
            else "No real-human, universal-control, or literal-thermodynamics claim"
        )
        lines.append("| %s | %s | `%s` | %s |" % (
            row["hypothesis"], row["status"], row["evidence"], extension,
        ))
    lines.extend([
        "", "## Global prohibited claims", "",
        "- V8 studied real human operators or establishes human usability.",
        "- Entropy directly selects the best intervention or recovers V7 selective safety.",
        "- FP16 savings were obtained without an actual serializer.",
        "- Communication savings exclude headers, integrity bytes, forwarding, drops, or operational traffic.",
        "- Development or validation evidence is a sealed-holdout confirmation.",
        "- Information entropy is literal physical thermodynamics.",
    ])
    return "\n".join(lines) + "\n"


def _paper_outline() -> str:
    return """# Provisional paper outline (20–30 pages; not a manuscript)

## 1. Introduction (2–3 pages)

Motivate distributed belief monitoring for independent autonomous agents under
communication constraints. State the conditional primary claim and preserve
the V7 selective-safety no-go result.

## 2. Related work (3–4 pages)

Event-triggered estimation, age of information, distributed detection,
multi-agent communication, decentralized RL, information-theoretic
disagreement, quantized belief exchange, and selective autonomy.

## 3. Problem and information boundaries (2–3 pages)

Private beliefs and memories, authorized local observations, ad-hoc network,
distributed estimate, evaluator-only latent state, actual-wire accounting,
matched stochastic tapes, and panel-level estimands.

## 4. Generalized-information scheduling (3 pages)

Shannon/Tsallis spectrum, Jensen–Shannon drift, confidence and age terms,
hysteresis, maximum silence, partition recovery, binary serialization, and
strong non-entropic schedules. Explain why mode changes can preserve Shannon
entropy.

## 5. Applications and autonomous agents (4–5 pages)

Humanitarian multi-commodity logistics and abstract defensive utility
restoration, complexity/topology regimes, persistent decentralized agents,
typed actions, sequential IPPO, causal message-to-action tests, and frozen
policy isolation across communication arms.

## 6. Prospective design (3–4 pages)

Pilot selection, power, five training seeds, validation and holdout sealing,
H1–H3 margins, matched bytes, bootstrap/randomization inference, Holm
correction, replay, conservation, and stopping rules.

## 7. Results (4–6 pages)

Wire encoding, activation, communication/error tradeoff, H1–H3 forest,
autonomous outcomes, scale/topology/partition robustness, seed stability,
negative results, and stage-specific evidence labels.

## 8. Mechanism and limitations (2–3 pages)

Which trigger components matter, matched-budget entropy-specific evidence,
abstract simulator boundaries, one decentralized RL architecture, no Qwen or
human claim, compute limits, and external-validity requirements.

## 9. Conclusion (1 page)

State only the highest-stage supported communication-monitoring claim.

## Supplemental material

Serializer schema, all thresholds and seeds, all training curves, per-panel
effects, topology sensitivities, causal ledgers, failure/deviation registry,
checksums, complete figure source data, and reproduction environment.
"""


def _no_go_paper_outline() -> str:
    return """# Provisional boundary-result paper outline (20–25 pages; not a manuscript)

## Evidence boundary

This outline describes a possible engineering/boundary paper. V8 stopped at a
prospectively declared pilot gate; it is not an outline for a positive
confirmatory communication result or an Artificial Intelligence submission.

## 1. Introduction (2 pages)

Motivate exact communication accounting for distributed belief monitoring.
Preserve the V7 selective-safety no-go result and state V8's distinct RQ1–RQ5.

## 2. Related work (3 pages)

Event-triggered estimation, age of information, distributed detection,
decentralized agents, generalized information measures, and quantized wire
protocols.

## 3. Information boundary and system model (2–3 pages)

Independent private beliefs and memories, delivered-message-only state,
evaluator-only latent truth, network loss/latency/partitions, and matched tapes.

## 4. Serializer and scheduler family (3–4 pages)

TBV8 frame, CRC and exact bytes; FP32/FP16/uint8 simplex; Shannon, Tsallis,
Jensen–Shannon, confidence and age components; baselines; hysteresis, cooldown,
maximum silence, and partition recovery.

## 5. Applications and causal plumbing (3–4 pages)

Humanitarian logistics and defensive utility restoration, V7 persistent-agent
architecture, local estimate reconstruction, message-to-belief-to-action
tests, and the limits of rule-policy pilot outcomes.

## 6. Prospective pilot design (2–3 pages)

Append-only iterations, exact candidate rules, fixed 5% information-event and
10% nominal-traffic gates, six panels/application, and the declared hard stop.

## 7. Development findings (3–4 pages)

Wire round trips, invalidated age-driven result, two state-machine repairs,
information-event fractions, nominal chatter, diagnostic communication/error
tradeoffs, and panel-bootstrap uncertainty. Explicitly label every result as
development pilot evidence.

## 8. Failure mechanism and design implications (2–3 pages)

Why entropy change can miss mode changes, why latch semantics can suppress
scores, why high-excursion semantics can chatter, and what a future scheduler
must pre-register before an untouched evaluation.

## 9. Reproducibility and limitations (2 pages)

Exact replay, conservation, privacy, compact ledgers, source data, PDF QA,
small independent n, no five-seed training, no Qwen/humans, and no formal
validation or holdout.

## 10. Conclusion (1 page)

State the engineering contribution and failed feasibility boundary only.

## Supplemental material

Every pilot configuration, invalidation chronology, trigger traces, binary
schema tests, per-panel diagnostics, compact ledgers, hashes, and figure data.
"""


def _no_go_closeout_manifests(
    results_root: Path, tables: Mapping[str, Any],
    invalidated_counts: Mapping[str, Any], rl_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        frame = _frame(results_root, stage)
        if frame.empty:
            continue
        columns = [
            value for value in (
                "application", "environment_seed", "complexity",
                "topology_family", "coupling", "fragmentation",
                "network_disruption",
            ) if value in frame.columns
        ]
        for row in frame[columns].drop_duplicates().to_dict("records"):
            seed_rows.append({
                "stage": stage, **row,
                "evidence_class": "development_only",
            })
    if rl_manifest:
        seed_rows.append({
            "stage": "sequential_rl_engineering_pilot",
            "application": "both",
            "environment_seed": "see training curve",
            "rl_seed": rl_manifest.get("rl_seed"),
            "evidence_class": "engineering_only_not_multiseed",
        })
    write_csv(results_root / "manifests" / "development_seed_registry.csv", seed_rows)
    write_csv(results_root / "manifests" / "formal_rl_seed_disposition.csv", [
        {
            "rl_seed": seed, "status": "prospectively_selected_but_not_run",
            "reason": "pilot trigger-feasibility gate failed before training",
        } for seed in (88201, 88202, 88203, 88204, 88205)
    ])
    failure_rows = [
        {
            "stage": "interrupted_development", "run_count": 6,
            "classification": "infrastructure_interruption_partial_temp_files",
            "eligible": False,
            "evidence": "negative_results/interrupted_development/interrupted_attempts.csv",
        },
        {
            "stage": "development_final_pre_hysteresis_invalidated",
            "run_count": int(invalidated_counts.get(
                "development_final_pre_hysteresis_invalidated", {}
            ).get("partial_run_directories", 0)),
            "classification": "partial_runs_not_episodes", "eligible": False,
            "evidence": "negative_results/development_final_pre_hysteresis_invalidated/invalidation.json",
        },
        {
            "stage": "hysteresis_repair_pilot_v3", "run_count": 2,
            "classification": "candidate_level_scientific_gate_failure",
            "eligible": True,
            "evidence": "tables/trigger_feasibility.csv",
        },
    ]
    write_csv(results_root / "manifests" / "failed_run_registry.csv", failure_rows)
    exclusion_rows = [
        {
            "stage": "development_final_pre_hysteresis_invalidated",
            "complete_episodes": int(invalidated_counts.get(
                "development_final_pre_hysteresis_invalidated", {}
            ).get("complete_episodes", 0)),
            "reason": "pre-repair trigger implementation; retained, not inferential",
            "outcome_based": False,
        },
        {
            "stage": "development_final_hysteresis_suppression_invalidated",
            "complete_episodes": int(invalidated_counts.get(
                "development_final_hysteresis_suppression_invalidated", {}
            ).get("complete_episodes", 0)),
            "reason": "mechanism audit found age-driven traffic after suppressed information latch",
            "outcome_based": False,
        },
    ]
    write_csv(results_root / "manifests" / "exclusion_ledger.csv", exclusion_rows)
    write_csv(results_root / "manifests" / "protocol_deviation_ledger.csv", [
        {
            "sequence": 1, "event": "local execution interruption",
            "timing": "before completed replacement batch",
            "resolution": "partial temporary files retained; fresh prescribed stage namespace",
            "formal_outcome_affected": False,
        },
        {
            "sequence": 2, "event": "off-latch age term suppressed information events",
            "timing": "pilot mechanism audit before formal freeze",
            "resolution": "invalidated batch retained; prospectively declared repair pilot",
            "formal_outcome_affected": False,
        },
        {
            "sequence": 3, "event": "active latch suppressed renewed high excursions",
            "timing": "repair pilot before formal freeze",
            "resolution": "state-machine semantics repaired before final fixed-gate pilot",
            "formal_outcome_affected": False,
        },
        {
            "sequence": 4, "event": "final pilot gate failed",
            "timing": "before formal development/training/validation/holdout",
            "resolution": "study stopped exactly as prospectively declared",
            "formal_outcome_affected": False,
        },
    ])
    compute = {
        "manifest_accounted_episode_cpu_hours": float(sum(
            float(value["episode_cpu_hours_sum"])
            for value in tables["stage_rows"]
        )),
        "gpu_hours": 0.0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "incremental_cloud_cost_usd": 0.0,
        "qwen_used": False,
        "formal_multiseed_training_used": False,
        "execution_host": "local CPU; established RunPod endpoint refused connection",
        "scope_note": "invalidated partial execution wall time is retained but not reconstructed as episode CPU time",
    }
    atomic_json(results_root / "manifests" / "compute_manifest.json", compute)
    disposition = {
        "pilot_trigger_feasibility": "failed",
        "replacement_formal_development": "not_run_locked",
        "five_seed_training": "not_run_locked",
        "validation": "not_run_locked",
        "holdout": "not_run_locked",
        "ablations": "not_run_locked",
        "confirmatory_evidence": False,
    }
    atomic_json(results_root / "manifests" / "stage_disposition.json", disposition)
    return {"compute": compute, "disposition": disposition}


def _build_no_go_reporting(
    repository: Path, results_root: Path, no_go: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build an explicitly development-only package after the pilot stop."""
    tables = _stage_tables(results_root)
    intervals_path = results_root / "statistics" / "v8_pilot_no_go_intervals.csv"
    intervals = pd.read_csv(intervals_path).to_dict("records")
    feasibility = pd.read_csv(results_root / "tables" / "trigger_feasibility.csv")
    replay = _json(results_root / "reproducibility" / "replay" / "replay_summary.json")
    protocol = _json(results_root / "protocol" / "v8_development_stop_manifest.json")
    full_tests = _test_count(results_root / "reproducibility" / "pytest_full.xml")
    focused_tests = _test_count(results_root / "reproducibility" / "pytest_v8.xml")
    test_text = "not yet recorded"
    if full_tests and focused_tests:
        test_text = (
            f"{full_tests['tests']} full-suite tests ({full_tests['failures']} failures, "
            f"{full_tests['errors']} errors, {full_tests['skipped']} skipped); "
            f"{focused_tests['tests']} focused V8 tests ({focused_tests['failures']} failures, "
            f"{focused_tests['errors']} errors, {focused_tests['skipped']} skipped)"
        )
    application_lines = []
    for value in no_go["application_summaries"]:
        application = str(value["application"])
        application_lines.append(
            "- %s (n=%d panels): exact sketch-wire byte reduction %s; "
            "distributed-state error change %s; pre-disruption transmission rate "
            "for tau=0.115 %.1f%%."
            % (
                application.replace("_", " "), int(value["independent_panels"]),
                _interval_text(intervals, application, "sketch_byte_reduction"),
                _interval_text(intervals, application, "primary_estimation_error_increase"),
                100.0 * float(feasibility[
                    feasibility.candidate_name.eq("generalized_0115_u8")
                    & feasibility.application.eq(application)
                ].pre_disruption_noninitial_transmission_rate.iloc[0]),
            )
        )
    stage_counts = {value["stage"]: value["episodes"] for value in tables["stage_rows"]}
    invalidated_counts: Dict[str, Any] = {}
    for stage in (
        "development_final_pre_hysteresis_invalidated",
        "development_final_hysteresis_suppression_invalidated",
    ):
        record = _json(
            results_root / "reproducibility" / "compaction" / (stage + ".json")
        )
        if record:
            invalidated_counts[stage] = {
                "complete_episodes": int(record.get("complete_episodes", 0)),
                "partial_run_directories": int(
                    record.get("partial_run_directories", 0)
                ),
            }
    rl_manifest = _json(
        results_root / "pilots" / "rl_pilot" / "training" / "manifests"
        / "v8-ippo-seed-88101.json"
    )
    closeout = _no_go_closeout_manifests(
        results_root, tables, invalidated_counts, rl_manifest,
    )
    hypotheses = [
        {
            "hypothesis": "H1 communication-efficient estimation",
            "status": "not_tested_formally_trigger_gate_failed",
            "evidence": "negative_results/v8_stop_decision.json",
        },
        {
            "hypothesis": "H2 generalized-information superiority at matched bytes",
            "status": "not_tested_formally_no_frozen_comparator",
            "evidence": "statistics/v8_pilot_no_go_summary.json",
        },
        {
            "hypothesis": "H3 frozen decentralized-agent retention",
            "status": "not_tested_training_locked",
            "evidence": "training/NOT_RUN.md",
        },
    ]
    write_csv(results_root / "tables" / "hypothesis_outcomes.csv", hypotheses)
    write_csv(results_root / "tables" / "gate_outcomes.csv", [
        {
            "stage": "pilot", "gate": "information_score_fraction_at_least_0.05",
            "status": "pass", "evidence": "tables/trigger_feasibility.csv",
        },
        {
            "stage": "pilot", "gate": "pre_disruption_noninitial_rate_at_most_0.10",
            "status": "fail", "evidence": "tables/trigger_feasibility.csv",
        },
        {
            "stage": "pilot", "gate": "formal_development_unlocked",
            "status": "fail", "evidence": "negative_results/v8_stop_decision.json",
        },
    ])
    readme = f"""# Entropy-triggered belief monitoring V8

## Scientific disposition

V8 is a **development-stage no-go study**. It asked whether locally deployable
generalized-information scheduling could reduce exact belief-sketch traffic
while preserving distributed estimation and frozen-agent performance. V7
commit `e46b6738231883e92b9b525ab1c3c190e38391e7` and all earlier namespaces
remain unchanged.

The final pilot repaired two genuine hysteresis defects. The repaired trigger
became information-driven, but it transmitted during nominal pre-disruption
operation at 71–86%, far above the prospectively fixed 10% limit. Neither
declared candidate passed in both applications. Formal development, five-seed
training, validation, and holdout were therefore not run. No confirmatory V8
claim is supported.

## Trigger and actual wire protocol

The candidate score combined 0.45 Jensen–Shannon drift, 0.25 maximum normalized
Tsallis-spectrum drift over q={{0.5,1,1.5,2,3}}, 0.15 confidence drift, and
0.15 bounded message age. Pilot iteration 3 evaluated `tau_on` 0.11 and 0.115,
with `tau_off=0.04`, two-step cooldown, a 30-step maximum-silence deadline,
partition-recovery refresh, two-hop forwarding, and deterministic uint8 simplex
encoding. The serializer is an actual `TBV8` big-endian binary frame with IDs,
step, confidence, encoding, hop count, payload, and CRC32; byte counts use
`len(serialized_frame)`, not a formula.

The strongest non-entropic comparator was not frozen because the trigger gate
failed. KPI-change 0.12 is shown only as a development diagnostic.

## Development evidence

{chr(10).join(application_lines)}

The tau=0.115 point estimates reduced exact sketch bytes by 24.0% in
humanitarian and 27.8% in utility restoration, but the confidence intervals are
based on only six independent panels per application and the scheduler violated
the nominal-traffic gate. Fully combined byte reductions were 22.5% and 22.9%.
These are pilot diagnostics, not H1 evidence. H2 and H3 were not formally tested.

## Autonomous agents and stage boundary

The retained pilots used the independent persistent V7 agents and deterministic
decentralized rule policy; delivered sketches updated recipient beliefs and
could alter consequential actions. A one-seed, six-episode sequential-IPPO
engineering pilot exercised 1,320 transitions and all four delegation actions.
The prospective five-seed formal training was locked, so V8 does not claim
multi-seed learned-agent replication or frozen-policy noninferiority. Qwen and
human participants were not used.

## Integrity and compute

- Development protocol: `{protocol.get('development_protocol_version', 'pending final manifest')}`;
  hash `{protocol.get('manifest_sha256', 'pending final manifest')}`.
- Completed stage counts: `{json.dumps(stage_counts, sort_keys=True)}`.
- Retained invalidated/partial stage accounting:
  `{json.dumps(invalidated_counts, sort_keys=True)}`.
- Sequential RL engineering pilot: one seed, six episodes,
  `{rl_manifest.get('sequential_transitions', 'not recorded')}` temporally linked transitions;
  formal five-seed training was not run.
- Tests: {test_text}.
- Replay: {replay.get('episodes_replayed', 0)} ledgers,
  {replay.get('replay_mismatches', 0)} mismatches; maximum conservation residual
  {replay.get('maximum_conservation_residual', 'pending')}.
- GPU hours, LLM calls, prompt/generated tokens, and cloud cost: zero. Execution
  used local CPU NumPy and the existing RunPod endpoint was unreachable.

## Reproduction order

```bash
./scripts/run-v8-tests.sh
./scripts/run-v8-hysteresis-repair-pilot.sh
./scripts/run-v8-hysteresis-repair-pilot-v2.sh
./scripts/run-v8-hysteresis-repair-pilot-v3.sh
./scripts/analyze-v8-no-go.sh
./scripts/replay-v8-results.sh
./scripts/generate-v8-figures.sh
./scripts/validate-v8-pdfs.sh
./scripts/record-v8-manual-pdf-qa.sh
./scripts/close-v8-development-no-go.sh
./scripts/compact-v8-artifacts.sh
./scripts/build-v8-report.sh
./scripts/index-v8-artifacts.sh
```

The earlier pilots and invalidated development attempts are retained with their
own manifests. `training/NOT_RUN.md`, `validation/NOT_RUN.md`, and
`holdout/NOT_RUN.md` record the prospective stop. See `CLAIMS_MATRIX.md` for
prohibited extensions and `INDEX.csv` for every artifact.

## Limitations and prohibited claims

The final mechanism pilot has only six panels per application, used rule-policy
outcomes, and never reached validation. It cannot establish communication-
efficient monitoring, entropy-specific superiority, downstream learned-policy
retention, real-world utility or humanitarian performance, Qwen behavior, or
human effectiveness. Information entropy is not literal thermodynamics.
"""
    claims = """# V8 claims-to-evidence matrix

| Claim | Status | Evidence |
|---|---|---|
| Actual deterministic binary belief serialization was implemented and audited | Supported engineering claim | `thermoagent/v8_wire.py`; `tests/test_v8_wire.py` |
| The corrected trigger became information-score driven | Supported development-only mechanism observation | `tables/trigger_feasibility.csv` |
| The candidate satisfied nominal communication feasibility | Failed | `negative_results/v8_stop_decision.json` |
| H1 communication-efficient estimation | Untested formally | `statistics/v8_pilot_no_go_summary.json` |
| H2 generalized-information superiority | Untested formally | no frozen matched-byte comparator |
| H3 frozen learned-agent retention | Untested | `training/NOT_RUN.md` |
| Validation or locked-holdout replication | Untested | `validation/NOT_RUN.md`; `holdout/NOT_RUN.md` |

## Prohibited extensions

- Do not call pilot byte-reduction point estimates confirmatory H1 support.
- Do not claim generalized entropy beat a frozen non-entropic comparator.
- Do not claim multi-seed RL, Qwen, human-operator, or real-world evidence.
- Do not omit operational, forwarding, dropped, stale, header, or integrity traffic.
- Do not reinterpret the failed nominal-traffic gate after observing it.
"""
    summary = """# V8 paper summary

**Working direction:** Entropy-triggered distributed belief monitoring for
independent autonomous agents.

**Disposition:** development-stage no-go; not ready for a positive paper.

V8 implemented an auditable binary belief-sketch wire protocol and exposed an
important trigger-design boundary. An initially suppressed hysteresis latch
made traffic age-driven; repairing that latch made generalized-information
events dominate traffic but caused 71–86% pre-disruption transmission. The
unchanged prospective nominal limit was 10%, so no candidate qualified and all
formal stages stopped. The honest contribution is an engineering platform and
negative boundary result, not evidence for communication-efficient entropic
monitoring.

**Supported engineering evidence:** deterministic binary belief serialization;
delivered-message-only distributed estimates; trigger/hysteresis/partition
tests; exact traffic accounting; and replayable causal communication plumbing.

**Unsupported scientific claims:** H1 communication-efficient estimation, H2
entropy-specific matched-byte superiority, H3 frozen learned-policy retention,
multi-seed replication, validation, holdout confirmation, Qwen evidence, and
human effectiveness.
"""
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "README.md").write_text(readme, encoding="utf-8")
    (results_root / "CLAIMS_MATRIX.md").write_text(claims, encoding="utf-8")
    (results_root / "PAPER_SUMMARY.md").write_text(summary, encoding="utf-8")
    (results_root / "PAPER_OUTLINE.md").write_text(
        _no_go_paper_outline(), encoding="utf-8",
    )
    build = {
        "highest_primary_stage": None,
        "evidence_status": "development_pilot_no_go",
        "hypotheses": hypotheses,
        "stage_counts": stage_counts,
        "retained_invalidated_counts": invalidated_counts,
        "compute": closeout["compute"],
        "stage_disposition": closeout["disposition"],
        "readme": "README.md", "paper_summary": "PAPER_SUMMARY.md",
        "paper_outline": "PAPER_OUTLINE.md", "claims_matrix": "CLAIMS_MATRIX.md",
    }
    atomic_json(results_root / "reproducibility" / "report_build_summary.json", build)
    return build


def build_v8_reporting(repository: Path, results_root: Path) -> Dict[str, Any]:
    no_go = _json(results_root / "statistics" / "v8_pilot_no_go_summary.json")
    if no_go:
        return _build_no_go_reporting(repository, results_root, no_go)
    tables = _stage_tables(results_root)
    stage = _highest_primary_stage(results_root)
    primary = _primary_rows(results_root, stage)
    gates = _gate_rows(results_root)
    hypotheses = _hypotheses(results_root, stage)
    protocol = _json(results_root / "protocol" / "v8_frozen_protocol.json")
    trigger_configuration = protocol.get("primary_trigger_configuration", {})
    freeze = _json(results_root / "protocol" / "freeze_manifest.json")
    replay = _json(results_root / "reproducibility" / "replay" / "replay_summary.json")
    training = _json(results_root / "training" / "training_summary.json")
    full_tests = _test_count(results_root / "reproducibility" / "pytest_full.xml")
    focused_tests = _test_count(results_root / "reproducibility" / "pytest_v8.xml")
    report = _json(results_root / stage / "primary_gate_results.json") if stage else {}
    supported_level = stage or "development monitoring only"
    applications = ("humanitarian", "utility_restoration")
    application_lines = "\n".join(
        "- %s: message reduction %s; actual sketch-wire byte reduction %s; "
        "distributed-state error increase %s; downstream service degradation %s."
        % (
            application.replace("_", " "),
            _interval_text(primary, application, "message_reduction"),
            _interval_text(primary, application, "wire_byte_reduction"),
            _interval_text(primary, application, "primary_error_increase"),
            _interval_text(primary, application, "relative_service_degradation"),
        ) for application in applications
    )
    test_text = "not yet recorded"
    if full_tests and focused_tests:
        test_text = (
            "%d full-suite tests (%d failures, %d errors, %d skipped); "
            "%d focused V8 tests (%d failures, %d errors, %d skipped)"
            % (
                full_tests["tests"], full_tests["failures"], full_tests["errors"], full_tests["skipped"],
                focused_tests["tests"], focused_tests["failures"], focused_tests["errors"], focused_tests["skipped"],
            )
        )
    stage_counts = {value["stage"]: value["episodes"] for value in tables["stage_rows"]}
    readme = f"""# Entropy-triggered belief monitoring V8

## Research question and relationship to V7

Can a locally deployable generalized-information event trigger reduce actual
belief-sketch communication while preserving distributed estimation and the
performance of frozen decentralized autonomous agents? V8 is based on frozen
V7 commit `e46b6738231883e92b9b525ab1c3c190e38391e7`; no V1–V7 result artifact
was modified. V7 remains an immutable negative selective-safety study. V8
does not attempt to rescue that failed endpoint and makes no human-operator
claim.

## Method

The primary scheduler combines Jensen–Shannon drift from the last transmitted
belief, maximum normalized entropy-spectrum change for
`q={{0.5,1,1.5,2,3}}`, confidence change, and bounded message age. It uses a
two-threshold hysteresis, cooldown, maximum-silence refresh, and explicit
partition-recovery behavior. The exact frozen trigger is
`{protocol.get('primary_trigger', 'not frozen')}`; the strongest development-
selected non-entropic comparator is
`{protocol.get('strongest_nonentropic_comparator', 'not frozen')}`.
The complete trigger parameters are
`{json.dumps(trigger_configuration, sort_keys=True)}`.

Belief vectors use the deterministic `{protocol.get('primary_encoding', 'pilot encoding not frozen')}`
wire serializer. Header, payload, integrity, forwarding, drops, useful,
redundant, stale, operational, and fully combined traffic are all recorded.
The primary scheduler and always-on control use the identical payload format.

Each application contains independent persistent agents with private
observations, beliefs, memory, utility, authority, inboxes, and action
processes. Formal downstream evidence uses the same frozen unweighted ensemble
of all five prospectively selected decentralized sequential-IPPO seeds under
every scheduler. The action policy receives only local state and delivered
messages; evaluator truth is used only offline for scoring.

## Evidence and disposition

- Highest primary evidence stage: **{supported_level}**.
- Episode counts by completed stage: `{json.dumps(stage_counts, sort_keys=True)}`.
- H1 communication-efficient estimation: `{report.get('H1_communication_efficient_estimation_pass', 'not evaluated')}`.
- H2 entropy-specific matched-byte extension: `{report.get('H2_entropy_specific_extension_pass', 'not evaluated')}`.
- H3 frozen-agent retention: `{report.get('H3_downstream_policy_retention_pass', 'not evaluated')}`.
- Progression at the highest completed primary stage: `{report.get('progression_pass', 'not evaluated')}`.

{application_lines}

The H2 extension is not a progression gate. If H1 and H3 pass while H2 fails,
the allowed claim is communication-efficient event-triggered belief
monitoring—not generalized-entropy superiority.

## Integrity, model, and compute

- Protocol: `{protocol.get('protocol_version', 'not frozen')}`; checksum
  `{freeze.get('protocol_sha256', 'not available')}`.
- Tests: {test_text}.
- Replay: `{replay.get('episodes_replayed', 0)}` episodes,
  `{replay.get('replay_mismatches', 0)}` mismatches; maximum conservation
  residual `{replay.get('maximum_conservation_residual', 'not available')}`.
- RL: `{training.get('completed_seeds', 0)}` completed seeds,
  `{training.get('failed_seeds', 0)}` failed seeds,
  `{training.get('total_sequential_transitions', 0)}` sequential transitions.
- GPU hours, LLM calls, prompt/generated tokens, and incremental cloud cost:
  zero for the CPU NumPy-IPPO V8 execution. Qwen was optional and not used for
  the primary claim.
- Planned Qwen reference only: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`; no V8 Qwen evidence is implied.

## Reproduction order

```bash
./scripts/run-v8-tests.sh
./scripts/run-v8-pilots.sh
./scripts/analyze-v8-pilots.sh
./scripts/run-v8-development.sh
./scripts/run-v8-final-development.sh
./scripts/freeze-v8-protocol.sh
./scripts/train-v8-multiseed.sh
./scripts/run-v8-development-agents.sh
./scripts/run-v8-seed-stability.sh
./scripts/run-v8-validation.sh   # refuses unless development gates unlock it
./scripts/run-v8-holdout.sh      # refuses unless validation passes
./scripts/run-v8-ablations.sh
./scripts/analyze-v8-calibration.sh <highest-completed-primary-stage>
./scripts/replay-v8-results.sh
./scripts/generate-v8-figures.sh
./scripts/validate-v8-pdfs.sh
./scripts/compact-v8-artifacts.sh
./scripts/index-v8-artifacts.sh
./scripts/build-v8-report.sh
```

## Directory guide and limitations

`protocol/` and `manifests/` contain the frozen design; `development*/`,
`validation/`, and `holdout/` preserve stage-separated evidence; `training/`
contains compact checkpoint hashes and curves; `raw/packed/` contains
losslessly replayable ledgers; `statistics/` and `tables/` contain panel-level
analysis; `figures/` contains vector PDFs, previews, and exact source CSVs;
`negative_results/` retains invalidated and failed runs.

These are abstract simulations and one lightweight decentralized RL
implementation, not real logistics deployment, real utility validation, or
real-human evidence. Development is not confirmation; validation is not a
sealed holdout. See `CLAIMS_MATRIX.md` for every prohibited extension.
"""
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "README.md").write_text(readme, encoding="utf-8")
    (results_root / "CLAIMS_MATRIX.md").write_text(
        _claims_matrix(hypotheses), encoding="utf-8",
    )
    (results_root / "PAPER_OUTLINE.md").write_text(_paper_outline(), encoding="utf-8")
    supported = [value for value in hypotheses if str(value["status"]).startswith("supported")]
    failed = [value for value in hypotheses if not str(value["status"]).startswith("supported")]
    paper_summary = f"""# V8 paper summary

**Working title:** Communication-Efficient Distributed Belief Monitoring for
Independent Autonomous Agents under Network Disruption

**Highest evidence level:** {supported_level}

**Supported claims:** {supported or ['none']}

**Unsupported or untested claims:** {failed or ['none']}

The paper-facing contribution is limited to the evidence level above. The
primary estimand is panel-paired communication reduction with estimation and
frozen-agent noninferiority. Generalized-entropy superiority is separately
eligible only through H2. All traffic uses measured serialized bytes; all
operators are autonomous policies and no human participants were studied.

Primary numerical results and confidence intervals are in
`tables/primary_effects.csv`; communication accounting is in
`tables/communication_accounting.csv`; gate decisions are in
`tables/gate_outcomes.csv`; figures and exact source data are under `figures/`.
"""
    (results_root / "PAPER_SUMMARY.md").write_text(paper_summary, encoding="utf-8")
    build = {
        "highest_primary_stage": stage,
        "hypotheses": hypotheses,
        "gate_rows": len(gates),
        "primary_rows": len(primary),
        "stage_counts": stage_counts,
        "readme": "README.md", "paper_summary": "PAPER_SUMMARY.md",
        "paper_outline": "PAPER_OUTLINE.md", "claims_matrix": "CLAIMS_MATRIX.md",
    }
    atomic_json(results_root / "reproducibility" / "report_build_summary.json", build)
    return build
