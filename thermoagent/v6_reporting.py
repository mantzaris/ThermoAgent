"""Build the publication-facing V6 package from immutable stored evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .v5_experiments import atomic_json, write_csv
from .v6_artifacts import build_index


APPLICATIONS = ("commercial", "humanitarian", "utility_restoration")


def _json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False, lineterminator="\n")
    except TypeError:  # pandas < 1.5
        frame.to_csv(path, index=False, line_terminator="\n")


def _copy_csv(source: Path, destination: Path) -> None:
    if source.exists():
        _write_frame(destination, pd.read_csv(source))


def _safe_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not available"
    return ("%%.%df" % digits) % float(value)


def _stage_design(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for manifest in sorted((root / "pilots").glob("*/stage_manifest.json")):
        value = _json(manifest)
        rows.append({
            "stage": value["stage"], "evidence_status": "pilot/development iteration",
            "episodes": value.get("episodes", 0),
            "decision_rows": value.get("candidate_decisions", 0),
            "failed_episodes": value.get("failed_episodes", 0),
            "confirmatory": False, "simulated_operator": True,
        })
    for name, label in (
        ("formal_reference", "frozen development reference"),
        ("sketch_reference", "frozen development communication ablation"),
        ("dynamic", "cross-fitted dynamic development evaluation"),
    ):
        path = root / "development" / name / "stage_manifest.json"
        if path.exists():
            value = _json(path)
            rows.append({
                "stage": value["stage"], "evidence_status": label,
                "episodes": value.get("episodes", 0),
                "decision_rows": value.get("candidate_decisions", 0),
                "failed_episodes": value.get("failed_episodes", 0),
                "confirmatory": False, "simulated_operator": True,
            })
    qwen_path = root / "qwen" / "qualification_summary.json"
    if qwen_path.exists():
        value = _json(qwen_path)
        rows.append({
            "stage": "real_qwen_development_qualification",
            "evidence_status": "real open-weight LLM autonomous agents",
            "episodes": value.get("episodes", 0),
            "decision_rows": value.get("decision_epochs", 0),
            "failed_episodes": value.get("failed_episodes", 0),
            "confirmatory": False, "simulated_operator": True,
        })
    training_path = root / "training" / "training_summary.json"
    if training_path.exists():
        value = _json(training_path)
        manifest = _safe_frame(root / "training" / "seed_manifest.csv")
        rows.append({
            "stage": "sequential_decentralized_ppo",
            "evidence_status": "multi-seed development training/evaluation",
            "episodes": int(manifest.get("training_episodes", pd.Series(dtype=float)).fillna(0).sum()
                            + manifest.get("evaluation_episodes", pd.Series(dtype=float)).fillna(0).sum()),
            "decision_rows": int(manifest.get("training_decision_epochs", pd.Series(dtype=float)).fillna(0).sum()
                                 + manifest.get("evaluation_decision_epochs", pd.Series(dtype=float)).fillna(0).sum()),
            "failed_episodes": value.get("failed_runs", 0),
            "confirmatory": False, "simulated_operator": False,
        })
    gate_path = root / "development" / "gate_status.json"
    validation_unlocked = _json(gate_path).get("validation_unlocked", False) if gate_path.exists() else False
    for stage in ("validation", "holdout"):
        manifest = root / stage / "stage_manifest.json"
        if manifest.exists():
            value = _json(manifest)
            rows.append({
                "stage": stage, "evidence_status": "formal %s" % stage,
                "episodes": value.get("episodes", 0),
                "decision_rows": value.get("candidate_decisions", 0),
                "failed_episodes": value.get("failed_episodes", 0),
                "confirmatory": stage == "holdout", "simulated_operator": True,
            })
        else:
            rows.append({
                "stage": stage,
                "evidence_status": "not run—prospectively locked" if not validation_unlocked else "not yet run",
                "episodes": 0, "decision_rows": 0, "failed_episodes": 0,
                "confirmatory": stage == "holdout", "simulated_operator": True,
            })
    frame = pd.DataFrame(rows)
    _write_frame(root / "tables" / "experimental_design.csv", frame)
    return frame


def _entropy_table(root: Path) -> None:
    rows = [
        {"measure": "Shannon entropy", "symbol": "H_1", "status": "prespecified reference", "interpretation": "normalized uncertainty within one belief"},
        {"measure": "Tsallis entropy", "symbol": "H_q", "status": "q in {0.5,1,1.5,2,3}", "interpretation": "tail-sensitive generalized uncertainty"},
        {"measure": "Gini-Simpson impurity", "symbol": "G", "status": "primary Gini-family measure", "interpretation": "normalized q=2 Tsallis equivalent"},
        {"measure": "Jensen-Shannon disagreement", "symbol": "D_1", "status": "prespecified reference", "interpretation": "epistemic disagreement among agents"},
        {"measure": "Jensen-Tsallis disagreement", "symbol": "D_q", "status": "prespecified family", "interpretation": "generalized pooled-minus-local uncertainty"},
        {"measure": "Graph-weighted disagreement", "symbol": "D_graph", "status": "distributed measure", "interpretation": "edge-available pairwise disagreement"},
        {"measure": "Operational energy", "symbol": "E", "status": "secondary KPI-derived ablation", "interpretation": "normalized operational stress"},
        {"measure": "Free-energy-style diagnostic", "symbol": "F", "status": "exploratory only", "interpretation": "not literal physical free energy"},
    ]
    write_csv(root / "tables" / "entropy_measure_definitions.csv", rows)


def _action_accounting(root: Path) -> None:
    frame = _safe_frame(root / "development" / "dynamic" / "completed_actions.csv")
    if frame.empty:
        return
    proposal = frame["proposal"].astype(str) if "proposal" in frame else pd.Series("", index=frame.index)
    rows: List[Dict[str, Any]] = []
    for (application, controller), subset in frame.groupby(["application", "controller"], sort=True):
        rows.append({
            "application": application, "controller": controller,
            "completed_typed_actions": len(subset),
            "accepted_physical_actions": int(subset["accepted_physical_action"].astype(bool).sum()),
            "beneficial_actions": int(subset["beneficial"].astype(bool).sum()),
            "neutral_actions": int(subset["neutral"].astype(bool).sum()),
            "harmful_actions": int(subset["harmful"].astype(bool).sum()),
            "reached_next_stage": int(subset["reached_next_stage"].astype(bool).sum()),
            "reached_service": int(subset["reached_service"].astype(bool).sum()),
            "changed_commitment": int(subset["changed_commitment"].astype(bool).sum()),
            "mean_causal_effect": float(subset["causal_effect"].mean()),
        })
    write_csv(root / "tables" / "dynamic_action_accounting.csv", rows)


def _qwen_table(root: Path) -> None:
    path = root / "qwen" / "qualification_summary.json"
    if not path.exists():
        return
    report = _json(path)
    rows = [{"application": application, **values} for application, values in report["applications"].items()]
    write_csv(root / "tables" / "qwen_agent_qualification.csv", rows)


def _compute_table(root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for name, path in (
        ("formal_reference", root / "development" / "formal_reference" / "episode_summary.csv"),
        ("sketch_reference", root / "development" / "sketch_reference" / "episode_summary.csv"),
        ("dynamic_development", root / "development" / "dynamic" / "episode_summary.csv"),
    ):
        frame = _safe_frame(path)
        if not frame.empty:
            rows.append({
                "workflow": name, "episodes": len(frame), "llm_calls": 0,
                "prompt_tokens": 0, "generated_tokens": 0,
                "operational_messages": int(frame.total_messages.sum() - frame.sketch_messages.sum()),
                "thermodynamic_sketch_messages": int(frame.sketch_messages.sum()),
                "total_bytes": int(frame.total_bytes.sum()),
                "wall_seconds": float(frame.wall_seconds.sum()), "gpu_hours": 0.0,
            })
    training = _safe_frame(root / "training" / "seed_manifest.csv")
    if not training.empty:
        rows.append({
            "workflow": "sequential_decentralized_ppo", "episodes": int(training.training_episodes.fillna(0).sum() + training.evaluation_episodes.fillna(0).sum()),
            "llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0,
            "operational_messages": 0, "thermodynamic_sketch_messages": 0,
            "total_bytes": 0, "wall_seconds": float(training.wall_seconds.fillna(0).sum()),
            "gpu_hours": float(training.loc[training.device.astype(str).str.contains("cuda"), "wall_seconds"].fillna(0).sum() / 3600.0),
        })
    qwen_path = root / "qwen" / "qualification_summary.json"
    if qwen_path.exists():
        value = _json(qwen_path)
        rows.append({
            "workflow": "real_qwen_qualification", "episodes": value["episodes"],
            "llm_calls": value["llm_calls"], "prompt_tokens": value["prompt_tokens"],
            "generated_tokens": value["generated_tokens"], "operational_messages": 0,
            "thermodynamic_sketch_messages": 0, "total_bytes": 0,
            "wall_seconds": value["wall_seconds_including_model_load"],
            "gpu_hours": value["wall_seconds_including_model_load"] / 3600.0,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    frame["estimated_cost_usd_at_0_34_per_gpu_hour"] = frame.gpu_hours * 0.34
    _write_frame(root / "tables" / "compute_token_communication_accounting.csv", frame)
    totals = {
        "workflows": len(frame), "episodes": int(frame.episodes.sum()),
        "llm_calls": int(frame.llm_calls.sum()), "prompt_tokens": int(frame.prompt_tokens.sum()),
        "generated_tokens": int(frame.generated_tokens.sum()),
        "wall_seconds_sum": float(frame.wall_seconds.sum()),
        "single_gpu_hours": float(frame.gpu_hours.sum()),
        "estimated_cost_usd_at_0_34_per_gpu_hour": float(frame.estimated_cost_usd_at_0_34_per_gpu_hour.sum()),
    }
    atomic_json(root / "reproducibility" / "compute_summary.json", totals)
    return totals


def _stage_disposition(root: Path, gates: Mapping[str, Any]) -> None:
    validation = root / "validation"
    holdout = root / "holdout"
    if not gates.get("validation_unlocked", False):
        validation.mkdir(parents=True, exist_ok=True)
        (validation / "NOT_RUN.md").write_text(
            "# Validation prospectively not run\n\nRequired frozen V6 development gates failed. No validation outcome was generated or inspected.\n",
            encoding="utf-8",
        )
        holdout.mkdir(parents=True, exist_ok=True)
        (holdout / "NOT_RUN.md").write_text(
            "# Sealed holdout prospectively not run\n\nValidation was not unlocked, so the sealed V6 holdout was never executed or inspected.\n",
            encoding="utf-8",
        )
    atomic_json(root / "manifests" / "stage_disposition.json", {
        "development": "complete",
        "validation": "complete" if (validation / "stage_manifest.json").exists() else "prospectively_not_run",
        "holdout": "complete" if (holdout / "stage_manifest.json").exists() else "prospectively_not_run",
        "development_gates_all_passed": bool(gates.get("all_required_development_gates_passed", False)),
        "validation_unlocked": bool(gates.get("validation_unlocked", False)),
        "holdout_unlocked": bool(gates.get("holdout_unlocked", False)),
    })


def _claims(root: Path, gates: Mapping[str, Any]) -> pd.DataFrame:
    gate_map = {int(value["gate"]): value for value in gates.get("gates", [])}
    statuses = [
        ("H1", "Generalized-entropic control reduces harm at matched coverage in both primary applications", 5),
        ("H2", "Selective abstention preserves utility and service under a fixed operator budget", 6),
        ("H3", "Incremental value is greater under fragmented than public information", 7),
        ("H4", "Event-triggered sketches preserve safety with lower complete communication cost", 8),
        ("H5", "A prespecified generalized entropy family adds value beyond Shannon", 5),
        ("H6", "Distributed estimates remain bounded and degrade predictably under partitions", 8),
        ("H7", "The selective-safety effect replicates in humanitarian and utility restoration", 10),
        ("H8", "RL and real-Qwen agents satisfy decentralized agentic validity", 4),
    ]
    rows = []
    for hypothesis, statement, gate_number in statuses:
        gate = gate_map.get(gate_number, {})
        rows.append({
            "hypothesis": hypothesis, "statement": statement,
            "evidence_stage": "development",
            "status": "supported_in_development" if gate.get("passed") else "unsupported_or_mixed",
            "supporting_gate": gate_number,
            "supporting_files": "development/gate_status.json;development/gate_checks.csv",
            "confirmatory": False,
        })
    frame = pd.DataFrame(rows)
    _write_frame(root / "tables" / "hypothesis_outcomes.csv", frame)
    lines = ["# V6 claims-to-evidence matrix", "", "All operator evidence uses a simulated operator; no human participants were studied.", "", "| Claim | Development disposition | Evidence |", "|---|---|---|"]
    for row in frame.itertuples(index=False):
        lines.append("| %s: %s | %s | `%s` |" % (row.hypothesis, row.statement, row.status, row.supporting_files))
    lines.extend(["", "No claim in this matrix is confirmatory unless a separately locked holdout is recorded.\n"])
    (root / "CLAIMS_MATRIX.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def _readme(root: Path, gates: Mapping[str, Any], design: pd.DataFrame, compute: Mapping[str, Any]) -> None:
    failed = [value for value in gates.get("gates", []) if not value.get("passed")]
    disposition = gates.get("scientific_disposition", "not_yet_evaluated")
    dynamic = _safe_frame(root / "development" / "dynamic" / "paired_dynamic_effects.csv")
    effects = []
    for app in ("humanitarian", "utility_restoration", "commercial"):
        subset = dynamic[(dynamic.application == app) & (dynamic.information_condition == "private_fragmented")] if not dynamic.empty else pd.DataFrame()
        if len(subset):
            row = subset.iloc[0]
            effects.append(
                "- %s: harm-rate reduction %s (95%% CI %s to %s); relative service-loss change %s; net causal-utility change %s."
                % (app.replace("_", " ").title(), _fmt(row.harm_rate_reduction), _fmt(row.harm_ci95_low), _fmt(row.harm_ci95_high), _fmt(row.relative_service_loss_degradation), _fmt(row.net_causal_utility_gain))
            )
    qwen_path = root / "qwen" / "qualification_summary.json"
    qwen = _json(qwen_path) if qwen_path.exists() else {}
    failed_text = ", ".join("Gate %d (%s)" % (value["gate"], value["name"]) for value in failed) or "none"
    text = f"""# Generalized Entropic Consensus V6

## Research question

Can generalized measures of decentralized uncertainty and consensus identify when independent autonomous-agent recommendations are unsafe, improving selective autonomy, abstention, communication, and bounded simulated-operator escalation at matched action coverage and operator budget?

V6 is scientifically distinct from V5. V5's immutable negative result remains unchanged: KPI plus Shannon entropy and Jensen–Shannon disagreement did not improve direct intervention ranking. V6 instead asks whether uncertainty and consensus predict *when to delegate*, not which domain action is correct.

## Evidence status

The authoritative disposition is **{disposition.replace('_', ' ')}**. Failed required gates: {failed_text}. This package distinguishes pilots, frozen development, real-Qwen qualification, multi-seed sequential PPO, validation, and sealed holdout. It never treats candidate decisions within one panel as independent replicates.

Validation status: **{'unlocked' if gates.get('validation_unlocked') else 'prospectively not run'}**. Holdout status: **{'unlocked' if gates.get('holdout_unlocked') else 'prospectively not run'}**.

## Applications and independence

- Humanitarian logistics and abstract defensive utility restoration are the primary replication applications.
- Commercial logistics is a prespecified boundary application where ordinary KPIs may suffice.
- Every organization has a private observation, belief distribution, memory vault, utility, inbox/outbox, commitments, role-specific typed tools, and separate decision authority. Partitions block delivery. The simulator validates actions but does not replace rejected decisions with oracle actions.
- Utility cyber events are abstract simulator state changes only. No real infrastructure, protocol, credential, device, or external target was accessed.

## Measures

For belief `p_i` over six incident modes, V6 computes normalized Shannon entropy and Tsallis entropy at q = 0.5, 1, 1.5, 2, and 3. Gini–Simpson impurity is the normalized q=2 case. Reliability-weighted pooled beliefs support Jensen–Shannon and Jensen–Tsallis disagreement, graph-weighted disagreement, consensus residuals, and temporal slopes. Operational energy and free-energy-style quantities remain secondary diagnostics and are not literal thermodynamics.

All distributed-sketch messages, bytes, latency, operational messages, LLM calls, prompt tokens, generated tokens, GPU time, and simulated-operator minutes are counted.

## V5 fair-abstention addendum

`v5_reanalysis/` preserves the original V5 findings and adds same-score, coverage-matched, mandatory-action, and operator-budget-matched comparisons. It does not unlock V5 validation or revise V5 gates.

## Frozen development findings

{chr(10).join(effects) if effects else '- Formal development effects have not been generated.'}

The gate table at `development/gate_checks.csv` is authoritative. Negative, zero, and harmful actions are retained. Simulated-operator results are not evidence about real-human usability, workload, trust, or effectiveness.

## Real Qwen and learned agents

Primary model: `Qwen/Qwen2.5-7B-Instruct`, immutable revision `a09a35458c702b33eeacc393d103063234e8bc28`, bitsandbytes NF4, BF16 computation. Real-Qwen qualification contains {qwen.get('episodes', 0)} episodes and {qwen.get('decision_epochs', 0)} independent-agent decision records. Sequential role-specific PPO uses local execution observations, action masks, discounted trajectories, GAE, clipping, and five independent seeds per method. It is not the V5 contextual actor-critic.

## Compute

- Recorded single-GPU hours: {_fmt(compute.get('single_gpu_hours', 0), 4)}.
- LLM calls: {int(compute.get('llm_calls', 0)):,}.
- Prompt tokens: {int(compute.get('prompt_tokens', 0)):,}; generated tokens: {int(compute.get('generated_tokens', 0)):,}.
- Approximate GPU cost at the recorded $0.34/hour accounting rate: ${_fmt(compute.get('estimated_cost_usd_at_0_34_per_gpu_hour', 0), 2)}.

## Artifact map

- `protocol/`: frozen protocol and checksums.
- `manifests/`: sealed input manifests and stage disposition.
- `v5_reanalysis/`: fair V5 safety reanalysis and implementation audit.
- `pilots/`: retained design iterations, including failures and superseded pilots.
- `development/`: frozen reference, cross-fitting, dynamic evaluation, learnability, communication, gates, and power evidence.
- `training/`: five-seed sequential PPO manifests, curves, and small checkpoints.
- `qwen/`: real open-weight agent decision and episode summaries.
- `raw/`: compressed event-sourced episodes.
- `tables/` and `statistics/`: publication-facing numerical summaries.
- `figures/pdf/` and `figures/png/`: vector figures and 240-DPI previews.
- `dashboard_exports/`: populated deterministic replay export.
- `reproducibility/`: replay, checksum, environment, compute, PDF QA, deviations, exclusions, and failures.

## Reproduction

```bash
./scripts/run-v6-tests.sh
./scripts/run-v6-v5-reanalysis.sh
./scripts/run-v6-pilot.sh
./scripts/freeze-v6-protocol.sh
./scripts/run-v6-development.sh
./scripts/analyze-v6-development.sh
./scripts/train-v6-multiseed.sh
./scripts/run-v6-real-qwen.sh
./scripts/replay-v6-results.sh
./scripts/evaluate-v6-gates.sh
./scripts/build-v6-report.sh
./scripts/generate-v6-figures.sh
./scripts/validate-v6-pdfs.sh
./scripts/index-v6-artifacts.sh
```

Validation and holdout scripts enforce gate locks and refuse execution when not prospectively unlocked.

## Limitations and readiness

All domains and operators are simulations. The models are abstractions, not validated logistics or critical-infrastructure digital twins. No human participants were studied. Development evidence cannot establish confirmatory generalization. `PAPER_SUMMARY.md` gives the evidence-specific publication disposition; `PAPER_OUTLINE.md` is a writing plan, not a completed manuscript.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _paper_files(root: Path, gates: Mapping[str, Any]) -> None:
    all_passed = bool(gates.get("all_required_development_gates_passed", False))
    failed = ["Gate %d (%s)" % (value["gate"], value["name"]) for value in gates.get("gates", []) if not value.get("passed")]
    summary = f"""# Provisional paper summary

## Working title

Generalized Entropic Consensus for Risk-Controlled Human Oversight of Decentralized Autonomous Agents

## Evidence-aware abstract

We study selective autonomy among independent agents with private observations, beliefs, memories, utilities, commitments, and bounded authority. We compare Shannon, Tsallis, Gini–Simpson, Jensen–Shannon, Jensen–Tsallis, and graph-weighted distributed uncertainty measures as predictors of unsafe autonomous recommendations. The evaluation uses matched dynamic counterfactual trajectories, action-coverage and operator-budget controls, complete sketch accounting, five-seed sequential decentralized PPO, and a substantial pinned-Qwen qualification across humanitarian logistics, abstract defensive utility restoration, and a commercial boundary application. The current evidence is {'eligible to proceed beyond development' if all_passed else 'a development-stage no-go'}; {'all frozen development gates passed' if all_passed else 'required gates failed: ' + ', '.join(failed)}. No real human operators were studied, and no confirmatory claim is made without an executed sealed holdout.

## Verified contribution boundary

- The software contribution supports privacy-preserving generalized-entropic consensus, dynamic selective delegation, exact event replay, true resource accounting, and bounded simulated-operator escalation.
- Scientific claims are limited to the stages actually executed.
- Entropy/disagreement are primary information measures; energy and free-energy-style summaries are secondary diagnostics.
- Commercial logistics remains a prespecified boundary rather than a required positive domain.

## Journal readiness

{'Development gates passed, but validation and a sealed holdout remain necessary before an AIJ submission claim.' if all_passed else 'The no-go does not support the intended positive AIJ claim. The artifact is suitable as a rigorous engineering and boundary-study package; further preregistered redesign would be needed before journal submission.'}

See `CLAIMS_MATRIX.md`, `tables/hypothesis_outcomes.csv`, and `development/gate_status.json` for the exact claim-to-evidence mapping.
"""
    (root / "PAPER_SUMMARY.md").write_text(summary, encoding="utf-8")
    outline = """# Artificial Intelligence article outline (planned 24–28 pages)

1. **Introduction and claim boundary (2 pages).** Selective autonomy, scarce oversight, V5 negative motivation, conditional—not universal—thermodynamic observability claim.
2. **Related work (3 pages).** Multi-agent autonomy, selective prediction and abstention, human-on-the-loop systems, distributed detection, entropy families, cyber-physical restoration simulations.
3. **Problem formulation (2 pages).** Independent-agent contract, private information, two-level action/delegation policy, matched coverage and operator budget, applications and outcomes.
4. **Generalized entropic consensus (3 pages).** Shannon/Tsallis definitions, Gini–Simpson equivalence, weighted pooling, Jensen–Tsallis disagreement, graph estimator, temporal measures, operational-energy boundary.
5. **System architecture (3 pages).** Agent memory and tools, ad-hoc messaging, partitions, simulated operator, dashboard information boundary, evaluator-only counterfactuals, event sourcing.
6. **Dynamic environments (3 pages).** Humanitarian logistics, abstract defensive utility restoration, commercial boundary; simultaneous incidents, resources, harms, stochastic tapes, conservation.
7. **Learning and baselines (2 pages).** Interpretable cross-fitting, conformal/uncertainty comparators, sequential decentralized PPO, pinned Qwen, oracle bounds.
8. **Prospective protocol and statistics (2 pages).** Gates, seed isolation, nested grouped fitting, paired cluster bootstrap, hierarchy/Holm, risk/utility coverage, communication accounting.
9. **Engineering and agentic qualification (1.5 pages).** Tests, privacy, replay, conservation, Qwen behavior, PPO stability.
10. **Development/validation/holdout results (3 pages).** Report only stages run; effects, confidence intervals, gates, harms, service, operator load, costs.
11. **Mechanism and robustness (2 pages).** Fragmented-versus-public interaction, partitions, stale/corrupt messages, q spectrum, sketch policies, abstention.
12. **Case studies and dashboard replay (1.5 pages).** Humanitarian and utility sequences; authorized versus evaluator-only views.
13. **Limitations, ethics, and future human study (1.5 pages).** Simulated operator, abstract cyber events, external validity, IRB boundary.
14. **Conclusion (0.5 page).** Evidence-calibrated contribution or no-go boundary.

Optional supplement: full schemas, prompts, all seed curves, mathematical tests, environment parameters, intervention catalog, replay audit, protocol deviations, and PDF QA.
"""
    (root / "PAPER_OUTLINE.md").write_text(outline, encoding="utf-8")


def build_v6_reporting(repository: Path) -> Dict[str, Any]:
    root = repository / "results" / "generalized_entropic_consensus_v6"
    gate_path = root / "development" / "gate_status.json"
    if not gate_path.exists():
        raise FileNotFoundError("development gate report is required before final reporting")
    gates = _json(gate_path)
    design = _stage_design(root)
    _entropy_table(root)
    for source, name in (
        (root / "development" / "risk_analysis" / "primary_matched_effects.csv", "risk_coverage_primary_effects.csv"),
        (root / "development" / "risk_analysis" / "prediction_metrics.csv", "risk_prediction_metrics.csv"),
        (root / "development" / "dynamic" / "paired_dynamic_effects.csv", "dynamic_paired_effects.csv"),
        (root / "development" / "dynamic" / "regime_dynamic_effects.csv", "regime_dynamic_effects.csv"),
        (root / "development" / "dynamic" / "fragmentation_interaction.csv", "fragmentation_interaction.csv"),
        (root / "development" / "communication" / "sketch_costs.csv", "sketch_communication_costs.csv"),
        (root / "development" / "communication" / "distributed_estimation_error.csv", "distributed_estimation_error.csv"),
        (root / "development" / "permutation" / "refit_permutation_family_test.csv", "refit_permutation_family_test.csv"),
        (root / "development" / "gate_checks.csv", "development_gate_checks.csv"),
        (root / "training" / "seed_manifest.csv", "rl_seed_manifest.csv"),
    ):
        _copy_csv(source, root / "tables" / name)
    _action_accounting(root)
    _qwen_table(root)
    compute = _compute_table(root)
    _stage_disposition(root, gates)
    _claims(root, gates)
    _readme(root, gates, design, compute)
    _paper_files(root, gates)
    index = build_index(root)
    return {
        "scientific_disposition": gates.get("scientific_disposition"),
        "development_episodes": int(design.loc[design.evidence_status.str.contains("development", case=False), "episodes"].sum()),
        "index": index,
    }
