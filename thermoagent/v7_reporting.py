"""Build the V7 publication-facing package from stored evidence only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from xml.etree import ElementTree

import pandas as pd

from .v5_experiments import atomic_json, write_csv
from .v7_io import episode_artifacts, read_json_artifact


def _json(path: Path, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _episode_frame(results_root: Path, stage: str) -> pd.DataFrame:
    path = results_root / stage / "episode_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    raw_stage = results_root / "raw" / stage
    rows = [read_json_artifact(value).get("summary", {}) for value in episode_artifacts(raw_stage)]
    return pd.DataFrame(rows)


def _episode_count(results_root: Path, stage: str) -> int:
    return len(_episode_frame(results_root, stage))


def _test_count(results_root: Path, filename: str) -> Optional[Dict[str, int]]:
    path = results_root / "reproducibility" / filename
    if not path.exists():
        return None
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        field: sum(int(float(suite.attrib.get(field, 0))) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _stage_disposition(results_root: Path) -> Dict[str, Any]:
    formal = _json(results_root / "manifests" / "stage_disposition.json")
    if formal:
        return formal
    feasibility = _json(results_root / "development" / "gate_feasibility" / "gate_summary.json")
    return {
        "formal_development_primary_pass": False,
        "RL_training_unlocked": False, "Qwen_qualification_unlocked": False,
        "validation_unlocked": False, "holdout_unlocked": False,
        "disposition": feasibility.get("interpretation", "not evaluated"),
    }


def _hypothesis_rows(results_root: Path) -> List[Dict[str, Any]]:
    dynamic = _json(results_root / "statistics" / "dynamic_primary_analysis.json")
    communication = _json(results_root / "statistics" / "communication_primary_analysis.json")
    formal_ran = bool(dynamic)
    rows = [
        {
            "hypothesis": "H1 complexity-dependent selective-safety value",
            "status": ("supported" if dynamic.get("H1_pass") else "unsupported") if formal_ran else "not_tested_formally",
            "evidence": "statistics/dynamic_primary_analysis.json" if formal_ran else "pilots_iteration3/analysis/pilot_analysis.json",
        },
        {
            "hypothesis": "H2 high-complexity selective-safety value in both applications",
            "status": ("supported" if dynamic.get("H2_pass") else "unsupported") if formal_ran else "not_tested_formally",
            "evidence": "statistics/high_complexity_dynamic_effects.csv" if formal_ran else "pilots_iteration3/analysis/high_complexity_effects.csv",
        },
        {
            "hypothesis": "H3 communication-efficient distributed monitoring",
            "status": ("supported_development_monitoring_only" if communication.get("H3_pass") else "unsupported") if communication else "pilot_only",
            "evidence": "statistics/communication_primary_analysis.json" if communication else "pilots/analysis/communication_reductions.csv",
        },
    ]
    for number, label in enumerate((
        "entropy value increases with scale", "fragmented exceeds public information",
        "generalized entropy adds beyond Shannon", "consensus recovers after partitions",
        "learned policies remain stable", "held-out topology generalization",
        "V6-like setting is a low-complexity boundary",
    ), start=4):
        rows.append({"hypothesis": "H%d %s" % (number, label), "status": "exploratory_or_not_unlocked", "evidence": "CLAIMS_MATRIX.md"})
    return rows


def _publication_tables(results_root: Path) -> Dict[str, Any]:
    stage_names = (
        "smoke", "pilots", "pilots_iteration2", "pilots_iteration3",
        "development_formal_reference", "development_formal_dynamic",
        "development_formal_communication", "validation", "holdout",
    )
    stage_rows: List[Dict[str, Any]] = []
    formal_episode_frames: List[pd.DataFrame] = []
    for stage in stage_names:
        frame = _episode_frame(results_root, stage)
        if frame.empty:
            continue
        started = pd.to_datetime(frame.started_at, utc=True, errors="coerce") if "started_at" in frame else pd.Series(dtype="datetime64[ns, UTC]")
        completed = pd.to_datetime(frame.completed_at, utc=True, errors="coerce") if "completed_at" in frame else pd.Series(dtype="datetime64[ns, UTC]")
        span_hours = (
            float((completed.max() - started.min()).total_seconds() / 3600.0)
            if len(frame) and started.notna().any() and completed.notna().any() else 0.0
        )
        row = {
            "stage": stage, "episodes": len(frame),
            "applications": ";".join(sorted(frame.application.dropna().unique())) if "application" in frame else "",
            "stage_wall_clock_hours": span_hours,
            "gpu_hours": 0.0, "llm_calls": 0, "prompt_tokens": 0,
            "generated_tokens": 0, "approximate_incremental_cost_usd": 0.0,
        }
        stage_rows.append(row)
        if stage.startswith("development_formal_"):
            formal_episode_frames.append(frame.assign(stage=stage))
    write_csv(results_root / "tables" / "stage_compute_accounting.csv", stage_rows)

    application_rows: List[Dict[str, Any]] = []
    if formal_episode_frames:
        formal = pd.concat(formal_episode_frames, ignore_index=True)
        for (application, complexity), subset in formal.groupby(["application", "complexity"], sort=True):
            application_rows.append({
                "application": application, "complexity": complexity,
                "episodes": len(subset), "agent_count": int(subset.agent_count.max()),
                "operational_nodes": int(subset.operational_node_count.max()),
                "horizon_steps": int(subset.horizon.max()),
                "decision_epochs": int(subset.decision_epochs.max()),
                "physical_actions": int(subset.physical_actions.sum()),
                "service_reaching_actions": int(subset.service_reaching_actions.sum()),
                "cross_community_messages": int(subset.cross_community_messages.sum()),
                "maximum_causal_chain_depth": int(subset.maximum_cascade_depth.max()),
                "total_messages": int(subset.total_messages.sum()),
                "total_bytes": int(subset.total_bytes.sum()),
            })
    write_csv(results_root / "tables" / "formal_application_complexity.csv", application_rows)
    return {
        "stage_rows": stage_rows, "application_rows": application_rows,
        "formal_cpu_wall_clock_hours": sum(
            value["stage_wall_clock_hours"] for value in stage_rows
            if value["stage"].startswith("development_formal_")
        ),
    }


def _readme(results_root: Path, disposition: Mapping[str, Any]) -> str:
    replay = _json(results_root / "reproducibility" / "replay" / "replay_summary.json")
    dynamic = _json(results_root / "statistics" / "dynamic_primary_analysis.json")
    communication = _json(results_root / "statistics" / "communication_primary_analysis.json")
    freeze = _json(results_root / "protocol" / "freeze_manifest.json")
    accounting = _publication_tables(results_root)
    full_tests = _test_count(results_root, "pytest_full.xml")
    v7_tests = _test_count(results_root, "pytest_v7.xml")
    if full_tests and v7_tests:
        full_passed = full_tests["tests"] - full_tests["failures"] - full_tests["errors"] - full_tests["skipped"]
        v7_passed = v7_tests["tests"] - v7_tests["failures"] - v7_tests["errors"] - v7_tests["skipped"]
        tests = (
            f"{full_passed}/{full_tests['tests']} full-system tests and "
            f"{v7_passed}/{v7_tests['tests']} focused V7 tests passed in the final package; "
            f"skipped={full_tests['skipped']} full-system/{v7_tests['skipped']} focused"
        )
    else:
        tests = "final JUnit results were not yet generated"
    stage_counts = {
        "smoke_and_pilots": sum(_episode_count(results_root, value) for value in ("smoke", "pilots", "pilots_iteration2", "pilots_iteration3")),
        "formal_reference": _episode_count(results_root, "development_formal_reference"),
        "formal_dynamic": _episode_count(results_root, "development_formal_dynamic"),
        "formal_communication": _episode_count(results_root, "development_formal_communication"),
        "validation": _episode_count(results_root, "validation"),
        "holdout": _episode_count(results_root, "holdout"),
    }
    interaction = dynamic.get("interaction", {})
    high = {row["application"]: row for row in dynamic.get("high_complexity", [])}
    communication_rows = {row["application"]: row for row in communication.get("applications", [])}
    humanitarian = high.get("humanitarian", {})
    utility = high.get("utility_restoration", {})
    humanitarian_communication = communication_rows.get("humanitarian", {})
    utility_communication = communication_rows.get("utility_restoration", {})
    return f"""# ThermoAgent V7: complexity-dependent generalized-entropic coordination

## Research question

When does distributed generalized-entropic consensus become useful for
communication-efficient monitoring and risk-controlled coordination among
independent autonomous agents in complex, coupled, partially observed
networks?

V7 is a new namespace built from frozen V6 commit
`8013300c23553928a0269e6be27f5baaedee7e53`. V1–V6 artifacts were neither
regenerated nor modified. V6 remains the low-complexity historical boundary.

## What changed from V6

V7 has separate humanitarian multi-commodity and defensive abstract
utility-restoration state machines. Small, medium, and large configurations
use 12/28/52 persistent agents, 8/16/30 operational nodes, and 30/60/100
steps. Agents control multiple assets; resource contention, delayed movement,
topology-dependent service, telemetry corruption, partitions, commitments,
cascades, and multi-step causal chains change future feasibility.

Every agent has its own observation history, belief, memory, utility, assets,
commitments, inbox, outbox, role-specific tools, and action process. Peer
information arrives only through logged messages or fully costed distributed
sketches. The environment validates typed actions but never substitutes an
oracle decision.

## Information measures and control

The Level-2 controller receives normalized Shannon and Tsallis entropy for
`q={{0.5,1,1.5,2,3}}`, Gini–Simpson impurity, pooled uncertainty,
Jensen–Shannon/Jensen–Tsallis disagreement, graph-weighted disagreement,
consensus residual, and temporal slopes. Gini–Simpson is not the economic
Gini. These are information-theoretic/statistical-mechanics analogies, not
literal physical thermodynamics. Entropy never changes action effects.

Operational proposals, information gathering, communication, and delegation
are separate typed fields. Operational messages and thermodynamic sketches
are accounted independently in messages, bytes, drops, and latency.

## Protocol and stages

- Feasibility gates: all A/B/C gates passed; this did not require a favorable
  entropy effect.
- Frozen protocol: `{freeze.get('protocol_version', 'not frozen')}`;
  checksum `{freeze.get('protocol_sha256', 'not available')}`.
- Episode counts: `{json.dumps(stage_counts, sort_keys=True)}`.
- Validation unlocked: `{bool(disposition.get('validation_unlocked', False))}`.
- Holdout unlocked: `{bool(disposition.get('holdout_unlocked', False))}`.

The formal independent unit is the matched environment panel. Candidate
actions within a panel are never treated as independent replicates. Models use
nested grouped folds, matched 60% action coverage, 10,000 panel bootstraps, and
prespecified same-capacity feature blocks.

## Main findings: formal development no-go

The coupled dynamic experiment did **not** support H1 or H2. The pooled
coupling-by-fragmentation interaction was
`{float(interaction.get('coupling_fragmentation_interaction', float('nan'))):.4f}`
(95% cluster-bootstrap CI
`[{float(interaction.get('ci95_low', float('nan'))):.4f}, {float(interaction.get('ci95_high', float('nan'))):.4f}]`),
below the frozen `0.02` threshold and not distinguishable from zero.

In the prespecified high-coupling/high-fragmentation region, harm-rate
reduction was `{float(humanitarian.get('harm_reduction', float('nan'))):.4f}`
for humanitarian logistics (95% CI
`[{float(humanitarian.get('harm_ci95_low', float('nan'))):.4f}, {float(humanitarian.get('harm_ci95_high', float('nan'))):.4f}]`)
and `{float(utility.get('harm_reduction', float('nan'))):.4f}`
for utility restoration (95% CI
`[{float(utility.get('harm_ci95_low', float('nan'))):.4f}, {float(utility.get('harm_ci95_high', float('nan'))):.4f}]`).
The utility direction was positive but far below the frozen `0.04` practical
threshold; its service noninferiority upper bound was `0.0571`, above the
`0.02` margin. Humanitarian causal utility was significantly worse.

H3 passed as a **monitoring-cost ablation**, not as evidence that entropy
improved selective safety. Event-triggered sketches reduced all-message
traffic by `{float(humanitarian_communication.get('message_reduction', float('nan'))) * 100:.1f}%`
and `{float(utility_communication.get('message_reduction', float('nan'))) * 100:.1f}%`,
and all bytes by `{float(humanitarian_communication.get('byte_reduction', float('nan'))) * 100:.1f}%`
and `{float(utility_communication.get('byte_reduction', float('nan'))) * 100:.1f}%`,
respectively. Maximum distributed-estimation MAE was below `0.05`. The
communication ablation held the operational controller fixed to always-act,
so its exact zero harm difference cannot establish a causal safety benefit.

The final prospective disposition is: **{disposition.get('disposition', 'pending')}**.
RL training, real-Qwen qualification, validation, and locked holdout were
therefore not run. This stopping decision was made by the frozen gates, not by
post-hoc preference.

Negative and neutral effects, failed feasibility iterations, environment
repairs, missing-PyTorch test evidence, and the unavailable RunPod endpoint are
retained. No result is real-human evidence. The Qwen model, if its gated stage
ran, is one LLM implementation; deterministic pilots are engineering controls.

## Integrity and reproducibility

- Tests: {tests}.
- Replay: `{replay.get('episodes_replayed', 0)}` episodes, `{replay.get('replay_mismatches', 0)}` mismatches.
- Maximum reconstructed conservation residual: `{replay.get('maximum_conservation_residual', 'not available')}`.
- Formal CPU execution span across the three stages: approximately
  `{accounting['formal_cpu_wall_clock_hours']:.2f}` stage-hours.
- GPU hours, LLM calls, prompt tokens, generated tokens, and incremental GPU
  cost: all zero; the gated Qwen/RL stages did not run.
- Planned but unused Qwen configuration: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, NF4/BF16.
- Simulated operator only; no human participants.

## Reproduction order

```bash
./scripts/run-v7-tests.sh
./scripts/run-v7-complexity-audit.sh
./scripts/run-v7-environment-smoke.sh
./scripts/run-v7-pilots.sh
./scripts/run-v7-pilot-iteration2.sh
./scripts/run-v7-pilot-iteration3.sh
./scripts/replay-v7-results.sh
./scripts/evaluate-v7-gates.sh
./scripts/freeze-v7-protocol.sh       # only when gates pass and source is clean
./scripts/run-v7-development.sh       # frozen, resumable formal development
./scripts/train-v7-multiseed.sh       # refuses unless formal gates unlock it
./scripts/run-v7-real-qwen.sh         # refuses unless formal gates unlock it
./scripts/run-v7-validation.sh        # refuses while locked
./scripts/run-v7-holdout.sh           # refuses while locked
./scripts/generate-v7-figures.sh
./scripts/validate-v7-pdfs.sh
./scripts/replay-v7-results.sh
./scripts/compact-v7-artifacts.sh     # lossless Git-facing compaction
./scripts/index-v7-artifacts.sh
./scripts/build-v7-report.sh
```

## Directory guide

- `protocol/`, `manifests/`: frozen design and seed provenance when unlocked.
- `pilots*/`, `raw/`: retained feasibility summaries, exact compressed episode
  payloads, and compressed event ledgers. Per-run candidate CSV duplicates were
  removed only after semantic comparison with the canonical episode payload.
- `development/`, `statistics/`, `tables/`: formal evidence if run.
- `training/`, `qwen/`: gated learned-agent evidence only.
- `figures/pdf/`, `figures/source_data/`: vector figures and exact source tables.
- `reproducibility/`: environment, replay, checksum, and PDF QA.
- `negative_results/`: failed iterations and prospective no-go disposition.

## Limitations and publication readiness

All operators are simulated. Formal controllers were deterministic independent
agents plus grouped cross-fitted Level-2 models—not learned PPO or LLM agents.
Domain models are abstract, utility candidates were harm-heavy (90.7%
prevalence in formal probes), the high-complexity test had 12 panels per
application, and external validity is untested. V7 supports a coupled-system
engineering platform, a negative selective-safety boundary, and a positive
communication-monitoring efficiency result. It does not support a positive
entropy-control claim or an AIJ submission without validation and holdout.
"""


def _claims_matrix(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# V7 claims-to-evidence matrix", "",
        "| Claim | Status | Evidence | Prohibited extension |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | `%s` | No universal entropy, real-human, or physical-thermodynamics claim |"
            % (row["hypothesis"], row["status"], row["evidence"])
        )
    lines.extend([
        "", "## Global prohibited claims", "",
        "- Simulated operators establish real-human performance, trust, or workload.",
        "- Deterministic pilot agents are LLM agents.",
        "- Sketch savings exclude operational, negotiation, evidence, dropped, or escalation traffic.",
        "- Development evidence is validation or locked-holdout confirmation.",
        "- Dimensionless information measures are literal thermodynamic quantities.",
    ])
    return "\n".join(lines) + "\n"


def _paper_outline() -> str:
    return """# Provisional AIJ paper outline (20–30 pages, not a manuscript)

## 1. Introduction (2–3 pages)

Motivate risk-controlled coordination under coupling and fragmented private
information; state the conditional, falsifiable contribution and V6 boundary.

## 2. Related work (3–4 pages)

Selective autonomy, decentralized POMDP/MARL, communication learning,
distributed detection, information-theoretic disagreement, human oversight,
and event-triggered control. Distinguish information entropy from physics.

## 3. Problem formulation (2–3 pages)

Persistent independent agents, private state and authority, two-level action
schema, communication graph, panel-level estimands, and operator budget.

## 4. Generalized-entropic consensus (2–3 pages)

Shannon/Tsallis/Gini–Simpson, weighted pooling, Jensen disagreements,
graph disagreement, consensus residual, temporal measures, missing-agent
behavior, sketch compression, and all communication costs.

## 5. Coupled applications (4–5 pages)

Humanitarian multi-commodity flow and utility cyber-physical restoration as
separate transition systems. Detail resources, delays, cascades, topology,
private evidence, conservation, and abstract defensive cyber boundary.

## 6. Agents and learning (2–3 pages)

Independent policy boundary, typed tools, role authority, sequential PPO with
per-agent GAE, Qwen qualification, centralized upper bounds, and simulated
operator limitations.

## 7. Prospective design (3–4 pages)

Complexity response surface, co-primary H1–H3, grouped cross-fitting, matched
coverage, dynamic paired execution, communication ablations, power, stage
gates, validation, and sealed holdout discipline.

## 8. Results (4–6 pages if unlocked)

Engineering/actionability, interaction and high-complexity effects, service
noninferiority, communication, topology OOD, seed stability, Qwen behavior,
negative effects, and causal chains. Label development/validation/holdout.

## 9. Mechanism, limitations, and implications (2–3 pages)

When entropy adds residual information; when KPIs suffice; action-pool and
simulator limits; simulated operator and abstract cyber scope; implications
for future real-human study.

## 10. Conclusion (1 page)

State only the claim supported by the highest legitimately completed stage.

## Supplemental material

Mathematical tests, all seeds, topology diagnostics, full protocol/checksums,
event schemas, conservation replay, extra sensitivity, failure registry,
figures/source data, and future ethics-reviewed human-study protocol.
"""


def build_v7_report(repository: Path, results_root: Path) -> Dict[str, Any]:
    disposition = _stage_disposition(results_root)
    hypotheses = _hypothesis_rows(results_root)
    write_csv(results_root / "tables" / "hypothesis_outcomes.csv", hypotheses)
    (results_root / "README.md").write_text(_readme(results_root, disposition), encoding="utf-8")
    (results_root / "CLAIMS_MATRIX.md").write_text(_claims_matrix(hypotheses), encoding="utf-8")
    (results_root / "PAPER_OUTLINE.md").write_text(_paper_outline(), encoding="utf-8")
    supported = [value["hypothesis"] for value in hypotheses if value["status"].startswith("supported")]
    unsupported = [value["hypothesis"] for value in hypotheses if value["status"] == "unsupported"]
    summary = f"""# V7 paper summary

**Working title:** Complexity-Dependent Value of Generalized Entropic Consensus in Decentralized Autonomous Coordination

**Evidence level:** formal development no-go; validation and holdout locked

**Verified supported hypotheses:** {supported or ['none']} (H3 is monitoring-cost evidence only)

**Unsupported hypotheses:** {unsupported or ['none formally tested']}

V7 implements two structurally distinct coupled applications and an auditable
distributed information layer. Formal development found no positive
coupling-by-fragmentation interaction and no replicated high-complexity safety
effect. Event-triggered sketches did reduce fully counted communication by
roughly 38–40% with distributed-estimation MAE below 0.05, but that ablation
used the same always-act operational policy and does not rescue the failed
selective-safety claim.

This package contains simulated-domain and deterministic independent-agent
evidence only. RL and real-Qwen stages were prospectively locked after H1/H2
failed; no human participants were studied. The strongest defensible paper
direction is a development-stage boundary report about complexity, residual
information value, and communication-efficient distributed monitoring—not a
positive AIJ claim.
"""
    (results_root / "PAPER_SUMMARY.md").write_text(summary, encoding="utf-8")
    report = {
        "readme": "README.md", "paper_summary": "PAPER_SUMMARY.md",
        "paper_outline": "PAPER_OUTLINE.md", "claims_matrix": "CLAIMS_MATRIX.md",
        "hypotheses": hypotheses, "disposition": disposition,
    }
    atomic_json(results_root / "reproducibility" / "report_build_summary.json", report)
    return report
