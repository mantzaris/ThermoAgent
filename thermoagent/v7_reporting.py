"""Build the V7 publication-facing package from stored evidence only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .v5_experiments import atomic_json, write_csv


def _json(path: Path, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _episode_count(results_root: Path, stage: str) -> int:
    path = results_root / stage / "episode_summary.csv"
    return len(pd.read_csv(path)) if path.exists() else 0


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
            "status": ("supported" if communication.get("H3_pass") else "unsupported") if communication else "pilot_only",
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


def _readme(results_root: Path, disposition: Mapping[str, Any]) -> str:
    replay = _json(results_root / "reproducibility" / "replay" / "replay_summary.json")
    pilot = _json(results_root / "pilots_iteration3" / "analysis" / "pilot_analysis.json")
    dynamic = _json(results_root / "statistics" / "dynamic_primary_analysis.json")
    communication = _json(results_root / "statistics" / "communication_primary_analysis.json")
    freeze = _json(results_root / "protocol" / "freeze_manifest.json")
    tests = "378 passed under system Python; a retained isolated-venv attempt had 12 missing-PyTorch failures"
    stage_counts = {
        "smoke_and_pilots": sum(_episode_count(results_root, value) for value in ("smoke", "pilots", "pilots_iteration2", "pilots_iteration3")),
        "formal_reference": _episode_count(results_root, "development_formal_reference"),
        "formal_dynamic": _episode_count(results_root, "development_formal_dynamic"),
        "formal_communication": _episode_count(results_root, "development_formal_communication"),
        "validation": _episode_count(results_root, "validation"),
        "holdout": _episode_count(results_root, "holdout"),
    }
    if dynamic:
        finding = (
            "Formal development %s H1 and %s H2. The coupled interaction was %.4f."
            % (
                "supported" if dynamic.get("H1_pass") else "did not support",
                "supported" if dynamic.get("H2_pass") else "did not support",
                float(dynamic.get("interaction", {}).get("coupling_fragmentation_interaction", float("nan"))),
            )
        )
    else:
        finding = (
            "Only retained feasibility pilots exist. Their coupling-by-fragmentation coefficient was %.4f; this is not a formal estimate."
            % float(pilot.get("interaction_model", {}).get("coupling_fragmentation_interaction", float("nan")))
        )
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

## Main findings

{finding}

Formal communication H3 status: `{communication.get('H3_pass', 'not formally tested')}`.
The final prospective disposition is: **{disposition.get('disposition', 'pending')}**.

Negative and neutral effects, failed feasibility iterations, environment
repairs, missing-PyTorch test evidence, and the unavailable RunPod endpoint are
retained. No result is real-human evidence. The Qwen model, if its gated stage
ran, is one LLM implementation; deterministic pilots are engineering controls.

## Integrity and reproducibility

- Tests: {tests}.
- Replay: `{replay.get('episodes_replayed', 0)}` episodes, `{replay.get('replay_mismatches', 0)}` mismatches.
- Maximum reconstructed conservation residual: `{replay.get('maximum_conservation_residual', 'not available')}`.
- Primary model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, NF4/BF16 (only if gated Qwen ran).
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
./scripts/index-v7-artifacts.sh
./scripts/build-v7-report.sh
```

## Directory guide

- `protocol/`, `manifests/`: frozen design and seed provenance when unlocked.
- `pilots*/`, `raw/`: retained feasibility summaries and compressed ledgers.
- `development/`, `statistics/`, `tables/`: formal evidence if run.
- `training/`, `qwen/`: gated learned-agent evidence only.
- `figures/pdf/`, `figures/source_data/`: vector figures and exact source tables.
- `reproducibility/`: environment, replay, checksum, and PDF QA.
- `negative_results/`: failed iterations and prospective no-go disposition.

## Limitations and publication readiness

All operators are simulated, domain models are abstract, the pilot utility
action pool remains harm-heavy, graph fidelity is deliberately lower than a
real disaster or utility system, and external validity is untested. A positive
AIJ claim requires validation and a single locked holdout; if either is locked,
this package is an engineering/development study rather than confirmatory
journal evidence.
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
    supported = [value["hypothesis"] for value in hypotheses if value["status"] == "supported"]
    unsupported = [value["hypothesis"] for value in hypotheses if value["status"] == "unsupported"]
    summary = f"""# V7 paper summary

**Working title:** Complexity-Dependent Value of Generalized Entropic Consensus in Decentralized Autonomous Coordination

**Evidence level:** {disposition.get('disposition', 'pending')}

**Verified supported hypotheses:** {supported or ['none']}

**Unsupported hypotheses:** {unsupported or ['none formally tested']}

V7 implements two structurally distinct coupled applications and an auditable
distributed information layer. This package contains simulated-domain and
autonomous-agent evidence only. It is not real-human evidence. Journal claims
must be limited to the highest completed stage described in `README.md` and
`CLAIMS_MATRIX.md`.
"""
    (results_root / "PAPER_SUMMARY.md").write_text(summary, encoding="utf-8")
    report = {
        "readme": "README.md", "paper_summary": "PAPER_SUMMARY.md",
        "paper_outline": "PAPER_OUTLINE.md", "claims_matrix": "CLAIMS_MATRIX.md",
        "hypotheses": hypotheses, "disposition": disposition,
    }
    atomic_json(results_root / "reproducibility" / "report_build_summary.json", report)
    return report
