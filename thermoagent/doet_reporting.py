"""Evidence-bound README and paper-summary generation for DOET v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def _percent(value: Any, digits: int = 1) -> str:
    return ("%%.%df%%%%" % digits) % (100.0 * float(value))


def _number(value: Any, digits: int = 3) -> str:
    return ("%%.%df" % digits) % float(value)


def _truth(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no", "", "nan"):
            return False
    return bool(value)


def _primary_rows(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "statistics" / "main_paired_comparisons.csv")
    return frame[
        (frame["method"] == "doet_rule")
        & (frame["scenario"] == "all_non_nominal")
    ].sort_values("application")


def _readiness(hypotheses: pd.DataFrame, primary: pd.DataFrame) -> str:
    outcomes = dict(zip(hypotheses["hypothesis"], hypotheses["outcome"]))
    if all(outcomes.get(name) == "supported" for name in ("H1", "H2", "H3", "H5", "H6")):
        return "strong AIJ direction"
    supported_apps = int(primary["noninferior"].map(_truth).sum())
    savings_apps = int(primary["communication_target_20_percent"].map(_truth).sum())
    if supported_apps or savings_apps:
        return "narrower but potentially publishable direction"
    return "insufficient for an AIJ submission"


def _finding_table(primary: pd.DataFrame) -> str:
    lines = [
        "| Application | Loss degradation vs fixed | One-sided 95% upper | Non-inferior | Message reduction (95% CI) |",
        "|---|---:|---:|:---:|---:|",
    ]
    for _, row in primary.iterrows():
        lines.append(
            "| %s | %s | %s | %s | %s [%s, %s] |"
            % (
                str(row["application"]).capitalize(),
                _percent(row["mean_relative_degradation"], 2),
                _percent(row["noninferiority_upper_one_sided_95"], 2),
                "yes" if _truth(row["noninferior"]) else "no",
                _percent(row["mean_total_communication_messages_reduction"], 1),
                _percent(row["total_communication_messages_reduction_ci95_low"], 1),
                _percent(row["total_communication_messages_reduction_ci95_high"], 1),
            )
        )
    return "\n".join(lines)


def _cost_table(primary: pd.DataFrame) -> str:
    metrics = [
        ("total_communication_bytes", "Structured bytes"),
        ("prompt_tokens", "Prompt tokens"),
        ("generated_tokens", "Generated tokens"),
        ("llm_calls", "LLM calls"),
        ("llm_latency_seconds", "Inference latency"),
        ("wall_clock_seconds", "Wall-clock time"),
    ]
    lines = ["| Application | " + " | ".join(label for _, label in metrics) + " |"]
    lines.append("|---|" + "---:|" * len(metrics))
    for _, row in primary.iterrows():
        values = [
            _percent(row["mean_" + metric + "_reduction"], 1)
            for metric, _ in metrics
        ]
        lines.append(
            "| %s | %s |"
            % (str(row["application"]).capitalize(), " | ".join(values))
        )
    return "\n".join(lines)


def run(root: Path) -> Dict[str, Any]:
    analysis_path = root / "statistics" / "analysis_manifest.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            "DOET reporting requires completed locked analysis: %s" % analysis_path
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    primary = _primary_rows(root)
    hypotheses = pd.read_csv(root / "tables" / "hypothesis_outcomes.csv")
    compute = json.loads(
        (root / "tables" / "compute_token_accounting.json").read_text(
            encoding="utf-8"
        )
    )
    total_compute = json.loads(
        (root / "tables" / "total_compute_accounting.json").read_text(
            encoding="utf-8"
        )
    )
    training = json.loads(
        (root / "training" / "training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    selection = json.loads(
        (root / "validation" / "trigger_selection.json").read_text(
            encoding="utf-8"
        )
    )
    design = json.loads(
        (root / "protocol" / "holdout_design_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frontier = pd.read_csv(
        root / "statistics" / "pareto_frontier_hypervolume.csv"
    )
    config_path = Path(str(design["config_path"]))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = _readiness(hypotheses, primary)
    failed = int(analysis["failed_episode_count"])
    trigger = selection["selected_trigger"]["parameters"]
    supported = hypotheses[hypotheses["outcome"] == "supported"]["hypothesis"].tolist()
    unsupported = hypotheses[hypotheses["outcome"] != "supported"]["hypothesis"].tolist()
    finding_table = _finding_table(primary)
    cost_table = _cost_table(primary)
    all_frontiers = bool(frontier["doet_improves_frontier"].map(_truth).all())

    readme = f"""# Distributed Operational Entropy Triggering (DOET)

Status: locked v2 analysis `{analysis['status']}`; {failed} failed holdout episodes. Journal-readiness classification: **{readiness}**.

This directory is a separate result namespace for the second ThermoAgent study. The frozen v1 data and interpretation remain unchanged in the parent results tree.

## Research question and method

Can a privacy-preserving distributed operational-entropy estimate tell genuinely independent logistics agents when to leave quiet local operation, start targeted bilateral coordination, or activate bounded crisis-coalition coordination—while preserving always-on fixed communication performance with less communication and inference?

DOET maintains a separate state machine inside every agent. It consumes only that agent's gossiped entropy estimate, entropy change, nominal calibration, local surprisal, consensus disagreement, communication availability, and time since coordination. It never receives evaluator-global entropy or the true disruption label. Low-cost entropy sketches are explicitly counted. Structured neighbor alerts are bounded and recipients independently decide whether to escalate. The trigger changes communication eligibility; it does not accept contracts or make domain decisions for an agent.

Each organization retains its own identity, private observations/memory/utility, planning context, inbox/outbox, trust state, commitment ledger, policy recurrent state, and accept/reject/counter/withdraw authority. The quantitative simulator validates typed tools and conservation but cannot invent a domain action.

## Frozen original-study evidence

V1 remains a mixed/negative result: 101 tests passed; 1,096 post-freeze ledgers replayed exactly; operational entropy monitored disruption well (AP 0.934, ROC AUC 0.863), the free-energy gap did not; small in-distribution ThermoAgent improvements did not survive correction; the 80-episode holdout tied the matched no-entropy policy exactly; and fixed communication won every necessity-map cell.

The v2 tie diagnosis found that the raw IEEE-754 primary outcomes—not rounded displays—were identical in all 16 ThermoAgent/no-entropy pairs even though option choices diverged on 58.2% of common commercial epochs and 46.2% of humanitarian epochs. Fifty-four of 57 learned-policy material actions failed; the three successful ThermoAgent shipments reached only intermediate nodes and none reached demand. The policies therefore communicated differently without changing demand-reaching material flow. Entropy features varied, were not masked, and changed only 2.69%/3.49% of actor actions when zeroed. V1 had one RL seed (3001).

Monitoring controls show that full/global ordinary KPIs already classify disruption perfectly in these synthetic trajectories, so entropy has no incremental global predictive value. Under the execution-realistic restriction to one agent's private local KPIs, adding distributed entropy improved development AP by about 0.097 in both applications and AUC by about 0.17. The seen v1 holdout retained calibration gains but not a robust ranking gain. The defensible contribution is therefore a compressed distributed trigger under restricted information—not a claim that entropy universally beats centralized KPIs.

## Hardware, model, and design

- GPU: existing RunPod RTX 4090, 24 GB; CUDA 12.8; no additional Pod.
- Planner: `{config['model']['identifier']}` at immutable revision `{config['model']['revision']}`.
- Precision: {config['model']['precision']}.
- Prompt revision: `{config['prompt_template_revision']}`; deterministic decoding; max {config['model']['max_input_tokens']} input and {config['model']['max_new_tokens']} generated tokens.
- Selected trigger: `{selection['selected_method_variant']}` (`{trigger['trigger_type']}`, `{trigger['direction']}` residual, `rho={trigger['rho']}`, `kappa={trigger['kappa']}`, `tau_on={trigger['tau_on']}`, `tau_off={trigger['tau_off']}`, crisis threshold `{trigger['tau_crisis']}`, dwell `{trigger['minimum_dwell']}`, cooldown `{trigger['cooldown']}`, propagation `{trigger['propagation']}`).
- Learned replications: {training['seeds_per_variant']} independent RL seeds per learned method; {training['completed_trainings']}/{training['planned_trainings']} fixed-budget trainings completed; no outcome-selected checkpoints.
- New holdout: {design['base_scenario_panels']} base matched panels and {design['episode_count']} method episodes on unseen `tri_region_bridge_v2`; 16 seeds per application in each isolated, partition, correlated, and compound-OOD regime plus 8 nominal seeds per application.
- Primary benchmark: `fixed_always_on`; 2% relative non-inferiority margin.
- Primary unit: one complete multi-agent episode. Analysis uses paired panels, 10,000 hierarchical bootstrap replicates, explicit RL-seed variation, one-sided non-inferiority bounds, and Holm correction for H1/H2 across applications.

## Locked-holdout primary findings

{finding_table}

Positive reductions mean DOET used less than fixed communication. Every message total includes operational packets, alerts, and entropy-sketch gossip.

{cost_table}

Did DOET strictly improve every frozen normalized loss-cost frontier comparison? **{'yes' if all_frontiers else 'no'}**. The hypervolume analysis compares DOET against periodic, budget-matched random, learned non-entropic, and local-KPI-CUSUM communication for messages, prompt tokens, LLM calls, and latency in each application.

## Preregistered hypotheses

"""
    for _, row in hypotheses.iterrows():
        readme += f"- `{row['hypothesis']}` — **{row['outcome']}**: {row['criterion']}\n"
    readme += f"""

Supported: {', '.join(supported) if supported else 'none'}. Unsupported: {', '.join(unsupported) if unsupported else 'none'}.

Negative, mixed, and failed findings are retained in `tables/hypothesis_outcomes.csv`, `tables/failed_runs.csv`, the validation candidate table, and the mechanistic outputs. No claim of literal thermodynamics, realistic humanitarian behavior, or autonomous-agent necessity is made unless the corresponding evidence supports it.

## Compute and communication accounting

- Holdout episodes: {compute['episodes']} ({compute['failed_episodes']} failed).
- Summed episode wall time: {_number(compute['wall_clock_hours_sum'], 3)} hours.
- LLM calls: {compute['llm_calls']:,}; prompt tokens: {compute['prompt_tokens']:,}; generated tokens: {compute['generated_tokens']:,}.
- All counted messages: {compute['messages_including_sketches']:,}; structured bytes: {compute['structured_bytes']:,}.
- Approximate GPU cost at $0.34/hour: ${_number(compute['approximate_gpu_cost_usd_at_0_34_per_hour'], 2)}. One-time model load and non-GPU local diagnostics are reported separately in manifests.
- Total additional validation/training/holdout/authorized-ablation Pod time including model loads: {_number(total_compute['additional_single_gpu_hours_including_model_load'], 3)} single-GPU hours; approximate cost ${_number(total_compute['approximate_gpu_cost_usd_at_0_34_per_hour'], 2)}. CPU-bound staged PPO time on the reserved Pod is included. This is the value audited against the 35-hour cap.

## Artifacts

- `protocol/`: selected trigger, power/precision analysis, matched design, and immutable holdout freeze.
- `diagnostics/`: exact v1 tie mechanism and action/communication divergence.
- `monitoring/`: entropy-versus-KPI detectors, incremental value, lead time, and localization.
- `training/` and `checkpoints/`: all RL seeds, curves, fixed-budget selection, and small policy checkpoints.
- `validation/`: all trigger candidates, selected operating point, and budget-matched control rates.
- `holdout_locked/` and `raw/holdout_locked/`: episode summaries and event-sourced ledgers.
- `processed/`, `statistics/`, and `tables/`: paired analysis, bootstrap output, Pareto frontiers, mechanisms, failures, and CSV/LaTeX tables.
- `figures/pdf/` and `figures/previews/`: vector paper figures and rendered previews; `reproducibility/pdf_qa/` contains mechanical and visual QA.
- `logs/` and `manifests/`: restartable run status, exact model/config/seeds/tokens/checksums, and failure records.

## Reproduction commands

```bash
./scripts/run-doet-calibration.sh
./scripts/run-doet-validation.sh
./scripts/train-doet-multiseed.sh
./scripts/design-doet-holdout.sh
./scripts/freeze-doet-holdout.sh
./scripts/run-doet-holdout.sh
./scripts/rebuild-doet-results.sh
```

For filtered RunPod deployment, use `./scripts/runpod-sync.sh`, then `./scripts/runpod-sync-v2-controls.sh bootstrap`. Fetch only this study with `./scripts/runpod-fetch-v2-results.sh`; the command never overwrites the frozen v1 namespace. Exact sequencing and restart instructions are in `notes/14_entropy_trigger_protocol.md` and `notes/15_entropy_trigger_implementation.md`.

## Limitations and readiness

These are abstract logistics simulators, one 7B model family, deterministic decoding, a small discrete coordination policy, synthetic disruption processes, and a single 4090 execution environment. Full/global KPI detectors can dominate entropy when centralized observability is available. Balanced learned-checkpoint evaluation exposes five training seeds but does not cross every checkpoint with every panel. Communication cost uses measured messages/bytes/tokens/calls/latency and a transparent hourly-rate estimate, not a deployment-specific network tariff.

Current classification: **{readiness}**. See `PAPER_SUMMARY.md` and `notes/19_entropy_trigger_paper_claims.md` for the exact allowed claims and remaining work.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    abstract_result = (
        "The locked evaluation supports the event-triggered contribution."
        if readiness == "strong AIJ direction"
        else "The locked evaluation establishes a bounded or negative result rather than the full proposed contribution."
    )
    paper = f"""# Paper-oriented summary

## Working title

Distributed Operational Entropy Triggering for Communication-Efficient Autonomous Logistics Coordination

## Provisional abstract

Autonomous logistics agents can benefit from rich communication during disruptions, but always-on coordination imposes message, inference, and negotiation costs. We introduce Distributed Operational Entropy Triggering (DOET), a decentralized stateful trigger computed from privacy-preserving, locally gossiped operational macrostate sketches. DOET regulates quiet, targeted, and crisis communication modes while leaving planning and commitment authority inside independent agents. We evaluate against always-on fixed communication, periodic and budget-matched controls, learned non-entropic coordination, the original ThermoAgent policy, and a local-KPI trigger across commercial and abstract humanitarian logistics. The preregistered holdout contains {design['episode_count']} episodes and five independently trained seeds per learned method. {abstract_result} Exact effect estimates and uncertainty are reported in the tables below rather than summarized with an unsupported positive claim.

## Verified contributions

1. A decentralized, privacy-preserving, fully counted event-trigger architecture with independent agent authority.
2. A causal diagnosis of why the original frozen holdout tied despite policy divergence.
3. Monitoring evidence separating globally redundant entropy from its incremental value under private local information.
4. A multiple-training-seed, paired, frozen-holdout comparison against always-on communication and budget-matched controls.
5. Exact replay, conservation, communication/inference accounting, and mechanistic trigger analyses.

## Primary numerical results

{finding_table}

{cost_table}

## Hypothesis outcomes

"""
    for _, row in hypotheses.iterrows():
        paper += f"- `{row['hypothesis']}`: **{row['outcome']}** — {row['criterion']}\n"
    paper += f"""

## Figure plan

The paper-facing set comprises the DOET architecture; original tie diagnosis; monitoring baselines and incremental value; trigger dynamics; loss–communication Pareto frontier; non-inferiority forest; communication reduction; multiple-seed curves and variability; locked primary results; partition robustness; trigger ablations; commercial/humanitarian event studies; and an entropy-triggered network sequence. All are vector PDFs with rendered QA previews.

## Table plan

Experimental design, RL seeds, trigger parameters, communication budgets, monitoring controls, paired comparisons, non-inferiority, reductions, Pareto points/hypervolume, holdout summaries, trigger ablations, compute/tokens, failed runs, and hypothesis outcomes are under `tables/`.

## Limitations and recommendation

The strongest limitations are synthetic environments, one primary language model, abstract humanitarian roles, deterministic decoding, and restricted topology/model diversity. The result does not establish literal thermodynamic behavior or general autonomous-agent necessity. Recommendation: **{readiness}**. Any manuscript must retain the original negative study, global-KPI redundancy, all failed/unstable seeds, and application/regime-specific exceptions.
"""
    (root / "PAPER_SUMMARY.md").write_text(paper, encoding="utf-8")
    return {
        "status": "written",
        "readiness": readiness,
        "readme": str(root / "README.md"),
        "paper_summary": str(root / "PAPER_SUMMARY.md"),
    }
