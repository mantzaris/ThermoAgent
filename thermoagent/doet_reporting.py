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


def _readiness(
    hypotheses: pd.DataFrame,
    primary: pd.DataFrame,
    mechanism_demonstrated: bool = True,
) -> str:
    outcomes = dict(zip(hypotheses["hypothesis"], hypotheses["outcome"]))
    # Non-inferiority and savings cannot validate an event-trigger contribution
    # if the preregistered trigger never activates.  Keep the default True for
    # backwards-compatible unit-level use; the evidence-bound report passes the
    # observed mechanistic result explicitly.
    if not mechanism_demonstrated:
        return "insufficient for the intended AIJ submission"
    if all(outcomes.get(name) == "supported" for name in ("H1", "H2", "H3", "H5", "H6")):
        return "strong AIJ direction"
    jointly_useful_apps = int((
        primary["noninferior"].map(_truth)
        & primary["communication_target_20_percent"].map(_truth)
    ).sum())
    # A cheaper but materially inferior trigger is not, by itself, the
    # narrower publishable boundary contemplated in the protocol. Require at
    # least one jointly useful application, or a confirmed frontier/partition
    # boundary, before returning the intermediate classification.
    if (
        jointly_useful_apps >= 1
        or outcomes.get("H3") == "supported"
        or outcomes.get("H5") == "supported"
    ):
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
    execution_source = json.loads(
        (root / "reproducibility" / "execution_source.json").read_text(
            encoding="utf-8"
        )
    )
    frontier = pd.read_csv(
        root / "statistics" / "pareto_frontier_hypervolume.csv"
    )
    pareto = pd.read_csv(root / "statistics" / "pareto_points.csv")
    mechanistic = pd.read_csv(root / "processed" / "mechanistic_events.csv")
    config_path = Path(str(design["config_path"]))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    core_mechanistic = mechanistic[
        mechanistic["method"].isin(["doet_rule", "doet_rl"])
    ].copy()
    primary_mechanistic = core_mechanistic[
        core_mechanistic["method"] == "doet_rule"
    ]
    primary_triggered_episodes = int(
        primary_mechanistic["trigger_activations"].gt(0).sum()
    )
    mechanism_demonstrated = primary_triggered_episodes > 0
    readiness = _readiness(
        hypotheses, primary,
        mechanism_demonstrated=mechanism_demonstrated,
    )
    failed = int(analysis["failed_episode_count"])
    trigger = selection["selected_trigger"]["parameters"]
    supported = hypotheses[hypotheses["outcome"] == "supported"]["hypothesis"].tolist()
    unsupported = hypotheses[hypotheses["outcome"] != "supported"]["hypothesis"].tolist()
    finding_table = _finding_table(primary)
    cost_table = _cost_table(primary)
    all_frontiers = bool(frontier["doet_improves_frontier"].map(_truth).all())
    h3_rows = hypotheses.loc[
        hypotheses["hypothesis"] == "H3", "outcome"
    ]
    h3_supported = bool(len(h3_rows) == 1 and h3_rows.eq("supported").all())
    no_comm_dominates = pareto[
        pareto["method"] == "doet_rule"
    ]["dominated_by"].fillna("").str.contains("autonomous_no_comm")
    no_comm_dominates_both = bool(
        len(no_comm_dominates) == 2 and no_comm_dominates.all()
    )
    mechanism_lines: List[str] = []
    for method in ("doet_rule", "doet_rl"):
        rows = core_mechanistic[core_mechanistic["method"] == method]
        mechanism_lines.append(
            "- `%s`: %d/%d episodes activated; %d total activations; "
            "mean quiet-mode fraction %.3f; maximum observed trigger residual "
            "%.3f versus `tau_on=%.3f`."
            % (
                method,
                int(rows["trigger_activations"].gt(0).sum()),
                len(rows),
                int(rows["trigger_activations"].sum()),
                float(rows["quiet_mode_fraction"].mean()),
                float(rows["maximum_trigger_residual"].max()),
                float(trigger["tau_on"]),
            )
        )
    mechanism_summary = "\n".join(mechanism_lines)
    ablation_path = root / "ablations" / "episodes.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        cell_keys = ["application", "scenario_name"]
        selected_loss = ablation[
            (ablation["method"] == "doet_rule")
            & (ablation["method_variant"] == "selected")
        ].groupby(cell_keys)["primary_outcome"].mean().rename("selected_loss")
        ablation_cells = ablation.groupby(
            cell_keys + ["method", "method_variant"]
        ).agg(
            primary_outcome=("primary_outcome", "mean"),
            messages=("total_communication_messages", "mean"),
        ).reset_index().merge(
            selected_loss.reset_index(), on=cell_keys, validate="many_to_one"
        )
        ablation_cells["relative_loss_change"] = (
            ablation_cells["primary_outcome"]
            - ablation_cells["selected_loss"]
        ) / ablation_cells["selected_loss"].abs().clip(lower=1e-9)
        selected_messages = float(
            ablation_cells[
                (ablation_cells["method"] == "doet_rule")
                & (ablation_cells["method_variant"] == "selected")
            ]["messages"].mean()
        )

        def control_values(method: str, variant: str) -> Dict[str, float]:
            rows = ablation_cells[
                (ablation_cells["method"] == method)
                & (ablation_cells["method_variant"] == variant)
            ]
            raw = ablation[
                (ablation["method"] == method)
                & (ablation["method_variant"] == variant)
            ]
            return {
                "episodes": float(len(raw)),
                "activated": float(raw["trigger_activations"].gt(0).sum()),
                "activations": float(raw["trigger_activations"].sum()),
                "relative_loss_change": float(rows["relative_loss_change"].mean()),
                "message_change": float(
                    rows["messages"].mean() / max(selected_messages, 1e-9) - 1.0
                ),
            }

        global_oracle = control_values("global_entropy_trigger_oracle", "base")
        label_oracle = control_values("disruption_label_oracle", "base")
        kpi_control = control_values("kpi_cusum_trigger", "private_local_kpi")
        local_variants = ablation[
            ablation["method"] == "doet_rule"
        ]
        ablation_summary = (
            "- All %d local DOET-variant episodes and all %d exact-global-entropy "
            "oracle episodes had zero activations.\n"
            "- The private-KPI control activated in %d/%d episodes and changed "
            "mean loss by %s while using %s messages than selected DOET.\n"
            "- The putative disruption-label oracle activated in %d/%d episodes "
            "and changed mean loss by %s while using %s messages than selected "
            "DOET. Ledger timing shows both active controls first fired at period "
            "0, eight periods before disruption; these are false activations, not "
            "timely alarms. The binary-label oracle inherited the selected "
            "low-direction transform, so label 0 was treated as anomalously low; "
            "it is retained as an invalid exploratory oracle implementation, not "
            "an upper bound."
            % (
                len(local_variants), int(global_oracle["episodes"]),
                int(kpi_control["activated"]), int(kpi_control["episodes"]),
                _percent(kpi_control["relative_loss_change"], 3),
                _percent(abs(kpi_control["message_change"]), 1)
                + (" more" if kpi_control["message_change"] >= 0 else " fewer"),
                int(label_oracle["activated"]), int(label_oracle["episodes"]),
                _percent(label_oracle["relative_loss_change"], 3),
                _percent(abs(label_oracle["message_change"]), 1)
                + (" more" if label_oracle["message_change"] >= 0 else " fewer"),
            )
        )
    else:
        ablation_summary = "Extended signal/oracle controls were not run."

    readme = f"""# Distributed Operational Entropy Triggering (DOET)

Status: locked v2 analysis `{analysis['status']}`; {failed} failed holdout episodes. Journal-readiness classification: **{readiness}**.

This directory is a separate result namespace for the second ThermoAgent study. The frozen v1 data and interpretation remain unchanged in the parent results tree.

## Research question and method

Can a privacy-preserving distributed operational-entropy estimate tell genuinely independent logistics agents when to leave quiet local operation, start targeted bilateral coordination, or activate bounded crisis-coalition coordination—while preserving always-on fixed communication performance with less communication and inference?

DOET maintains a separate state machine inside every agent. Its private state records that agent's gossiped entropy estimate and recent change, nominal center and variance, local surprisal, consensus confidence, communication availability, and time since its own last intensive coordination event. The selected trigger statistic uses only the locally calibrated entropy residual, surprisal, consensus confidence, and explicitly delivered bounded neighbor alerts; the retained auxiliary fields support audit and candidate ablations. It never receives evaluator-global entropy or the true disruption label. Low-cost entropy sketches are explicitly counted. Recipients independently decide whether to escalate. The trigger changes communication eligibility; it does not accept contracts or make domain decisions for an agent.

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
- New holdout: {design['base_scenario_panels']} base matched panels and {design['episode_count']} method episodes on unseen `tri_region_bridge_v2`; four compute-priority methods use all panels, while secondary comparators use the same preregistered non-nominal subset. There are 16 primary-method seeds per application in each isolated, partition, correlated, and compound-OOD regime plus 8 nominal seeds per application.
- Primary benchmark: `fixed_always_on`; 2% relative non-inferiority margin.
- Primary unit: one complete multi-agent episode. Analysis uses paired panels, 10,000 hierarchical bootstrap replicates, explicit RL-seed variation, one-sided non-inferiority bounds, and Holm correction for H1/H2 across applications.

## Locked-holdout primary findings

{finding_table}

Positive reductions mean DOET used less than fixed communication. Every message total includes operational packets, alerts, and entropy-sketch gossip.

{cost_table}

## Critical mechanistic result

{mechanism_summary}

The selected entropy trigger never activated in the locked holdout. This repeats the preregistered validation warning: all four entropy candidates also had zero activations there, but the frozen selector lacked a minimum-activation eligibility gate and chose among them on non-inferiority and communication. The formal H1/H2 endpoint results therefore show that the frozen sparse quiet-mode schedule was close to fixed communication while using less communication; they do **not** show that operational entropy successfully triggered timely coordination. This is the central negative finding and prevents the intended causal event-trigger contribution.

### Exploratory signal and oracle controls

{ablation_summary}

Did DOET satisfy the complete preregistered Pareto criterion? **{'yes' if h3_supported else 'no'}**. Normalized hypervolume increased in all frozen comparator/cost cells (**{'yes' if all_frontiers else 'no'}**), but H3 also required loss-message nondominance. No communication dominated DOET-rule on loss and messages in both applications (**{'yes' if no_comm_dominates_both else 'no'}**), so a hypervolume-only positive claim is not permitted.

## Preregistered hypotheses

"""
    for _, row in hypotheses.iterrows():
        readme += (
            f"- `{row['hypothesis']}` — **{row['outcome']}**. "
            f"Frozen success criterion: {row['criterion']}\n"
        )
    readme += f"""

Supported: {', '.join(supported) if supported else 'none'}. Unsupported: {', '.join(unsupported) if unsupported else 'none'}.

Negative, mixed, and failed findings are retained in `tables/hypothesis_outcomes.csv`, `tables/failed_runs.csv`, the validation candidate table, and the mechanistic outputs. No claim of literal thermodynamics, realistic humanitarian behavior, or autonomous-agent necessity is made unless the corresponding evidence supports it.

## Compute and communication accounting

- Holdout episodes: {compute['episodes']} ({compute['failed_episodes']} failed).
- Summed episode wall time: {_number(compute['wall_clock_hours_sum'], 3)} hours.
- LLM calls: {compute['llm_calls']:,}; prompt tokens: {compute['prompt_tokens']:,}; generated tokens: {compute['generated_tokens']:,}.
- All counted messages: {compute['messages_including_sketches']:,}; structured bytes: {compute['structured_bytes']:,}.
- Approximate GPU cost at $0.34/hour: ${_number(compute['approximate_gpu_cost_usd_at_0_34_per_hour'], 2)}. One-time model load and non-GPU local diagnostics are reported separately in manifests.
- Total additional model-smoke/profile/validation/training/holdout/authorized-ablation Pod time including model loads: {_number(total_compute['additional_single_gpu_hours_including_model_load'], 3)} single-GPU hours; approximate cost ${_number(total_compute['approximate_gpu_cost_usd_at_0_34_per_hour'], 2)}. CPU-bound staged PPO time on the reserved Pod is included. This is the value audited against the 35-hour cap.

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

### Figure guide

- [`doet_architecture.pdf`](figures/pdf/doet_architecture.pdf): decentralized sketches, local trigger state, three communication modes, and retained agent authority.
- [`original_holdout_tie_diagnostics.pdf`](figures/pdf/original_holdout_tie_diagnostics.pdf): v1 action and communication divergence despite exact service-outcome ties.
- [`monitoring_baseline_comparison.pdf`](figures/pdf/monitoring_baseline_comparison.pdf): entropy detectors against thresholds, rolling statistics, change detectors, and multivariate KPI models.
- [`entropy_incremental_value.pdf`](figures/pdf/entropy_incremental_value.pdf): incremental entropy value under global and private-local observability.
- [`trigger_dynamics.pdf`](figures/pdf/trigger_dynamics.pdf): aligned entropy, trigger statistic, active agents, messages, and service loss; it explicitly displays the absent activation.
- [`performance_communication_pareto.pdf`](figures/pdf/performance_communication_pareto.pdf): common-panel loss-message frontier, including no communication and strong controls.
- [`noninferiority_forest.pdf`](figures/pdf/noninferiority_forest.pdf): paired DOET-rule degradation intervals against the frozen 2% margin by application and regime.
- [`communication_reduction.pdf`](figures/pdf/communication_reduction.pdf): fully counted reductions in messages, bytes, tokens, calls, latency, and wall time.
- [`multiple_seed_learning_curves.pdf`](figures/pdf/multiple_seed_learning_curves.pdf): all five independent seeds for each learned method.
- [`training_seed_variability.pdf`](figures/pdf/training_seed_variability.pdf): checkpoint-level outcome and communication variability in locked evaluation.
- [`holdout_primary_results.pdf`](figures/pdf/holdout_primary_results.pdf): common matched-panel primary outcomes with seed points.
- [`partition_robustness.pdf`](figures/pdf/partition_robustness.pdf): consensus error, loss degradation, and the zero-activation result under partitions.
- [`trigger_ablation_effects.pdf`](figures/pdf/trigger_ablation_effects.pdf): validation candidates plus prospectively specified exploratory signal/oracle controls.
- [`commercial_event_case_study.pdf`](figures/pdf/commercial_event_case_study.pdf) and [`humanitarian_event_case_study.pdf`](figures/pdf/humanitarian_event_case_study.pdf): disruption-aligned episode sequences with the absent trigger visibly annotated.
- [`network_snapshots_entropy_trigger.pdf`](figures/pdf/network_snapshots_entropy_trigger.pdf): deterministic physical/communication network snapshots showing that quiet mode persisted rather than implying an unobserved escalation.

### Table guide

- `experimental_design.csv`: episode counts, matched seeds, systems, regimes, and methods; the `.tex` companion is publication-ready.
- `rl_training_seed_results.csv`: evaluation variability for all independent learned checkpoints; `training/seed_manifest.csv` records selection and hashes.
- `trigger_parameters.csv`, `communication_budgets.csv`, and `achieved_budget_match.csv`: frozen trigger and communication-control settings plus realized matching error.
- `monitoring_comparison.csv`: detector performance; the richer source tables under `monitoring/` retain prevalence, timing, localization, and incremental-value analyses.
- `main_paired_comparisons.csv`, `noninferiority_analysis.csv`, and `communication_reductions.csv`: paired effects, hierarchical intervals, formal margins, and fully counted savings.
- `pareto_operating_points.csv`: common-panel loss/cost points and dominance flags.
- `holdout_results.csv`: application/scenario/method summaries; episode-level rows remain under `holdout_locked/` and `processed/`.
- `mechanistic_summary.csv` and `rl_option_selection.csv`: trigger/collapse/mode behavior and learned option distributions.
- `trigger_ablation_results.csv` and `extended_ablation_results.csv`: validation candidates and post-holdout exploratory signal/oracle controls.
- `compute_token_accounting.*` and `total_compute_accounting.*`: holdout-only and complete additional-resource accounting.
- `failed_runs.csv`: all locked failures (empty because all 696 completed); training attempts and any retained setup failures remain in their stage logs.
- `hypothesis_outcomes.csv`: frozen H1--H6 decisions; the `.tex` companion is publication-ready.

## Reproduction commands

```bash
./scripts/run-entropy-trigger-diagnostics.sh
./scripts/run-monitoring-validation-v2.sh
./scripts/run-doet-calibration.sh
./scripts/run-doet-profile.sh
./scripts/run-doet-validation.sh
./scripts/train-doet-multiseed.sh
./scripts/design-doet-holdout.sh
./scripts/freeze-doet-holdout.sh
./scripts/run-doet-holdout.sh
./scripts/run-doet-ablations.sh  # exploratory, after the locked holdout
./scripts/rebuild-doet-results.sh
```

The locked episodes were executed from commit `{execution_source['commit']}` with source checksum `{execution_source['source_checksum']}`; the authoritative values are also stored in `reproducibility/execution_source.json` and every run manifest. For filtered RunPod deployment, use `./scripts/runpod-sync.sh`, then `./scripts/runpod-sync-v2-controls.sh bootstrap`. Fetch only this study with `./scripts/runpod-fetch-v2-results.sh`; the command never overwrites the frozen v1 namespace. Exact sequencing and restart instructions are in `notes/14_entropy_trigger_protocol.md` and `notes/15_entropy_trigger_implementation.md`.

## Limitations and readiness

These are abstract logistics simulators, one 7B model family, deterministic decoding, a small discrete coordination policy, synthetic disruption processes, and a single 4090 execution environment. Full/global KPI detectors can dominate entropy when centralized observability is available. Balanced learned-checkpoint evaluation exposes five training seeds but does not cross every checkpoint with every panel. Communication cost uses measured messages/bytes/tokens/calls/latency and a transparent hourly-rate estimate, not a deployment-specific network tariff.

Current classification: **{readiness}**. The completed platform and boundary result are suitable as an engineering demonstration, but the intended entropy-triggered positive contribution is not presently sufficient for an AIJ submission. See `PAPER_SUMMARY.md` and `notes/19_entropy_trigger_paper_claims.md` for the exact allowed claims and remaining work.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    abstract_result = (
        "The locked evaluation supports the event-triggered contribution."
        if readiness == "strong AIJ direction"
        else (
            "The locked evaluation establishes a negative mechanistic boundary: "
            "the selected trigger never activated, so observed savings cannot be "
            "attributed to entropy-triggered coordination."
        )
    )
    paper = f"""# Paper-oriented summary

## Working title

Distributed Operational Entropy Triggering for Communication-Efficient Autonomous Logistics Coordination

## Provisional abstract

Autonomous logistics agents can benefit from rich communication during disruptions, but always-on coordination imposes message, inference, and negotiation costs. We introduce Distributed Operational Entropy Triggering (DOET), a decentralized stateful trigger computed from privacy-preserving, locally gossiped operational macrostate sketches. DOET regulates quiet, targeted, and crisis communication modes while leaving planning and commitment authority inside independent agents. We evaluate always-on fixed, learned non-entropic, DOET-rule, and DOET-RL on every panel; periodic, budget-matched, original ThermoAgent, no-communication, and local-KPI controls use a prospectively fixed common subset under the compute cap. The study spans commercial and abstract humanitarian logistics. The preregistered holdout contains {design['episode_count']} episodes and five independently trained seeds per learned method. {abstract_result} Exact effect estimates and uncertainty are reported in the tables below rather than summarized with an unsupported positive claim.

## Verified contributions

1. An implemented decentralized, privacy-preserving, fully counted event-trigger architecture with independent agent authority; the locked run did not demonstrate successful trigger activation.
2. A replay-backed mechanistic diagnosis of why the original frozen holdout tied despite policy divergence.
3. Monitoring evidence separating globally redundant entropy from its incremental value under private local information.
4. A multiple-training-seed, paired, frozen-holdout comparison against always-on communication and budget-matched controls.
5. Exact replay, conservation, communication/inference accounting, and mechanistic trigger analyses.

## Primary numerical results

{finding_table}

{cost_table}

## Mechanistic interpretation

{mechanism_summary}

H1, H2, and the formal cross-application endpoint H6 are supported as preregistered statistical statements. They do not validate the proposed entropy mechanism: both DOET variants remained in quiet mode in every locked episode, H4 and H5 failed, H3 failed, and the common-panel no-communication control dominated DOET-rule on loss and messages in both applications.

### Exploratory control result

{ablation_summary}

## Hypothesis outcomes

"""
    for _, row in hypotheses.iterrows():
        paper += (
            f"- `{row['hypothesis']}`: **{row['outcome']}**. "
            f"Frozen success criterion: {row['criterion']}\n"
        )
    paper += f"""

## Figure plan

The paper-facing set comprises the DOET architecture; original tie diagnosis; monitoring baselines and incremental value; trigger dynamics; loss–communication Pareto frontier; non-inferiority forest; communication reduction; multiple-seed curves and variability; locked primary results; partition robustness; trigger ablations; commercial/humanitarian event studies; and an entropy-triggered network sequence. All are vector PDFs with rendered QA previews.

## Table plan

Experimental design, RL seeds, trigger parameters, communication budgets, monitoring controls, paired comparisons, non-inferiority, reductions, Pareto points/hypervolume, holdout summaries, trigger ablations, compute/tokens, failed runs, and hypothesis outcomes are under `tables/`.

## Limitations and recommendation

The strongest limitations are the absent trigger activation, synthetic environments, one primary language model, abstract humanitarian roles, deterministic decoding, and restricted topology/model diversity. The result does not establish literal thermodynamic behavior, useful entropy-triggered coordination, or general autonomous-agent necessity. Recommendation: **{readiness}**. Any manuscript must retain the original negative study, global-KPI redundancy, all failed/unstable seeds, the zero-activation mechanism, no-communication dominance, and application/regime-specific exceptions.
"""
    (root / "PAPER_SUMMARY.md").write_text(paper, encoding="utf-8")
    return {
        "status": "written",
        "readiness": readiness,
        "readme": str(root / "README.md"),
        "paper_summary": str(root / "PAPER_SUMMARY.md"),
    }
