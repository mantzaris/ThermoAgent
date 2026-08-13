# Entropy-trigger diagnostics

## Scope and evidence boundary

This note records retrospective analyses of the frozen v1 study at commit
`d555ac04927968ad577707b5c7e9e7b1162069e6`. No original event ledger, episode,
manifest, statistic, or figure is modified. Derived artifacts live only under
`results/entropy_triggered_v2/` and record checksums of their v1 inputs.

The original v1 holdout is already seen. It is used solely to diagnose why the
matched learned policies had exact primary-outcome ties and to establish prior
monitoring evidence. It is ineligible for v2 calibration, threshold selection,
checkpoint selection, or confirmatory testing.

## Questions fixed before analysis

The diagnostic reports exact, not rounded, primary outcomes; decision and event
divergence; entropy-feature ranges; checkpoint sensitivity; action-mask effects;
communication and inference costs; operational-tool consequences; deterministic
exogenous-state equality; and the number of v1 RL training seeds. Definitions
and denominators are emitted with the result tables rather than selected after
viewing a favorable metric.

## Current status

Complete. No new simulator episode or LLM call was required.

The derived diagnostic, monitoring, and monitor-only calibration artifacts were
generated locally under Python 3.8.10 with NumPy 1.22.1, SciPy 1.7.3, pandas
1.3.4, scikit-learn 1.0, Matplotlib 3.1.2, and CPU PyTorch 1.10.1 where used.
These exact analysis versions are now recorded in their manifests. The v2
RunPod execution environment is separate and pinned in
`requirements-runpod.txt`; no v1 raw artifact was regenerated.

## Exact holdout-tie diagnosis

- All 16 matched ThermoAgent/no-entropy primary outcomes are equal at the raw
  IEEE-754 value, not merely after rounding. All service and exogenous-demand
  trajectories are also exact matches.
- The actors were not behaviorally identical. Commercial options differed on
  331/569 (58.2%) common and 637/875 (72.8%) union decision epochs.
  Humanitarian options differed on 257/556 (46.2%) common and 574/873 (65.8%)
  union epochs. More than 92% of simulator periods had a different option
  multiset in each application.
- ThermoAgent used 7,416 versus 917 total commercial messages and 6,904 versus
  883 humanitarian messages. The excess includes all 11,988 mandatory entropy-
  sketch transmissions. It also used 1,459 versus 1,414 LLM calls overall.
- Across both learned methods, 57 material tool calls were attempted. Fifty-four
  failed deterministic route/capacity/inventory checks. ThermoAgent's three
  successful shipments went only to intermediate nodes; no successful learned-
  method material call reached a demand node. Thus divergent coordination did
  not change service.
- Entropy fields varied and remained inside their explicit design bounds.
  Fixed-mask zeroing changed only 2.69% of commercial and 3.49% of humanitarian
  ThermoAgent choices: the checkpoint was measurably but weakly behaviorally
  sensitive to the monitor block. Exact training-feature ranges were not
  retained by v1; v1-main support comparisons are reported instead.
- No action mask had only one enabled option. Masks often altered a raw argmax,
  but they did not force universal identity. The downstream lack of valid,
  demand-reaching operational action—not masking—caused the primary ties.
- Both learned v1 checkpoints used the single RL training seed `3001`.

The complete derived evidence, denominators, semantic event definitions, and
input SHA-256 values are in `results/entropy_triggered_v2/diagnostics/`.

## Monitoring validation against ordinary KPIs

The comparison uses one frozen scripted trajectory per scenario, not repeated
method copies. Original-main results are leave-one-environment-seed-out; the
seen original holdout remains diagnostic only.

- A full evaluator-KPI logistic detector (current service loss, backlog/unmet-
  need pressure, maximum impairment, message volume, rolling moments, and EWMA)
  achieved AP/AUC 1.000/1.000 in both applications. Adding distributed entropy
  changed AP and AUC by exactly zero. Entropy is therefore not independently
  predictive once these global KPIs are available.
- High-direction distributed entropy retained useful ranking in connected and
  correlated cases, but direction and threshold transfer failed under compound
  partitions. The original-holdout ranking was nearly perfect, while the main-
  derived 95th-percentile threshold had zero recall. Absolute-deviation
  calibration false-alarmed on every original-holdout negative timepoint.
- Under the execution-relevant restricted-information test, each row contains
  only one agent's private KPIs. Adding that agent's final-round gossip entropy
  raised original-main out-of-seed disruption AP by 0.0974 commercial and 0.0980
  humanitarian, and AUC by 0.1714 and 0.1696. On the seen holdout both local-KPI
  models already ranked perfectly, but entropy improved Brier score by 0.0148
  and 0.0197.
- Restricted entropy modestly improved future-loss prediction on original main
  but worsened holdout RMSE/R-squared. This predictive result is mixed.
- At disruption onset, ordinary impairment localized the true source perfectly.
  Local surprisal top-1 was perfect commercially but only 0.125 humanitarian in
  original main; top-3 was perfect. Localization value is therefore role- and
  application-dependent.

The scientifically supportable rationale is narrow: entropy is a compressed,
distributed system statistic that adds information to an isolated agent's local
view. It is not superior to an evaluator with all ordinary KPIs. DOET must be
compared directly with a local-KPI trigger and count all sketch traffic.
