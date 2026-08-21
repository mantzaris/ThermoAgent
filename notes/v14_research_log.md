# V14 research log

## 2026-08-20 — provenance and design

- Fetched `origin` and verified `origin/collective-agent-statmech-v13` at `20a9ca66041b1636bed15d5916aabcb605e6a063`.
- Verified the V13 worktree was clean and reconciled before branching.
- Created local branch `memory-quench-agent-statmech-v14`; no files were staged, committed, or pushed.
- Audited the V12/V13 source, aggregate results, protocol, tests, scripts, manuscript, and external artifact availability.
- Confirmed the existing authorized RTX 4090 Pod is online and idle, with the pinned environment and sufficient disk space. Process inspection used names only; no authenticated URL, token, or secret was printed.
- Chose a focused prospective design: six new matched quench clusters × four conditions × 16 agents × 45 sweeps = 17,280 formal decisions. This is projected below the 10-hour generation ceiling.
- Preserved the V12 discovery and V13 replication roles for memory. No new V14 memory trajectories are planned.
- Predeclared H2–H4 as the Holm-corrected V14 confirmatory family and exact cluster sign-flip inference as primary.
- Added source, configuration, external-artifact controls, estimators, deterministic replay, figure generation, reporting, and focused tests under V14-only namespaces.
- Focused V14 CPU tests initially passed (22 passed, two generation-dependent QA tests skipped).

## Outcome-inspection boundary

No V14 LLM quench outcome has been generated or inspected at the time of the design entries above. The engineering pilot is restricted to validity, occupancy, transition counts, scheduling/restoration, isolation, delivery, runtime, and token projection.

## 2026-08-20 — retained engineering pilot attempt 1

- The first Qwen pilot process loaded the pinned model but stopped before its first decision. PyTorch deterministic mode rejected a cuBLAS matrix operation because `CUBLAS_WORKSPACE_CONFIG` was not defined at process startup.
- No pilot transition and no scientific outcome was generated. The complete traceback is retained externally in `logs/pilot_attempt_1.log`.
- Before rerunning the unchanged pilot, the V14 pilot and formal launchers were amended to export the CUDA-documented deterministic workspace setting `:4096:8`. No prompt, seed, perturbation, observable, estimator, or scientific threshold changed.

## 2026-08-20 — retained engineering pilot attempt 2

- The corrected process completed all 256 pilot decisions, but summary construction then stopped on a mismatched configuration-key name (`maximum_projected_prompt_tokens` in code versus the declared `maximum_projected_formal_prompt_tokens`). The raw calls and traceback are retained externally; no scientific quench contrast was inspected.
- The key reference was corrected, and transition-table persistence was moved before summary calculation so any future summary fault cannot discard the compact transition record. To respect the 300-decision pilot ceiling, the 256 completed decisions are not regenerated. A deterministic recovery path instead validates their content hashes, prompts, inference seeds, and sampling temperatures while replaying them through the unchanged decentralized transition code; only then may it construct the summary.

## 2026-08-20 — engineering disposition and protocol freeze

- Deterministic recovery reproduced all 256 pilot transitions with exact prompt, inference-seed, and sampling-temperature checks. First-pass and final validity were both 1.0; latent `+1` occupancy was 0.50390625; both belief-transition directions occurred 31 times; privacy, message delivery, quench, partition, corruption, and restoration checks passed.
- Mean measured generation latency was 1.60553 seconds per decision, projecting 7.70657 formal generation hours. Mean prompt length was 541.797 tokens, projecting 9,362,250 formal prompt tokens. Both projections were below their frozen ceilings.
- Protocol `v14-memory-quench-agent-statmech-1.0` was frozen before any V14 formal outcome at SHA-256 `5d8440dedbf389c02f3b448f38abfd1a370b8f9c4fafdba4760afc706e0bcfdf`.
- The formal execution-source checksum is `2b4276dd323bb048d8c98834ed0e9f8bfe5a0ed46e8735db3b471a6dc97e91ad`. The local and RunPod frozen protocol files and source checksums matched exactly.
- The complete 24-trajectory, 17,280-decision formal experiment began only after the freeze. The first 720-row nominal trajectory completed atomically without inspection of its scientific outcomes.

## 2026-08-20 — outcome-blind post-freeze implementation audit

- A source-only audit performed while the formal run was still blinded identified a limitation in frozen H3: the maximum recovery-period distance minus the final-five-sweep mean is structurally nonnegative except in a constant trajectory. Its magnitude remains a descriptive early-to-late contrast, but an exact sign-flip result for this estimand cannot by itself establish relaxation. The protocol and implementation were not changed after freeze; the final claims will instead interpret the full recovery trajectory, final residual, and threshold-crossing behavior.
- The frozen implementation saves macrostate trajectories for rolling windows 3, 5, and 7, but its nominal-distance robustness table recomputes covariance, ridge, nominal-window, and leave-observable-family variants only at the primary five-sweep window. It also implements family deletion rather than single-observable deletion. These omissions do not alter the frozen primary H2 or H4 estimands, but the unexecuted sensitivities will be reported explicitly and will remain necessary follow-up rather than being reconstructed post hoc.
- No V14 scientific outcome was opened for this audit, and no frozen source, estimator, hypothesis, seed, prompt, or protocol field was changed.

## 2026-08-20 — formal execution and deterministic replay

- Completed the frozen design without selective reruns: six independent clusters, four matched conditions, 24 trajectories, 45 sweeps per trajectory, and 17,280 decisions.
- All 17,280 structured responses were valid on the first pass; no repair or exclusion was required. Privacy-state mutation, scheduler substitution, and message-accounting violations were zero.
- Formal accounting was 17,280 model calls, 9,510,001 prompt tokens, 1,391,607 generated tokens, and 7.78995 generation GPU-hours. Including the 256-decision engineering pilot, total metered generation was 7.90412 GPU-hours.
- Deterministic replay regenerated all 17,280 analyzed rows across all 24 trajectories with zero mismatches.
- Raw transitions and trajectories remain external under `/workspace/ThermoAgent-v14-artifacts/`; their compact repository manifest records tree SHA-256 `cc42761c0ce2b0873f72d571f74a04f5bcbf345e07b4651606a4398bb121dd05`.

## 2026-08-20 — frozen analysis disposition

- H2 field-reversal maximum departure minus nominal was 133.80477 regularized distance units (95% interval 107.94944 to 184.07463; exact one-sided sign-flip `p=0.015625`; Holm-adjusted `p=0.046875`). All six paired cluster effects were positive.
- H3 early counter-quench peak minus the final-five-sweep mean was 134.11032 (106.65373 to 184.19622). Its structurally nonnegative definition makes its sign-flip result non-diagnostic by itself. The scientific relaxation evidence instead comes from the full paths: every cluster crossed its held-out nominal threshold six sweeps after restoration and ended at mean distance 1.560.
- H4 full-minus-order-only leave-one-cluster-out balanced accuracy was 0.33333 (0.16667 to 0.45833; exact `p=0.03125`; Holm-adjusted `p=0.046875`). Absolute balanced accuracy was 0.125 for order-only, 0.458 for simple uncertainty, and 0.458 for the full representation. The full representation therefore improved on order alone but did not beat the smaller uncertainty representation.
- Field reversal produced two large pulses, at the quench and counter-quench, and then recovered. Partition and corruption remained close to nominal under this design; no omnibus-disruption claim is made.
- Of 276 frozen macrostate-distance sensitivity cells, 266 retained a positive field-minus-nominal ordering and ten were exactly zero; none reversed it. Removing the entropy/dependence family nearly eliminated the primary-geometry distance contrast, while no post-outcome feature or estimator tuning was performed.
- The V12 memory discovery and V13 prospective replication remain separate: 0.01790 (0.00357 to 0.03399) and 0.04030 (0.02883 to 0.05856) nats/update. Their 0.02936 (0.01872 to 0.03999) fixed-effect synthesis is descriptive only. V14 generated no new memory trajectories.

## 2026-08-20 — paper-facing package and QA

- Generated 26 data-derived vector PDF candidates and compact source CSVs, then selected eight principal figures for the 17-page manuscript.
- Applied presentation-only refinements to ten figures from frozen source data; no numerical value, estimand, or scientific source changed.
- Automated PDF QA passed for 27 PDFs and 43 pages. Every figure and manuscript page was manually inspected at original size and at 300 DPI; fonts are embedded, text is extractable, and no material clipping or overlap remains.
- The manuscript explicitly distinguishes effective reference energy from physical energy and coarse-grained pathwise irreversibility from exact thermodynamic entropy production.
