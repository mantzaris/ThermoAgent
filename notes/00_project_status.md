# ThermoAgent project status

Last updated: 2026-08-13 00:02 America/New_York

## Current phase

Stage 0 engineering, final nominal calibration, CUDA inference, and the strict
real-LLM Stage 1 v6 validation are complete. V6 includes an independently
accepted coalition membership under prompt v5; the earlier v5 failure is
retained. Paired-v5 was interrupted before treatment analysis after an
actor-feature privacy defect was found. Link-local monitor correction and
matched PPO retraining completed, but a subsequent factorial audit found a
privacy/economic-state confound. Its correction, recalibration, and retraining
are complete. Paired-v7 exposed inconsistent partition timing; the communication
correction and matched retraining are complete. Paired-v8 qualification and the
pre-freeze public-route central-LLM smoke are complete. The protocol is
immutably frozen and verified locally and remotely. Main evaluation completed
944/944 and its immutable event-ledger replay passed 944/944 before any outcome
inspection. Frozen ablation evaluation completed 72/72 and locked holdout
completed 80/80, both with exit status 0 and exact replay. Final analysis,
figures, and visual QA are complete; documentation and repository hygiene are
the current phase.
Paired-v8 completed 84/84 and replayed 84/84 on its immutable launch snapshot;
its mixed/negative pilot outcomes and resource projection are now documented.
Pilot v1 and corrected-shock v2 are archived diagnostics. Final-v3 was
intentionally interrupted after a pre-freeze validity audit; every artifact is
retained and its named rows are prospectively excluded. Corrected paired-v5
policy/planner qualification has completed.
The outcome seal was lifted only after all 1,096 post-freeze ledgers passed
replay. Final evidence is mixed/negative: entropy features have suggestive
in-distribution effects versus the matched actor but no Holm-confirmed or
holdout benefit, and autonomous agents do not beat strong simpler controls.

## Completed work

- Inspected and preserved the pre-existing dirty worktree at commit `8bd699b`.
- Connected to the existing RunPod Pod and established a noninteractive direct
  SSH path to `/workspace/ThermoAgent` without copying credentials.
- Audited RTX 4090/CUDA 12.8, CPU, RAM, disk, Python, and installed packages.
- Created `/workspace/ThermoAgent/.venv` with system PyTorch 2.8.0+cu128 and
  isolated experiment dependencies; the model cache is outside the repository.
- Implemented two material-conserving simulators, independent private agent
  state, role-scoped typed tools, explicit messaging and commitments, event
  sourcing/replay, entropy/free-energy monitoring, distributed gossip, PPO,
  baselines, restartable matrices, manifests, paired statistics, and figure QA.
- Passed the current 101-test suite locally and on RunPod. It covers independence,
  privacy, resource conservation, role-scoped tools, counteroffers, topology
  shocks, purpose-specific RNG streams, replay, monitoring, staged publication,
  and PDF validation.
- Fixed the monitor formulation before treatment comparison: pooled,
  role-normalized occupancy with a one-period window.
- Ran Qwen2.5-7B-Instruct at its immutable revision in NF4. Corrected model
  smoke achieved 100% JSON/schema/dynamic validity and 42.3 generated tokens/s.
- Trained the pre-monitor-boundary matched staged PPO checkpoints for 192 episodes per policy
  from the same balanced initialization. Both checkpoints are retained; training
  losses are engineering diagnostics, not comparative outcome evidence.
- Ran a fresh 128-episode deterministic analysis/figure preflight outside the
  research results tree. All 128 quantitative replays passed; all ten required
  vector PDFs opened, exposed fonts, rendered at 150 DPI, and passed a manual
  visual review after layout correction.
- Passed the real-LLM Stage 1 v3 gate. Qwen agents submitted a private-cost
  offer, rejected and countered privately evaluated offers, proposed a temporary
  coalition, observed a closed-route failure, and successfully replanned to a
  reachable target. Five- and six-agent application episodes conserved material.
- Completed pilot v1: 54/54 episodes, no failures, 1,344 real-model calls,
  97,188 generated tokens, and 19.96 min sweep wall time including model load.
  It is archived as a planner/monitoring diagnostic because the physical shock
  mechanism was corrected while it was running.
- Found and corrected two pre-freeze validity defects: centralized-LLM access
  to hidden coordinator reports and a shared RNG that coupled message counts to
  exogenous trajectories. Final-v3 stopped after 19 new rows; its exact config,
  logs, raw events, and exit-130 manifest are preserved and machine-excluded.
- Retrained matched 192-episode policy pairs after the actor-information,
  information-factor, and partition-timing corrections. Superseded pairs are
  archived. The final active no-entropy/ThermoAgent hashes are
  `edbe570a...7096` and `62d5a1c7...74d2`; both use the same 2,304-row
  behavioral-cloning initialization.
- Completed a final local tool/metric/baseline audit without changing the
  active remote v8 source: actual coalition joins now define formation,
  agreement rationality checks both private parties, delivery verification and
  shipment authority are enforced, the full-information bound replans every
  period, and the legal central LLM has one typed slot per reported demand.
- Completed a fresh 128-episode disposable preflight through replay, analysis,
  all ten PDFs, mechanical PDF validation, and manual contact-sheet review. It
  caught and resolved one Matplotlib-3.3 rendering compatibility issue before
  protocol freeze.
- Added fail-closed central dispatch under absent reports, complete coordinator
  result events, a fixed-cell (not per-seed oracle) necessity-map comparator,
  and actual-join-only coalition visualization semantics.
- Tightened the Stage 1 gate so a coalition requires a delivered proposal plus
  a separate invitee LLM's validated join. The real-Qwen v6 validation passes;
  its preceding v5 strict-validation failure remains retained. Failed-run
  manifests now retain full reproducibility fields rather than a reduced subset.

## Active jobs

No research or analysis job is active. Main, ablation, and holdout detached jobs
all exited 0. Retained failed smoke/pilot predecessors remain artifacts, not
background processes. One obsolete pilot status-watcher shell was terminated
after confirming it had no experiment child. No tmux session or CUDA compute
process remains; it is safe to stop the Pod after the final result fetch.

## Next actions

All requested implementation, evaluation, analysis, documentation, and
verification actions are complete. Optional next action: review and commit the
prepared tree, then stop the Pod.

## Blockers

No external blocker.

## Latest valid results

- `results/reproducibility/replay_report.json`: all 1,096 post-freeze ledgers
  replayed exactly with zero metric or tool-result mismatches; maximum absolute
  material residual was below `4.55e-13`.
- `results/reproducibility/ablations_replay_report.json`: all 72 ablation
  ledgers replayed exactly with zero metric or tool-result mismatches; maximum
  absolute conservation residual was approximately `3.41e-13`.
- `results/manifests/{main,ablations,holdout}_sweep.json`: 944/944, 72/72, and
  80/80 rows completed without failure in 18.592 including-load GPU-hours.
- `results/statistics/primary_paired_comparisons.csv`: ThermoAgent versus
  no-entropy improvements were `0.187` commercial and `106.51` humanitarian,
  but Holm-adjusted p-values were `0.0856` and `0.0584`; holdout effects were
  exact ties.
- `results/statistics/monitoring_summary.csv`: operational entropy AP `0.934`
  and ROC AUC `0.863`; free-energy gap AP `0.577` and ROC AUC `0.393`.
- `results/reproducibility/pdf_qa/report.json`: all ten final PDFs open, expose
  fonts, render through PyMuPDF, and have passed manual preview inspection.
- `results/reproducibility/protocol_freeze.json`: protocol frozen at
  `2026-08-12T08:48:40.949536+00:00`; all 36 listed files verify locally and on
  RunPod. The frozen source checksum is
  `4b76671d2d1cbaa7b213d2b11917ab02a440d91b3d46d5646dfacdf934599c55`.
- `results/smoke/model_smoke.json`: 100% valid JSON/static/dynamic tools, 192
  generated tokens in 4.54 s, 7.07 GB peak allocated GPU memory.
- `results/reproducibility/macrostate_calibration.json`: final thresholds fixed
  from nominal seeds 101--105 after the transport-arc correction; checksum
  `58dabe4de322ead381965809018c98931cc2b95acc6335c50385f76ec0cc0769`.
- `results/pilot/monitor_formulation_comparison.json`: pooled/window-1 selected;
  moderate disruption increased entropy by 0.062971 but decreased the calibrated
  free-energy gap by 0.030783, an important mixed monitoring result.
- `results/checkpoints/coordination_{no_entropy,thermo}.pt`: privacy-corrected,
  matched 192-episode staged checkpoints, with checksums
  `edbe570a...7096` and `62d5a1c7...74d2`; the recorded training losses do not
  establish treatment performance.
- `results/smoke/stage1/stage1_agentic_smoke.json`: failed Stage 1 v2 record;
  both application episodes conserved material, but four capability checks
  were absent. It is evidence about planner failure, not a valid smoke pass.
- `results/smoke/stage1_v3/stage1_agentic_smoke.json`: all nine required checks
  true; six direct capability proposals succeeded on their first bounded attempt.
- `results/smoke/stage1_v4_routefix/stage1_agentic_smoke.json`: all nine checks
  true under `planner-json-v4`; the agent attempted the explicitly unreachable
  route, received `no_route`, then replanned to a reachable retailer on its first
  follow-up proposal. The retained predecessor in `stage1_v4/` passed 7/9 and
  exposed the option-semantics harness defect.
- `results/smoke/stage1_v6_invitees_only/stage1_agentic_smoke.json`: all ten
  strict checks true under `planner-json-v5`; a warehouse proposed a coalition,
  its invitation crossed the ordinary channel, and a carrier's separate LLM
  context executed `join_coalition` on the first attempt. The preceding
  `stage1_v5_actual_join` failure is retained and documents two strict
  `self_member` rejections.
- `results/manifests/pilot_sweep_v1.json`: 54/54 complete, 19.96 minutes,
  1,598,502 prompt and 97,188 generated tokens. Pilot raw free-energy detection
  was mixed (AUC 0.404), while absolute deviation reached AUC 0.808.
- `results/manifests/pilot_sweep_v2.json`: 90/90 cumulative complete and zero
  failures; the 36-row route/compound extension took 14.38 minutes.

## Estimated remaining time

No work remains for the frozen design. No GPU process is active.
