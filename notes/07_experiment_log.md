# Experiment log

All timestamps are recorded in America/New_York unless explicitly marked UTC.

## 2026-08-11

- 21:45: First attempt through stale alias `runpod-thermo` was refused. This is
  an access failure, not an experiment failure.
- 21:49: The operator-provided RunPod proxy authenticated with an existing local
  identity, and a PTY shell reached the requested Pod. No credential was copied.
- 21:52: Completed read-only hardware, package, disk, and remote-tree audit.
  No active GPU process was present. No research package was installed and no
  experimental run was started.
- 22:19: Completed isolated environment setup. The first resolver attempt was
  stopped and quarantined after it selected a redundant CUDA 13 PyTorch stack.
  The corrected pip setup preserved PyTorch 2.8.0+cu128 and passed all 26 tests.
- 22:20: Resolved the public Qwen2.5-7B-Instruct model to immutable Hub commit
  `a09a35458c702b33eeacc393d103063234e8bc28`. No model experiment has run yet.
- 22:50: Real-model CUDA smoke completed. NF4 loading used 7.07 GB peak
  allocated VRAM; a batch of two produced 168 tokens in 4.74 seconds (35.4
  generated tokens/s); both responses were valid JSON and the CUDA matrix was
  finite. One statically valid plan targeted itself with `due_step=500`; this
  is retained as `model_smoke_initial_semantic_failure.json`. Dynamic self and
  relative-deadline checks plus clearer prompt constraints were added before a
  semantic-validation rerun.
- 22:56: Tightened semantic smoke passed: 100% JSON, static-schema, and
  dynamic-tool validity; 192 generated tokens in 4.54 seconds (42.3 tokens/s),
  7.07 GB peak allocated VRAM. One justification miscompared inventory and
  capacity, retained as a planner numerical-reasoning limitation.
- 23:04: First Stage 1 real-agent run conserved material in both applications,
  produced a coalition, two counters, a route failure, and a successful replan.
  It failed the required rejection check and the initial seller requested a
  quote instead of submitting an offer. The run is retained under
  `stage1_initial_no_rejection/`; the private reservation rule and quote-request
  fixture were made explicit before rerun.
- 23:04: First PPO job failed before episode 0 with `NameError` because selected
  monitor settings were initialized in the wrong function scope. No checkpoint
  was produced. The fix passed an 8-episode local training validation with
  nonzero delayed rewards and per-agent GAE groups.
- 23:07: Corrected PPO job completed both 96-episode staged variants in 6.6 s
  each on CPU/GPU-light mock-planner rollouts. Checkpoints are about 30 KiB.
  Training logs retain every episode and PPO update. The identical reported
  final-window mean (3.940) is descriptive only, not a paired treatment result.
- 23:09: Stage 1 v2 failed its explicit gate. Qwen submitted the seller offer,
  but for two buyers it explained that prices exceeded reservation values and
  nevertheless emitted `accept_offer`. It also emitted `join_coalition` with a
  fabricated ID and an invalid extra `members` field, then an unavailable
  `submit_quote` tool during the route probe. Both application episodes still
  conserved resources. The artifact and exit status 1 are retained.
- 23:24: Added state-dependent planner affordances and private offer consistency
  validation, a fixed three-attempt replan budget, and an isolated closed-route
  probe. All 36 local tests pass. No v3 result exists yet.
- 23:31: Stage 1 v3 passed all nine gates. Each direct capability succeeded on
  attempt 1/3: offer submission, rejection, counteroffer, coalition proposal,
  closed-route failure, and successful alternative dispatch. Commercial and
  humanitarian short episodes conserved material. Qwen still misstated 4.5 as
  equal to a 0.45 reservation value in rejection prose; typed action validation,
  not prose, remains the authority.
- 23:38: Expanded the pre-pilot suite to 39 tests and recorded remote exit 0.
  Added explicit communication bytes/delivery, agreement rationality, utility,
  efficiency, memory, trust, and coalition metrics; failed-run retention;
  seed-cluster bootstrap; sign-flip tests; multi-signal monitoring; predictive,
  calibration, convergence, and source-localization outputs.

## 2026-08-12

- 00:02: Pilot v1 completed 54/54 rows with exit 0. It took 1,197.36 s,
  issued 1,344 Qwen calls, and generated 97,188 tokens. Exact v1 config,
  completion tables, sweep manifest, source checksum, logs and raw ledgers were
  archived before source synchronization.
- 00:06: Pilot-v1 analysis found 99.70% Thermo structured validity but only
  85.86% tool validity versus 94.79% for learned-no-entropy. Every three-seed
  primary pair favored ThermoAgent, but policy action collapse and the old
  permissive route mechanism make those effects diagnostic only. Raw free
  energy had disruption AUC 0.404; absolute nominal deviation reached 0.808.
- 00:08: The corrected source passed 48/48 tests on RunPod. Launched a detached
  v2 extension; v1 run IDs resumed and only 36 route/compound rows executed.
- 00:22: V2 completed 90/90 cumulative rows, exit 0, in 863.01 s. All 36 new
  rows completed and conserved material. Archived exact v2 config/manifest and
  completion tables before further changes.
- 00:13--00:20: PPO diagnostic sequence. A 384-episode longer-training retry
  collapsed both variants to request-reallocation. Unbalanced imitation and
  initial imitation-anchor variants remained semantically concentrated. These
  candidates were not promoted and are recorded as negative results.
- 00:24--00:32: Added private action masks, balanced scripted initialization,
  imitation anchoring, affected-agent event replanning, explicit delivery
  events, bounded bargaining, resource-direction-safe counteroffers, coalition
  ledgers, and temporary route authority. The suite grew to 56 passing tests.
  Independent deterministic seeds 71--73 confirmed action diversity; the
  entropy candidate did not outperform no-entropy, so checkpoint selection was
  not based on favorable treatment outcome.
- 00:43: Recalibrated local surprisal using nominal role-conditioned
  references, then completed the final identically initialized 192-episode PPO
  pair. Both jobs exited 0; final windows and checkpoint checksums are retained
  in the training logs. No process remained active afterward.
- 00:48: The saved direct port had become stale. Reconnected through the stable
  RunPod proxy, recovered the current direct mapping in local operator state,
  and fetched calibration, checkpoints, and logs. No credentials or endpoint
  details are retained in Git-facing artifacts.
- 00:52: Remote suite passed 59/59 after adding matched memory and metapolicy
  controls. Launched the restartable final-v3 qualification: 84 new matched
  core-method episodes, with 90 archived diagnostic rows resumed rather than
  rerun.
- 01:05: Pre-freeze audit found that the centralized LLM baseline derived
  coarse reports from exact simulator state under strong privacy. The running
  rows are not being deleted or overwritten. A public-report-only correction,
  explicit tests, and 12 separately named v4 qualification rows were specified
  before main freeze.
- 01:08: Added evaluator-only current/delayed/noisy/no-estimate comparison and
  top-3 localization outputs. These fields use a separate RNG and cannot alter
  environment or agent trajectories.
- 01:11: A second pre-freeze audit found that one simulator RNG served
  action-dependent message delivery as well as later exogenous demand and
  production. Interrupted final-v3 after 19 new rows, retained every artifact,
  recorded exit 130, and prospectively excluded its named rows from outcome
  inference without inspecting effect direction.
- 01:16: Split initialization, exogenous dynamics, private observation noise,
  and communication into deterministic derived streams; added a regression test
  showing extra messages cannot shift the exogenous trajectory. Renamed the
  replacement qualification `paired_*_v5`; calibration and PPO will be rebuilt
  before it runs.
- 01:21: Recalibration selected pooled/window-1 again. Moderate-shock entropy
  shifted +0.063 and free energy -0.031; both directions are retained. The
  regenerated matched 192-episode PPO pair completed with exit 0.
- 01:27: Full mock preflight exposed redundant per-episode CUDA/provenance
  queries and stalled after 23 completed rows. Stopped only that preflight,
  retained its restartable `/tmp` outputs, and added process-local memoization
  for immutable hardware/dependency/source/Git records before relaunch.
- 01:31: Cached preflight completed 128/128 mock episodes, but replay caught two
  rows whose raw output had been written immediately before an intentional
  process stop and whose manifests were absent. Added a staged
  manifest-before-publication transaction; legacy missing-manifest rows now
  remain explicit failures rather than silently resuming.
- 01:32: Used the excluded v3 rows only for planner diagnostics. Learned-option
  prompts showed 76.3% (267/350) and 69.5% (89/128) structured validity, stale
  time fields, and transport calls against a graph with no transport outbound
  arcs. Added exact time/argument guidance, concise v4 outputs, 2,560 input and
  160 output token bounds, and executable non-producing transport arcs. Local
  suite passed 66/66; recalibration/retraining are required again.
- 01:41: Final environment calibration and matched PPO pair completed; pooled
  window-1 remained selected. A fresh preflight staged its first row correctly
  but blocked in the one-time direct PyTorch driver probe. Replaced that probe
  with a bounded ten-second `nvidia-smi` query while retaining the separate
  complete CUDA/model-smoke evidence.
- 01:48: Definitive mock preflight completed 128/128 episodes and 128/128
  quantitative replays; analysis produced all ten PDFs. Mechanical QA stopped
  because Poppler binaries were absent. Pinned PyMuPDF 1.28.2 in the isolated
  venv and added a backend-recording fallback plus a regression test.
- 02:00: PyMuPDF mechanical QA passed all ten mock-preflight PDFs. The first
  visual pass identified several clipped or crowded annotations; layouts were
  revised, regenerated, rerendered, and inspected again. All ten now pass visual
  QA. These artifacts remain in `/tmp` and support engineering readiness only.
- 02:01: Launched detached job `stage1-agentic-v4-20260812` to revalidate all
  nine Stage 1 real-model capability gates under `planner-json-v4` and the
  final transport topology.
- 02:03: Stage 1 v4 retained a 7/9 pass: offer, rejection, counteroffer,
  coalition, both applications, and conservation passed, but the route probe
  selected valid coalitions on all attempts and therefore never observed a
  failed shipment or replan. Audit found the harness requested option 6
  (coalition reallocation), whose deliberately narrowed affordance permits only
  coalition actions absent a proposal. The probe expectation was inconsistent
  with the frozen option semantics; no planner output was repaired or removed.
- 02:07: Changed only the route probe from option 6 to option 0, added an
  affordance-regression test, passed 69/69 locally and the complete remote test
  command, and launched `stage1-agentic-v4-routefix-20260812`. The prior run is
  not overwritten and the model/prompt/seed/decoding remain fixed.
- 02:10: Corrected Stage 1 v4 completed with exit 0 and all nine checks true.
  Qwen scheduled the unreachable shipment, received deterministic `no_route`,
  recalled that failure, and scheduled the same quantity to the newly delivered
  reachable retailer on the first replan. Both short application episodes again
  conserved material.
- 02:12: Launched detached `pilot-paired-v5-20260812`. The planned matrix has
  174 cumulative run IDs: 90 immutable v1/v2 diagnostics plus 84 new legal,
  purpose-RNG-separated `paired_*_v5` rows. No outcome threshold is required
  for continuation; the pilot gates protocol validity and measures cost.
- 02:18: Added and locally validated the prospective budget profiler while the
  pilot remained active. The suite now passes 71/71. Projection rules, method
  analogues, 24-hour cutoff, cost assumptions, and message-match definition
  were fixed before pilot outcome analysis.
- 02:22: Added a run-family filter to quantitative replay so the final paired-v5
  pilot can be replayed without falsely applying the current simulator to
  archived pre-correction diagnostics. The filter and separate report filename
  are explicit in the artifact; 72/72 local tests pass.
- 02:21--02:29: A pre-freeze privacy audit found that actor consensus error
  compared gossip with evaluator-only global occupancy and interaction entropy
  was global. Interrupted paired-v5 after 16 rows, including one no-entropy row;
  no treatment outcome was inspected. Added prospective exclusions, exact v5
  config/checksum, and an interruption manifest. Implemented link-local
  residuals, per-agent interaction entropy, reliability-sampled gossip on
  separate RNG streams, explicit sketch events, and zeroed controls. The local
  suite passes 79/79.
- 02:33: Remote corrected suite exited 0. Archived the active pre-boundary
  checkpoint pair and logs (checksums `f8d9326c...8e57` and
  `119069db...dd2`), then launched matched 192-episode v6 PPO retraining.
- 02:40: Privacy-corrected matched PPO retraining exited 0. The active
  no-entropy and ThermoAgent checkpoints have SHA-256 checksums
  `231f451497349ef8d87de768f9d11fdf139f767f6f70fa754398688ce3ad4373`
  and `09899d19c432489e7f6afb24923b6e97c30838a8210962548b11b9af8d91706b`.
  Both used the identical 2,304-row initialization (validation accuracy
  0.6791); final-window training outcomes were 718.25 and 724.60 and were not
  used to select a checkpoint.
- 02:41: Verified the remote checkpoint hashes, retraining exit code 0, and an
  idle GPU, then launched detached `pilot-paired-v6-20260812`. This 84-row
  run is the first comparative pilot eligible under the corrected local-only
  information boundary.
- 02:44: A continuing pre-freeze factorial audit found that privacy severity
  also multiplied true marginal costs and increased agents' own forecast noise.
  Stopped paired-v6 after 13 atomically published rows without comparing method
  outcomes. Archived the exact configuration/checksum and exclusion rule.
  Corrected privacy to affect public observability only, added an invariant
  test, and named the clean replacement paired-v7.
- 02:49: Local and remote suites passed 82/82 after adding the information-
  regime invariant, fail-closed freeze verification, and coalition withdrawal
  affordance. Archived the superseded checkpoint pair and launched detached
  `recalibrate-train-factor-v7-20260812`.
- 02:54: Corrected calibration and matched retraining exited 0. Pooled/window-1
  remained selected; moderate entropy shifted +0.062971 and free energy
  -0.030783. New calibration/no-entropy/Thermo hashes are `786a65d1...4d01`,
  `220e6055...9467`, and `06ac5212...2d65`. Training outcomes were not used to
  choose either policy. Added failure-aware paired reporting; local suite is
  now 83/83.
- 03:03: Synced the final pre-v7 source, remote suite exited 0, and launched
  detached `pilot-paired-v7-20260812`. This is the first qualification eligible
  under both the local-monitor and isolated-information-factor boundaries.
- 03:04: Continuing audit found inconsistent partition onset: messages were
  impaired from period 0 while sketches switched at the disruption. Stopped v7
  after 10 rows without comparing outcomes. Added prospective exclusions and
  exact provenance, unified both channels on the disruption onset/component
  graph, and named the clean replacement paired-v8. Local suite passes 84/84.
- 03:09: Final aligned calibration/policy rebuild exited 0. Calibration,
  no-entropy, and ThermoAgent hashes are `58dabe4d...0769`,
  `edbe570a...7096`, and `62d5a1c7...74d2`. Added explicit sketch-message/byte
  accounting and public-dashboard ledger events; local suite passes 86/86.
- 03:15: Final 86-test suite exited 0 on RunPod and detached
  `pilot-paired-v8-20260812` launched. V8 is the first qualification eligible
  under all prospectively documented validity and accounting corrections.
- 03:31: While v8 remained isolated on its launch source, a local pre-freeze
  audit found two descriptive-metric defects (one-sided agreement rationality
  and proposal-as-formation), incomplete post-arrival verification, permissive
  shipment/coalition target authority, and artificially bandwidth-limited
  centralized baselines. No v8 outcome comparison had been inspected. Added
  strict deterministic validation, two-party evaluator metrics, completed
  shipment records, an every-period full-information receding-horizon bound,
  per-demand legal central-LLM slots, and a deployable comparator definition
  for the necessity map. The expanded local suite passes 93/93. These changes
  have not been synchronized into the still-running remote v8 process.
- 03:39: The new disposable full preflight completed 128 mock episodes, replay,
  and analysis but exposed a Matplotlib-3.3 incompatibility in the seventh
  figure (`Figure.supxlabel`). Replaced it with the compatible shared-text API;
  all ten PDFs then opened, exposed fonts, rendered, and passed contact-sheet
  visual inspection. The local suite remains 93/93. Paired-v8 had 37/84
  atomically published rows and zero recorded failures at the sealed progress
  check; no episode outcome fields were inspected.
- 03:49: The final paper/control audit added an adversarial fail-closed test for
  blind central dispatch under fully private reports, event sourcing for all
  coordinator outcomes, one fixed rather than per-seed-clairvoyant comparator
  in the necessity map, and actual-join-only coalition outlines. Focused tests
  pass and the complete local suite now contains 96 tests. None of these edits
  has been synchronized into the isolated active v8 process, and no v8 outcome
  value has been read.
- 03:54: Re-audited the Stage 1 coalition criterion against the corrected
  actual-join metric. The retained v4 real-model run had a proposal but did not
  solicit a separate invitee decision. Extended the harness to require a
  delivered invitation and an independently authored `join_coalition`; the
  full mock harness passes, and a real-Qwen v5 rerun is queued before freeze.
  Also completed failed-run manifest fields so failed seeds retain the same
  provenance, RNG, topology, model, checkpoint, token, and hardware record as
  successful seeds. The local suite now contains 98 tests.
- 04:06: A fresh post-correction disposable workflow completed 128/128 mock
  episodes and quantitative replays, generated all ten required vector PDFs,
  passed mechanical open/font/render checks, and passed original-resolution
  contact-sheet inspection. The expanded local suite passes 99/99. Paired-v8
  had 81/84 atomically published manifests with zero recorded failures; outcome
  fields remained unopened pending complete execution and immutable replay.
- 04:11: Paired-v8 exited 0 with all 84/84 planned episodes complete and no
  failure. Before source synchronization or outcome inspection, the original
  remote snapshot replayed 84/84 ledgers without a metric or tool-result
  mismatch; maximum absolute conservation residual was `3.41e-13`.
- 04:20: Opened and documented the three-seed pilot outcomes. ThermoAgent and
  learned/no-entropy were mostly tied, while the strong central, scripted, and
  no-communication controls generally did better; accepted coalitions were
  frequent but did not establish useful recovery. The negative finding is
  retained. The p90 design projection is 20.269 GPU-hours, so all 1,096 planned
  post-freeze episodes remain. Fixed the random activity control once at
  624/1,151 = `0.542137`; the initial messages/proposals calculation was
  invalid because coalition fanout can exceed one and monitor sketches are a
  separate mandatory channel.
- 04:27: The stricter real-Qwen Stage 1 v5 exited 1. Offer, rejection,
  counteroffer, route failure, replan, both applications, and conservation all
  passed, but Qwen included the proposing warehouse in the coalition invitee
  list twice; strict validation returned `self_member` and no coalition formed.
  Retained the entire failure. Added explicit invitee-only guidance and a
  concrete eligible-ID list as `planner-json-v5`; no action was repaired and
  the simulator remains fail-closed. The local suite passes 100/100 before the
  uniquely named v6 rerun.
- 04:35: Stage 1 v6 (`planner-json-v5`) exited 0 with every strict gate true.
  The warehouse's first proposal listed only seven other organizations; after
  ordinary message delivery, carrier_05's separate private context selected
  and successfully executed `join_coalition` on its first attempt. Offer,
  rejection, counteroffer, route failure, replan, both application episodes,
  and conservation also passed. Launched the separately named exact/coarse/
  absent-report central-LLM smoke before protocol freeze.
- 04:40: Focused central-LLM prompt-v5 smoke exited 0 and replayed 3/3. Exact
  reports exposed eight organizations and scheduled three shipments; absent
  reports exposed zero and every mutation was blocked. Coarse reports scheduled
  once but selected a public source without a route twice. The prompt lacked
  public topology, weakening the comparator. Added and executor-enforced public
  `eligible_source_ids` per demand as central-only `planner-json-v6`; private
  operational visibility is unchanged. The expanded local suite passes 101/101
  before the uniquely named central smoke v2.
- 04:45: Central prompt-v6 public-route smoke completed and replayed 3/3.
  Exact and coarse reports each scheduled 3/3 shipments from executor-verified
  route-eligible sources. Absent reports exposed zero agents and no domain tool
  was called; Qwen attempted blind dispatch on all three epochs and the
  executor rejected each with `coordinator_no_public_demand`. A fresh
  post-correction 128-episode mock workflow also replayed 128/128, generated all
  ten PDFs, passed Poppler open/font/render checks, and passed contact-sheet
  inspection. Final environment/source capture was requested before freeze.
- 04:50: Compared the final local and remote execution inputs byte-for-byte.
  Both report source checksum `4b76671d...9c55`, calibration checksum
  `58dabe4d...0769`, no-entropy checkpoint `edbe570a...7096`, and ThermoAgent
  checkpoint `62d5a1c7...74d2`. Wrote the non-overwriting protocol manifest at
  `2026-08-12T08:48:40.949536+00:00`; all 36 frozen artifacts independently
  verified both on RunPod and on the local research copy. No main result was
  opened before this boundary.
- 04:52: Launched detached job `main-frozen-v1-20260812`. `run-main.sh`
  re-verifies the immutable manifest before model loading, and the tmux session
  is active. This run targets all 944 preregistered rows. Monitoring will inspect
  completion/failure state and resource use; every outcome will be retained.
- 20:18: Frozen main job exited 0 after all 944/944 planned rows atomically
  published. Monitoring throughout read only row count, job state, and hardware
  health; no treatment outcome was opened and no row was restarted. Peak
  observed allocated VRAM was 23.5 GiB during one maximum-context episode;
  steady-state later stabilized near 13.5 GiB, with no OOM.
- 20:20: Before fetching or opening outcomes, replayed all 944 main event
  ledgers from the immutable remote copy. All 944 passed with zero tool-result
  mismatches, zero metric mismatches, and maximum absolute conservation
  residual `4.547473508864641e-13`.
- 20:21: Launched detached frozen job `ablations-frozen-v1-20260813` for all 72
  preregistered ablation rows. The outcome seal remains in place through the
  locked holdout launch.
## 2026-08-12 22:29 EDT -- Ablation replay gate and holdout launch

- `ablations-frozen-v1-20260813` completed all 72 prespecified rows and wrote
  detached-job exit status 0. During outcome-blind monitoring, peak observed
  GPU memory was 23,636 MiB; no OOM or infrastructure error appeared.
- Before inspecting treatment outcomes, replay reconstructed 72/72 ablation
  ledgers with no metric mismatches and no tool-result mismatches. Maximum
  absolute material residual was approximately `3.41e-13`.
- Launched `holdout-frozen-v1-20260813` through `scripts/run-holdout.sh`; the
  wrapper verified the immutable protocol checksum before executing the frozen
  80-row, nine-agent unseen-scenario matrix.
- Main and ablation outcome columns remain unopened. No row was rerun or
  selected based on performance.

## 2026-08-13 00:02 EDT -- Holdout, unsealing, analysis, and figure QA

- Locked holdout completed 80/80 and exited 0. Before outcomes were opened,
  replay passed 80/80 with zero metric/tool mismatches. A combined final report
  then replayed all 1,096 post-freeze episodes successfully.
- Lifted the outcome seal and ran the frozen episode-level analysis. Main
  ThermoAgent/no-entropy differences favored ThermoAgent by `0.187` commercial
  AUC and `106.51` humanitarian weighted-need units, but Holm-adjusted p-values
  were `0.0856` and `0.0584`. Strong fixed, scripted, central, and lookahead
  controls generally matched or beat ThermoAgent. Locked holdout effects versus
  no entropy were exact ties in both applications.
- The descriptive necessity surface was negative in every privacy/misalignment
  cell. Operational entropy detected disruption well; the calibrated free-
  energy high-direction signal did not. No outcome-responsive rerun or protocol
  edit was made.
- A yielded analysis call was mistakenly interpreted as complete and retried,
  causing two identical read-only analyzers to overlap. Both deterministic
  processes finished; the index was rebuilt once afterward. Raw and protocol
  artifacts were untouched. Constant-input monitoring correlations remain
  missing rather than imputed.
- Generated all ten final vector PDFs. Mechanical PyMuPDF validation passed.
  Manual review found a Pareto-label collision and clipped network coalition
  outlines. Added a separately documented presentation-only save wrapper,
  leaving all 36 frozen checksums unchanged. Regenerated three PDFs; all ten
  then passed original-resolution manual QA.
- Measured post-freeze execution: 18.592 including-load GPU-hours, 69,533 LLM
  calls, 116,469,832 prompt tokens, and 4,468,148 generated tokens. No research
  or analysis process remains active.
- Final verification passed all 101 tests and all 36 protocol hashes; local and
  remote aggregate source checksum is `4b76671d...9c55`. The 1,096 final
  manifests contain every required provenance field. Sensitive-pattern and
  filename scans returned zero matches. The largest Git-facing file is the
  30.5 MB processed time-series table; raw ledgers are gzip-compressed.
- Terminated one obsolete sleep-only pilot status watcher (no experiment child,
  no GPU use). Final tmux and CUDA-compute checks are empty. The final result
  tree was fetched locally; the Pod can safely be stopped.
