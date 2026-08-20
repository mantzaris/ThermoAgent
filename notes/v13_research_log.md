# V13 research log

## 2026-08-19 — Stage 1 provenance and design

- Fetched `origin` and verified local and remote V12 at
  `457f6d635b60292623c8d97aa3b0c60d8d0aac4e` with a clean worktree and index.
- Inspected the live RTX 4090 Pod using redacted process diagnostics. No tmux,
  Python experiment, or CUDA process was active; GPU utilization was 0% and
  memory use was 1 MiB. No credential-bearing process arguments were printed.
- Compared the non-Git RunPod snapshot with the pushed V12 tree. Preserved all
  older remote files; no synchronization or deletion was performed.
- Audited the V12 protocol, agent state, prompt, random-sequential update,
  delivery graph, transition estimators, fitted surrogate, replay path, and
  formal results. V13 reuses tested infrastructure through imports but writes
  only new source and artifact namespaces.
- Fixed an initial 32,224-decision formal design before any V13 scientific outcome.
  Pilot inspection is restricted to engineering estimability and throughput.

Further entries will record the engineering pilot, protocol freeze, complete
formal execution, analysis, QA, and deviations in chronological order.

## 2026-08-19 — Stage 2 engineering pilot

- Deployed only the new V13 configuration, source, tests, and scripts to the
  existing non-Git RunPod snapshot. Local and remote execution-source trees
  both hashed to
  `7295953a75b6fe3ddddeba992c687dd887809aafd150c0b4288e5026c05dee04`.
- Ran 192 prospectively bounded Qwen decisions. All 192 were valid on first
  pass; latent + occupancy was 0.578125; 13 minus-to-plus and 12 plus-to-minus
  belief transitions occurred; every disruption schedule and message-delivery
  check passed; and no peer-private mutation was observed.
- Mean generation latency was 1.607 seconds per decision, projecting 14.384
  metered generation-hours. Mean prompt length was 544.04 tokens, projecting
  17,531,031 formal prompt tokens. Both estimates passed their prespecified
  limits. No H1--H7 contrast, irreversibility contrast, disruption separation,
  or surrogate agreement was inspected.

## 2026-08-19 — Stage 3 protocol freeze and formal launch

- Froze protocol `v13-collective-agent-statmech-1.0` at
  `2026-08-19T11:42:47.656990+00:00`, before any V13 formal outcome existed.
  Protocol SHA-256 is
  `0d17d12d046c6dd424b5bdb2db6c9d4d82b6f72e095ceb726234afc689a65885`.
- The frozen source checksum remains `7295953a75b6fe3ddddeba992c687dd887809aafd150c0b4288e5026c05dee04`.
- Launched the initial 32,224-decision formal experiment under an exclusive
  writer lock. The only permitted early stops remain corruption, systematic
  invalidity, privacy/substitution failure, hardware failure, or the frozen
  resource ceiling; scientific effect signs are not monitored as stop rules.

## 2026-08-19 — Pre-network statistical amendment

- During the still-incomplete microscopic-response block, identified that four
  matched memory clusters make the smallest exact one-sided sign-flip value
  `1/16=0.0625`. H3 could therefore never meet its own 0.05 criterion. No
  network panel, factor contrast, memory contrast, or disruption outcome had
  yet been generated or inspected.
- Interrupted generation after 163 microscopic raw records. The aggregate
  microscopic table did not exist, the exclusive lock cleared normally, and
  every raw record was retained externally. No process remained active.
- Amendment 01 increases the H3 memory clusters from four to six. To preserve
  resource limits, it reduces only the secondary ring trajectories from 20 to
  15 sweeps and the ordered-start subset from two clusters to one. The
  modular-primary H1/H2 design and all disruption/recovery panels are
  unchanged. The amended design contains 32,672 analyzed decisions and still
  projects below 18 million prompt tokens.
- Froze operative protocol `v13-collective-agent-statmech-1.1` at SHA-256
  `e21bf4eefad193c4004394c05daa907c6fd86f82ad69caf1d79de4be9bbd8512`.
  Its amended execution-source SHA-256 is
  `7eeacc8003b82af67162eadb270a17f6dc6cdbaa46e46ed94ef014a0550efe10`,
  identical locally and on RunPod. The original v1.0 freeze is retained under
  its original filename and is explicitly invalidated, not overwritten.
- Before restarting generation, clarified the global token accounting in
  Amendment 02: the 18,000,000-token project ceiling includes 104,455 pilot
  prompt tokens and all retained interrupted calls. The executable formal-raw
  allowance is therefore 17,895,545 prompt tokens. This amendment changes no
  scientific panel, hypothesis, seed, estimator, or contrast; v1.1 is retained
  as a superseded pre-outcome freeze.
- Froze the final operative protocol `v13-collective-agent-statmech-1.2` at
  SHA-256 `a5259bbfd49da20b23a79646c248e0723a7fc382fa37b6165e3e936b4b669e3a`
  and source SHA-256
  `72c76946020e6ff7137848de25f384dd9cb25b5c3999754cac0b1b65ae7a4cc9`.
  The interrupted run-state and 163-record checksum list were preserved under
  the external `invalidated/formal_v1.0_interrupted/` namespace. The complete
  amended formal execution then began with a fresh run-state namespace.
- Reverified the pushed V12 parent after the V13 restart:
  `origin/llm-agent-stochastic-thermodynamics-v12` resolves exactly to
  `457f6d635b60292623c8d97aa3b0c60d8d0aac4e`; the V13 index remains empty.
- Validated a PTY-safe RunPod transfer path by reproducing the operative
  protocol locally with identical SHA-256. This will transfer only compact
  aggregate results and vector figures after analysis, never raw transcripts.
- Re-ran 125 applicable V10--V13 regression tests before outcome analysis;
  all passed with zero failures, errors, or skips. The execution-source
  checksum remained exactly the frozen v1.2 value after paper-only edits.
- Expanded the outcome-independent manuscript appendices to state the autonomy
  audit, matched quench operators, hypothesis hierarchy, amendment timing,
  and leave-cluster-out nominal-manifold geometry. The pre-result manuscript
  compiles at 18 pages with embedded fonts and no LaTeX warnings.
- After two complete network panels, a token-only resource audit (no belief,
  action, factor, memory, or disruption outcome was read) found 549.34 prompt
  tokens per Markovized network decision, versus the lower pilot projection.
  With the observed 169,296-token microscopic grid, 95,809 retained
  interrupted tokens, 104,455 pilot tokens, and the V12 measured memory-prompt
  increment, the complete frozen design projects to about 18.19 million total
  prompt tokens (a conservative longer-memory bound is about 18.24 million).
  This is approximately 1.1--1.3% above the nominal 18 million target but well
  within the 18-hour GPU ceiling. The user's rule expressly permits a documented
  benchmark exception where the complete design is infeasible within 18 million.
  No prompt, panel, seed, endpoint, or scientific criterion was changed. The
  runner checks the ceiling before each atomic panel; the final 720-decision
  panel is expected to begin below the formal-raw allowance and complete as one
  indivisible unit. The final actual overage will be reported as a protocol
  deviation rather than hidden or retroactively normalized.
- The halfway resource audit refined that projection using three completed
  memory pairs. Markovized memory prompts averaged 550.23 tokens and bounded-
  memory prompts averaged 605.07 tokens, a larger increment than the V12-based
  estimate. The updated total is approximately 18.39 million prompt tokens
  (about 2.2% above target). Generation remains projected at 14.55 hours. This
  update again used only token, latency, validity, and privacy metadata; no
  scientific state or contrast was inspected. The atomic-panel completion
  rule and all scientific settings remain unchanged.

## 2026-08-20 — Formal completion and frozen analysis

- Completed all 32,672/32,672 analyzed formal decisions: a 288-decision
  microscopic response grid and 72 interacting graph trajectories comprising
  2,264 sweeps and 32,384 network updates. The trajectories belong to 21 unique
  graph/environment clusters. No structured response remained invalid after
  repair, and no privacy mutation or scheduler substitution occurred.
- Including the retained 163 pre-amendment attempts and the 192-decision pilot,
  the study made 33,027 model calls, used 18,387,880 prompt tokens and
  2,652,913 generated tokens, and consumed 14.7351 metered generation GPU-hours.
  The final prompt total exceeded the 18-million target by 387,880 tokens
  (2.2%), matching the pre-completion token-only audit; no scientific outcome
  was inspected for that decision and no formal panel changed.
- Content-addressed deterministic replay checked all 32,672 analyzed rows in
  73 formal units with zero mismatches. CPU analysis and dense surrogate
  simulation used 1,060.1 seconds and approximately 50 million surrogate
  updates.
- H1 was not supported. The $J=0.80$ minus $0.35$ contrasts were -0.000592 for
  absolute belief magnetization (95% CI -0.006866 to 0.005682), -0.003455 for
  susceptibility (-0.013395 to 0.008776), and -3.5555 attempted updates for
  correlation time (-8.9104 to -0.2368).
- H2 was not supported and its nonzero contrasts reversed the V12 direction.
  The $\tau=0.85$ minus $0.50$ contrasts were 0.007753 for order (0.001302 to
  0.015211), 0.015952 for susceptibility (0.002516 to 0.033099), and 3.2319
  updates for correlation time (0.7517 to 5.4844).
- H3 replicated the V12 memory direction. Persistent minus Markovized adjusted
  block irreversibility was 0.040298 nats per attempted update (95% CI
  0.028825 to 0.058556; Holm-adjusted p=0.047995) over six matched clusters.
- H4 passed its frozen aggregate interval criterion, but scientific inspection
  showed that mean maximum macrostate distance was 45.055 for field reversal,
  2.315 for message corruption, 1.633 for partition, and 1.789 for nominal
  operation. The supported interpretation is field-quench response, not generic
  disruption sensitivity.
- H5 and H6 passed their frozen interval criteria: full-representation accuracy
  was 0.500 and exceeded the strongest reduced representation by 0.1875 (95% CI
  0.0625 to 0.2500). With only four held-out clusters, the full representation
  classified every field reversal but only 4/12 partition, corruption, and
  nominal panels. These findings are explicitly qualified as preliminary.
- H7 was not supported. The surrogate coupling effect (+0.01497) opposed the
  direct LLM effect (-0.00059); the surrogate captured the positive noise
  direction but overestimated its magnitude (+0.02955 versus +0.00775).
- Generated 22 candidate vector PDFs and exact source CSVs. Presentation-only
  refinements clarify the architecture boundary, phase legends, network arrows,
  surrogate axes, and finite-size labels without changing frozen numerical
  inputs or the execution-source checksum.
- Re-ran the final focused V10--V13 regression suite after all source and paper
  edits: 125 tests passed with zero failures, errors, or skips.
- Performed a fresh final PDF audit over 23 PDFs and 43 pages: all files opened,
  had embedded fonts and extractable text, rendered successfully at 300 DPI,
  and passed page-by-page manual inspection at original rendered resolution.
