# V15 fresh-RunPod reconstruction and collective-observable extension

This is an append-only execution record for reconstruction of the frozen V15
study from committed source after deletion of the original external RunPod
artifacts. It does not redefine the frozen V15 protocol or hypotheses.

## 2026-08-22: provenance and access audit

- Verified local branch `jstat-scientific-audit-v15` at
  `b309f0ab76cb24377de5872eebc811582af1f43f`.
- Verified the remote tracking branch at the same commit and parent
  `103e4c4598ecc26a98c37a8d03ee3663f9be1070`.
- Verified a clean starting worktree before any reconstruction edits.
- Found no repository `AGENTS.md` file.
- Confirmed the frozen protocol SHA-256
  `863f54a05dbbe9f23a0d3fe6d4344b71409796340c6659c51247d9e8949f89c9`.
- Confirmed that the historical execution digest includes one ignored
  root-level bytecode cache. The committed audit reconstructs the legacy
  digest from its path and digest and reports clean semantic-source digest
  `f8d4fa546ba46a42cd4234dd8af6ad60309c231f2997e10d0d25830f6dddb2f2`.
- The configured RunPod SSH endpoint repeatedly returned TCP connection
  refusal. No remote filesystem, GPU, CUDA, cache, process, or disk claim has
  yet been made. The user was asked to refresh the saved SSH endpoint or Pod
  SSH service without sharing credentials.

## 2026-08-22: reconstruction safeguards before model execution

- Repaired the setup script so a fresh image no longer assumes that the
  tested CUDA-enabled PyTorch build survived. It accepts an already exact
  `2.8.0+cu128` build, otherwise installs the official CUDA 12.8 wheel, then
  verifies every pinned package and writes an external environment manifest.
- Added a pinned model-prefetch verifier for Qwen revision
  `a09a35458c702b33eeacc393d103063234e8bc28` and Granite revision
  `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.
- Selected clean external roots before any fresh generation:
  `/workspace/ThermoAgent-v15-reconstruction-b309f0ab` for execution artifacts
  and `/workspace/ThermoAgent-v15-model-cache/huggingface` for model snapshots.
- Added an audited reconstruction wrapper. It runs formal trajectories only
  from a clean checkout of commit `b309f0ab...`, verifies the clean semantic
  source, and substitutes the historical digest only for the in-process
  checksum lookup. It neither fabricates bytecode nor changes scientific
  source or protocol.
- Added a reconstruction comparator that ignores only wall-clock latency and
  requires deterministic agreement of every scientific and non-latency
  accounting column in the committed aggregate tables.

## 2026-08-22: secondary formal extension fixed before reconstructed outputs

The extension is explicitly descriptive and does not alter H1-H4. Its
machine-readable configuration is
`configs/statmech_v15/collective_extension.yaml`.

- Connected correlation uses post-update binary belief vectors, per-agent
  phase means, unordered node pairs, and unweighted shortest paths on each
  trajectory's actual reciprocal modular graph.
- Autocorrelation uses attempted-update-level belief magnetization. The
  primary truncation is two sweeps (`32` updates for `N=16`); one and three
  sweeps are fixed sensitivities. Zero-variance series remain undefined.
- Binder cumulants use attempted-update magnetization within baseline,
  disruption, and recovery phases. A second moment at or below `1e-12` is
  reported as undefined.
- Every observable is computed for a complete model/graph/environment
  trajectory before cluster summaries. Nodes, pairs, and updates are not
  inferential replicates.
- Cluster-bootstrap intervals use 10,000 replicates and seed `15170301`.
- Prohibited interpretations include a thermodynamic-limit transition,
  critical slowing down, a universal critical point, and exact thermodynamic
  entropy production.

## Pending remote stages

The following remain pending until the configured Pod becomes reachable:

1. hardware/disk/CUDA audit;
2. pinned environment installation and model snapshot verification;
3. two engineering pilots;
4. 48 formal frozen trajectories and 34,560 attempted decisions;
5. content-addressed replay of all decisions;
6. committed-versus-reconstructed comparison;
7. extension analysis from reconstructed raw panel trajectories;
8. final figures, manuscript build, PDF QA, and complete test/replay checks.

## 2026-08-22: local audit completion while the Pod endpoint is stale

- RunPod's current connection documentation confirms that exposed TCP port
  mappings change when a Pod resets. Repeated allowlisted SSH probes continued
  to reach the retired mapping and were refused; no fallback credential or Pod
  identifier was available locally. The remote scientific stages therefore
  remain access-blocked rather than failed.
- Independently recomputed H1--H4 from the committed aggregate tables using
  the frozen pairing, exhaustive sign flips, deterministic 10,000-replicate
  cluster bootstrap, and Holm procedure. Maximum numerical disagreement with
  the committed effects was below `1e-16`; every reported p-value and cluster
  count matched exactly.
- Verified disjoint Qwen and Granite panel, graph, and control seed namespaces
  and four-arm seed matching within all 12 model/graph/environment clusters.
  The pooled H2--H4 unit is therefore the prospectively specified
  model-by-graph/environment cluster; the new model-stratified table remains a
  descriptive heterogeneity analysis.
- Reproduced the H3 boundary at block length four and pseudocount one: pooled
  `0.00024188`, Qwen `0.00390737`, and Granite `-0.00342360` nats per attempted
  update. All six Granite arm means and most Qwen arm means lie below their
  matched shuffle floors, so positive paired contrasts are not evidence of
  positive absolute entropy production.
- Added a trajectory-level audit that reconstructs every genuine and
  synthetic displayed history from compact panel data, verifies its checksum,
  entry count, and character count, and tests for future timestamps. The
  synthetic control is explicitly classified as own-opportunity-time,
  deterministic synthetic history rather than a permutation of genuine
  content. Final token/state balance estimates await reconstructed panels.
- Hardened new semantic source hashing against both `__pycache__` directories
  and root-level `.pyc`/`.pyo` files, without changing the audited legacy hash
  used by the frozen formal runner.
- Twenty-eight focused V15 CPU tests pass locally, including synthetic
  correlation, autocorrelation, Binder, history-control, source-hash,
  leakage, end-to-end analysis, reconstruction-comparison, and vector-figure
  tests. Full regression and PDF claims remain pending reconstructed outputs.

## 2026-08-22: focused-suite closure before remote execution

- Thirty-two focused V15 CPU tests now pass after adding deterministic checks
  for seed-namespace independence, reconstruction comparison, extension-figure
  source data, and exclusion of generated LaTeX bibliography files from the
  repository-facing scientific inventory.
- Python compilation, shell syntax checks, and `git diff --check` pass. These
  checks validate code readiness only; they do not substitute for the pending
  fresh-RunPod model pilots, formal trajectories, exact replay, or PDF QA.

## 2026-08-22: isolated repository-regression audit

- The focused V15 suite now contains 33 tests, all passing locally. A new
  reconstruction-base check rejects any change outside V15 paths and the two
  shared setup files changed for the fresh environment.
- A detached disposable worktree was used for the monolithic repository run
  so artifact-generating historical tests could not touch the review tree.
  Of 525 collected tests, 512 passed. Twelve failures require PyTorch, which
  is absent from the workstation's legacy Python 3.8 environment and will be
  rerun in the pinned RunPod environment. One V14-only branch-relative test
  rejects all later V15 paths by construction; it is preserved unchanged and
  replaced in the V15 runner by the reconstruction-base check rather than
  misreported as a V15 regression.

## 2026-08-22: pre-remote publication-integrity audit

- The combined focused V10--V15 runner completed with 189 executed tests and
  no failures. The only deselection is the preserved V14 branch-relative
  immutability assertion whose scope ends at V14; V15 has its own exact-base
  namespace test. Expected robust-covariance warnings remain visible.
- A clean temporary LaTeX build produced an 18-page manuscript with embedded
  fonts, extractable text, resolved references, and no overfull, underfull, or
  undefined-reference warning in the final log. Figures 13 and 14 remain
  placeholders in this pre-reconstruction build because their scientific
  source tables require fresh panel trajectories.
- Synthetic 200-DPI visual inspection found the two extension figures
  readable after moving the graph-correlation legend inside its panel. The
  persistence figure now includes the matched nominal late window alongside
  the three field arms; this changes presentation only, not an estimator or
  scientific contrast.
- Found and repaired a package-integrity sequencing defect: automated PDF-QA
  records are regenerated after the report manifest, and the verifier writes
  its own final attestation. PDF QA now atomically refreshes both manifest
  copies, the self-referential final verification file is explicitly excluded
  from its own checksum index, and verification checks for unindexed and
  unexpected files as well as missing and mismatched files.
- Added a content-addressed raw-generation accounting audit. Every completed
  transition digest must resolve to one model record and agree with completion
  calls, tokens, and latency. Atomic-panel call records left by an interrupted
  attempt are retained as a separate non-scientific compute category instead
  of disappearing from the total. The frozen runner and panel outcomes are
  unchanged.
- The focused V15 suite now contains 35 passing tests after adding the manifest
  re-entrancy and interrupted-record accounting checks.
- The configured RunPod exposed-TCP SSH mapping remains refused after repeated
  safe probes. The Pod may be online, but remote reconstruction cannot begin
  until the local `runpod-thermo` alias is updated to the current mapping.

## 2026-08-23: fresh RunPod reconstruction and compute-ceiling audit

- The user supplied a fresh reachable endpoint. Read-only inspection found an
  RTX 4090 with 24,564 MiB VRAM, driver 580.178.04, Python 3.12.3, about 270 GB
  RAM, and no surviving repository, artifact root, frozen checkout, virtual
  environment, or model cache at any expected V15 path.
- The remote branch and both execution checkouts resolve exactly to
  `b309f0ab76cb24377de5872eebc811582af1f43f`; the parent remains
  `103e4c4598ecc26a98c37a8d03ee3663f9be1070`. The network-mounted volume was
  too slow for working-tree materialization, so two interrupted clone trees
  were preserved under clearly marked external paths and the verified working
  and frozen trees were materialized on Pod-local overlay storage. Symlinks at
  the expected `/workspace` paths expose those exact trees. No scientific data
  were deleted.
- The setup script reconstructed Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
  Transformers 4.55.4, bitsandbytes 0.47.0, and every frozen Python package.
  The missing PDF/LaTeX test dependencies were installed as system packages:
  `poppler-utils`, `latexmk`, the base/recommended/extra LaTeX collections, and
  recommended fonts.
- Fresh setup exposed one repository-hygiene defect: editable installation
  generated `thermoagent.egg-info` in the checkout. The generated metadata was
  moved to the external invalidated-artifact area, and setup now installs a
  site-packages path file instead. The rerun is idempotent and leaves no package
  metadata in Git-facing paths.
- A historical V11 test compiled its manuscript in place. Its generated PDF
  and intermediates were preserved externally and the exact committed V11 PDF
  was restored byte-for-byte from the untouched frozen checkout. Focused V15
  tests pass 35/35 after restoration. Historical artifact-generating tests will
  be run only in a guarded disposable context.
- Exact external model snapshots were resolved and downloaded: Qwen revision
  `a09a35458c702b33eeacc393d103063234e8bc28` (15,242,807,270 bytes) and
  Granite revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`
  (16,346,537,590 bytes). Model weights remain outside Git.
- Both frozen 128-decision engineering pilots passed every engineering gate and
  reproduced the committed transition counts, occupancy, prompt-token counts,
  generated-token counts, validity, and repair counts exactly. Qwen required
  3.938106 seconds per decision and projected 18.902909 formal generation
  hours; Granite required 5.882377 seconds per decision and projected 28.235410
  hours. The combined current projection is 47.138319 generation hours, well
  above the frozen 25-hour ceiling.
- A non-scientific 15-second GPU diagnostic reached P2, 2,535 MHz SM clock,
  10,251 MHz memory clock, 447.42/450 W, and 100 percent utilization. The pilot
  slowdown is therefore not attributable to a simple power, clock, or thermal
  throttle. No formal V15 trajectory has been started pending explicit authority
  to exceed the frozen compute ceiling or a separately authorized scope change.

## 2026-08-23: explicit reconstruction-budget authorization

- The user explicitly instructed the reconstruction to run immediately and not
  request further compute approval. The operational reconstruction ceiling was
  therefore raised to 50 measured generation hours, which covers the
  47.138319-hour two-model pilot projection.
- This authorization changes only the external next-panel GPU-hour guard. The
  frozen protocol file and checksum, scientific execution source, model weights,
  prompts, seeds, panel order, decision ceiling, prompt-token ceiling, invalidity
  ceiling, estimands, and analyses remain unchanged. The wrapper records both
  the frozen 25-hour ceiling and authorized 50-hour reconstruction ceiling in
  the external compatibility manifest.
- The sequential Qwen-then-Granite formal supervisor started at
  `2026-08-23T04:58:04Z`. It writes atomic panel checkpoints and raw call records
  under the fresh external artifact root; Qwen began generating before this log
  entry and no scientific outcome was inspected.

## 2026-08-23: primary-source manuscript check during formal generation

- Current IOP Publishing guidance was checked directly. It permits common TeX
  variants and states that the `iopjournal` class is optional; retaining the
  standard 12-point working manuscript is therefore technically acceptable for
  scientific review.
- The primary arXiv records for the cited 2025--2026 LLM-collective studies
  were independently checked for title, author list, identifier, and date:
  `2510.22422`, `2601.05606`, `2605.10528`, `2608.02827`, and `2608.16578`.
  The Qwen technical report record `2412.15115` and the pinned IBM Granite model
  card were also checked. No unverifiable recent reference was retained.
- Added the original Binder finite-size distribution reference with verified
  DOI `10.1007/BF01293604` to ground the descriptive cumulant definition. This
  does not change any estimator, outcome, or claim disposition.
- Paper builds now fix `SOURCE_DATE_EPOCH`, `FORCE_SOURCE_DATE`, and UTC in the
  build wrapper. Two consecutive clean local builds produced identical main
  and supplement PDF SHA-256 values. First-pass LaTeX warnings visible inside
  the `latexmk` transcript resolve on its later passes; final PDF and log QA
  will be rerun after reconstructed figures replace the two placeholders.
- The out-of-sample kinetic-closure figure was expanded from five shared
  coordinates to the full requested time-resolved comparison: belief and
  action magnetization, belief--action overlap, effective reference energy,
  configuration entropy, susceptibility, correlation-time estimate, and the
  common response distance. Coefficients remain fitted only to the immutable
  V13 microscopic-response data; no V14/V15 quench outcome is used for fitting.

## 2026-08-23: regression isolation and pre-analysis checks

- The expanded V15 focused suite passes 37/37 on the workstation. A broader
  V10--V15 run exposed a Git 2.25 compatibility defect in the disposable sparse
  worktree setup: that Git release does not create the per-worktree `info`
  directory before `sparse-checkout init`. The runner now creates that
  disposable metadata directory explicitly. Historical suites then passed in
  isolation; their known robust-covariance numerical warnings remain warnings,
  not test failures.
- The manuscript currently builds deterministically to 18 main pages and a
  two-page supplement. Final source-data figures, cross-references, PDF QA, and
  visual inspection remain deferred until the reconstructed raw trajectories
  pass replay and the committed-result comparison.
- The fresh environment audit also recorded direct numerical and tokenization
  dependencies that the earlier requirements file left implicit: NumPy 2.1.2,
  joblib 1.5.3, threadpoolctl 3.6.0, and tokenizers 0.21.4. They were already
  present during both engineering pilots and formal generation; the setup and
  verification manifests now pin them explicitly for a future clean rebuild.
- PDF QA now separates automated rendering/font/text checks from the later
  visual disposition. Verification refuses to pass while manual status remains
  pending, and the attestation command rechecks every PDF digest before it can
  record inspection of the original vectors and external 300-DPI renderings.

## 2026-08-23: generation-window integrity checks

- The V15 focused suite now contains 41 passing tests.  The added checks cover
  the Poppler font-table parser, digest-bound manual visual-QA attestation, and
  disposable compilation of both manuscripts with embedded fonts and
  extractable text.  Compilation occurs only in a temporary copy, so a test
  cannot mutate a repository-facing manuscript PDF or leave LaTeX
  intermediates in the review tree.
- A literal whole-repository invocation reached 145 passes and two skips before
  the historical V14 branch-relative immutability assertion rejected the
  presence of later V15 paths.  That frozen assertion is intentionally not
  edited.  The guarded V15 harness instead executes every historical suite in
  a detached sparse worktree and excludes only that one obsolete assertion;
  this distinction will remain explicit in final test accounting.  The first
  guarded run passed 197 tests (41 V15 plus 156 isolated V10--V14 tests); its
  robust-covariance determinant diagnostics are retained as numerical warnings.
- The frozen formal supervisor continues sequential Qwen generation with
  atomic panel checkpoints.  Health monitoring inspects only call counts,
  completed-panel counts, GPU use, and prompt-free error markers; no scientific
  panel outcome is read before the full reconstruction is complete.
- Primary arXiv records again confirmed the recent LLM-collective bibliography:
  2510.22422, 2601.05606, 2605.10528, 2608.02827, and 2608.16578.  Titles,
  author lists, dates, and identifiers in the manuscript match those records.
- The reconstruction comparator's optional-field policy had not been applied
  symmetrically: a derived audit field present only in the post-reconstruction
  JSON could have caused a false mismatch with the older committed package.
  The comparator now aligns optional audit fields on both sides, compares them
  when both packages contain them, and continues to require every frozen
  scientific table plus confirmatory, privacy, call, and token fields.  A
  regression fixture covers both the historical-versus-extended and
  extended-versus-extended cases; the nine-target self-comparison passes.
- The frozen path-reversal calculation retained only the mean of each
  500-permutation time-shuffle floor.  A post-reconstruction reporting audit
  now reproduces the identical raw divergence, floor mean, and adjusted value
  while retaining the empirical null interval, null standard deviation,
  Monte Carlo standard error of the floor mean, and its normal-approximation
  interval.  These columns do not alter H2 or H3; a regression test requires
  agreement with the frozen estimator to machine precision.
- The final PDF workflow previously created external 300-DPI QA renderings
  after the compact external-artifact digest had been assembled.  Verification
  now refreshes the report and repository index after digest-bound manual QA,
  without rebuilding any PDF, so the final external tree checksum covers the
  inspection renderings as well as raw, formal, replay, and analysis artifacts.
- An independent table-level recomputation, separate from the report builder,
  recovered all four committed cluster means, 10,000-replicate bootstrap
  intervals, exhaustive one-sided sign-flip probabilities, and the H2--H4 Holm
  adjustments to numerical precision.  The positive-cluster counts are 6/6,
  11/12, 10/12, and 12/12 for H1--H4, respectively.  This checks the reference
  statistics while the fresh raw reconstruction remains outcome-blinded.

## 2026-08-23: guarded regression verification during reconstruction

- The focused V15 suite passes 45/45 tests in the exact reconstructed RunPod
  environment.  The guarded historical harness also passes every available
  V10--V13 test and every V14 test except the one frozen branch-relative
  immutability assertion that intentionally rejects the existence of any later
  V15 namespace.  The excluded assertion remains unmodified; V15 supplies a
  reconstruction-base immutability check instead.
- Historical V13 logistic-fit deprecation warnings and V14 robust-covariance
  determinant warnings were retained as diagnostics.  They did not produce a
  test failure and do not enter the frozen V15 LLM execution.
- Formal health monitoring remains outcome-blinded.  It records only raw-call
  counts, completed atomic-panel counts, GPU memory use, and prompt-free error
  markers while the sequential Qwen reconstruction continues.
- After the first two atomic Qwen panels completed, the external artifact tree
  was copied incrementally, without deletion, to the workstation scratch path
  `/tmp/ThermoAgent-v15-reconstruction-b309f0ab-backup`.  This is a recovery
  copy outside Git, not a repository-facing result or an additional scientific
  run.  Subsequent completed-panel milestones will refresh it.
- A read-only audit of the committed V14 `INDEX.csv` verified all 142 indexed
  paths against both recorded byte counts and SHA-256 digests with zero
  mismatch.  This check uses only repository-facing provenance because the old
  external V14 artifact root did not survive the Pod replacement.
- The first three completed Qwen panels used 2.412496 measured generation
  hours.  A latency-only extrapolation gives 19.299967 hours for the Qwen half;
  combined with the blinded Granite pilot projection, the two-family forecast
  remains below the user-authorized 50-hour reconstruction guard.  No state,
  effect, or trajectory observable was read for this forecast.
- The separate post-replay analysis checkout now matches the workstation's
  review source at SHA-256
  `1912a7a39a74f7d336dc454c66cce56faec2a72c83e394d1ab3b1c1d74300f31`.
  The immutable formal process continues from the clean frozen semantic tree;
  syncing this review source cannot change a loaded formal panel.  The focused
  V15 suite again passes 45/45 after synchronization.
- All four matched arms of Qwen graph/environment cluster 0 completed
  atomically before cluster 1 began.  Their aggregate generation latency was
  3.217907 hours, giving an outcome-blind 19.307444-hour projection for all 24
  Qwen panels.  The restricted workstation scratch backup was refreshed after
  the complete cluster checkpoint.
- The remaining top-level and V9 repository regressions pass 449/449 in the
  reconstructed environment.  Together with the 201-test guarded V10--V15
  harness, 650 available tests pass; the count excludes only the one frozen
  V14 assertion whose original branch-relative scope rejects every later V15
  path by construction.  No pre-V15 worktree path changed during these runs.
- The workstation reran the focused and guarded V15 regression command while
  formal generation continued.  Its first sandboxed invocation could not
  create the disposable Git worktree because `.git/worktrees` was read-only;
  rerunning the identical command with narrowly scoped filesystem permission
  completed successfully.  The retained scikit-learn covariance warnings are
  unchanged and no test failure occurred.
- All four matched arms of Qwen graph/environment cluster 1 then completed
  atomically, bringing the outcome-blind checkpoint to eight panels and 5,760
  recorded decisions.  GPU memory and temperature remained stable and the
  prompt-free error scan remained empty.  The restricted workstation scratch
  backup was refreshed after this second complete-cluster checkpoint; it
  contains 6,063 files and occupies 37 MiB outside Git.
- A compliance review found that the secondary Binder calculation was already
  phase- and cluster-resolved but did not make its requested temporal-window
  and pooling-rule sensitivity explicit.  Before inspecting any reconstructed
  trajectory outcome, extension version 1.1 superseded the unexecuted 1.0
  draft with deterministic
  full-phase, early-half, and late-half windows and with cluster-mean versus
  pooled-moment summaries.  Both bootstrap constructions resample complete
  graph/environment clusters.  This remains a post-outcome descriptive audit
  of the original V15 data, not a new confirmatory hypothesis.
- The new moment-pooling fixture recovers the exact two-state Binder value of
  2/3 and leaves a zero-second-moment case undefined.  All 46 focused V15 tests
  pass after the addition.  The separate RunPod review checkout and workstation
  analysis source now agree at SHA-256
  `3c7cadbe0b6c94307da6bf06ed84533ede6385f934faf7a014dca0e3acd1984c`;
  the immutable formal checkout and its frozen scientific source are unchanged.
- The extension identifier was then advanced from the unexecuted draft 1.0 to
  `v15-collective-observables-1.1` so the new sensitivity is versioned rather
  than silently folded into its predecessor.  After this metadata-only change,
  the workstation and separate review checkout agree at analysis-source
  SHA-256 `2890b8d2cc494708a03913511290a05810c47270068c169773f7fd89e57402f6`.
- Figure 14's machine-readable source now includes the Binder window and
  pooling-rule sensitivity rows even though the already crowded six-panel
  visualization continues to show the primary phase-level estimate.  The
  small CPU end-to-end analysis checks the expected 216 trajectory-window rows
  and 144 model/condition/phase/window/pooling summaries.  The final tested
  workstation and review-checkout analysis source for this stage is
  `9cd634b6658f798e64c941c4a501243cec729a7ea5468d3b75889660e788f17c`.
- The updated guarded command passes 202 tests: 46 focused V15 tests and 156
  isolated V10--V14 regressions under the documented exclusion of the one
  frozen branch-relative V14 assertion.  The manuscript also compiles cleanly
  to 18 main pages and a two-page supplement.  Final output-dependent tests,
  indexes, and PDF inspection remain scheduled after raw reconstruction.
- Qwen graph/environment cluster 2 completed all four matched arms atomically,
  bringing the outcome-blind formal checkpoint to 12 panels and 8,640 complete
  decisions.  The scratch-only checkpoint guard automatically refreshed the
  restricted off-Pod backup before the next cluster proceeded.  The backup now
  contains 8,949 files and occupies 53 MiB outside Git.
- Qwen graph/environment cluster 3 likewise completed all four arms, bringing
  the formal checkpoint to 16 panels and 11,520 complete decisions.  The guard
  refreshed the restricted backup to 11,838 files and 69 MiB before panel 17
  progressed.  No scientific panel content or contrast was inspected.
- Qwen graph/environment cluster 4 completed all four arms, reaching 20 panels
  and 14,400 complete decisions.  The checkpoint guard refreshed the backup to
  14,734 files and 86 MiB before the final Qwen cluster began.  Health checks
  remained restricted to counts, GPU use, and prompt-free error markers.
- The complete Qwen half finished 24/24 trajectories and 17,280/17,280
  decisions with 17,280 calls, 9,964,681 prompt tokens, 1,385,868 generated
  tokens, zero invalid outputs after repair, and 19.542721 measured generation
  GPU-hours.  No prompt-free error marker was present.  The guard backed up the
  complete half before Granite loaded; Granite's expected higher GPU-memory
  footprint then appeared and its error scan was empty.  Scientific outcomes
  remained sealed through the model-family transition.
- The post-Qwen restricted backup contains 17,612 files and occupies 102 MiB
  outside Git.
- Granite graph/environment cluster 0 completed all four matched arms, bringing
  the outcome-blind checkpoint to 28/48 panels and 20,160/34,560 decisions.
  Its four panels used 4.826418 measured generation hours; together with the
  completed Qwen half, the latency-only projection is 48.501232 hours for the
  unchanged two-model design, below the user-authorized 50-hour reconstruction
  guard.  No state, effect, or trajectory observable entered this projection.
  The scratch-only checkpoint was refreshed to 20,501 files and 118 MiB before
  the next Granite cluster progressed.
- Running the focused suite under the workstation's older Matplotlib exposed a
  portability defect that the pinned RunPod Matplotlib 3.10 environment did not:
  its colormap class lacks a ``copy()`` method.  The review source now uses a
  standard-library shallow copy before changing the heatmap's missing-value
  color.  This changes no data or frozen execution path.  All 46 focused tests
  pass in the older host environment, and the workstation and separate RunPod
  review checkout agree at analysis-source SHA-256
  `5d8f9d30a6c33067494c284674853319924499679d7c9ed0405ef7a9a9a6f0b3`.
- A transient clean-source verifier invocation against the intentionally dirty
  review tree had overwritten its repository-facing diagnostic with an expected
  failure.  That transient output was not scientific evidence and was removed;
  `verification_clean.json` is again byte-identical to the committed frozen
  clean-checkout audit.  Fresh reconstruction provenance remains external until
  replay and the committed-result comparison complete.
- The generated reproduction block now includes the authorized external
  `THERMO_V15_AUTHORIZED_GPU_HOURS=50` guard on both formal commands.  Omitting
  it would make the exact fresh reconstruction instructions stop at the frozen
  historical 25-hour guard on this slower replacement Pod.  The text explicitly
  distinguishes this operational authorization from a scientific protocol
  change.  The focused reporting/reconstruction tests pass 11/11, and the
  synchronized review-source checksum is
  `0206e352b59b87adabcd27a3b694437d8a7541c2783058e207f997d7fdfb45cb`.
- The complete guarded V10--V15 command was rerun after the portability and
  reproduction-command changes.  It again passes 202 tests (46 current V15
  tests plus 156 isolated V10--V14 regressions), with only the previously
  documented Matplotlib deprecations and V14 robust-covariance determinant
  warnings.  Historical suites ran in a disposable worktree and did not modify
  any frozen result namespace.
- The remaining top-level and V9 command was also rerun in the disposable
  regression worktree and passes 449/449.  The available unique regression
  total is therefore again 651 (449 top-level/V9 plus 202 guarded V10--V15),
  with the single frozen V14 branch-relative assertion excluded for the
  previously documented scope reason.
- The completed fresh Qwen half independently reproduces the committed
  non-latency accounting exactly: 17,280 calls, 9,964,681 prompt tokens,
  1,385,868 generated tokens, and zero invalid outputs after repair.  Its
  measured generation latency is 19.542721 hours rather than the historical
  7.807737 hours on the replacement Pod.  This check inspected accounting
  fields only; trajectory states and scientific contrasts remain sealed until
  both model families finish and the scheduled replay begins.
- The five recent LLM-collective references were rechecked against their
  primary arXiv records on 24 August 2026; titles, complete listed authors,
  years, identifiers, and arXiv-issued DataCite DOI metadata agree with the
  bibliography.  The current official IOP/JSTAT page continues to define the
  journal for the broad statistical-physics community.  No unverified venue or
  novelty claim was added.
- At 31 complete panels, Granite stopped while atomically recording the next
  decision because the Pod's per-workspace quota was exhausted.  Global
  filesystem statistics misleadingly still showed 334 TB free: the two exact
  model snapshots occupied about 30 GB of the Pod allocation.  The interrupted
  panel had 197 complete raw calls and no panel CSV; those calls are retained as
  orphaned compute and cannot enter a scientific estimate.  The failed 5,238th
  record was a zero-byte temporary file, preserved with an external compact
  incident record; its token and latency accounting is unavailable.
- No model or scientific artifact was deleted.  The completed 15 GB Qwen cache
  was moved to the container overlay, which had 35 GB free, and linked back
  into its expected Hugging Face cache namespace.  This freed about 15 GB of
  persistent quota while retaining the exact pinned Qwen snapshot for final
  verification.  The Granite snapshot remained unchanged in the persistent
  cache.
- A first resume-script construction expanded shell variables too early and
  exited before loading a model or issuing any decision.  Its 41-byte status
  record is retained with the incident artifacts.  The corrected literal
  script then resumed only Granite through the unchanged frozen wrapper at
  09:55 UTC.  All 31 complete panels were validated and skipped; the incomplete
  panel restarted from its original seed, schedule, graph, and condition.
- The reporting audit now distinguishes the 197 durable orphan calls from the
  single Granite generation that failed before its atomic call record could be
  written.  That additional model call is counted explicitly, while its prompt
  tokens, generated tokens, latency, GPU time, and corresponding cost remain
  unknown.  Reconstruction token and metered-time totals will therefore be
  labeled measured lower bounds rather than treating the unavailable values as
  zero.  Two focused accounting tests pass.  This reporting-only change does
  not enter the frozen prompts, agent transitions, panels, or confirmatory
  analysis; the current review-source checksum is
  `d43b37609d90cd33861bc5b4fe29eada1b3823e4854cdf93c309343abfc3e1e2`.
- A follow-up wording guard makes the lower-bound qualifier conditional on an
  actual unrecorded generation, so clean reconstructions are not mislabeled.
  The reporting behavior and focused incident test remain unchanged; the
  superseding review-source checksum is
  `e44218d33aed2ea5d364467fef50973da430229e4f844dfdeab5b7a597380715`.
- The guarded current-and-historical regression command passes 203 tests after
  the incident-accounting regression was added: 47 current V15 tests and the
  same 156 isolated V10--V14 tests.  The V14 robust-covariance determinant
  warnings remain the known numerical warnings; no frozen historical checkout
  file was changed.
- The restarted Granite arm completed atomically without modifying any of the
  preceding 31 panels.  The reconstruction is therefore at 32/48 formal
  trajectories and 23,040/34,560 retained decisions.  The eight completed
  Granite panels account for 5,760 retained decisions, 5,764 calls (four
  bounded repairs), 3,648,836 prompt tokens, 503,301 generated tokens, and
  9.661756 measured generation hours.  These are accounting fields only; no
  scientific trajectory coordinate or contrast was inspected.
- The outcome-blind off-Pod checkpoint completed immediately at the 32-panel
  boundary.  It contains 23,594 files, occupies 135 MiB, and includes all 32
  atomic panel CSV files plus their durable external records and incident
  metadata.
- Granite cluster 2 completed all four matched arms without another
  infrastructure interruption, bringing the reconstruction to 36/48 panels
  and 25,920/34,560 retained decisions.  The 12 completed Granite panels use
  8,644 calls, 5,469,656 prompt tokens, 754,398 generated tokens, and
  14.600494 measured generation hours.  Together with Qwen, retained measured
  formal generation through this boundary is 34.143215 hours; no scientific
  trajectory value was inspected.
- The 36-panel off-Pod checkpoint contains 26,478 files and occupies 151 MiB.
  All 36 atomic panel CSVs and their durable raw call records are present in
  that recovery copy.
- A pre-outcome wording audit narrowed the Figure 13 catalog claim: the plotted
  contrast compares phases within the persistent-history field arm and does
  not by itself identify a causal memory contrast.  No formal state or effect
  was inspected.  The corresponding review/analysis source checksum is
  `4e3ba87140b72d66c81df1864b23d1297bd0a2c5742708bdd1fe2eb3b92a7397`.
- Granite cluster 3 completed all four matched arms, bringing the reconstruction
  to 40/48 trajectories and 28,800/34,560 retained decisions.  The 16 completed
  Granite panels use 11,524 calls, 7,292,526 prompt tokens, 1,005,566 generated
  tokens, and 19.463296 measured generation hours.  Together with Qwen, the
  retained formal total through this outcome-blind boundary is 39.006017 hours;
  no state, observable, or scientific contrast was inspected.
- The 40-panel off-Pod checkpoint contains 27,940 files and occupies 162 MiB.
  It includes every atomic panel CSV and durable raw call record through this
  boundary, plus the separately preserved quota-failure audit.
- Granite cluster 4 completed all four matched arms, bringing the
  reconstruction to 44/48 trajectories and 31,680/34,560 retained decisions.
  The 20 completed Granite panels use 14,405 calls, 9,118,214 prompt tokens,
  1,256,824 generated tokens, two invalid outputs after repair, and 24.320625
  measured generation hours.  Together with Qwen, retained formal generation
  through this outcome-blind boundary is 43.863346 hours.  No state,
  observable, hypothesis value, or scientific contrast was inspected.
- The 44-panel off-Pod checkpoint contains 32,261 files and occupies 184 MiB.
  It includes every completed panel and durable raw record through the
  boundary, plus the separately preserved quota-failure audit.
- A pre-outcome interpretation audit now states that agent indices in the
  cluster-averaged connected-correlation heatmaps align only by predefined
  community, and that quench-window autocorrelation sums may contain
  nonstationary relaxation.  They are therefore finite-window persistence
  diagnostics, not equilibrium correlation times.  The manuscript also
  identifies the exact sign-flip enumeration as conditional on a sign-symmetry
  null rather than treatment-label randomization.  These reporting changes do
  not enter the frozen execution or hypotheses; the synchronized review-source
  checksum is
  `0464809f0ce1988c65f770f0559187db1258edf4bc97bce11647bc54f2dd586f`.
- The final Granite cluster completed all four matched arms.  The fresh
  reconstruction therefore contains all 48/48 frozen trajectories and
  34,560/34,560 retained attempted decisions.  Formal generation used 34,565
  retained calls, 20,908,194 prompt tokens, 2,893,967 generated tokens, and
  48.737223 measured generation GPU-hours: 19.542721 for Qwen and 29.194502
  for Granite.  Qwen retained zero invalid outputs after repair; Granite
  retained two under the frozen no-action rule.  The final completed-panel
  accounting is exactly equal to the committed reference accounting outside
  latency.  Scientific states and contrasts remained sealed through this
  completion check.
- The outcome-blind checkpoint guard observed zero CUDA memory after the
  worker exited and copied the complete 48-panel artifact tree off the Pod.
  The on-Pod reconstruction occupied 214 MiB across 35,141 files immediately
  before replay and derived analysis.  The quota-failure records remain
  separately retained: 197 durable interrupted-panel records plus one
  post-generation call whose atomic accounting record could not be written.

## 2026-08-25: exact replay, comparison, and outcome release

- Replayed every completed panel through the content-addressed decision
  records: 48/48 trajectory units and 34,560/34,560 retained decision rows
  reproduced with zero state, action, memory, message, or accounting mismatch.
- The first reconstruction comparator used an absolute `1e-12` threshold and
  rejected a maximum cross-platform difference of `1.7337e-12` in a
  covariance-derived value, plus JSON decimal round trips of at most about
  `6.3e-13`. The failed audit is retained externally. The final comparator uses
  only an explicit binary64 machine-relative tolerance (`64*eps`,
  `1.421e-14`) plus `1e-12` absolute tolerance, compares nested numeric JSON
  cluster values, and passes every committed aggregate science and accounting
  output. No raw or derived number was edited to force agreement.
- Independently reproduced H1 `42.2633393 [24.3164388, 59.3034029]`, H2
  `0.0543793 [0.0345859, 0.0754842]`, H3 `0.0484494 [0.0219049,
  0.0752601]`, and H4 `52.5405965 [35.4062727, 68.6731577]`, with the
  frozen exact sign-flip and Holm results. The predeclared model heterogeneity,
  negative arm-level adjusted divergences, block/pseudocount sensitivity, and
  incomplete Granite threshold re-entry remain unchanged.
- Formal retained generation used 34,565 calls, 20,908,194 prompt tokens,
  2,893,967 generated tokens, and 48.737223 measured GPU-hours. Including the
  two successful pilots and 197 durably recorded interrupted calls gives at
  least 49.414869 measured hours. One additional post-generation call has no
  durable token or latency record, so totals are explicitly lower bounds.

## 2026-08-25: secondary collective-observable extension

- Added a versioned descriptive extension without modifying the frozen V15
  protocol or H1-H4. The three implemented quantities are connected belief
  correlation by actual graph distance, attempted-update magnetization
  autocorrelation with a prespecified truncated integrated time, and the
  Binder cumulant of the finite-window order-parameter distribution.
- All estimators are evaluated within a complete trajectory and phase before
  cluster pooling. Zero-variance and zero-second-moment cases remain undefined
  rather than being imputed. One- and three-sweep autocorrelation truncations,
  half-window Binder estimates, and moment-pooled Binder estimates are retained
  as sensitivities.
- Distance-one connected-correlation changes are weak and model-dependent.
  Qwen disruption-minus-baseline is `0.00238 [-0.01750, 0.02277]`; Granite is
  `-0.02032 [-0.05486, 0.00947]`. Qwen recovery-minus-baseline is `0.00080
  [-0.01254, 0.01453]`; Granite is `-0.02748 [-0.05776, -0.00346]`.
- Persistent-minus-Markovized recovery-window integrated autocorrelation is
  `-2.46995 [-5.79214, 0.17724]` attempted updates for Qwen. Only two Granite
  clusters are defined because four have zero magnetization variance; their
  estimate is `-4.14109 [-19.88286, 11.60067]`. These are nonstationary
  finite-window persistence diagnostics, not equilibrium relaxation times.
- Qwen field-Markovized Binder changes are strongly negative under the primary
  finite-window rule (`-4.80504` during disruption and `-3.99611` during
  restoration relative to baseline), while Granite intervals span zero. The
  magnitude is explicitly identified as sensitive to a small second moment,
  window choice, and pooling; no critical point, crossing, or phase-transition
  claim is made.
- Generated source-data-complete Figures 13 and 14. Figure 13 is recommended
  for the main paper as the spatial-organization result; Figure 14 is a useful
  supplementary persistence/order-shape diagnostic because of undefined
  Granite windows and Binder sensitivity.

## 2026-08-25: final regression and presentation audit

- The isolated current-and-historical command passed 207 executed tests: all
  51 V15 tests and 156 V10-V14 tests. One branch-relative historical V14
  immutability assertion is intentionally excluded and replaced by the V15
  base-immutability test. The remaining top-level and V9 suite passed 449
  tests. Total available executed regressions are therefore 656/656, with no
  failure or skip. Only documented scikit-learn numerical/deprecation warnings
  were emitted.
- Inspected all 14 vector figure PDFs, all 21 main-manuscript pages, and both
  supplement pages at original resolution and in external 300-DPI renderings.
  No material clipping, overlap, missing glyph, unreadable legend, empty
  scientific panel, or rasterized text was found. Automated checks cover 16
  PDFs and 37 pages.
- A byte-level rebuild audit initially found that Matplotlib embedded the wall
  clock in PDF metadata. The figure wrapper now pins `SOURCE_DATE_EPOCH` and
  UTC consistently with the manuscript build. Two complete successive builds
  then produced identical SHA-256 manifests for all 62 selected science
  tables, figure source files, figure PDFs, and manuscript PDFs. The final
  review/analysis source checksum is
  `4f23c83ca08a4f4b11253247bf1534ac047892a9c3f6b8ffcad5b97167473bc8`
  over 42 enumerated V15 execution/review source files; it is distinct from
  the frozen formal execution-source checksum.
- Final package verification passes every required check. The repository-facing
  package contains 152 inventoried files and about 13.34 MB. The final external
  manifest covers 35,246 files and 208,149,915 bytes with tree SHA-256
  `c9fc7857daa9672b4d255a2ebeb3d2a7f24ed63df8fb6996d0455d768217ea2e`.
  The recorded analysis-source hash differs from the current review-source
  hash after documented presentation, testing, and verification changes; this
  inequality is reported as informational rather than relabeled as equality.
  Exact replay and aggregate reconstruction comparison remain required and
  both pass.
