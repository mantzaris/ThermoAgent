# V12 research log

## 2026-08-18: provenance and scope

- Fetched `origin` and verified both local and remote
  `evidence-grounded-llm-entropy-v11` at
  `0d73693160c25e251533d6f6720fdd78b349605e`.
- Verified an empty V11 worktree and clean `git diff --check`; no reconciliation
  was required.
- Created local branch `llm-agent-stochastic-thermodynamics-v12` from that
  exact commit. No file was staged, committed, pushed, tagged, or published.
- Audited the V10 exact reference, V10 modular scaling anomaly, V11 evidence
  protocol, V11 qualification output, and the unexecuted V11 formal template.
- Chose a categorical choice-based stochastic process rather than a Bayesian
  calibration study. This decision was made before any V12 nonreciprocity
  outcome existed.

## 2026-08-18: engineering design

- Added narrow V12 ignore rules before generation.
- Defined a state-complete Markovized prompt and a bounded-memory robustness
  regime. Natural-language transcripts remain external.
- Chose two fixed-degree graph families only: ring and modular. The directed
  perturbation is a divergence-free antisymmetric cycle flow, so alpha cannot
  change support, row sums, column sums, or opportunity count.
- Chose actual fixed-size binary packets with CRC framing for byte accounting.
- The initial profile specified approximately 36,000 formal decisions. After
  attempt 2 established a 1.60-second median-scale decision runtime, and before
  any V12 current or irreversibility calculation, the small-system trajectories
  were lengthened from 36 to 60 sweeps to improve finite-state transition
  support. The frozen candidate now contains 40,704 decisions plus a 5% repair
  reserve, with a conservative 20.0 GPU-hour and USD 6.80--13.80 projection,
  below the 24-hour ceiling.
- The collective grid was changed prospectively from a fixed coupling with
  three replications to the complete 2-by-2 combination of coupling
  `{0.35,0.80}` and decoding-noise `{0.50,0.85}`, with two matched graph
  replications and eight sweeps. This meets the requested coupling/noise
  factorial without materially increasing compute. The final candidate has
  42,048 decisions, a 20.7-hour projection, and no inspected V12 formal
  outcome.
- The memory robustness block was prospectively paired with an otherwise
  identical Markovized arm rather than borrowing a nonmatched collective arm.
  This brings the final design to 44,352 decisions (46,570 calls with reserve),
  a 21.8-hour projection, and an estimated USD 7.41--15.04. It remains below
  the hard cap and makes the memory contrast identifiable.
- The first valid pilot design used four reciprocal four-agent panels (128
  attempted updates). Its code records no current, entropy-production,
  path-reversal, or nonreciprocity statistic.

## 2026-08-18: RunPod continuation

- Reached the already authorized Pod through the existing `ssh.runpod.io`
  proxy. Host identity remained `2acac16f37c7`; the RTX 4090 was idle at 1 MiB
  and 0% utilization, with no V10--V12 research process or tmux session.
- The Pod's `/workspace/ThermoAgent` is a deployed source snapshot without a
  `.git` directory, not a competing Git checkout. The local Git repository
  remains authoritative.
- Verified PyTorch 2.8.0+cu128, Transformers 4.55.4, and CUDA availability.
- Transferred only the new V12 package/config/tests as an external temporary
  archive. Tar could not preserve the local uid/gid on the network volume, but
  file extraction succeeded; this is a transport warning, not a scientific
  error. A remote quality test that reads `.gitignore` failed only because the
  deployed snapshot intentionally did not receive the local root ignore file;
  all 20 computational V12 tests passed there. Local tests, including the
  repository-hygiene check, passed 21/21.
- Started the reciprocal-only pilot in persistent tmux session `v12-pilot`.
  No new Pod was created and no lifecycle change was made.

## 2026-08-18: retained pilot attempt 1 failure

- Pilot attempt 1 stopped before its first generation because the deterministic
  tape's namespaced seed exceeded NumPy's accepted 32-bit range. The model may
  have loaded, but zero raw call records and zero scientific decisions were
  produced. The external log is retained.
- Corrected the deterministic mapping to `(namespace_seed * 100000 + update)
  mod (2^32-1)`. This engineering repair preceded every V12 response, current,
  irreversibility estimate, and nonreciprocity comparison.

## 2026-08-18: retained pilot attempt 2 and prospective repair

- Attempt 2 completed 128 reciprocal decisions in 205.2 seconds of measured
  generation latency (133 calls including repair attempts). It used 67,638
  prompt tokens and 10,377 generated tokens in the transition ledger.
- Inspection remained limited to the frozen engineering fields. Latent-plus
  belief and action occupancy were both 0.508; privacy mutations were zero;
  126 messages were delivered; first-pass validity was 0.961. There were three
  transitions in each belief direction and four in each action direction.
- Two decisions remained invalid after one sampled repair, yielding 0.984
  after-repair validity rather than the predeclared 0.990 target. The pilot
  therefore did not pass. No probability current, entropy-production,
  path-reversal, or nonreciprocity statistic was calculated.
- Before any such outcome was available, attempt 3 was defined to address only
  those estimability failures: the bounded repair became greedy, the response
  allowance increased from 112 to 144 tokens, and the reciprocal sample grew
  from four to six panels (192 decisions). All original occupancy, validity,
  privacy, delivery, and bidirectional-transition thresholds were retained.
  Attempts 1 and 2 remain in the external artifact tree.

## 2026-08-18: retained pilot attempt 3 and schema repair

- Attempt 3 completed 192 reciprocal decisions. Occupancy remained centered
  (belief 0.513; action 0.519), both belief directions occurred five times,
  both action directions eight times, 187 messages were delivered, and no
  privacy mutation occurred. It nevertheless failed the unchanged validity
  criteria: 0.948 first-pass and 0.974 after-repair validity.
- A diagnostic inspected only exception classes and validation messages, never
  response content or scientific outcomes. Every one of the five invalid
  records was syntactically valid but selected inconsistent plan-labelled
  values in the redundant `action_choice` and `tool_action` fields, both before
  and after repair.
- Before any current or irreversibility calculation, attempt 4 replaced the
  redundant plan-labelled tools with `execute_selected` or `no_action`. The
  LLM still independently chooses the categorical action; `execute_selected`
  applies exactly that model-selected choice. Choosing the existing action is
  the explicit retain operation. The scheduler derives no scientific action.
  Attempt 4 retains all original sample size and estimability thresholds.

## 2026-08-18: pilot attempt 4 passed and replay was frozen into source

- Attempt 4 completed all 192 reciprocal engineering decisions. First-pass
  validity was 0.9948 and validity after the one bounded repair was 1.0000.
  Latent-plus occupancy was 0.4792 for beliefs and 0.4844 for actions.
  Belief transitions occurred six times in each direction; action transitions
  occurred twelve times in each direction. All 192 valid packets were
  delivered and no unrelated peer-private mutation occurred.
- The attempt used 104,793 prompt tokens, 15,430 generated tokens, and 307.645
  seconds of measured generation latency. Every unchanged estimability target
  passed. Its retained summary explicitly records that entropy production,
  probability currents, time-reversal divergence, and a nonreciprocity effect
  were not computed.
- Before protocol freeze, deterministic replay was strengthened to resolve each
  external LLM interaction by its full content SHA-256, reconstruct the exact
  prompt, seed, temperature, autonomous decision, network delivery, and local
  state transition, and compare every formal table field. Raw prompt and
  completion text remains external. The replay implementation and its negative
  corruption tests are part of the frozen execution-source checksum.

## 2026-08-19: formal completion, exact replay, and analysis repair 1

- The frozen formal run completed 44,352 decisions in 401 units (394 dynamic
  panels), using 44,354 model calls, 24,230,610 prompt tokens, 3,543,392
  generated tokens, and 20.0923 single-GPU hours including model loading. Two
  invalid first-pass objects were repaired greedily; no decision remained
  invalid. The run remained below the prospectively frozen 24-hour ceiling.
- Content-addressed deterministic replay regenerated all 44,352 transitions
  across all 401 units with zero mismatches before any formal effect result was
  opened.
- The first analysis invocation failed before writing its result tables with
  `KeyError: 'replicate'`. The panel-design table contained the prospectively
  generated replicate identifier, but `_panel_statistics` had not copied that
  bookkeeping field into the derived panel-summary row consumed by the frozen
  factor analysis. No raw trajectory, scientific state, estimand, contrast,
  bootstrap rule, hypothesis, threshold, seed, or exclusion changed.
- The failed invocation is retained in the external failure registry. Repair 1
  is the explicit wrapper `scripts/analyze-statmech-v12-repair1.py`: it copies
  `panel_definition['replicate']` into each summary row and then calls the
  otherwise unchanged frozen analysis. It lives outside the frozen execution
  checksum so the formal source remains exactly
  `0796286362ec4dde0eb4f2dc88ecea4c3bf53859e618aec1928ba8b9e8b0a154`.
  A focused self-test was added to the V12 test runner. This repair was defined
  solely from the exception and code path, before any V12 effect estimate or
  confidence interval was inspected.

## 2026-08-19: frozen analysis disposition

- Repair 1 completed the prospectively frozen analysis without changing any
  formal decision, panel, seed, estimator, bootstrap rule, or exclusion. H1
  was supported: the paired neighbor-field response was 0.08333 latent-plus
  choice per unit field (95% cluster-bootstrap CI 0.04167--0.125; 96
  independent information-state clusters).
- H2 was not supported. The alpha=0.8 minus reciprocal bias-adjusted
  length-three block-KL contrast was -0.0002327 nats/update
  (-0.001441--0.001013; 8 small-system clusters) and -0.0007589
  (-0.01115--0.009718; 32 collective clusters).
- H3 and H4 were not supported. The small-system dose slope was +0.0007068
  (-0.0005011--0.002331) and the collective slope was -0.00005391
  (-0.01449--0.01456). A positive small-system quadratic coefficient did not
  outperform the linear model under held-cluster prediction; the collective
  quadratic interval crossed zero.
- H5 was supported as a finite-size response family after Holm correction.
  Stronger coupling increased absolute belief order (+0.03298),
  susceptibility (+0.04550), and correlation time (+2.319 attempted updates).
  Higher decoding noise reduced them by 0.05387, 0.07356, and 5.990 updates,
  respectively. H6 was not supported because collective orientation/size
  intervals crossed zero and the point-estimate direction changed at N=16.
- The primary Markovized small trajectories occupied only 1.80 projected
  states and 3.20 transition pairs on average; collective trajectories
  occupied 5.26 states and 8.04 pairs. First-order closure and early/late
  stationarity diagnostics were substantially weaker collectively. Projected
  Markov entropy production therefore remains secondary, while the frozen
  bias-adjusted block divergence is described only as coarse-grained pathwise
  irreversibility.
- Persistent memory increased adjusted block irreversibility by 0.01790
  nats/update (0.00357--0.03399; 24 clusters). All content, temporal, sender,
  placebo, and no-message control families were retained; with three clusters
  per control, none survived Holm correction.

## 2026-08-19: paper package and quality assurance

- Generated sixteen vector PDF figures from compact source-data tables and an
  18-page JSTAT-oriented manuscript. The bibliography audit checked all recent
  LLM/statistical-physics entries against primary arXiv records and all journal
  DOI metadata used in the manuscript. The manuscript follows the current IOP
  expectations available to this environment: standard LaTeX, sequential
  references, keywords, code/data statement, and an AI-assistance disclosure.
- Figure 9 uses a post-analysis matched panel selected solely so a projected
  current is visible. It is explicitly marked descriptive, excluded from
  inference, and accompanied by a machine-readable selection record. This
  presentation decision does not alter any statistic.
- Original-resolution inspection found two presentation-only overlaps: the
  Figure 12 caveat crossed a fitted-surrogate line, and the Figure 9 current
  label crossed a displayed edge. The caveat was moved above the axes and the
  current label received an opaque backing. Neither change recalculated or
  modified data.
- Automated QA opened all 17 paper-facing PDFs, confirmed embedded fonts and
  extractable text, and rendered all 34 pages (16 figure pages plus 18
  manuscript pages) at 300 DPI outside the repository. Every final figure and
  manuscript page was then inspected at original rendered resolution; no
  clipping, unreadable legend, unresolved overlap, or empty panel remained.
- The first full formal test pass was 102/102 across V10--V12, including 33
  focused V12 tests. A final post-documentation integrity run is recorded in
  `notes/v12_final_status.md` and the external JUnit report.
