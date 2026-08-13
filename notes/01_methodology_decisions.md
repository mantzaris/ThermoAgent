# Methodology decision log

Decisions are recorded when made. Entries marked **provisional** may change
only in response to Stage 0 tests or documented pilot evidence. Entries marked
**frozen** cannot change after the main evaluation begins.

## D001: Separate quantitative dynamics from planner language

- Status: provisional, 2026-08-11.
- Alternatives: free-form text actions; LLM-authored simulator updates; typed
  tool calls against a deterministic simulator.
- Decision: every operational mutation will occur through a typed, validated
  tool call against deterministic state transitions. Language can justify or
  propose an action but cannot itself mutate the world.
- Rationale: resource conservation, replay, privacy tests, and causal
  comparison require a deterministic validator.

## D002: Independent contexts with one shared inference model

- Status: provisional, 2026-08-11.
- Alternatives: one coordinator prompt containing all agents; one model
  process per agent; a shared batched model with separate prompt/memory state.
- Decision: use one frozen inference model for efficiency while maintaining a
  distinct identity, private state, memory, utility, inbox, commitment ledger,
  RNG stream, and planning loop for every agent. Batch inference must preserve
  one input/output record per agent and never concatenate private contexts.
- Rationale: this meets the independence contract without duplicating 7B-class
  weights in VRAM.

## D003: Staged coordination-policy training

- Status: provisional pending throughput and distribution-shift pilot.
- Alternatives: online LLM-in-the-loop PPO; scripted-planner PPO followed by
  LLM evaluation; no learned policy.
- Decision: initially train a small PPO actor/critic using the deterministic
  planner, then evaluate it with the frozen real LLM. A pilot will quantify
  feature/action distribution shift and may add a bounded LLM-coupled
  refinement set. The decentralized actor will receive local features only;
  a global critic is training-only.
- Rationale: online generation in every PPO rollout is computationally wasteful
  and would confound planner and metapolicy learning.

## D004: Candidate primary language model and backend

- Status: installed; structured-output smoke test pending, 2026-08-11.
- Alternatives: Qwen2.5-7B-Instruct with Transformers 4-bit inference; a vLLM
  server; a smaller 1--4B model; another non-gated 7--9B instruct model.
- Decision: test `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28` using Transformers 4.55.4,
  Accelerate 1.10.1, and bitsandbytes 0.47.0 NF4. Prefer this
  mature non-gated model because it fits 24 GB comfortably and has reliable
  instruction/JSON behavior. Use deterministic decoding for comparisons.
- Rationale: the live base image has PyTorch 2.8/CUDA 12.8; preserving that
  stack and adding a lightweight Transformers environment is lower risk than
  replacing Torch for vLLM. Exact versions/revision will be recorded after a
  successful smoke test.

## D005: Statistical-mechanics primary formulation

- Status: provisional until nominal calibration pilot.
- Alternatives: pooled role-normalized occupancy; role-conditioned occupancy;
  rolling-window versions of each.
- Decision: implement all requested estimators and provisionally use pooled
  role-normalized 3 x 3 x 3 macrostates with Laplace alpha 0.1 over a short
  rolling window. Thresholds and healthy ensembles must be calibrated only on
  nominal training episodes. Selection will use nominal stability and
  disruption sensitivity, not final method effects.
- Rationale: 8--14 agents are too sparse for an unsmoothed 27-state snapshot.

## D006: Main design budget rule

- Status: provisional until pilot profiling; rule itself is frozen.
- Alternatives: full Cartesian design; fractional factorial; arbitrary small
  demonstration.
- Decision: run a paired, high-information fractional design spanning both
  applications and the four factor extremes. Preserve at least eight paired
  environment seeds for primary comparisons where measured cost permits. If
  the projected design exceeds 24 GPU-hours, reduce scenario cells and
  secondary ablations before reducing primary paired seeds.
- Rationale: paired seeds and boundary conditions yield more inferential value
  than a broad, under-replicated Cartesian sweep.

## D007: Primary outcomes

- Status: frozen before seeing any experimental method comparison.
- Commercial: service-loss area under the fulfillment curve (lower is better).
- Humanitarian: cumulative unmet weighted need (lower is better).
- Rationale: both integrate disruption magnitude and recovery rather than
  rewarding only final-state performance.

## D008: Honest result retention

- Status: frozen.
- Decision: every launched seed will appear in its manifest and completion
  ledger; failures and timeouts remain analysis rows. No seed will be rerun or
  excluded because of an unfavorable method result. Pilot corrections will be
  logged before protocol freeze.

## D009: Macrostate estimator selected before method comparison

- Status: frozen for pilot and main, 2026-08-11.
- Alternatives tested: pooled versus role-conditioned shrinkage; one-period
  versus three-period rolling windows.
- Decision: pooled role-normalized occupancy with a one-period window.
- Evidence: nominal/moderate scripted seeds 501--503 selected this formulation
  by the preregistered absolute-sensitivity score with nominal-CV penalty. Its
  nominal entropy CV was 0.044 and the moderate-shock entropy shift was +0.061.
- Caveat: its free-energy shift was -0.030. Selection used absolute sensitivity,
  as fixed in advance; a positive free-energy response was not required.

## D010: Private offer constraints and dynamic tool affordances

- Status: revised during pilot and pending final-v3 qualification, 2026-08-12.
- Alternatives: accept any schema-valid LLM tool; silently rewrite inconsistent
  actions; expose tools valid for delivered state and reject inconsistent plans.
- Decision: derive an explicit offer rule inside each agent from only its own
  pending commitment and private reservation price. The LLM must still emit the
  action. A mismatch becomes a logged failure and is never silently rewritten.
  Likewise, coalition response tools are exposed only after an explicit
  proposal; otherwise the available coalition action is `propose_coalition`.
- Rationale: Stage 1 v2 showed correct prose paired with an irrational accept,
  and a fabricated coalition ID. State-dependent affordances preserve agent
  authority while preventing prose/action contradictions from mutating state.
- Pilot correction: counteroffers now preserve resource owner/recipient,
  seller responses use private marginal cost, recipients evaluate the cheapest
  pending delivered offer, and incompatible bargaining terminates after two
  counter rounds. This fixes the observed indefinite reservation/cost loop
  without forcing agreement. Proposed commitments expire at their deadline.

## D011: Shuffled-signal and random-activity controls

- Status: final activation rate calibrated once from paired-v8 before protocol
  freeze, 2026-08-12.
- Decision: the shuffled-entropy control receives another agent's prior-period
  distributed monitor vector. This is causal and parameter matched, but breaks
  both current-event and identity alignment without leaking future state. The
  random gate will activate with a probability fixed from ThermoAgent's pilot
  communication-active decision epochs, not chosen from main outcomes.
- Pilot-v1 calibration: ThermoAgent sent 121 messages over 672 agent decision
  epochs, so the message-attempt-matched probability is `121/672 = 0.1801`;
  configuration uses `0.18`. This is preferable to matching the 0.4003 rate of
  all non-local/non-silent options because emergency option 7 often dispatches
  material without communicating.
- Paired-v8 final calibration: coalition invitations create multi-recipient
  fanout, so raw message attempts divided by decision proposals was 1.274 and
  is not a Bernoulli probability. The final reproducible definition is the
  fraction of independent agent decision epochs with at least one validated,
  policy-originated ordinary message. Paired-v8 observed 624/1,151 =
  `0.542137`. Mandatory entropy sketches and automatic breach/late-delivery
  notices are separate accounting channels and are excluded. This matches
  random gate *activity*, not exact fanout or bytes; realized communication is
  reported and the control is not described as exactly message-count matched.

## D012: Operational-energy sensitivity set

- Status: frozen before the main experiment, 2026-08-12.
- Primary weights remain backlog/shortfall/delay/commitment =
  `(0.35, 0.30, 0.20, 0.15)` (stored in implementation order as
  `(backlog=.35, delay=.20, shortfall=.30, commitment=.15)`).
- Fixed alternatives are balanced `(0.25, 0.25, 0.25, 0.25)`,
  backlog-and-service-heavy `(0.40, 0.10, 0.40, 0.10)`, and
  delay-and-commitment-heavy `(0.15, 0.35, 0.15, 0.35)`.
- These variants are evaluator-only and cannot affect agent actions. For every
  variant, both operational energy and a separately induced healthy-reference
  free-energy gap are saved. They test robustness; no variant will replace the
  primary construct based on final treatment outcomes.

## D013: Pilot-driven metapolicy stabilization and event triggering

- Status: pending final-v3 real-model qualification, 2026-08-12.
- Evidence motivating change: the 96-episode v1 PPO checkpoints selected only
  continue/emergency on 671/672 ThermoAgent pilot epochs; a 384-episode retry
  collapsed both variants onto request-reallocation and was rejected. Merely
  training longer did not solve the failure.
- Alternatives considered: retain the collapsed checkpoints; hand-script the
  final policy; tune against pilot outcome; use local feasibility masks plus
  identical offline initialization and PPO. The selected last alternative
  remains an RL policy and applies exactly the same procedure to entropy and
  no-entropy variants. It was selected for semantic coverage and stability,
  not because ThermoAgent won: on independent deterministic qualification the
  no-entropy candidate was better on humanitarian loss.
- Decision: mask only options structurally unavailable from private/local
  state; behavior-clone a balanced set of scripted local-observation traces;
  run 192 PPO episodes; add a one-epoch imitation anchor after each PPO update.
  Global evaluator state never reaches the actor or demonstrations.
- Event scheduling: periodic decisions remain every four periods, while only
  affected agents replan after explicitly delivered offers, rejections,
  coalition events, commitment failures, or a failed tool. Active actions
  accrue reward until that agent replans. This is both more faithful and less
  expensive than globally replanning every agent after every message.
- Domain semantics: continue executes an accepted commitment or an explicitly
  delivered need, not an uncommunicated global dispatch. Temporary coalition
  membership can grant a bounded recovery route until contract expiry.

## D014: Role-conditioned local-surprisal reference

- Status: fixed before final-v3 qualification, 2026-08-12.
- Alternatives: pooled Gibbs reference for every identity; raw empirical
  per-role occupancy; shrink empirical role occupancy toward the Gibbs healthy
  ensemble. The selected shrinkage estimator meets the specified `q_role`
  boundary while avoiding unstable estimates for one- or two-agent roles.
- Decision: `q_role = 0.5 * p_role,nominal + 0.5 * q_Gibbs`, with Laplace
  `alpha=0.1`. All inputs come from nominal scripted seeds 101--105. This
  affects local surprisal only; global entropy, energy, and free-energy gap
  retain the preregistered pooled Gibbs ensemble.

## D015: Parameter-matched pre-freeze qualification and ablations

- Status: fixed before final-v3 qualification, 2026-08-12.
- Decision: the two final pilot cells evaluate all seven core methods on the
  same applications, agent counts, scenario seeds, horizons, and disruption
  conditions. This adds centralized-LLM, no-communication, and fixed-periodic
  controls without rerunning or relabeling earlier diagnostic pilot cells.
- The ablation cell uses the same four seeds and topology for ThermoAgent,
  learned coordination without entropy, entropy exposed to an LLM with a
  heuristic metapolicy, an otherwise matched actor/planner without episodic
  memory retrieval, fixed and absent communication, an activity-matched
  random gate, shuffled delayed entropy, and the exact-global oracle.
- The `no_episodic_memory` condition suppresses retrieval but retains identity,
  working memory, private observation, utilities, commitments, and independent
  action authority. This isolates memory without violating the independence
  contract.
- Rationale: the matched cell isolates entropy, RL coordination, episodic
  memory, communication, and estimation quality. It was specified before final
  qualification and cannot be selected based on main outcomes.

## D016: Evaluator-only estimator robustness and joint regimes

- Status: fixed during the pre-freeze pilot audit, 2026-08-12.
- Decision: retain the actors' normal distributed Metropolis-gossip estimate,
  and additionally record evaluator-only errors for an exact global reference,
  the current distributed estimate, its one-period lag, an independently
  seeded Gaussian perturbation with standard deviation 0.01 followed by
  simplex projection, and the zero/no-estimate condition. These measurements
  never enter an actor or planner. Top-1 and top-3 local-surprisal source
  localization are recorded.
- Joint operational/interaction regimes use 75th-percentile thresholds fixed
  from final nominal scripted pilot episodes.
- Rationale: the initial outputs contained current distributed error but did
  not explicitly quantify delayed/noisy comparisons or top-k localization.
  Adding evaluator-only fields cannot change a treatment trajectory.

## D017: Legal-information centralized LLM baseline

- Status: corrected before protocol freeze, 2026-08-12.
- Failure found: the initial centralized-LLM implementation constructed coarse
  inventory/need bins directly from simulator state even when the information
  regime marked those values strongly private. Its pilot-v3 rows are retained
  as invalid diagnostics and excluded from analysis.
- Decision: the corrected coordinator receives only `public_identities()`.
  Shared regimes expose declared numeric reports, moderate regimes expose
  declared bins, and strongly private regimes expose identity/role with values
  marked `unreported`. It may select a dispatch, but the normal tool validator
  can reject infeasible choices; no hidden state repairs the action.
- Validation: add matched nominal and compound `paired_*_v5` pilot cells before
  freeze. This correction is based on privacy validity, not method performance.

## D018: Independent deterministic random-number streams

- Status: corrected before protocol freeze, 2026-08-12.
- Failure found: the original simulator used one seeded generator for
  initialization, private forecast noise, packet delivery, production, and
  demand. Because methods send different numbers of messages, action-dependent
  communication draws shifted later exogenous demand and production draws.
  Matching the nominal seed therefore did not guarantee a paired exogenous
  trajectory.
- Alternatives considered: pre-sample every stochastic trajectory; use a
  counter-based generator keyed by event; or use deterministic purpose-specific
  streams. Purpose-specific streams are the smallest auditable correction and
  preserve the existing simulator API.
- Decision: derive four documented streams from every environment seed:
  initialization (`+0`), exogenous dynamics (`+1,000,003`), observation noise
  (`+2,000,003`), and communication (`+3,000,017`), modulo `2^32`. Run
  manifests record all derived seeds. Actions may alter endogenous state, but
  message and observation draw counts can no longer alter the exogenous stream.
- Validation: an automated paired-trajectory test injects extra messages into
  one copy of an otherwise identical simulation and verifies equal demand,
  cumulative demand, production, and inventory in the absence of operational
  tool actions.
- Consequence: the 19 completed `final_*_v3` rows are retained as invalid
  diagnostics and excluded by the prospective registry in
  `results/reproducibility/excluded_runs.json`. Calibration, PPO checkpoints,
  and qualification are regenerated after the correction. New qualification
  IDs use `paired_*_v5`; no outcome magnitude or direction informed this
  exclusion or naming decision.

## D019: Cache immutable run provenance within each sweep process

- Status: corrected before paired-v5 qualification, 2026-08-12.
- Finding: the first full mock preflight re-ran dependency imports, CUDA
  hardware queries, Git provenance, and the complete source checksum for every
  episode manifest. After 23 valid mock episodes, a repeated driver query
  waited indefinitely with no GPU workload. This is needless fragility for a
  944-episode main sweep.
- Decision: memoize the dependency map and hardware record once per process and
  memoize Git/source provenance by resolved repository root. Every episode
  still embeds the same full values; a resumed process recomputes them, and a
  new sweep process necessarily observes any source change.
- Validation: the partial mock preflight is restartable by run ID. After the
  change, rerun all sweeps, replay, analysis, ten figures, and PDF QA from the
  same output directory. This change cannot alter simulator or planner state.

## D020: Pilot-driven planner schema compression and executable transport arcs

- Status: corrected before paired-v5 qualification, 2026-08-12.
- Evidence: the 19 invalid-for-outcome final-v3 rows remain valid planner
  diagnostics. Fixed/no-communication planners produced nearly all valid JSON,
  but the learned variants selected harder offer/coalition options: structured
  validity was 267/350 for no-entropy and 89/128 for ThermoAgent, with repeated
  128-token truncation/recovery, stale deadlines, one-period coalition expiry,
  extra fields, and message-kind names mistaken for tools. Valid tool rates were
  consequently poor. This was analyzed without using treatment outcome values.
- Prompt alternatives: parser-side mutation of model actions was rejected
  because it would conceal semantic failures; simply increasing generation
  length would increase cost without addressing stale private guidance. The
  chosen v4 prompt puts exact current time bounds, private utility guidance,
  and the permitted one-tool schemas first; constrains both text fields to 12
  words and the response to 90 tokens; expands input capacity from 2,048 to
  2,560 tokens; and raises the generation ceiling from 128 to 160 only as a
  guard against truncation. No invalid domain action is auto-repaired.
- Transport audit: carrier/transport roles owned bounded initial operating
  stock and were authorized to schedule/transfer it, but the physical graph
  gave them no outbound route. This made every such call fail regardless of
  planning. Transport roles are now explicit non-producing resource owners
  with outbound demand arcs; centralized and autonomous controllers see the
  same graph. Automated commercial and humanitarian tests execute these arcs.
- Consequence: prompt revision is frozen as `planner-json-v4`. The environment
  graph change requires nominal recalibration and identically initialized PPO
  retraining before qualification. Selection was driven by capability validity,
  not comparative logistics performance.

## D021: Manifest-before-publication episode transaction

- Status: corrected before paired-v5 qualification, 2026-08-12.
- Finding: intentionally stopping the first mock preflight exposed a narrow
  window in which `episode.json` could exist before its required manifest. Two
  of 128 mock rows then failed replay because the resume path treated the raw
  file alone as complete.
- Decision: write episode/event files into `results/.staging/<stage>`, write
  the checksum-bearing manifest, and only then atomically rename the staged
  directory into `results/raw/<stage>`. An interrupted staging directory is
  retained but excluded from analysis and Git. Legacy complete episodes that
  lack manifests are recorded as failures and are never silently rerun.
- Validation: a fresh exact-source mock preflight must pass all episode replays
  before the real paired-v5 pilot is launched.

## D022: Bounded GPU provenance query

- Status: corrected before paired-v5 qualification, 2026-08-12.
- Finding: a fresh preflight showed that even the first cached call to
  `torch.cuda.is_available()` can wait indefinitely in this container after
  repeated CUDA process initialization. Manifest creation must never depend on
  an unbounded driver call.
- Decision: record PyTorch and compile-time CUDA versions without initializing
  CUDA, and obtain GPU name, memory, and driver through `nvidia-smi` with a
  ten-second subprocess timeout. A timeout is recorded as unavailable in that
  episode rather than hanging the experiment. Full hardware capture and the
  CUDA/model smokes remain separate authoritative artifacts.

## D023: Isolated Python PDF validation fallback

- Status: fixed before paired-v5 qualification, 2026-08-12.
- Finding: all 128 mock episodes replayed and all ten vector PDFs generated,
  but the container no longer included Poppler's `pdfinfo`, `pdffonts`, and
  `pdftoppm`; mechanical QA correctly failed rather than claiming success.
- Alternatives: install a global OS package or keep QA within the isolated
  project environment. PyMuPDF 1.28.2 was the current PyPI release on
  2026-08-12 and provides PDF opening, font enumeration, metadata, and 150-DPI
  rendering without mandatory external dependencies.
- Decision: pin `PyMuPDF==1.28.2` and use it only when all three Poppler tools
  are unavailable. The report records the backend for every figure. A unit test
  verifies open/font/render behavior; visual review remains a separate required
  step.

## D024: Mock preflight is engineering evidence, not research evidence

- Status: completed before paired-v5 qualification, 2026-08-12.
- Evidence: the exact-source bounded preflight completed 128/128 deterministic
  mock episodes and 128/128 quantitative replays. It generated all ten required
  vector PDFs. PyMuPDF opened each PDF, enumerated fonts, and rendered a 150-DPI
  preview. Visual review found and corrected label overlap or clipping in the
  architecture, monitoring, ablation, and network figures; a second inspection
  passed all ten layouts.
- Decision: use this run only to validate restartability, replay, analysis,
  plotting, mechanical PDF checks, and the visual-QA workflow. Mock episode
  values are never included in scientific comparisons. Every final research
  figure will undergo a new mechanical and visual review after real data replace
  the preflight data.

## D025: Prospective pilot-to-main resource projection

- Status: specified while the paired qualification was running and before its outcomes are
  inspected, 2026-08-12.
- Decision: `scripts/profile-budget.sh` will require all 84 valid paired-v8
  episodes and zero conservation failures. For each core method it records the
  empirical mean and 90th percentile of episode wall time, calls, and tokens.
  The strengthened centralized-LLM baseline additionally scales by three
  demand-assignment slots, and the full-information controller by its fourfold
  increase from five pilot decision epochs to every-period replanning. Other
  cells scale by agent count and decision epochs. Parameter-matched ablations
  use their named core-method analogue. Model-load overhead and raw-ledger size
  are added.
- Budget rule: the conservative sum of method-specific 90th-percentile
  projections across main, ablation, and holdout must not exceed 24 GPU-hours.
  If it does, reduce the matrix before freeze without consulting method outcome
  direction. The report includes advertised RTX 4090 starting-rate assumptions
  of USD 0.34 and 0.69 per hour, checked 2026-08-12; actual console pricing is
  authoritative.
- Communication match: set the random-gate probability once to the fraction of
  paired-v8 ThermoAgent agent-decision epochs with at least one validated
  policy-originated ordinary message, then freeze it. Monitor sketches and
  automatic notices are excluded. This pre-main correction is required because
  coalition fanout makes messages/proposals exceed one and therefore invalid as
  a Bernoulli parameter. No main outcome informs the value.

## D026: Actor monitoring must be locally observable

- Status: corrected before any main run, 2026-08-12.
- Finding: the distributed estimator used coarse sketches, but the actor field
  called `consensus_error` was evaluator RMSE against exact global occupancy.
  The interaction-entropy field likewise used the evaluator's global graph.
  Neither value was locally observable. No-entropy and random-gate controls
  also retained global interaction entropy in planner context.
- Decision: actors now receive neighbor-disagreement residual computed only
  from estimates exchanged on their current links. An isolated agent sees a
  no-neighbor uncertainty marker; a disconnected component cannot know that a
  different component disagrees. Actor interaction entropy uses only its own
  explicit inbox/outbox with temporal decay. No-monitor controls receive zeros
  for all six monitoring fields. Exact global errors remain analysis-only.
- Communication implementation: Metropolis gossip samples each round's links
  from a purpose-specific RNG stream at the configured reliability. Every
  agent's link-local round update and neighbor set is an explicit private
  `macrostate_sketch` event. Monitor RNG draws cannot perturb operations.
- Consequence: paired-v5 stopped after 16 published rows (one learned row),
  before comparing outcomes. All v5 rows are prospectively excluded and
  retained. Policies are retrained from the same initialization; the clean
  replacement is separately named paired-v6.

## D027: Information regime must not change the underlying economy

- Status: corrected before any main run and before paired-v6 outcome review,
  2026-08-12.
- Finding: a factorial audit found that `private_information` multiplied the
  simulator's true marginal costs and increased the noise in each agent's own
  private forecast. The intended factor is what other organizations or a legal
  coordinator can observe, so these changes would confound observability with
  a harder economic state and poorer local information.
- Alternatives: model privacy as a compound difficulty construct, attempt a
  post-hoc adjustment, or isolate disclosure. The first changes the research
  question and the second cannot guarantee causal separation.
- Decision: the scenario seed now fixes identical costs, demand, capacity,
  utility (at fixed objective regime), exogenous dynamics, and local forecast
  quality across privacy levels. `private_information` controls only exact,
  coarse, or absent sharing through the public-information interface. A test
  compares complete initial states and private observations at privacy 0 and 1.
- Consequence: paired-v6 was interrupted after 13 atomically published rows,
  before comparing method outcomes. All rows and logs remain retained and are
  prospectively excluded. Nominal calibration and both matched policies will
  be regenerated, and the clean qualification is named paired-v7.

## D028: Failure-aware reporting supplements complete-case effects

- Status: fixed before paired-v7 and protocol freeze, 2026-08-12.
- Decision: numeric mean differences and confidence intervals remain paired
  complete-case estimates because humanitarian weighted unmet need has no
  common finite failure ceiling. A separate failure-aware paired table includes
  every planned matched row: a single-method failure is a loss for that method,
  two failures or equal outcomes are ties, and two complete episodes are ranked
  by the preregistered lower-is-better outcome. Completion rates are reported
  by stage, application, and method.
- Rationale: silently dropping asymmetric failures could favor an unreliable
  system; inventing a common numeric penalty across two differently scaled
  primary outcomes would be arbitrary. Both views are retained explicitly.

## D029: All communication channels share one partition onset

- Status: corrected before any main run and before paired-v7 outcome review,
  2026-08-12.
- Finding: ordinary message delivery applied the partition probabilities from
  period 0, while the entropy-sketch graph remained connected until the
  disruption step. An agent could therefore observe a reliable gossip channel
  and an already partitioned ordinary channel in the same period.
- Decision: a partition is a time-varying communication shock. Before the
  common disruption onset, both ordinary messages and sketch links use 0.98
  reliable delivery. At and after onset, both remove cross-partition links and
  retain within-component links at 0.85 reliability. The agent's private
  reliability observation is computed from this same function. A regression
  test checks both channels immediately before and at onset.
- Consequence: paired-v7 was interrupted after 10 atomically published rows,
  before outcome comparison. The exact configuration, rows, and logs are
  retained and excluded. PPO policies are rebuilt because partition timing is
  part of their training distribution; nominal calibration is unaffected. The
  clean replacement is paired-v8.

## D030: Distributed monitoring traffic is not free

- Status: fixed before paired-v8 and protocol freeze, 2026-08-12.
- Decision: the monitor retains a low-bandwidth channel separate from each
  agent's bounded domain-negotiation budget, because mandatory consensus rounds
  cannot be avoided by choosing a silent coordination option. Every directed
  edge-round sketch transmission is nevertheless counted, along with the byte
  length of its deterministic compact 27-state payload. Results report
  operational messages, sketch messages, and combined messages/bytes; the
  communication Pareto figure uses the combined count.
- Scope: only methods that consume a distributed monitor pay this accounting
  cost. Shadow evaluator estimates for no-monitor controls do not. The exact
  global oracle remains explicitly unattainable and is not assigned fictional
  gossip traffic.
- Related provenance: shared/coarse operational dashboards are now logged as
  explicit public-signal events, and complete run manifests include the exact
  coordination-checkpoint SHA-256 when a learned policy is used.

## D031: Agentic quality metrics require completed social actions

- Status: corrected before protocol freeze and before inspecting paired-v8
  outcomes, 2026-08-12.
- Finding: the provisional `coalition_formation_rate` treated a unilateral
  proposal as formation, and individually rational agreement checked only the
  party that called `accept_offer`. Both would overstate completed social
  coordination.
- Decision: a coalition forms only after at least one invited organization
  independently executes a successful `join_coalition`; proposal count remains
  a separate metric. Useful-coalition precision and recall use the first actual
  join, not the proposal. A priced agreement is individually rational only if
  its price covers the resource owner's private marginal cost and does not
  exceed the recipient's private reservation price. A zero-price humanitarian
  pledge represents voluntary owner consent and still checks recipient utility.
- Scope: these are evaluator definitions and do not enter any actor feature,
  reward, prompt, or simulator transition. Paired-v8 remains valid for its
  prespecified behavior/throughput qualification, but its provisional versions
  of these two descriptive metrics will not be pooled with main results.

## D032: Strong central comparators and deployable necessity boundary

- Status: fixed before protocol freeze and before inspecting paired-v8 outcome
  direction, 2026-08-12.
- Finding: the provisional full-information controller shared the ordinary
  four-period decision interval, and the legal central-LLM coordinator could
  propose only one small dispatch per epoch. Those action-bandwidth limits made
  the central comparisons weaker than their intended scientific roles.
- Decision: the full-information numerical controller is an explicitly
  unattainable receding-horizon upper bound. It observes exact costs, routes,
  inventories, in-transit material, priorities, and demand and replans every
  period, but it still uses the same validated routes, inventory, handling
  capacity, and lead times. The central LLM receives only exact, coarse, or
  absent reports legally exposed by the information regime and gets one typed
  dispatch slot per reported demand organization, batched through the same
  frozen model. Under strong privacy it receives no operational reports and can
  only pause.
- Figure boundary: the agentic-necessity map compares ThermoAgent with the best
  deployable legal-central-LLM or scripted-independent comparator. The
  full-information controller remains in performance figures/tables as an
  oracle bound and is not mislabeled as deployable under privacy.
- Validation: the local suite now passes 93 tests, including every-period
  upper-bound replanning, legal coordinator assignment slots, strict offer and
  coalition targets, two-party rationality, shipment authority, and post-arrival
  delivery verification. A real-Qwen central-planner smoke is required after
  paired-v8 completes and before freeze.

## D033: Figure tooling must support the documented local environment

- Status: fixed before protocol freeze, 2026-08-12.
- Finding: the fresh disposable end-to-end preflight completed all three mock
  matrices, replay, and statistical analysis, but local Matplotlib 3.3 lacks
  `Figure.supxlabel`; figure generation therefore stopped after six PDFs.
- Decision: use the stable `Figure.text` API for the one shared ablation-axis
  label. This is a rendering-only compatibility correction and cannot affect an
  episode, metric, treatment comparison, or analysis result.
- Validation: regeneration produced all ten required vector PDFs. Each opened,
  exposed embedded/subset fonts to Poppler, rendered at 150 DPI, and passed a
  full contact-sheet visual inspection. The final research figures will receive
  the same mechanical and manual QA after the actual analyses are rebuilt.

## D034: Central privacy and paper figures fail closed

- Status: fixed before protocol freeze and while paired-v8 outcomes remained
  sealed, 2026-08-12.
- Central privacy: an adversarial coordinator-planner regression showed that
  the ordinary mock obeyed the no-report/no-dispatch instruction, but the
  executor did not independently enforce it. A typed `central_dispatch` is now
  rejected unless that request slot has an exact legally reported demand
  target. Coordinator structured-output hashes and every coordinator result,
  including a no-op or validation failure, are event-sourced.
- Necessity map: choosing the lower central/scripted loss separately for each
  seed would create a clairvoyant ensemble. The map now selects one comparator
  by its across-seed mean in each factor cell, with a deterministic tie-break,
  then preserves the original seed pairing. It is labeled the best *observed*
  fixed deployable comparator; this descriptive selection is not an inferential
  test.
- Network figures: coalition outlines and response labels now require a
  successful independent `join_coalition` event. A unilateral proposal remains
  visible as negotiation traffic but cannot be depicted as a formed coalition.
- Validation: two adversarial and reconstruction regression tests were added;
  the current local suite contains 96 tests.

## D035: Stage 1 coalition evidence requires independent consent

- Status: corrected before protocol freeze, 2026-08-12.
- Finding: the retained Stage 1 v4 harness called a successful unilateral
  proposal `coalition_observed`. After D031 fixed the research metric, that is
  insufficient evidence that a temporary coalition actually formed.
- Decision: the harness now lets the proposal traverse the ordinary reliable
  communication channel, selects an invitee that actually received it, gives
  that organization its own stressed private observation, and asks its separate
  LLM context to join or refuse through the typed tools. The completion gate
  requires `coalition_joined`; the simulator never inserts membership.
- Validation plan: the deterministic mock harness passes every gate and a new
  regression test verifies that exactly one independently authored join is in
  the record. A uniquely named real-Qwen Stage 1 v5 rerun is mandatory before
  freeze; the older v4 record remains valid for its other eight capabilities but
  is no longer cited as formed-coalition evidence.

## D036: Paired-v8 qualification and prospective resource freeze

- Status: fixed after all 84 paired-v8 rows and their immutable replay passed,
  and before any main episode, 2026-08-12.
- Outcome-independent launch decision: retain the complete 944-main +
  72-ablation + 80-holdout design. The empirical p90 projection is 20.2692
  GPU-hours, below the prespecified 24-hour ceiling; its expected projection is
  16.7201 GPU-hours. No cell was removed based on pilot effect direction.
- Random control: fix `random_gate_probability=0.542137`, from 624 of
  1,151 ThermoAgent decision epochs with a validated ordinary message. Raw
  coalition fanout, mandatory gossip sketches, and simulator notices are not a
  Bernoulli gate and are accounted separately.
- Interpretation rule: paired-v8 is a three-seed qualification/pilot, not a
  confirmatory estimate. Its mixed or negative outcomes are retained in full.
  The main protocol is not tuned to reverse them. Pre-freeze evaluator-metric
  and central-control corrections documented in D031--D034 apply prospectively
  to main and do not rewrite the paired-v8 episode records.

## D037: Coalition member lists are invitees, not current members

- Status: corrected after the retained real-Qwen Stage 1 v5 failure and before
  protocol freeze, 2026-08-12.
- Finding: Qwen correctly selected `propose_coalition` twice but included the
  proposing warehouse in `members`. The strict environment rejected both with
  `self_member`; all other Stage 1 gates passed. The prior prompt removed self
  from public candidates and said never to target self, but did not define the
  coalition field as an invitee-only list.
- Decision: prompt revision `planner-json-v5` explicitly states, in both private
  action guidance and the system instruction, that the proposer is already a
  member and `members` must contain only other IDs from a concrete
  `eligible_invitee_ids` list. The simulator remains strict and no output is
  rewritten. Main, ablation, holdout, and focused real-model smoke manifests
  use the new revision.
- Validation: the failed v5 output, log, and exit code remain under their unique
  path. The 100-test suite passes after the prompt-only correction; a uniquely
  named v6 real-Qwen rerun must demonstrate an independently accepted join.

## D038: Legal central coordinator receives public route eligibility

- Status: corrected after focused real-model smoke and before protocol freeze,
  2026-08-12.
- Finding: prompt-v5 exact public reports produced three valid dispatches and
  absent reports produced no mutation, but the coarse-report coordinator chose
  a source without a route in two of three calls. The coordinator had legal
  operational reports but not the public physical topology needed to choose a
  feasible source. This made the intended strong deployable comparator weaker
  for an information reason unrelated to private state.
- Decision: `planner-json-v6` adds `eligible_source_ids` for each assigned
  demand, computed only from public physical edges. Candidate sources are
  restricted to this list, the prompt must select from it, and the executor
  independently rejects any outside source with `coordinator_source_route`.
  Inventory, capacity, demand, impairment, costs, and disruptions retain the
  exact/coarse/absent visibility imposed by the information regime.
- Validation: an adversarial regression verifies that a known but route-
  ineligible source cannot reach a domain tool. The prompt-v5 smoke and its
  3/3 replay remain retained; a separately named public-route v2 real-Qwen
  smoke is required before freeze.
- Result: public-route v2 completed and replayed 3/3. Exact and coarse reports
  each yielded three successful route-feasible shipments with 100% structured
  and tool validity. Absent reports exposed zero organizations and produced no
  domain tool call; three attempted blind dispatches were rejected with
  `coordinator_no_public_demand`. This is a pass of the privacy/execution gate
  and also retains the model's refusal-to-pause weakness as an agentic metric.

## D039: Immutable protocol boundary

- Status: frozen before the first main episode, 2026-08-12.
- Decision: freeze the complete main, ablation, and holdout configurations;
  prompt revision `planner-json-v6`; simulator, runner, analysis, replay, and
  figure code; calibration; both PPO checkpoints; dependency declarations; and
  execution scripts. The manifest is non-overwriting and its rule invalidates a
  main evaluation if any listed checksum changes after launch.
- Verification: local and remote source checksums both equal
  `4b76671d2d1cbaa7b213d2b11917ab02a440d91b3d46d5646dfacdf934599c55`.
  All 36 listed file hashes verify in both environments. Documentation and
  generated result artifacts may continue to accumulate, but frozen source,
  prompts, configurations, calibrations, checkpoints, and scripts must not be
  edited.
- Outcome rule: retain every completed, failed, and timed-out main or holdout
  seed. Do not selectively rerun an unfavorable row, and do not change the
  protocol in response to observed treatment direction.

## D040: Post-freeze figure polishing is presentation-only

- Status: adopted after all final outcomes were analyzed, 2026-08-13.
- Finding: the frozen generator produced all ten mechanically valid PDFs, but
  manual inspection found colliding Pareto x-axis labels and coalition outlines
  clipped by network subplot limits. These are presentation failures under the
  preregistered visual-QA rule, not simulation or statistical defects.
- Alternatives: (a) accept unreadable figures; (b) edit the frozen figure
  module and invalidate its protocol hash; (c) add an explicitly post-freeze
  save-time layout wrapper that changes no data, statistic, selected episode,
  or plotted encoding.
- Decision: use option (c).
  `results/reproducibility/tools/polish_figures.py` calls the frozen plotting
  functions and intercepts only their final save step to wrap Pareto labels,
  contain its legend, and expand network limits. It writes
  `results/reproducibility/postfreeze_figure_polish.json`. No listed protocol
  file changed; `verify-protocol` must continue to pass.
- Validation: the three PDFs were regenerated, all ten PDFs passed PyMuPDF
  open/font/render checks, and every 180-DPI preview passed original-resolution
  manual inspection. The QA report records the reviewer, note, and passed state
  for each figure.

## D041: No post-hoc second-model experiment

- Status: final, 2026-08-13.
- Decision: do not add a second open-weight model after outcomes were unsealed.
  The complete post-freeze design used 18.592 GPU-hours and the retained study
  used about 20.17 summed episode-hours. A new model would require a new prompt
  qualification, token/throughput profile, and separately labeled exploratory
  matrix close to the 24-hour planning ceiling. It would not strengthen the
  locked primary inference and could invite outcome-responsive exploration.
- Consequence: cross-model robustness remains a clearly named extension. The
  final evidence is about Qwen2.5-7B-Instruct at the immutable revision only.
