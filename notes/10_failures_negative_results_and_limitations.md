# Failures, negative results, and limitations

## Paired-v8 negative coordination result

The first fully eligible three-seed paired pilot did not show a stable logistics
benefit from entropy/free-energy features. ThermoAgent and learned coordination
without entropy were essentially tied, while the legal centralized LLM,
privileged lookahead, scripted independent agents, and no-communication agents
usually performed better. ThermoAgent formed many accepted coalitions but paid
more calls, invalid actions, ordinary messages, and mandatory sketch traffic.
This finding is retained without checkpoint, seed, or scenario selection. It
may mean that the studied cells do not require autonomous negotiation, that the
coordination policy over-negotiates, or that coalition actions are insufficiently
connected to material recovery. The frozen factorial main test distinguishes
these interpretations; it is not configured to guarantee a favorable result.

## PPO collapse diagnostics

- The original 96-episode actors coupled successfully to Qwen but collapsed to
  continue/emergency. ThermoAgent's apparent v1/v2 advantage was largely an
  emergency-dispatch difference rather than rich negotiation.
- A 384-episode retry collapsed both variants onto request-reallocation and
  worsened tool failures. It was not promoted.
- Initial unbalanced behavior-cloning and PPO-anchor candidates still favored
  request-reallocation or omitted negotiation on rollout. They were rejected
  using a predeclared semantic-coverage criterion, not comparative performance.
- The final deterministic qualification is mixed: the entropy candidate was
  slightly worse commercially and materially worse in humanitarian weighted
  unmet need than no-entropy. The main evaluation will retain this possibility.

## Bargaining failure found during pilot

The first counteroffer implementation could oscillate indefinitely between a
buyer's reservation price and a seller's marginal cost. It also let message
arrival order select the last offer rather than the cheapest delivered offer.
The final pre-main protocol limits incompatible bargaining to two counter
rounds, preserves resource direction, chooses the economically relevant
pending offer using only private/delivered information, and permits rejection.

This file is append-only in spirit: resolved failures remain documented.

## Stage 1 actual-join failure

The first stricter actual-join run (`stage1_v5_actual_join`) passed every
non-coalition gate but failed because Qwen treated the coalition `members`
field as including the proposer. The strict environment rejected both attempts
with `self_member`. The run remains an exit-1 artifact. Prompt revision v5 now
defines `members` as invitees only and enumerates legal invitee IDs; the
environment still rejects self, duplicate, and unknown IDs and never edits an
LLM proposal.

## Central-LLM public-route smoke failure

The first focused legal-coordinator smoke preserved the privacy boundary but
failed two of three coarse dispatches with `no_route`: the prompt exposed legal
coarse reports but omitted the public physical graph. That made the baseline
unnecessarily weak. The run and replay remain retained. Prompt v6 now supplies
only public route-eligible source IDs and the executor rejects any other source;
no private operational field was added.

## Infrastructure failures

1. A documented direct SSH mapping returned `Connection refused` on
   2026-08-11. The Pod itself was subsequently reached through the
   operator-supplied RunPod proxy, so the failure was a stale mapping rather
   than evidence of a stopped Pod.
2. The operator-supplied command named a local identity file that was absent;
   another existing operator-managed identity authenticated successfully. No
   identity material was copied or recorded in Git-facing artifacts.
3. The RunPod proxy rejects non-PTY command execution. Existing `rsync` and
   non-interactive execution scripts could not target it directly. This was
   resolved by reading the Pod's refreshed direct mapping and adding optional
   port/identity/known-host overrides to the scripts.
4. `uv pip` did not treat the inherited system PyTorch as satisfying dependency
   resolution and began downloading a redundant CUDA 13 stack. The attempt was
   stopped; its partial venv never passed import checks. Standard pip inside a
   clean system-site venv preserved PyTorch 2.8.0+cu128 and passed all tests.
5. The first real-model smoke returned syntactically valid tool JSON but one
   supplier requested a quote from itself with a deadline 500 periods away.
   Static schemas alone were insufficient. The initial artifact is retained;
   the simulator now rejects self-targeting and deadlines outside a relative
   six-period horizon, and the planner prompt excludes self from candidates.
6. In the corrected smoke, one Qwen justification called inventory of 53.4
   units "below" capacity of 7.7. The action itself was valid, but concise
   planner explanations are not numerically reliable and are treated as
   observable text, never as simulator authority.
7. The first Stage 1 negotiation did not yield a rejection: Qwen countered an
   offer at ten times the buyer's private reservation price. It also requested
   another quote instead of answering a delivered need with an offer. This run
   remains in `stage1_initial_no_rejection/`. The utility rule is now explicit
   in the planner contract; the rerun is required to pass every Stage 1 check.
8. PPO job `ppo-training-20260812` stopped before its first episode due to an
   undefined monitor-settings variable. No partial policy was used. The causal
   reward and per-agent GAE path subsequently passed unit tests and an
   independent eight-episode training validation.
9. Stage 1 v2 (`stage1-agentic-v2-20260812`) exited 1. Qwen's justifications
   correctly identified both overpriced offers but its structured action was
   `accept_offer` in both cases. It also combined `join_coalition` with a
   fabricated coalition ID and an invalid `members` field. The route probe
   produced a nonexistent `submit_quote` tool and then a safe no-op recovery.
   This shows that valid JSON and plausible prose do not imply semantic action
   validity. The run is retained and excluded from positive capability claims.
10. Stage 1 v3 passed, but its rejection explanation still said an offered
    price of 4.5 was equal to a private reservation value of 0.45. The emitted
    action matched the private constraint. This is a retained factor-of-ten
    explanation error and reinforces that generated justification is not used
    as numerical simulator authority.
11. The first final-v3 centralized-LLM pilot rows are invalid as a legal-
    information baseline: their coordinator received bins derived directly
    from simulator state even in a strongly private regime. They are retained,
    excluded from claims, and replaced only by separately named corrected-v4
    rows whose reports come from the public-information interface.
12. Final-v3 was stopped after 19 new rows because the shared simulator RNG let
    action-dependent message draws shift later demand and production. This
    defeats paired-seed inference even for methods that never received illegal
    information. All rows and logs are retained, but the complete v3 name
    family was prospectively excluded before inspecting method effects. The
    corrected paired-v5 run uses purpose-specific RNG streams and regenerated
    nominal calibration and PPO checkpoints.
13. The excluded v3 planner diagnostics found low structured validity for the
    learned-option methods (76.3% no-entropy; 69.5% ThermoAgent) and many
    coalition expiry/tool-schema failures. This is not hidden by parser repair:
    v4 shortens and reorders the prompt, supplies exact dynamic bounds, and is
    independently requalified. The old failures remain in raw ledgers.
14. The first full mock pipeline produced 128/128 episodes but failed replay on
    two raw rows written just before an intentional process stop; their
    manifests did not yet exist. The `/tmp` preflight is retained as an
    engineering diagnostic. Episode publication is now staged behind manifest
    creation, and a fresh exact-source preflight is required.
15. Stage 1 v4 passed seven of nine gates but did not exercise route failure or
    replanning. The model correctly followed option 6's coalition-reallocation
    affordance and proposed coalitions, while the outdated smoke probe expected
    a direct shipment. The failed artifact is retained. The corrected probe uses
    option 0 (continue the local execution plan), which is the option that
    actually permits direct shipment; this is a harness-semantic correction,
    not parser repair or outcome-driven prompt tuning.
16. Paired-v5 was stopped after 16 complete rows when a privacy audit found two
    evaluator-only actor features: exact-global consensus RMSE and global
    interaction entropy. One learned-no-entropy row had completed, but no method
    outcome comparison was inspected. All v5 rows and an interrupted staging
    copy remain retained and machine-excluded. Actor features are now link-local
    and per-agent; controls are fully zeroed; checkpoints and qualification are
    rebuilt under separately named v6 conditions.
17. Paired-v6 was stopped after 13 complete rows when a factorial audit found
    that the information-regime parameter also multiplied true marginal costs
    and changed agents' own forecast noise. No method outcome comparison was
    inspected. All v6 rows, logs, and exact configuration are retained and
    machine-excluded. The corrected v7 factor changes public observability only.
18. Paired-v7 was stopped after 10 complete rows when a communication audit
    found ordinary messages partitioned from period 0 while the sketch channel
    stayed connected until disruption onset. No outcome comparison was
    inspected. Both channels now share one onset and one component graph; v7 is
    retained and machine-excluded, and the clean qualification is named v8.

## Current scientific limitations

- The final 944-row main sweep produced suggestive ThermoAgent gains over the
  matched no-entropy actor, but neither application crossed the prespecified
  Holm-adjusted 5% threshold. Fixed communication, scripted agents, the legal
  central LLM, and especially privileged numerical lookahead usually matched
  or beat it. The locked holdout tied the no-entropy actor exactly. Independent
  autonomy is not justified by the current outcome evidence.
- Every privacy/misalignment response-surface cell was negative relative to the
  best fixed deployable benchmark selected once per cell. The hypothesized
  monotone increase in autonomous-agent value is unsupported in these domains.
- ThermoAgent formed more coalitions and revised more plans, but useful-
  coalition precision stayed below 6%, valid-tool rates were lower, failed
  actions were more frequent, and mandatory sketch traffic dominated the
  communication budget. Observable activity is not evidence of useful agency.
- Staged PPO used a deterministic planner. Distribution shift under the frozen
  LLM was observed in the pilot through semantic tool failures and remains an
  important limitation even though dynamic validation prevented illegal state
  mutations.
- The final free-energy gap retained the pilot's sign problem: it often fell
  under disruption and had ROC AUC below 0.5 when evaluated as a high-direction
  alarm. Operational entropy and energy were useful monitors, but the specific
  free-energy hypothesis is unsupported without a new prospectively calibrated
  formulation.
- Qwen2.5-7B can pair correct natural-language reasoning with the wrong tool.
  Dynamic validation prevents state mutation but may increase no-ops/retries.
- Only one open-weight planner, one deterministic LLM seed, and one active RL
  training seed were used in final inference. Eight environment seeds support
  the main pairing, but holdout and ablations have only four seed clusters each;
  broad model- and training-seed generalization is not established.
- Exact ties in many holdout and ablation cells reveal limited action
  sensitivity or saturation in those scenarios. They must not be interpreted
  as general equivalence between methods.
- The monitor uses a pooled 27-state coarse graining over small role
  populations. Commercial source localization was strong, but humanitarian
  top-1 localization failed in several large main cells even though top-3 was
  perfect.
- The numerical lookahead has unattainable full information and authority. It
  is an upper bound, not a deployable baseline. The legal central LLM is a more
  relevant centralized comparison but still assumes a coordinating role and
  structured reports.
- The environments are abstract quantitative testbeds. Humanitarian agents are
  not a model of real organizations or human behavior.

## Post-freeze operational issues

1. The first final analysis invocation yielded a remote session without visible
   output. A retry started a second identical frozen analyzer. Both were
   read-only over raw ledgers and deterministic over derived outputs; both were
   allowed to finish, and `INDEX.csv` was rebuilt once afterward. No episode,
   protocol, or inference result changed.
2. Monitoring-predictive analysis emitted `ConstantInputWarning` for cells in
   which a signal did not vary. Those Spearman values remain missing rather
   than being imputed or treated as zero.
3. Initial final visual QA found colliding Pareto x labels and clipped network
   coalition outlines. Frozen `thermoagent/figures.py` was not edited. A new,
   separately documented presentation-only save wrapper regenerated the three
   affected PDFs, after which all ten opened, exposed fonts, rendered, and
   passed original-resolution manual inspection.

## Entropy-triggered v2 pre-holdout issues

1. While the 144-episode real-LLM validation remained outcome-sealed, a source
   audit found that the staged DOET-RL trainer would read selected trigger
   parameters but silently omit the selected application/role normalizers.
   The automatic training watcher was stopped before it launched; validation
   was neither changed nor inspected. No checkpoint or training trajectory was
   produced under the faulty path. Training now resolves and checksums the
   referenced calibration fail-closed, and missing calibration is a tested
   error rather than a fallback to generic normalization.
