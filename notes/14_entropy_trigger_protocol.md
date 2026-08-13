# DOET protocol record

Status: **development/validation protocol implemented; holdout not frozen**.

## Research question

Can a decentralized trigger based on each agent's distributed operational-
entropy estimate preserve the logistics performance of always-on fixed
communication while reducing total communication and inference cost, including
the cost of entropy sketches?

## Immutable integrity rules

- Development may use v1 pilot/main data and new development scenarios.
- A new validation set selects the trigger statistic, thresholds, dwell/cooldown,
  propagation rule, operating points, matched budgets, and checkpoints.
- All code, prompts, checkpoints, parameters, statistics, figures, seeds, and the
  new holdout matrix are checksum-frozen before the first holdout episode.
- The seen v1 holdout is never used for v2 selection or confirmation.
- The new holdout is outcome-sealed until every planned job completes and every
  ledger passes engineering/replay gates.
- No failed or unstable training seed is removed. Any failed locked-holdout
  episode is retained and prevents a confirmatory-supported classification;
  complete matched pairs may still be summarized without imputation.
- The additional compute cap is 35 single-GPU hours; exceeding it requires user
  approval.

## Preregistered primary hypotheses

`DOET-rule` is the primary confirmatory method because it isolates the event
trigger contribution transparently. `DOET-RL` is the preregistered learned
secondary method and receives the same descriptive and non-inferiority
analyses, but it does not replace the primary family after outcomes are seen.

- **H1 non-inferiority:** normalized DOET performance degradation versus
  `fixed_always_on` has an upper 95% confidence bound below 2% in each application.
- **H2 communication superiority:** total counted communication is reduced with
  a confidence interval excluding zero and a practical target of at least 20%.
  Fully counted messages are the frozen primary communication unit; bytes,
  prompt/generated tokens, calls, and latency are required accounting outputs
  but cannot substitute for a failed message test.
- **H3 Pareto superiority:** DOET improves the frozen performance/communication
  frontier relative to periodic, random budget-matched, learned non-entropic,
  and KPI-triggered communication. Operationally, DOET-rule must be
  loss-message nondominated and must strictly increase normalized two-objective
  hypervolume for messages, prompt tokens, LLM calls, and inference latency in
  each application. Normalization is application-wise min-max on the frozen
  comparator set with reference point `(1.05, 1.05)`.
- **H4 timely activation:** at least 75% of non-nominal DOET-rule episodes have
  a first post-disruption activation strictly before visible collapse, no more
  than 10% activate falsely before the disruption, and no more than 10% of
  nominal episodes contain any activation. Visible collapse is the first
  post-disruption period whose normalized service loss exceeds that episode's
  pre-disruption mean by more than 0.10; activation in the collapse period does
  not count as lead time. A pre-disruption false alarm can never satisfy the
  timely-activation condition.
- **H5 distributed robustness:** degradation under delay/noise/partition follows
  consensus error while retaining useful trigger behavior. The primary
  partition criterion is non-inferiority in both partition families and both
  applications, plus a positive consensus-RMSE/degradation slope and Pearson
  `r >= 0.20` in each application. Noise/delay extensions remain secondary and
  run only inside the 35-hour cap.
- **H6 cross-application generality:** both applications support H1 and H2. This
  claim is not made if only one application succeeds.

Exact trigger parameters, validation criterion, design size, precision analysis,
and projected compute will be appended before protocol freeze.

## Diagnostic-driven constraints fixed before implementation

1. The contribution will not claim that entropy beats globally pooled ordinary
   KPIs. It tests a privacy-preserving distributed summary under restricted
   information.
2. `KPI_CUSUM_trigger` uses only the same agent's private KPI vector and receives
   the same intensive-communication budget as DOET.
3. Application-, topology-size-, and role-conditioned nominal normalization are
   candidates because v1 exposed threshold-transfer failure. No new-holdout
   value will enter calibration.
4. Both high/low directional behavior and absolute deviation remain development
   candidates. The new validation set—not the seen v1 holdout—selects direction.
5. Primary trigger selection is lexicographic on validation: first retain
   candidates whose paired mean degradation versus fixed always-on is at most
   1% in both applications and no regime exceeds the preregistered 2% margin;
   then maximize total counted-message reduction; then minimize nominal false-
   active epochs. If none qualify, select the smallest worst-case degradation
   and label validation failure before holdout.
6. Budget-matched random, periodic, and local-KPI controls take their activation
   rates only from validation DOET traffic. Matching uses total counted DOET
   messages—including entropy sketches—not an unpriced active-state fraction.
7. The primary holdout test retains the 2% relative non-inferiority margin. No
   application-specific replacement is currently justified because v1 fixed
   losses are nonzero and stable.

## Development calibration fixed before validation

New seeds `5101`--`5106` supplied 12 nominal calibration episodes (six per
application). New seeds `5201`--`5203` supplied 18 disrupted development
episodes. These seeds are disjoint from validation and the proposed holdout.
Sparse gossip uses one pairwise matching round every eight quiet periods, every
four targeted periods, and every two crisis periods; every directed sketch is
counted.

Application- and role-conditioned nominal means and standard deviations are in
`results/entropy_triggered_v2/calibration/trigger_nominal_calibration.json`.
A scale floor of 0.02 was fixed to prevent division by numerical noise. On the
development data, the low-direction statistic led the preregistered direction
ranking:

- commercial AP 0.6218 and ROC AUC 0.7101;
- humanitarian AP 0.5616 and ROC AUC 0.6774;
- positive prevalence 0.40 in each application.

This is not yet a positive trigger result. At a nominal 95th-percentile
threshold, recall was only 0.1780 commercial and 0.1549 humanitarian. The
preregistered development direction criterion therefore froze the low
direction before treatment validation. The compute-capped validation matrix
retains balanced local and neighbor-propagated low-direction CUSUM, a more
sensitive low-direction CUSUM, and low-direction simple hysteresis. Direction
will not be reconsidered after validation.

## Measured real-Qwen profile and prospective budget reduction

The real-Qwen profile completed 8/8 episodes with zero failures and exact 8/8
ledger replay. The sweep used 480 LLM calls, 934,041 prompt tokens, 35,378
generated tokens, 0.1606 summed episode GPU-hours, and 714.10 seconds including
one model load. The separate CUDA/model smoke loaded in 73.79 seconds, completed
its two-request batch in 3.80 seconds, and achieved 100% JSON and tool validity.
The profile's mean episode time was 72.27 seconds for eight agents over twelve
periods.

Scaling that measured mean by agent-periods and adding a 15% buffer projected
17.45 hours for the original 288-episode validation and 74.80 hours for the
original 1,296-episode holdout. Together with profile/smoke and the setup
reserve, the preferred design required at least 92.57 single-GPU hours before
counting PPO training. It therefore cannot be launched under the user's hard
35-hour cap.

The prospectively reduced design follows the authorized priority order rather
than selecting methods by outcome: validation is reduced to 144 episodes at a
16-period horizon, and the preferred locked holdout to 696 episodes at the same
horizon.
The profile extrapolation with a 15% buffer is 5.82 validation hours and 26.79
holdout hours. Including measured profile/smoke and the 0.1-hour setup reserve
totals 32.92 hours before CPU-bound PPO time. Before any validation outcome was
available, a runtime-only fallback ladder was frozen: keep all 576 priority-
method episodes and all five RL seeds, but reduce the identical secondary-
comparator subset from three environment seeds to two, then one, if needed.
That yields 696, 656, or 616 total episodes. The generator chooses the largest
fitting design using measured runtime only and fails closed unless validation,
all fifteen actual training runs, and the buffered holdout remain below 35
hours.

## Validation matrix and frozen selection rule

The real-LLM validation configuration is
`configs/entropy_trigger_validation.yaml`. It has three completely new seeds
(`6101`--`6103`), both applications, and nominal, isolated, correlated, and
compound-partition regimes at a 16-period horizon. It compares four
low-direction entropy candidates with always-on fixed communication and the
private-local-KPI CUSUM, for 144 prospective episodes. Selection is exactly:

1. retain candidates whose paired mean degradation is at most 1% in both
   applications and at most 2% in every application/regime cell;
2. maximize total counted-message reduction;
3. minimize the nominal false-active fraction;
4. if no candidate qualifies, choose minimum worst-cell degradation and label
   validation failure before any holdout run.

Budget-matched random and periodic rates are computed only from the selected
validation traffic. The selected DOET total counted messages per episode,
including sketches, are converted to an intensive-decision target using the
fixed control's validation messages per active decision. Random activation is
sampled at two-period opportunities; the periodic interval from 2 through 16
that most closely matches the target is selected. Both controls retain quiet
local planning every eight periods when intensive communication is inactive,
so matching communication does not disable local autonomy. Predicted and
achieved message-count mismatches are retained in the protocol and final
tables. The private-KPI CUSUM retains its independently calibrated KPI
normalizers and validation-fixed parameters; its local residual is multiplied
by the selected DOET/KPI validation message ratio, clipped to `[0.25, 4]`.
Because trajectory feedback makes this only an approximate rate match, its
actual mismatch is reported and no holdout retuning is allowed. The primary comparator is the new
`fixed_always_on` method, which sends explicit coarse three-dimensional status
packets to up to three active neighbors at every scheduled epoch, in addition
to autonomous negotiation. This replaces the weak alternating v1 fixed method
without changing any frozen v1 artifact.

## Proposed locked holdout design

The holdout generator is implemented but cannot run before real-LLM validation
and five completed checkpoints per learned method. It will create:

- 16 unseen seeds per application for each of isolated, communication-
  partition, correlated, and compound-OOD disruption;
- 8 unseen nominal seeds per application;
- 144 base matched panels and a preferred 696 total method episodes, all with a
  16-period horizon;
- fixed always-on, learned non-entropic, DOET-rule, and DOET-RL on every panel,
  because these are the preregistered compute-priority methods;
- no communication, periodic, random budget-matched, ThermoAgent v1, and local-
  KPI CUSUM on the same prospectively fixed non-nominal subset. The preferred
  24-panel subset uses seeds `8101`, `8106`, and `8111`; runtime-only fallbacks
  use `[8101, 8111]` (16 panels) and then `[8106]` (8 panels). Pareto analyses
  restrict every method to the exact selected common panels;
- five independent RL seeds (`7301`--`7305`) assigned round-robin without
  removal or outcome selection; full-panel learned methods receive 28 or 29
  panels per seed and the secondary ThermoAgent comparator remains balanced
  across every seed represented by the selected common subset;
- deterministic Qwen decoding with a new recorded and applied LLM seed `9101`;
- the unseen `tri_region_bridge_v2` topology, distinct from training and the
  seen v1 holdout topology.

A measured validation throughput estimate and simulation/precision analysis
must confirm that the measured real-Qwen profile and model smoke, validation,
full five-seed training, a 0.1-hour unmeasured setup reserve, and holdout stay
below 35 additional single-GPU hours. The precision
calculation uses 20,000 fixed-seed stratified Monte Carlo draws from validation
paired degradation, with 16 panels in each of four planned non-nominal regimes.
CPU-bound PPO time still counts because the paid GPU Pod is reserved. The
runtime fallback never reads validation performance values. If the minimum
616-episode design does not fit, the generator fails closed before freezing or
launching.

## Freeze and outcome-seal procedure

The filtered deployment does not copy `.git`. Before each source sync,
`capture-source-provenance` records the originating branch, commit, dirty flag,
and byte-level source checksum without reading remotes, environment variables,
SSH configuration, or credentials. `run_matrix` refuses execution if that
checksum differs from the deployed source.

After real validation and training, the v2 namespace is fetched locally, the
holdout matrix is generated, and all source/configuration changes are committed.
The provenance record is recaptured and the resume controls are synchronized.
`freeze-doet-holdout.sh` then requires exactly 15 checksum-valid learned
checkpoints, runs the full tests, and writes a non-overwritable freeze over the
selected trigger, budget matches, calibration, validation inputs, design,
precision analysis, checkpoints, protocol note, and source checksum.

During the holdout, `doet-job-status.sh` reports only job/process health and
counts of manifests/published directories. It intentionally displays no
partial loss, communication, or hypothesis values. The outcome seal lifts only
after all planned rows finish and causal ledger replay completes. Failed
manifests are retained without selective rerun.
