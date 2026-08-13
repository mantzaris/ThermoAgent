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
- No failed or unstable training seed is removed.
- The additional compute cap is 35 single-GPU hours; exceeding it requires user
  approval.

## Preregistered primary hypotheses

- **H1 non-inferiority:** normalized DOET performance degradation versus
  `fixed_always_on` has an upper 95% confidence bound below 2% in each application.
- **H2 communication superiority:** total counted communication is reduced with
  a confidence interval excluding zero and a practical target of at least 20%.
- **H3 Pareto superiority:** DOET improves the frozen performance/communication
  frontier relative to periodic, random budget-matched, learned non-entropic,
  and KPI-triggered communication.
- **H4 timely activation:** DOET activates after disruption and before severe
  service collapse with fewer unnecessary active epochs than always-on control.
- **H5 distributed robustness:** degradation under delay/noise/partition follows
  consensus error while retaining useful trigger behavior.
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
   rates only from validation DOET rates.
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
validation matrix therefore retains low-direction CUSUM, low-direction simple
hysteresis, absolute-deviation CUSUM, and change CUSUM candidates. Direction
will not be reconsidered after validation.

## Validation matrix and frozen selection rule

The real-LLM validation configuration is
`configs/entropy_trigger_validation.yaml`. It has four completely new seeds
(`6101`--`6104`), both applications, and nominal, isolated, correlated, and
compound-partition regimes. It compares seven entropy candidates with always-
on fixed communication and the private-local-KPI CUSUM. Selection is exactly:

1. retain candidates whose paired mean degradation is at most 1% in both
   applications and at most 2% in every application/regime cell;
2. maximize total counted-message reduction;
3. minimize the nominal false-active fraction;
4. if no candidate qualifies, choose minimum worst-cell degradation and label
   validation failure before any holdout run.

Budget-matched random and periodic rates are computed only from the selected
validation active-agent-step fraction. The primary comparator is the new
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
- 144 base matched panels and 1,296 total method episodes;
- nine core methods, including fixed, periodic, random budget-matched, local-
  KPI CUSUM, DOET-rule, and DOET-RL;
- five independent RL seeds (`7301`--`7305`) assigned round-robin to every
  learned method, 28 or 29 matched panels per seed;
- deterministic Qwen decoding with a new recorded LLM seed `9101`;
- the unseen `tri_region_bridge_v2` topology, distinct from training and the
  seen v1 holdout topology.

A measured validation throughput estimate and simulation/precision analysis
must confirm that validation plus holdout stays below 35 additional GPU-hours.
If it does not, the generator fails closed before freezing or launching.
