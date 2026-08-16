# Generalized Entropic Consensus V6 prospective protocol

Protocol version: `v6.0.1` (becomes frozen only when accompanied by
`frozen_protocol.yaml` and `freeze_manifest.json`).

Amendment `v6.0.1` was made before PPO training and before any formal
comparative outcome was inspected. It computes GAE separately within each
persistent agent trajectory instead of bootstrapping across interleaved
organizations. The active reference generator did not use this training code;
all hypotheses, gates, thresholds, seeds, applications, actions, environments,
feature blocks, and sealed validation/holdout inputs remain unchanged.

Evidence boundary: autonomous-agent simulations and bounded simulated
operators. No human participants are enrolled. Abstract cyber-physical events
do not model or enable real attacks.

## Motivation and separation from V5

V5 is an immutable development-stage no-go: KPI plus Shannon entropy and
Jensen–Shannon disagreement did not improve direct intervention ranking over
KPI-only triage. V6 neither retunes nor reinterprets that endpoint. It changes
the scientific question from choosing the best domain action to deciding when
an independently proposed action is safe enough to execute autonomously.

The primary applications are humanitarian logistics and abstract defensive
utility restoration. Commercial logistics is a prespecified boundary
application. A cross-application positive claim requires both primary
applications; a favorable commercial result is neither required nor silently
excluded.

## Independent-agent contract

Every organization has a separate identity, local observation, belief vector,
memory vault, utility, inbox, outbox, commitment ledger, role action mask, and
tool authority. An agent cannot dereference a peer vault. Information crosses
organizations only through logged messages or bounded sketches. A partition
blocks delivery. The simulator validates and applies typed actions but never
substitutes an evaluator action after rejection. Evaluator truth,
counterfactual branches, random tapes, and future outcomes are inaccessible to
agents, deployable controllers, and the normal operator view.

## Dynamic panel

Each 12-step panel contains four concurrent incidents and twelve autonomous
agents (three organizations with distinct roles per incident). Decisions occur
at steps 0, 2, 4, 6, 8, and 10; step 0 is a real pre-disruption epoch and the
registered disruption starts at step 2. Incidents overlap in observable severity but vary
in hidden mode and fragmented evidence. Physical actions have role permissions,
resource costs, delays, success uncertainty, and bounded harms. Verification
is delayed, fallible, and consumes capacity. The public condition shares
evidence but retains meaningful actions and nonzero intervention effects.

Humanitarian incidents combine uncertain access, resource scarcity, damaged
routes, heterogeneous urgency, and conflicting reports. Utility incidents use
abstract physical failures, communication partitions, telemetry-integrity
loss, defensive isolation, field verification, crews, spares, and critical
load. No real network or operational-technology system is contacted. Commercial
incidents combine inventories, route/warehouse failures, uncertain demand, and
telemetry ambiguity.

All action and no-action branches copy the same current agent, resource,
commitment, queue, and stochastic-tape state. Candidate causal utility is
future loss without action minus future loss with action. Formal policy
outcomes also arise from actual delayed state transitions over the horizon.

## Generalized uncertainty definitions

For agent belief `p_i=(p_i1,...,p_iK)`:

`H_1(p_i) = -sum_k p_ik log(p_ik) / log(K)`.

For `q != 1`, normalized Tsallis entropy is

`H_q(p_i) = (1-sum_k p_ik^q) / (1-K^(1-q))`.

The fixed development family is `q={0.5,1,1.5,2,3}`. The implementation uses
the Shannon limit around q=1 and verifies convergence numerically. Normalized
Gini–Simpson impurity is `(1-sum p_k^2)/(1-1/K)` and is exactly normalized
Tsallis q=2. An economic-style Gini concentration is exploratory and never
called Gini–Simpson.

Observable reliability weights use local telemetry confidence, delivered-edge
reliability, and message age—never true mode, correct action, future loss, or
counterfactual utility. The pooled belief is

`p_bar = sum_i w_i p_i / sum_i w_i`.

Average local uncertainty is `A_q=sum_i w_i H_q(p_i)/sum_i w_i`, pooled
uncertainty is `T_q=H_q(p_bar)`, and generalized disagreement is
`D_q=T_q-A_q`. At q=1 this is normalized weighted Jensen–Shannon disagreement.
Consensus is `C_q=1-D_q`; the normalized entropy gap supplies the common
[0,1] bound. Graph disagreement averages pairwise JS gaps only over delivered
active edges, weighted by modeled reliability. Missing and stale contributors
increase the separately reported consensus residual.

Temporal fields include first differences, acceleration, EWMA, persistence
above fixed monitoring levels, and recovery. The entropy spectrum is
`[H_0.5,H_1,H_1.5,H_2,H_3]`. q<1 emphasizes rare tail states; q>1 suppresses
small tail masses. No q is chosen from validation or holdout.

Operational energy is a dimensionless weighted stress index using visible
severity (0.31), backlog (0.22), delay (0.17), resource scarcity (0.14), safety
risk (0.10), and commitment strain (0.06). These domain-priority weights were
fixed before formal outcomes. Effective temperature is the bounded local
volatility/urgency proxy `0.25+0.375*delay+0.375*safety`. The exploratory
free-energy diagnostic `F=E-T_eff*H(p_bar)` is excluded from primary
controllers. None of these quantities is literal physical thermodynamics.

## Two-level autonomy

Level 1 is the organization-owned operational proposal: verify, request peer
evidence, allocate an emergency resource, reroute/reconfigure, deploy repair
capacity, isolate/quarantine, revise a commitment, defer, or no action, subject
to role masks. Level 2 decides execute, communicate, request evidence, defer,
abstain, or escalate. A bounded simulated operator has one service slot per
decision epoch, four episode interventions, queue delay, and 5–9 minutes per
response. It sees only the hashed authorized view and may be wrong.

Operational no-action proposals are never forced into the operator queue. For
actionable unserved proposals, escalation requires cross-fitted predicted harm
at least 0.80. The threshold came from the fixed pilot grid 0.50–0.80 in 0.05
increments using activation timing and false-activation rates only—never harm,
causal utility, or service outcomes. It was the highest threshold meeting at
least 75% activation, at least 75% activation by step 4, at most 10%
pre-disruption activation, and at most 10% nominal-panel activation for both
the predictive-uncertainty and combined controllers. Rare fixed-rate ambiguous
nominal signals keep those false-activation tests nondegenerate.

## Comparators and matching

Every primary comparison uses identical panels, stochastic tapes, action
coverage, one escalation slot per epoch, total operator budget, and sketch
policy. Reported references include always act; random abstention; fixed
severity; KPI confidence; action-value margin; maximum probability; predictive
action entropy; an ensemble-variance proxy; grouped one-sided split-conformal
risk; Shannon, pooled Shannon, JS, Gini–Simpson, every fixed Tsallis/Jensen–
Tsallis q, graph disagreement, the combined family, and an analysis-only oracle.

The primary non-entropic comparator is chosen once on development from KPI
logistic risk, KPI plus action-policy uncertainty, and grouped conformal risk:
lowest mean risk–coverage area across the two primary private applications,
then lowest Brier score, then lexical name. The low-consensus guard is fixed at
`C_1<0.88` or residual `>=0.25`; it is reported separately at matched coverage
and operator budget rather than being allowed to win by rejecting more cases.

Sketch policies are none, periodic, event triggered, and always on. Event
exchange sends at disruption onset and then no more often than every two steps
when belief total variation exceeds 0.35. Operational and sketch messages,
bytes, latency, calls, and tokens are all counted.

## Learning

Five learned delegation feature families are trained: KPI only, predictive
uncertainty, Shannon/JS, generalized Tsallis/Gini, and combined generalized
entropy. Each uses role-specific actor and critic networks, local execution
features, explicit action masks, full sequential trajectories, discounted
returns (`gamma=0.97`), GAE (`lambda=0.92`), PPO clipping (0.20), entropy
regularization (0.02), fixed optimizer settings, 200 training episodes, 60
evaluation episodes, and seeds 66201–66205. There is no best-seed selection.

Real-Qwen qualification uses `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, NF4 quantization and BF16
computation. It contains 60 humanitarian, 60 utility, and 30 commercial
episodes, 18 independent decisions per episode (2,700 before repair), separate
contexts/memories, at most one repair, multiple regimes, and both information
conditions.

## Independent units and inference

The panel is the unit. Compound split families make environment seed,
topology, scenario, and disruption families disjoint between folds. Nested
grouped fitting selects regularization inside training folds. A conformal
calibration family is distinct from fit and test families. Primary intervals
use 10,000 fixed-seed paired cluster bootstraps. Absolute effects, relative
effects, action counts, rates, wins/ties/losses, calibration, and heterogeneity
are reported. Family testing is hierarchical: combined generalized entropy
versus the strongest non-entropic comparator first; generalized additions
beyond the complete Shannon/JS reference second; individual q interpretation
only after a supported family result, with Holm correction. The full-refit
permutation test permutes generalized features within regime, compound split,
and severity strata; prediction-time shuffling is prohibited.

## Hypotheses and gates

H1 requires at least 0.03 absolute harm-rate reduction, paired 95% lower bound
above zero, and positive effects in at least three regimes in each primary
application at 50% autonomous coverage. H2 requires the relative service-loss
upper bound no greater than 0.02, causal-utility lower bound at least -0.02,
coverage at least 0.45, and at most 36 simulated-operator minutes. H3 requires
private-minus-public incremental harm reduction at least 0.02 with lower bound
above zero and meaningful public action effects. H4 requires at least 40%
event-sketch message and byte reduction, estimator MAE at most 0.12, and harm
degradation versus always-on no more than 0.015. H5 is a hierarchical secondary
test of generalized measures beyond Shannon. H6 concerns predictable
partition degradation. H7 requires both primary applications. H8 requires both
RL and Qwen agentic qualification.

The utility/trigger gate additionally requires at least 75% post-disruption
activation by step 4, no more than 10% pre-disruption or nominal false
activation, and no more than 3.5 mean escalations per disrupted episode. These
timing criteria were fixed from the outcome-blind pilot calibration.

The ten progression gates in the YAML protocol operationalize engineering,
mathematical validity, learnability, agentic validity, primary safety, utility
noninferiority, mechanism specificity, communication feasibility, multi-seed
stability, and cross-application replication. Thresholds may not be weakened
after formal outcomes. Any required development failure stops validation; any
required validation failure stops the sealed holdout.

## Stages and sealed inputs

Pilots use 609xx seeds and remain noninferential. Frozen development uses
66101–66130; RL uses 66201–66205 with training/evaluation environments 663xx
and 664xx; Qwen uses 665xx–667xx. Validation uses untouched 67101–67120 and
holdout uses untouched 68101–68124. Input manifests are generated and hashed
at protocol freeze. Formal execution is resumable after infrastructure failure
but one-shot after completion; no unfavorable outcome may be selectively
rerun. During sealed execution only health, counts, schema, finiteness, disk,
and catastrophic failures may be inspected.

## Power and compute

The pre-formal plan uses a conservative paired-SD floor of 0.10. At an effect
of 0.03, approximate power is 0.992 for 210 development panels, 0.908 for 120
validation panels, and 0.950 for 144 holdout panels per primary application
and information condition. The complete projection including a 15% reserve is
11.50 single-GPU hours, 2,730 LLM calls, 2.73 million prompt tokens, 253,000
generated tokens, 4.3 GiB, and about USD 3.91 at USD 0.34/hour. Measured
throughput supersedes projections for accounting but never changes gates.

## Prohibited interpretations

V6 cannot establish real-human usability or effectiveness, literal
thermodynamics, real critical-infrastructure validity, universal application
superiority, causal generalization beyond tested simulations, or communication
savings that omit sketch traffic. A stopped validation or holdout is not
missing data; it is the preregistered consequence of a failed gate.
