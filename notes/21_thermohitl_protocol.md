# ThermoHITL prospective protocol

Status: **development protocol executed; closed by prospective no-go rule**  
Created: 2026-08-14 before any v3 outcome experiment

## Question

Can privacy-preserving distributed operational entropy and energy triage scarce
simulated-operator attention among independent autonomous logistics agents and
improve primary logistics outcomes per operator minute relative to equally
informed KPI, periodic, random, learned non-thermodynamic, no-human, and
always-on-review controls?

The intended causal mechanism is:

`thermodynamic change -> independent request -> bounded queue allocation ->`
`authorized operator view -> bounded intervention -> agent acceptance/action ->`
`material movement -> demand arrival -> primary-outcome change`.

Every link will be measured. Resource savings alone are not success.

## Development-only changes motivated by v2

V2 often produced invalid or non-arriving material actions, coordination was
not necessary in the tested environment, and the selected trigger never
activated. V3 may therefore change only its own scenario mechanics, action
affordances, prompt revision, horizon, lead times, and typed tools. Frozen v1/v2
code meanings and artifacts remain untouched.

The v3 scenarios will include private route/need information, organizational
authority constraints, and disruptions that leave a legitimate but bounded
human-remediable action. They will also contain regimes in which intervention
is unnecessary or harmful, so an always-escalate rule is not favored by
construction.

## Information boundary

Each autonomous organization retains an independent identity, private vault,
utility, memory, commitments, inbox/outbox, RNG state, planner context, action
authority, distributed estimate, and escalation state. Only explicit messages,
bounded macrostate sketches, public infrastructure, and signed operator
directives cross boundaries.

Normal simulated operators may see only a serialized, schema-validated view
payload. Payloads are hashed and logged. No future disruption, evaluator-global
state, other agent's raw private observation, RNG state, or counterfactual
outcome is permitted. Oracle methods use separately labeled privileged schemas.

## V3 thermodynamic variables

Operational energy is a normalized severity score fixed before validation:

`E = 0.24 backlog + 0.22 unmet + 0.16 congestion + 0.14 lateness`
`    + 0.12 commitment risk + 0.12 route/safety risk`.

These weights reflect service-first application utilities rather than observed
treatment effects. Planned sensitivity sets perturb each weight by plus/minus
25% followed by renormalization.

The primary entropy representation combines three separately reported terms:

- flow entropy over outgoing material allocation;
- belief entropy over locally estimated disruption states;
- bounded Jensen-Shannon disagreement across received agent sketches.

Each is reported raw, normalized, as a two-sided nominal residual, and with a
finite-difference slope. Distributed estimates use bounded local sketches and
the existing link-respecting gossip machinery. Evaluator-global values are
analysis-only. The Helmholtz-like diagnostic is `F = E - T*S`, with temperature
defined as bounded recent disruption volatility; it is not a primary trigger.
None of these quantities is literal physical thermodynamics.

## Operator model and authority

The primary simulated operator is capacity constrained and sees the
`thermodynamic_plus_disagreement` view. Profiles vary slots, latency, service
time, accuracy, fatigue accumulation/recovery, risk tolerance, and intervention
cost. The primary profile is high-accuracy bounded, not an oracle.

Interventions are typed and bounded. They can authorize emergency routes or
resources, relax a documented constraint, authorize temporary coarse data
sharing, reprioritize demand, resolve an explicit commitment conflict, or return
authority. They never directly invent an autonomous organization's ordinary
domain decision. Advisory directives may be rejected; emergency overrides are
mandatory only within their recorded scope and duration.

## Candidate policies

- no human; fixed communication/no human;
- always-on, periodic, and random budget-matched human review;
- local-KPI, entropy-only, energy-only, free-energy, and disagreement triggers;
- `thermohitl_rule`: interpretable expected-benefit rule using energy,
  two-sided entropy anomaly/slope, disagreement, disruption risk, consensus
  confidence, and workload cost;
- `learned_no_thermodynamics`: same capacity, without entropy/energy features;
- `thermohitl_rl`: learned attention/escalation policy with thermodynamic
  features;
- bounded human and full-information oracles, labeled unattainable.

The rule-score coefficients are initialized from application utility weights
and tuned only on development/validation. The primary rule and a small frozen
set of operating points will be selected by validation utility subject to all
timing/false-alarm gates. The learned policy will use the existing compact PPO
infrastructure or a contextual bandit if profiling shows equivalent scientific
validity at materially lower variance. This choice will be documented before
training.

## Prospective development gates

All gates must pass, and each pass record must bind source/config/test checksums,
before validation or holdout commands unlock.

1. **Engineering:** all tests pass; exact replay and RNG restoration; material
   conservation; no tool/metric mismatch; deterministic dashboard replay; no
   operator-view privacy leak or nonfinite value.
2. **Actionability:** first-pass structured tool validity >=90%; validity after
   no more than one repair >=98%; >=70% of accepted material actions reach the
   next intended stage; >=30% reach demand or change horizon loss; paired
   feasible-action versus no-op tests change the primary outcome in each
   application.
3. **Coordination necessity:** strongest fixed-communication method improves
   aggregate primary loss by >=5% versus no communication in each application
   and improves at least two important disruption regimes per application.
4. **Human causal usefulness:** bounded authorized intervention improves
   aggregate loss versus autonomy in each application and in at least two
   important disruption regimes, through a logged authority/information/action
   change rather than an unexecuted message.
5. **Thermodynamic information value:** at identical private-local information,
   thermodynamic features improve AP or ROC AUC by >=0.05, or improve matched
   intervention utility by >=5% at the same operator budget, on development
   data. Calibration and decision utility are both reported.
6. **Trigger feasibility:** nonzero activation in every important regime;
   >=75% activate after onset and before sustained collapse; <=10%
   pre-disruption activation; <=10% nominal false activation; escalation is
   neither zero nor always-on; at least one causal material/outcome effect; and
   exact onset tests exclude a period-zero direction artifact.

If no candidate passes, the confirmatory holdout is not run.

## Data stages and outcome seal

- Development: original antecedent diagnostics plus new explicitly labeled v3
  development seeds; mechanics and candidate ranges may change.
- Validation: fresh seeds; selects trigger, operating points, budgets,
  checkpoints, and non-inferiority margin. No further method changes.
- Locked holdout: fresh topology, environment seeds, LLM seeds, operator seeds,
  nominal/isolated/partition/correlated/compound/OOD, overload/delay/imperfect
  profiles. Source, protocol, manifests, prompts, checkpoints, statistics,
  tables, and figure specifications are checksum-frozen first.

During locked execution only health, counts, schema/files, finiteness, disk, and
catastrophic engineering failures may be inspected. Comparative outcomes remain
sealed until every eligible ledger completes and replay passes.

## Primary outcomes and hypotheses

The episode is the experimental unit. Commercial primary loss is service-loss
AUC; humanitarian primary loss is cumulative weighted unmet need.

- H1: at a matched attention budget, ThermoHITL-rule reduces loss versus the
  strongest same-information local-KPI trigger in both applications.
- H2: ThermoHITL-rule is non-inferior to always-on review within a validation-
  frozen margin no larger than 0.2 fixed-baseline SD, while reducing operator
  interventions and minutes by at least 20%.
- H3: ThermoHITL improves the loss/effort Pareto frontier and is not dominated
  by no-human, no-communication, or another sparse baseline.
- H4: eligible episodes meet the gate-6 timing and false-alarm bounds.
- H5: paired branches show positive intervention effects and a complete causal
  funnel through demand-reaching material and primary outcome.
- H6: thermodynamic features add out-of-sample decision value beyond same-view
  local KPIs.
- H7: useful behavior persists under delay/noise/partitions, with worsening
  consensus error prospectively associated with worse allocation quality.
- H8: primary loss and operator-effort conclusions replicate in both
  applications.

The final frozen protocol will specify margins, families, estimands, seed sets,
sample size, and analysis hashes after development-only precision analysis but
before validation outcomes are opened. Holm correction applies to the primary
superiority family. Hierarchical bootstrap uses at least 10,000 fixed-seed
replicates and explicitly resamples environment, RL, and operator seeds.

## Compute rule

A measured profile must cover model load, development qualification, five
independent seeds for each primary learned method, validation, holdout, and a
15% reserve. The total may not exceed 40 additional single-GPU hours without
the user's advance approval.

## Final protocol disposition

The protocol was not promoted to a validation/holdout freeze. Gate 5 failed on
held-out development seeds after all candidate definitions were fixed. The
fail-closed guard therefore left training, validation, checkpoints, and holdout
as explicit `NOT_RUN` artifacts. The selected trigger parameters remain a
development candidate only and must not be described as holdout-frozen.
