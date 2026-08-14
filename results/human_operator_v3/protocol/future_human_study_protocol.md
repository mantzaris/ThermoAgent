# Proposed future human-operator study (not conducted)

Status: technical preparation only; no participants were recruited and no
human-subject evidence exists in v3. An institution must obtain the appropriate
ethics/IRB determination before recruitment or data collection.

## Objective and design

Evaluate whether authorized entropy–energy–disagreement views help trained
operators allocate scarce attention relative to the same local-KPI view.
A preregistered within-participant crossover design would expose each participant
to matched commercial and humanitarian scenario panels under:

1. local KPI only;
2. combined entropy and energy;
3. combined entropy, energy, and disagreement.

Condition order and scenario mapping must use blocked randomization. Scenario
instances, intervention menus, time budgets, operator slots, and exogenous RNG
must be matched. Practice trials must use separate seeds. Oracle and
evaluator-global views are excluded from participant trials.

## Candidate endpoints

The primary endpoint should be preregistered before data collection, preferably
paired logistics loss reduction per active operator minute. Secondary endpoints
may include intervention selection accuracy, false/missed interventions,
response time, queue delay, calibration, regret to a bounded oracle, and a
validated workload questionnaire. Simulation episodes—not clicks—remain the
operational unit; participant is an explicit random effect.

## Power analysis

Use the adjacent `power_analysis_template.csv` with a blinded pilot estimate of
within-participant variance, attrition, multiplicity family, desired precision,
and minimally important benefit. Do not use simulated-operator effects as though
they were human effect sizes. Final sample size, exclusion rules, stopping rule,
and analysis code must be frozen before the confirmatory participant set.

## Procedure and safeguards

- Obtain informed consent and permit withdrawal without penalty.
- Explain that scenarios are abstract simulations, not real deployment advice.
- Avoid collecting unnecessary personal data; assign pseudonymous participant IDs.
- Store consent linkage separately from trial data with restricted access.
- Log the exact hashed view, permitted actions, action/response timestamps, and
  simulator result; never expose private agent vaults.
- Predefine technical-failure retries and retain all attempts.
- Debrief participants about simulated agents and the nonphysical interpretation
  of entropy/energy.
- Review accessibility, fatigue limits, compensation, retention, and incident
  handling with the responsible institution.

## Instrumentation already prepared

The dashboard provides replay controls, scenario/condition selection, typed
operator actions, view hashes, response timestamps, workload integration
points, deterministic matched branches, SVG export, and the adjacent trial JSON
schema and randomization specification. Questionnaire answers are schema slots
only; no values have been fabricated.
