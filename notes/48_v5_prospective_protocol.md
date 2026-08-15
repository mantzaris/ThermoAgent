# V5 prospective protocol and decisions

Status: development protocol frozen before computing any V5 outcome.

Canonical configuration: `configs/human_operator_v5_development.yaml`.
Result-package pointer: `results/human_operator_v5/protocol/development_protocol.yaml`.

## Scientific question

V5 tests whether distributed local-belief entropy and inter-agent disagreement
improve scarce simulated-operator attention allocation beyond ordinary local
KPIs when information is fragmented. Humanitarian logistics and abstract
cyber-physical utility restoration are the two primary positive applications.
Commercial logistics is a prespecified boundary application.

The protocol was informed by the V4 no-go result but does not reuse V4 panels
as V5 evidence. V4 identified an engineered verification shortcut, constant KPI
columns, structural selection ties, excessive sketch traffic, a non-refit
permutation analysis, and very limited real-Qwen qualification. V5 addresses
those limitations prospectively.

## Frozen design decisions

- Each formal panel contains four simultaneous incidents. All four have
  nonzero, state-dependent intervention effects; at least two actions are
  plausible and the operator can select at most two incidents.
- Verification is delayed, imperfect, and costly. Incorrect actions can cause
  bounded harm. A stored stochastic tape is shared by paired branches.
- Public-information controls preserve meaningful interventions. Shared
  evidence makes ordinary features more sufficient; it does not mark incidents
  as pre-resolved.
- Local-belief entropy and Jensen-Shannon disagreement are primary monitoring
  measures. Operational energy and free energy are secondary ablations.
- Thermodynamic sketch messages and bytes are included in costs. None,
  periodic, event-triggered, and always-on exchange are compared.
- Grouped splits isolate environment seed, topology family, and scenario
  family. Candidate rows are never treated as independent panels.
- The primary model is a regularized action-value model fitted and selected
  inside grouped folds. The primary endpoint is actual budgeted causal utility,
  not disruption-label accuracy.
- The formal permutation analysis permutes within application, information,
  regime, and KPI-severity strata and refits the full pipeline.
- Five fixed RL seeds are retained without selecting the best seed. Centralized
  training may be used, but execution receives only per-agent local features.
- At least 108 real-Qwen decision epochs across 36 episodes are required for
  agent qualification. The model and immutable revision remain those of V1-V4.
- The low-consensus abstention mechanism is deliberately exercised in
  partition and telemetry-integrity panels.
- An incident is triage-eligible only at the first post-disruption decision
  epoch, with visible severity at least 0.30 and positive predicted action
  value. Thermodynamic policies abstain below consensus confidence 0.42. These
  rules were frozen before generating V5 outcomes.

## Gate rationale

Gate thresholds are specified in the canonical YAML. Coordination requires a
positive 90% paired cluster interval, at least 3% aggregate loss reduction,
changed outcomes in at least 25% of panels, improvement in three disrupted
regimes, and positive communication-adjusted utility. This formulation is not
the V4 5% threshold weakened after seeing V4: it separately constrains
direction, practical size, prevalence, regime breadth, and communication cost
for a newly designed multi-incident task.

Thermodynamic incremental value requires, in both primary applications, a
positive 95% paired cluster interval, at least 5% utility gain, regime breadth,
bounded harm and service degradation, common feature support, and no single
scenario-family explanation. A univariate entropy or disagreement AUC above
0.90 is disqualifying because it would indicate another near-label shortcut.

Mechanism specificity requires the fragmented-minus-public relative-gain
interaction to exceed five percentage points with a positive 95% cluster
interval. Public panels retain nonzero action value. A refit-based shuffled
thermodynamic block may reproduce at most half the observed gain.

Validation is locked unless all ten development gates pass. The sealed holdout
is locked unless the prespecified validation gates pass. Thresholds will not be
changed in response to V5 development, validation, or holdout results.

## Alternatives considered

- A single verification intervention was rejected because V4 made the
  thermodynamic result nearly deterministic.
- Treating incident rows as independent was rejected because intervention
  candidates share an environment and stochastic tape.
- High-frequency gossip was rejected as the primary method because V4 sketch
  traffic dominated ordinary communication.
- A full deep end-to-end policy was rejected as the only analysis because it
  obscures information value; interpretable grouped action-value models remain
  primary, with decentralized RL as a separate stability test.
- Free energy was retained only as an exploratory ablation because V1-V4 did
  not establish incremental value.

## Compute limit

The frozen cap is 50 additional single-GPU hours and approximately USD 40,
including a 15% reserve. A measured profile must precede the real-Qwen and
multi-seed runs. If the projection exceeds either cap, execution stops for user
approval before the large job.
