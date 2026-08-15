# Frozen v4 development protocol

Status at creation: specified before v4 development outcomes were generated.
Machine-readable parameters are in
`configs/human_operator_v4_development.yaml`. This note and that configuration
must not be weakened after development, validation, or holdout outcomes are
observed. Corrections to implementation bugs must be versioned and every prior
development attempt retained.

## Research question

Can privacy-preserving distributed energy, entropy, belief disagreement, and
consensus confidence improve allocation of scarce simulated-operator attention
beyond ordinary local KPIs when independent agents face private, fragmented,
uncertain, or potentially corrupted observations?

The primary generalization claim requires humanitarian and utility-restoration
support. Commercial supply-chain logistics is a pre-specified boundary
application, not a required positive domain.

## Frozen hypotheses

1. **H1 — coordination necessity.** Communication and negotiation materially
   improve service in private fragmented regimes.
2. **H2 — human causal usefulness.** Bounded simulated-operator interventions
   causally improve primary service outcomes in at least two applications.
3. **H3 — same-information incremental value.** At a fixed attention budget,
   KPI plus entropy/disagreement has greater causal intervention utility than
   KPI-only in humanitarian and utility restoration.
4. **H4 — fragmentation interaction.** Thermodynamic benefit increases with
   information fragmentation, ambiguity, or inter-agent disagreement.
5. **H5 — effort/safety.** Thermodynamic triage reduces harmful or unnecessary
   interventions without more than 2% service-loss degradation relative to the
   matched KPI comparator.
6. **H6 — distributed robustness.** Distributed estimates are calibrated when
   connected, degrade with partition error, and produce safe abstention at low
   confidence.
7. **H7 — causal chain.** The event ledger contains the full chain from
   thermodynamic change through alert, allocation, bounded intervention,
   changed agent action/commitment, feasible movement, demand/service arrival,
   and changed loss.
8. **H8 — commercial boundary.** If KPI-only remains sufficient commercially,
   that is evidence of conditional rather than universal thermodynamic value.

## Primary endpoints and statistical unit

One complete matched environment panel is the independent unit. Candidate
interventions and time steps nested inside a panel are not independent samples.
The primary Gate 5 endpoint is realized paired causal intervention utility at
the same operator budget. A regularized logistic model is the primary
interpretable ranking model; regularization is selected only within training
folds. AP, ROC AUC, Brier score, calibration, lead time, and localization are
secondary.

For humanitarian and utility restoration, Gate 5 requires all of: positive
paired mean utility gain; a 10,000-replicate cluster-bootstrap 95% lower bound
above zero; at least 5% relative gain; harmful-intervention-rate increase no
greater than two percentage points; no more than 2% relative service-loss
degradation; and no privileged feature. Commercial receives the identical
analysis without being required to pass.

## Feature blocks

The frozen blocks are local KPI only; energy only; entropy and disagreement
only; KPI plus energy; KPI plus entropy and disagreement; KPI plus energy,
entropy, disagreement, and confidence; and an exploratory free-energy
addition. Raw energy inputs remain available separately so aggregate energy
cannot masquerade as independent information. Correlation and condition-number
diagnostics precede model interpretation.

## Gates and stopping rule

All seven gates in the configuration are required before validation, learned
policy training, or holdout. In particular, Gate 5 must pass both humanitarian
and utility restoration, and Gate 7 must show stronger benefit under fragmented
than public information while shuffled/permuted thermodynamic features fail to
reproduce the gain. If a required gate fails, subsequent stages remain locked;
the result package, figures, and boundary analysis are still completed.

Development seeds may be used for debugging and calibration. Validation is one
complete batch. A fresh outcome-sealed holdout is generated only after all
gates, validation, and five-seed training qualify. No threshold, hypothesis,
feature definition, margin, or rerun rule may be revised after the relevant
outcomes are opened.

For Gates 3 and 4, a regime counts as improved only at a relative loss
reduction of at least 2%; this operational definition was added before formal
gate data.  Gate 3 requires the 5% aggregate target and two such regimes in
each application.  Gate 4 requires the 2% aggregate target, two such regimes,
and a complete causal chain in at least two applications.

The final formal-development attention budget is one intervention per episode.
An early implementation setting allowed two interventions for three incidents;
the conditional-permutation pilot showed that this was not genuinely scarce
attention, because even randomly permuted rankings selected two thirds of the
candidate set.  Reducing the budget to one makes the falsification meaningful
and strengthens, rather than relaxes, the operator constraint.  This choice was
made before formal gate data and applies identically to every view condition.

## Evidence labels

All large-scale operators are simulated. Real Qwen evidence qualifies
structured independent agent actionability only. Deterministic-agent evidence
supports simulator and mechanism development. Oracle conditions are
unattainable upper bounds. None of these establishes usability, trust,
fatigue, workload, or safety for real human operators.
