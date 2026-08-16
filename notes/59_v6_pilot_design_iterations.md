# V6 pilot design iterations

Evidence status: development-only engineering pilots. No V6 validation or
holdout outcome has been generated or inspected.

## Pilot history

Every pilot namespace is retained under
`results/generalized_entropic_consensus_v6/pilots/` and its raw ledgers remain
under `raw/`. Pilot V1 exposed role-mask fallbacks that selected unrelated
low-probability actions. Pilot V2 corrected authority overlap. Pilots V3 and
V4 separated grouped split families and removed direct telemetry-integrity
leakage. Pilot V5 separated the operational proposal from information-gathering
delegation and made peer proposals explicit messages.

Pilot V5 remained a negative mechanism diagnostic. At 50% action coverage,
the combined generalized-entropic block did not improve harm selection over
the strongest non-entropic predictive-uncertainty block in humanitarian or
utility restoration. Its disagreement measures varied only weakly with action
harm, and the private-versus-public interaction was absent.

## Pre-formal correction after Pilot V5

Decision recorded before any formal-development outcome: use the same true
incident, resource state, agent utilities, and stochastic tape for the matched
private-fragmented and public-shared conditions. Previously, the information
condition entered the random seed, so the comparison was not a strict matched
information intervention.

Evidence difficulty is now sampled broadly within every disruption family and
held fixed across the paired information conditions. Private agents receive
independent local signals; public agents receive the same shared signal. The
public intervention remains consequential and can still be wrong. This is a
mechanistic repair, not an outcome-threshold change: it makes disagreement an
observable consequence of fragmented evidence while preserving overlapping
severity and action outcomes.

Alternatives considered were (1) assigning harm directly from disagreement,
(2) exposing an integrity label, or (3) continuing with a narrow high-
fragmentation distribution. The first two would engineer label leakage; the
third had already falsified the intended manipulation. They were rejected.

The next pilot uses untouched seeds and a new namespace. Its results determine
whether the environment supports the proposed mechanism before the V6 protocol
is frozen. No q value, gate threshold, or validation decision will be selected
from a final holdout.

## Sketch-cost pilot

The retained `pilot_sketch_cost` run showed that the first event rule reduced
sketch traffic by only 7.94% relative to always-on exchange, despite nearly
identical estimation error (mean 0.0687 versus 0.0664). It therefore failed the
prospective 40% communication gate before that gate was frozen. The event rule
was changed, before formal safety evaluation, to send at disruption onset and
then at most every two steps when total variation exceeds 0.35 (belief-vector
L1 change exceeds 0.70). This is an interpretable communication threshold and
is selected from message/change distributions only, not from outcome effects.
The earlier rule and results remain retained.

## Final pre-freeze pilot analysis

After adding a genuine grouped split-conformal comparator and ensuring that
the Shannon/JS reference also receives the action-policy predictive-entropy
feature, the unchanged matched pilot rows were reanalyzed in the new
`pilot_v7_analysis` namespace. This did not overwrite Pilot V6. The strongest
non-entropic baseline remained predictive uncertainty. At 50% coverage the
combined block's pilot harm-rate reductions were 0.0333 in humanitarian and
0.0400 in utility restoration; both intervals still included zero. The
private-minus-public interaction was 0.0133 in humanitarian and 0.0400 in
utility restoration, again with intervals spanning zero. These are grounds
for a prospective formal test, not evidence of success.

The pilot also fixed the explicit low-consensus guard at consensus below 0.88
or consensus residual at least 0.25. These cutoffs were chosen from pilot
consensus distributions to ensure the mechanism is exercised, not from final
harm outcomes. Formal primary comparisons remain exactly coverage matched.

## Dynamic-causal Pilot V8

A final untouched pilot (seeds 60961–60965) replaced immediate-effect harm
labels with matched full-horizon action/no-action loss differences from the
same simulator state and stochastic tape. This was a required validity repair,
not an attempt to improve the proposed method. The strongest non-entropic
baseline changed to KPI confidence. At 50% coverage the combined controller's
harm-rate reduction was only 0.0133 in humanitarian (95% pilot interval
-0.0133 to 0.0400) and 0.0067 in utility restoration (-0.0200 to 0.0333).
The private-minus-public interactions were -0.0067 and 0.0067 respectively.

Thus the most scientifically faithful pilot is unfavorable. It does not
justify tuning the environment or thresholds toward a positive result. The
formal frozen development batch remains useful to estimate this boundary with
many independent panels, but validation will remain locked unless every
prospective gate passes.

## Window matching and timing Pilots V9–V11

Pilot V9 corrected a statistical implementation detail: 50% action coverage
is now enforced inside every decision window, not across an entire episode.
On the retained V8 rows, the strongest comparator was KPI confidence. The
combined block's harm-rate reductions were 0.0133 (95% interval -0.0267 to
0.0533) in humanitarian and 0.0267 (-0.0067 to 0.0600) in utility restoration.
Neither result met the frozen practical threshold or excluded zero.

Pilot V10 then exposed a prospective timing defect: the environment had no
pre-disruption decision, while every decision policy could consume an
operator slot. Before protocol freeze, the dynamic panel gained an explicit
nominal state and decision step 0. No future incident mode enters the step-0
belief. Operational `no_action` proposals are not force-escalated. Rare
fixed-rate ambiguous nominal evidence keeps false-activation measurement
nondegenerate. Pilot V10's new dynamic labels remained unfavorable: at matched
coverage, combined harm-rate reduction versus predictive uncertainty was
-0.0133 in humanitarian (interval -0.0533 to 0.0200) and approximately zero
in utility restoration (-0.0400 to 0.0400). This evidence is retained and no
safety outcome was used for escalation calibration.

Pilot V11 (`pilot_v11_timing_final`, untouched seeds 60981–60985) added nominal
panels and calibrated only activation timing and request burden on the fixed
0.50–0.80 grid. The highest threshold satisfying the prospective timing
constraints for both the predictive and combined controllers was 0.80. The
full selection table and panel rows are under
`pilots/pilot_v11_timing_final_analysis/`. This closes pilot iteration; the
environment, feature blocks, threshold, and gates must not change in response
to formal development outcomes.
