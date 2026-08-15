# V5 pilot findings and protocol revision

Evidence stage: development pilot only. No V5 formal-development, validation,
training, or holdout outcome had been generated when this revision was made.

## Retained iteration 1

The complete first pilot is preserved under
`results/human_operator_v5/development/pilot/` with 168 episodes, 6,048 action
candidates, and zero technical failures. It used protocol 5.0.0 and seeds
50901-50904 across every application, regime, and information condition.

The pilot found three prospective feasibility failures:

- utility-restoration fixed coordination reduced loss by 2.67% on average,
  below the unchanged 3% Gate 3 threshold;
- event-triggered sketches averaged roughly 46-48 messages per panel versus 60
  for always-on exchange, insufficient for the unchanged 50% byte-reduction
  requirement;
- low-consensus cases occurred in only 3.1% of incident observations in the
  most affected partition cells, so Gate 8 was not meaningfully exercised.

The pilot also showed that public-information intervention effects were nonzero,
beneficial and harmful candidates coexisted, and bounded operator interventions
had causal value. Average operator time exceeded the prospective 16-minute
panel cap in two applications because the implementation charged 8 plus action
delay minutes, inconsistent with the intended 6-minute bounded decision model.

## Protocol 5.0.1 changes

No gate, hypothesis, application, endpoint, seed set, or primary comparison was
changed. The following development mechanics were changed before formal V5
development:

1. Decentralized pooled beliefs now use each agent's locally observed telemetry
   confidence rather than unweighted averaging. Confidence is calibrated to
   evidence quality with noise; no agent receives the true state.
2. A severe abstract partition isolates two of three incident contributors
   after the split. Messages are still charged, and the estimator retains only
   delivered sketches. This deliberately creates the low-confidence cases
   required by the already frozen abstention gate.
3. Event exchange thresholds changed from entropy/distribution deltas
   0.075/0.050 to 0.120/0.140. The goal is to eliminate near-periodic gossip;
   Gate 9 remains unchanged.
4. Ordinary bounded interventions now use six plus action-delay operator
   minutes, with verification at six and peer evidence at four. This corrects
   the implementation to the intended time-accounting convention; the
   16-minute gate remains unchanged.

Protocol 5.0.0 and its complete outputs remain in the repository. Protocol
5.0.1 will be tested as a complete second pilot using all designated pilot
seeds; no individual episode is selectively rerun.

## Retained iteration 2 and protocol 5.0.2

The complete second pilot is preserved under
`results/human_operator_v5/development/pilot_iteration_2/` (168 episodes,
6,048 candidates, zero technical failures). It resolved sketch compression,
low-confidence coverage, and operator-time feasibility, but utility-restoration
coordination still achieved only 2.46% relative loss reduction and therefore
remained below the unchanged 3% Gate 3 threshold.

Inspection showed that the simulator gave agents strongly conflicting signals
whose errors remained correlated after exchange; reliability weighting could
not aggregate complementary evidence. Protocol 5.0.2 narrows private-evidence
noise and the rotating spurious-signal term while retaining overlapping beliefs
and continuous uncertainty. It also treats a negotiated proposal as rejected
only when every recipient rejects; a single rejection no longer silently
replaces an accepted decentralized commitment. Agents retain reject and
counteroffer authority, and no truth label is exposed. No gate or endpoint was
changed. A complete third pilot, rather than selected reruns, is required before
formal development.
## Retained iteration 3 and protocol 5.0.3

The complete third pilot is preserved under
`results/human_operator_v5/development/pilot_iteration_3/` (168 episodes,
6,048 candidates, zero failures). Humanitarian coordination passed the
prospective practical threshold, but utility restoration again reached only
2.34% aggregate reduction. Incident-level inspection showed only two more
correct pooled actions than local actions across 96 disrupted utility
incidents: the supposedly distinct utility roles still received statistically
interchangeable evidence.

Protocol 5.0.3 implements the preplanned fragmented-observability mechanism:
utility zone, field/communications, and cyber/resource roles receive different
evidence precision. The local zone observation is weakest; other private roles
contribute progressively more reliable but still noisy evidence. No agent sees
the truth label, correct action, peer observation, or evaluator state. This is
an application-mechanics correction, not a thermodynamic feature shortcut, and
the coordination gate is unchanged. All designated pilot cells are rerun as a
fourth complete iteration before formal development.

## Pilot iteration 4 disposition

The complete protocol-5.0.3 pilot is preserved under
`results/human_operator_v5/development/pilot_iteration_4/` (168 episodes,
6,048 candidates, zero technical failures). Aggregate private-fragmented
coordination loss reductions were 7.30% commercial, 8.04% humanitarian, and
11.64% utility restoration. The two primary applications therefore cleared the
unchanged 3% practical pilot target. Average bounded-operator time was 14.38
minutes in humanitarian and 13.58 minutes in utility restoration, below the
16-minute limit. Event-triggered exchange remained compressed, and severe
partitions produced nontrivial low-confidence coverage.

This passes engineering evaluability only. It is not a formal gate result and
does not support the paper claim. Protocol 5.0.3 is now frozen without further
simulator or threshold changes for the 20-seed formal-development batch.
