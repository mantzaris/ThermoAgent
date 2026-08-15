# V5 audit of the V4 mechanism

This audit was completed before the V5 simulator or protocol was frozen. It
uses the immutable V4 evidence at `8ccd27df...` and does not revise V4.

1. **Confirmed:** every dense counterfactual candidate was hard-coded as
   `authorize_verification` in `V4EpisodeRunner._dense_candidate_rows`.
2. **Confirmed:** entropy anomaly and disagreement tracked verification value.
   In private-fragmented panels, entropy/effect correlations were 0.839 and
   0.806 for humanitarian and utility; disagreement correlations were 0.766
   and 0.756.
3. **Confirmed:** all globally-public candidate effects were exactly zero in
   all three applications. The environment made ambiguous incidents already
   verified, so this was a trivial rather than informative public control.
4. **Partly confirmed:** commercial KPI actionability varied while it was
   constant in the two positive applications. Commercial KPI and augmented
   policies selected the same incident and achieved exact equal utility.
5. **Confirmed:** several humanitarian/utility KPI fields were constant:
   service deficit, lateness, safety stress, scarcity, and actionability had
   zero or numerical-zero variance; backlog variance was about 1.1e-5.
6. **Confirmed:** entropy/disagreement-only AP and AUC were 1.0/1.0 for
   humanitarian and 0.990/0.992 for utility. This is implausibly easy evidence,
   not a robust ranking challenge.
7. **Confirmed:** thermodynamic designs were rank deficient or nearly so.
   Direct private-panel condition numbers were 5.36e16 commercial, 8.90e15
   humanitarian, and 1.14e16 utility; reported block condition numbers reached
   1e17-1e30.
8. **Confirmed:** many of the 60 panels per application were same-choice ties.
   Candidate pools contained only one fixed action and repeated incident
   structures; the public condition was entirely an outcome tie.
9. **Confirmed:** `conditional_permutation_test` shuffled prediction inputs
   without refitting the complete cross-fit/model-selection pipeline. It is a
   sensitivity analysis, not a formal refit permutation test.
10. **Confirmed:** energy-only matched or underperformed KPI-only; free energy
    was highly collinear and showed no distinct causal decision value.
11. **Confirmed:** 1,584 formal episodes used deterministic planners. Real
    Qwen supplied only six one-call qualification episodes.
12. **Confirmed:** 771,840 sketch messages (169,821,262 bytes) dwarfed 588
    ordinary messages (131,544 bytes).
13. **Confirmed:** low-confidence decisions were zero, so abstention was not
    exercised.

## V5 consequences fixed before outcomes

- Four simultaneous incidents per panel, multiple plausible actions, continuous
  overlapping features, imperfect/costly verification, and bounded harm.
- Public information retains nonzero action effects; it changes information,
  not whether interventions matter.
- Entropy/disagreement are not direct encodings of the optimal action.
- Grouped model fitting and refit-based stratified permutation tests operate at
  environment-panel level.
- Event-triggered sketch exchange is the primary deployable monitor and every
  sketch byte is charged.
- Low-confidence partition cases are deliberately included.
- Deterministic agents remain controls; formal agent evidence requires
  decentralized learned policies and a substantially larger Qwen evaluation.

These changes are informed by the V4 boundary result and are not evidence about
V5 outcomes.
