# V11 audit of the V10 LLM evidence-use methodology

Date: 2026-08-17. This is a prospective V11 design audit. It does not alter
the frozen V10 source or results at commit
`4d372f00837bf75f90882392a92feac87dbc84b2`.

## What V10 actually established

V10's final balanced message pilot produced 48 valid Qwen decisions. Every
agent retained its prior binary belief, so both the right-minus-left choice
difference and paired switch fraction were zero. This is genuine evidence that
the *tested prompt and message interface* did not produce an observable binary
switch. It is not an identified estimate that delivered probabilistic evidence
had no influence.

The implementation did preserve important controls: paired inference seeds,
counterbalanced option order, three paraphrases, balanced prior beliefs, strict
JSON validation, one bounded repair, independent agent objects, explicit inbox
delivery, and no scheduler-side action substitution. The final 48 decisions
were 100% valid on the first pass. No remaining message-routing bug was found.

## Why the V10 null is not a calibrated evidence-use result

| Issue | V10 evidence | Consequence for inference | V11 repair |
|---|---|---|---|
| Semantically weak message | `llm_experiments.py::_message_counterfactual_rows` sent “my evidence supports plan_left/right” | The likelihood of the observation was undefined | Generate typed observations from an explicit binary latent-state model |
| Informal reliability | `DeliveredMessage.influence_weight` was described as stronger or weaker evidential weight | It was not a known probability of signal correctness | Use source reliability `r` with `P(signal=theta)=r` and known LLR |
| Prior and commitment anchoring | `AuthorizedAgentView` exposed current belief, action, confidence, commitment, and memory | Retaining the prior was made cognitively salient | Qualification belief elicitation hides prior action/commitment and begins uncommitted |
| Binary primary outcome | The message gate used belief-spin differences and switches | Rational subthreshold changes were unobserved | Primary outcome is signed change in reported log odds; repeated choice frequency is an independent measurement |
| Coarse self-report | Confidence took only 0.5 or 0.6 in the final message pilot | Confidence was neither directional nor calibrated | Require `probability_right` in `[0,1]` and compare it with repeated empirical choices |
| Belief/action conflation | One schema simultaneously elicited belief, action, commitment, message, and tool use | A commitment-preserving action could anchor belief reporting | Elicit belief first; validate belief, action, and commitment as separate state components |
| Weak plan grounding | `plan_left` and `plan_right` were generic coordination labels | Option semantics may have been too abstract to integrate evidence | Use route-viability and repair-hypothesis framings generated from the same probability model |
| Prompt-format effects | A clarification was needed after the first balanced pilot | Prompt interpretation and treatment were not cleanly separated | Freeze multiple templates and exact matched-prompt diffs before qualification |
| Stochastic sampling | One generation represented each local state | A binary response could miss a probabilistic policy change | Repeat independent samples and cluster by controlled local-information state |
| Self-reported probability risk | V10 did not request a directional probability | A plausible-looking number could be uncalibrated narration | Treat reported probabilities as descriptive unless they agree with empirical choice frequencies |

The final V10 confidence values do show why a continuous estimand matters, but
they do not rescue the old result. Mean paired confidence under right versus
left messages differed by only `-0.0083`, with pair differences spanning
approximately `[-0.10, 0.10]`; the response scale was too coarse and confidence
did not encode a signed probability. The corrected V10 private-evidence pilot
did show a binary right-choice difference of `0.35`, demonstrating that Qwen
could respond to locally worded evidence even though the delivered-message
interface failed.

## Possible explanations retained for V11

- **Genuine message non-use:** possible for the tested prompt; directly tested
  again with calibrated observations and placebo packets.
- **Insufficient message information:** strongly plausible because V10 messages
  had no likelihood model.
- **Binary information loss:** unavoidable in the V10 endpoint; V11 measures
  log-odds shifts and repeated-choice frequencies.
- **Prior/commitment inertia:** plausible because all were exposed and memory
  explicitly mentioned provisional commitment.
- **Option-order or paraphrase effects:** controlled reasonably in the final
  V10 design, but V11 estimates their continuous effects rather than only
  checking a binary coefficient.
- **Sampling variability:** V10 paired seeds reduced but did not remove it;
  V11 uses repeated samples and cluster inference.
- **Invalid self-reported probabilities:** an explicit V11 falsification risk,
  handled by calibration against empirical choices.
- **Implementation bugs:** the earlier V10 prior-balancing bug was retained and
  corrected. No scheduler substitution, inbox leakage, or final routing defect
  was found in the frozen implementation.

## V10 modular-network scaling anomaly

The anomaly is principally a topology-construction and normalization artifact,
with resulting freezing and metastability; it is not evidence for a universal
modular-network law.

`thermoagent/statmech/model.py::topology_adjacency` fixes within-community and
between-community probabilities at `0.22` and `0.025`. Consequently mean degree
grows with `N`, while the local interaction sum is not degree-normalized. In the
formal V10 grid, modular mean degree rose from about `1.81` at `N=8` to `15.52`
at `N=128`. Mean acceptance fell from about `0.39` to `0.0117`, absolute order
grew, and the pathwise irreversibility estimate approached zero.

A retained audit rerun used ten times the original burn-in and sampling length
at `T=1.65`, `alpha=0.35`. The original `N=64` graph had acceptance `0.0759`,
order `-0.914`, and EPR `1.16e-4` per update. At `N=128`, acceptance was
`0.00637`, order `0.994`, and EPR was numerically zero. Longer trajectories did
not restore mixing. Dividing the same adjacency by mean degree restored
acceptance to `0.464` and `0.467` and produced nonzero estimates `0.00135` and
`0.000579`, respectively.

Thus the apparent large-`N` collapse combines an increasing-degree ensemble,
unnormalized coupling, saturation of local fields, and metastable freezing.
The estimator itself passes exact and synthetic-chain checks. Insufficient
trajectory length aggravates uncertainty but is not the primary cause. V11
does not use the V10 modular scaling curve as evidence and will either fix mean
degree prospectively or normalize coupling when a modular topology is used.

