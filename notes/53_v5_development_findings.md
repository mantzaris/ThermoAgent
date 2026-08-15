# V5 development findings

Evidence boundary: development only. Operators are simulated. Formal causal
panels use deterministic independent-agent engineering controls; a separate
qualification uses the pinned real Qwen model; decentralized RL is reported
separately. No V5 validation or holdout evidence exists.

## What worked

- The valid formal stage completed 840 panels and 30,240 intervention
  candidates over commercial, humanitarian, and utility-restoration systems.
- Fixed decentralized coordination reduced loss relative to no communication
  by 7.72% commercially, 6.48% humanitarianly, and 12.17% in utility
  restoration. All 90% cluster intervals were positive, 54%-69% of panels
  changed outcome, and all six disrupted regimes improved.
- The bounded simulated-operator oracle reduced loss by 13.24%, 12.92%, and
  12.57%, respectively, using about 14.2-14.4 simulated operator minutes per
  panel. Complete alert-to-service causal chains occurred in 78.3%-82.9% of
  panels.
- Low-consensus abstention was exercised in 23.8%-25.0% of the selected
  stress-regime panels. It reduced harmful selections by 45.7% in humanitarian
  logistics and 69.0% in utility restoration while improving utility and not
  degrading service loss.
- Event-triggered sketches reduced bytes by 61.6%-63.5% versus always-on
  sketches in the private-information conditions. All sketch traffic is
  included in the accounting.

## Primary no-go result

At the same two-incident operator budget, KPI plus entropy/disagreement did
not improve causal intervention utility over KPI-only:

- commercial boundary: -0.0151, 95% CI [-0.0489, 0.0174];
- humanitarian: -0.0116, 95% CI [-0.0279, 0.0043];
- utility restoration: -0.0100, 95% CI [-0.0277, 0.0068].

The thermodynamic policy also selected harmful interventions more often in
both primary applications. Private-minus-public interactions were +0.0136
[-0.0047, 0.0321] humanitarianly and +0.0055 [-0.0157, 0.0269] in utility
restoration—directionally suggestive but not supported. Thus V5 does not show
that distributed entropy or disagreement improves scarce human-attention
allocation beyond same-boundary KPIs.

Trigger feasibility also failed: nominal false activations were 30.0%-32.5%,
and utility restoration achieved only 74.2% timely activation with 25.8%
misses. Event-triggered sketch compression cannot rescue negative causal
incremental value.

## Actionability and agent evidence

Deterministic independent-agent controls met formal action progression: over
92.9% of actions were accepted in every application and 76.6%-79.7% reached
service. Every panel contained negotiation or commitment revision, and six
autonomous action types were observed.

The real-Qwen qualification comprised 36 episodes and 108 decision epochs,
with 100% first-pass validity and no repairs. Material acceptance was
83.3%-86.1%. However, service-reaching rates were 22.2% commercial, 38.9%
humanitarian, and 63.9% utility; utility action divergence under different
private evidence was zero and action diversity was only three. Consequently,
the strict autonomous-agent qualification gate failed rather than being
waived.

Candidate intervention effects include bounded direct action harm plus the
opportunity cost of replacing an effective autonomous action. The latter can
make the paired relative causal loss exceed the direct 0.24 gross-harm term;
the result package reports both quantities and does not call the relative
effect the direct harm cap.
