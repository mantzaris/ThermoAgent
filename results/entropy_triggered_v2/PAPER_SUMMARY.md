# Paper-oriented summary

## Working title

Distributed Operational Entropy Triggering for Communication-Efficient Autonomous Logistics Coordination

## Provisional abstract

Autonomous logistics agents can benefit from rich communication during disruptions, but always-on coordination imposes message, inference, and negotiation costs. We introduce Distributed Operational Entropy Triggering (DOET), a decentralized stateful trigger computed from privacy-preserving, locally gossiped operational macrostate sketches. DOET regulates quiet, targeted, and crisis communication modes while leaving planning and commitment authority inside independent agents. We evaluate always-on fixed, learned non-entropic, DOET-rule, and DOET-RL on every panel; periodic, budget-matched, original ThermoAgent, no-communication, and local-KPI controls use a prospectively fixed common subset under the compute cap. The study spans commercial and abstract humanitarian logistics. The preregistered holdout contains 696 episodes and five independently trained seeds per learned method. The locked evaluation establishes a negative mechanistic boundary: the selected trigger never activated, so observed savings cannot be attributed to entropy-triggered coordination. Exact effect estimates and uncertainty are reported in the tables below rather than summarized with an unsupported positive claim.

## Verified contributions

1. An implemented decentralized, privacy-preserving, fully counted event-trigger architecture with independent agent authority; the locked run did not demonstrate successful trigger activation.
2. A replay-backed mechanistic diagnosis of why the original frozen holdout tied despite policy divergence.
3. Monitoring evidence separating globally redundant entropy from its incremental value under private local information.
4. A multiple-training-seed, paired, frozen-holdout comparison against always-on communication and budget-matched controls.
5. Exact replay, conservation, communication/inference accounting, and mechanistic trigger analyses.

## Primary numerical results

| Application | Loss degradation vs fixed | One-sided 95% upper | Non-inferior | Message reduction (95% CI) |
|---|---:|---:|:---:|---:|
| Commercial | 1.00% | 1.56% | yes | 72.4% [71.0%, 73.7%] |
| Humanitarian | 0.38% | 0.58% | yes | 74.2% [73.0%, 75.4%] |

| Application | Structured bytes | Prompt tokens | Generated tokens | LLM calls | Inference latency | Wall-clock time |
|---|---:|---:|---:|---:|---:|---:|
| Commercial | 68.6% | 37.8% | 41.5% | 36.8% | 38.8% | 38.8% |
| Humanitarian | 70.6% | 41.9% | 42.5% | 38.5% | 39.9% | 39.9% |

## Mechanistic interpretation

- `doet_rule`: 0/144 episodes activated; 0 total activations; mean quiet-mode fraction 1.000; maximum observed trigger residual 0.618 versus `tau_on=1.200`.
- `doet_rl`: 0/144 episodes activated; 0 total activations; mean quiet-mode fraction 1.000; maximum observed trigger residual 0.618 versus `tau_on=1.200`.

H1, H2, and the formal cross-application endpoint H6 are supported as preregistered statistical statements. They do not validate the proposed entropy mechanism: both DOET variants remained in quiet mode in every locked episode, H4 and H5 failed, H3 failed, and the common-panel no-communication control dominated DOET-rule on loss and messages in both applications.

### Exploratory control result

- All 60 local DOET-variant episodes and all 12 exact-global-entropy oracle episodes had zero activations.
- The private-KPI control activated in 12/12 episodes and changed mean loss by -0.023% while using 72.1% more messages than selected DOET.
- The putative disruption-label oracle activated in 12/12 episodes and changed mean loss by -0.300% while using 40.0% more messages than selected DOET. Ledger timing shows both active controls first fired at period 0, eight periods before disruption; these are false activations, not timely alarms. The binary-label oracle inherited the selected low-direction transform, so label 0 was treated as anomalously low; it is retained as an invalid exploratory oracle implementation, not an upper bound.

## Hypothesis outcomes

- `H1`: **supported**. Frozen success criterion: DOET-rule non-inferior to fixed in both applications after Holm correction
- `H2`: **supported**. Frozen success criterion: message reduction CI excludes zero and mean reduction >=20% in both applications after Holm correction
- `H3`: **unsupported**. Frozen success criterion: DOET-rule is loss-message nondominated and strictly increases the frozen normalized frontier hypervolume for messages, prompt tokens, calls, and latency in both applications
- `H4`: **unsupported**. Frozen success criterion: >=75% first post-disruption activation before sustained severe collapse (service loss >=0.90 for three consecutive periods), severe collapse observed in every non-nominal episode, <=10% pre-disruption false activation, and <=10% nominal episode false activation
- `H5`: **unsupported**. Frozen success criterion: non-inferior in partition and compound-partition regimes in both applications, with positive consensus-RMSE/degradation slope and Pearson r >=0.20 in each application
- `H6`: **supported**. Frozen success criterion: H1 and H2 supported in both applications


## Figure plan

The paper-facing set comprises the DOET architecture; original tie diagnosis; monitoring baselines and incremental value; trigger dynamics; loss–communication Pareto frontier; non-inferiority forest; communication reduction; multiple-seed curves and variability; locked primary results; partition robustness; trigger ablations; commercial/humanitarian event studies; and an entropy-triggered network sequence. All are vector PDFs with rendered QA previews.

## Table plan

Experimental design, RL seeds, trigger parameters, communication budgets, monitoring controls, paired comparisons, non-inferiority, reductions, Pareto points/hypervolume, holdout summaries, trigger ablations, compute/tokens, failed runs, and hypothesis outcomes are under `tables/`.

## Limitations and recommendation

The strongest limitations are the absent trigger activation, synthetic environments, one primary language model, abstract humanitarian roles, deterministic decoding, and restricted topology/model diversity. The result does not establish literal thermodynamic behavior, useful entropy-triggered coordination, or general autonomous-agent necessity. Recommendation: **insufficient for the intended AIJ submission**. Any manuscript must retain the original negative study, global-KPI redundancy, all failed/unstable seeds, the zero-activation mechanism, no-communication dominance, and application/regime-specific exceptions.
