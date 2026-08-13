# Main findings

Status: final frozen main, ablation, and locked-holdout evaluation is complete.
All 1,096 post-freeze episodes completed, and immutable event-ledger replay
passed 1,096/1,096 with zero metric mismatches, zero tool-result mismatches,
and maximum absolute material-conservation residual below `4.55e-13`. No seed
was selectively rerun. Primary outcomes and comparison families were fixed in
`01_methodology_decisions.md` before outcomes were opened.

Lower primary outcomes are better. In all comparison tables, positive
`mean_improvement` means the comparator's loss minus ThermoAgent's loss and
therefore favors ThermoAgent.

## Primary main comparisons

The parameter-matched entropy comparison was encouraging in the 944-episode
main sweep, but it did not cross the prespecified familywise 5% threshold.

- Commercial: ThermoAgent's mean service-loss AUC was `16.568` versus `16.755`
  without entropy/free energy. The paired improvement was `0.187` (1.12%),
  hierarchical bootstrap 95% CI `[0.049, 0.310]`, paired win rate `0.486`,
  probability of superiority `0.590`, and standardized seed-cluster effect
  `d_z=0.932`. The sign-flip p-value was `0.0428`, but Holm adjustment over the
  five preregistered commercial comparisons yielded `p=0.0856`.
- Humanitarian: mean cumulative unmet weighted need was `5815.83` versus
  `5922.34`; improvement `106.51` (1.80%), CI `[57.90, 156.58]`, paired win rate
  `0.514`, probability of superiority `0.715`, and `d_z=1.386`. The sign-flip
  p-value was `0.0195`, but Holm-adjusted `p=0.0584`.

These are suggestive main effects, not confirmatory rejections after the
prespecified multiplicity correction. Magnitude-weighted means can be positive
despite many ties or losses, which is why win rates and individual seed points
remain visible.

Strong alternatives generally did as well or better:

| Application | Comparator | Mean improvement | 95% CI | Holm p | Interpretation |
|---|---:|---:|---:|---:|---|
| Commercial | fixed communication | `-0.722` | `[-0.968, -0.478]` | `0.0584` | fixed communication lower loss |
| Commercial | scripted independent | `-0.681` | `[-1.148, -0.264]` | `0.0584` | scripted lower loss |
| Commercial | legal central LLM | `-0.310` | `[-0.532, -0.088]` | `0.0856` | central LLM lower loss |
| Commercial | full-information lookahead | `-15.370` | `[-15.770, -14.966]` | `0.0584` | upper bound overwhelmingly better |
| Humanitarian | fixed communication | `-3.47` | `[-50.25, 46.54]` | `0.9066` | practically tied |
| Humanitarian | scripted independent | `-143.12` | `[-294.31, 14.52]` | `0.2568` | numerically scripted-favoring, uncertain |
| Humanitarian | legal central LLM | `-182.67` | `[-248.16, -117.18]` | `0.0584` | central LLM lower loss |
| Humanitarian | full-information lookahead | `-4958.26` | `[-5193.08, -4741.31]` | `0.0584` | upper bound overwhelmingly better |

The legal central LLM is not privileged with hidden private state; it receives
only legally disclosed reports and the public physical graph. The numerical
lookahead is deliberately privileged and is interpreted only as an upper bound.

## Boundary between autonomy and simpler control

The descriptive necessity response surface is uniformly negative. Relative to
one fixed deployable benchmark selected by across-seed mean per factor cell,
ThermoAgent's normalized advantage ranged from `-3%` to `-9%` in commercial
and `-2%` to `-14%` in humanitarian logistics. Increasing private information
or objective misalignment did not create a cell in which ThermoAgent beat the
best observed fixed/legal-central/scripted alternative. This directly
contradicts the proposed monotone-autonomy hypothesis in these environments.

The response surface is descriptive rather than an additional inferential
family: the benchmark is chosen once per cell, never separately for each seed,
and original seed pairing is retained.

## Locked holdout

The 80-row holdout used unseen nine-agent topologies, four new seeds, and two
new correlated/compound disruption combinations. ThermoAgent and learned
coordination without entropy tied exactly in both applications across all eight
paired cells. ThermoAgent also tied scripted agents exactly. Fixed communication
was slightly better commercially (`-0.167`, CI `[-0.500, 0]`) and tied in
humanitarian logistics. Full-information lookahead remained much better:
`-10.703` commercial and `-3418.71` humanitarian. There were no failures.

Thus, the modest main entropy effect did not transfer to the locked holdout.
The many exact ties also indicate limited action sensitivity or saturation in
these particular unseen cells; they do not show equivalence outside this test.

## Ablations

Each compound-partition ablation has four paired seeds per application, so it
is deliberately limited. No Holm-adjusted ablation comparison was significant.

- Commercial ThermoAgent versus no entropy: `-0.076`, CI `[-0.579, 0.535]`.
- Humanitarian ThermoAgent versus no entropy: exact mean tie.
- Commercial exact-global entropy oracle: ThermoAgent `+0.088`, CI `[0, 0.264]`;
  humanitarian oracle favored the control by `42.96`, CI `[-128.87, 0]`.
- Entropy-to-LLM without an RL gate, activity-matched random gating, shuffled
  delayed entropy, and no episodic memory were mostly tied or imprecise.

The ablations do not isolate a reliable causal benefit for distributed
entropy, free energy, memory, or learned communication gating.

## Agentic behavior and communication

ThermoAgent exhibited more visible coordination, but activity was not reliably
useful. Main per-episode means were:

| Application | Method | structured valid | tool valid | revisions | proposals | formed | useful precision | breaches | failed actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Commercial | ThermoAgent | `0.928` | `0.534` | `16.13` | `28.07` | `8.54` | `0.059` | `0.93` | `63.49` |
| Commercial | no entropy | `0.922` | `0.589` | `14.89` | `19.57` | `6.69` | `0.032` | `1.96` | `53.40` |
| Humanitarian | ThermoAgent | `0.975` | `0.579` | `13.82` | `27.33` | `9.83` | `0.028` | `0.79` | `55.54` |
| Humanitarian | no entropy | `0.956` | `0.651` | `11.00` | `18.85` | `7.33` | `0.013` | `2.47` | `40.82` |

Across post-freeze rows, commercial ThermoAgent sent 15,449 ordinary messages
plus 303,482 mandatory sketches; humanitarian ThermoAgent sent 14,393 ordinary
messages plus 287,722 sketches. It reduced breaches and formed more coalitions,
but produced more failed actions and a lower valid-tool rate. Coalition utility
precision remained below 6%. The Pareto analysis therefore shows a large
communication cost without a broad logistics advantage.

## Monitoring and statistical mechanics

Operational entropy and energy were useful evaluative signals; the calibrated
free-energy gap was not a reliable high-direction detector.

- Exact operational entropy: disruption average precision `0.934`, ROC AUC
  `0.863`, precision `0.970`, recall `0.539`, and nominal false-alarm rate
  `0.030` at the nominal 95th-percentile threshold.
- Exact operational energy: AP `0.885`, ROC AUC `0.800`, but false alarms
  `0.755` at its nominal threshold.
- Exact free-energy gap: AP `0.577`, ROC AUC `0.393`, and false alarms `0.826`.
  The primary free-energy mean fell under disruption by `0.0108` commercially
  and `0.0046` in humanitarian logistics. This repeats the pilot sign problem.
- Interaction entropy alone was weak (AP `0.551`, ROC AUC `0.411`). Its joint
  regimes remain useful descriptively but not as a stand-alone alarm.

Energy results were fairly robust to fixed alternative weights. Free-energy
direction and discrimination stayed weak across all three sensitivity variants.
Where alarms fired, median delay was generally zero periods after shock onset
and before visible service collapse. The monitor therefore recognizes onset
before service degradation becomes obvious; it does not predict an exogenous
shock before the shock occurs.

Distributed gossip tracked evaluator-only values accurately when connected:

| Application | Links | entropy MAE | free-energy MAE |
|---|---|---:|---:|
| Commercial | reliable | `0.00524` | `0.00327` |
| Commercial | intermittent | `0.01833` | `0.01145` |
| Commercial | partition | `0.03976` | `0.02485` |
| Humanitarian | reliable | `0.00072` | `0.00045` |
| Humanitarian | intermittent | `0.02708` | `0.01692` |
| Humanitarian | partition | `0.06280` | `0.03924` |

Consensus RMSE and estimator error were strongly associated (`rho` about
`0.966` on reliable links and `0.985`--`0.996` with impaired links), an empirical
verification of expected degradation. Commercial source localization was
strong (top-1 `0.946`--`1.0`, top-3 `1.0`). Humanitarian top-1 localization was
poor in several large main cells (`0`--`0.089`) but top-3 was always `1.0` and
the small/holdout cells reached top-1 `1.0`; localization is therefore mixed.

## Compute and completion

Post-freeze sweeps used 18.592 wall/GPU-hours including three model loads,
69,533 LLM calls, 116,469,832 prompt tokens, and 4,468,148 generated tokens.
The main, ablation, and holdout stages took 15.405, 2.146, and 1.040 hours.
At the documented RTX 4090 rates of `$0.34` and `$0.69` per hour, active
post-freeze compute is approximately `$6.32`--`$12.83`; actual Pod-console
billing controls. All retained smoke, pilot, and final episode records total
20.174 summed episode-hours, 126,742,334 prompt tokens, and 4,952,073 generated
tokens, excluding one-time loads not attached to an episode.

## Bottom line

The architecture and distributed monitor work. Operational entropy detects
collective disruption and distributed estimates degrade coherently under
communication loss. The tested coordination policy, however, did not justify
its autonomy or communication cost relative to strong simpler controls. The
entropy-conditioned actor has suggestive in-distribution gains over its matched
non-entropic actor, but those gains fail familywise correction and disappear on
the locked holdout. The free-energy formulation is specifically unsupported as
a high-direction alarm. These negative and mixed findings are the final result,
not a trigger for post-hoc retuning.
