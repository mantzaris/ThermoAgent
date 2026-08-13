# Pilot findings

Status: pilot v1 completed at 2026-08-12 04:02 UTC. It contains 54/54 complete
episodes with no failed or timed-out runs: two applications, three paired
seeds, nominal/moderate/communication-loss conditions, and scripted,
learned-no-entropy, and ThermoAgent methods. Scripted episodes use the
deterministic planner; both learned methods use the frozen real Qwen model.

## Throughput and budget

- The sweep took 1,197.36 s (19.96 min) including one model load.
- The 36 real-model episodes used 1,344 calls, 1,598,502 prompt tokens and
  97,188 generated tokens. Mean call payload was 1,189 prompt and 72 generated
  tokens overall. Real-model episode wall time averaged 27.29 s at 8 agents,
  horizon 16, and decision interval 4.
- Scaling by agent decision epochs gives a conservative 45--55 s estimate for
  a full-size horizon-20 agentic episode. Even the current provisional main,
  ablation, and holdout matrices project well below 24 GPU-hours (roughly
  7--10 wall/GPU hours including overhead). The main design therefore need not
  be reduced solely for compute.

## Planner validity and behavior

- Learned-no-entropy: 672 proposals, 100.0% valid structured output, 94.79%
  valid tool execution, 35 failed actions, and 15 later plan revisions.
- ThermoAgent: 672 proposals, 99.70% valid structured output, 85.86% valid tool
  execution, 95 failed actions, and 72 later plan revisions. The lower tool
  validity is retained rather than hidden; the entropy-conditioned checkpoint
  chose emergency actions much more often and exposed more route/tool failures.
- ThermoAgent sent 121 messages across 672 decision epochs (0.1801 per epoch,
  6.72 per episode). This fixes the provisional random-gate probability at
  0.18 for a message-attempt-matched ablation.
- Both learned policies were highly concentrated: learned-no-entropy selected
  option 0 on 592/672 epochs and emergency option 7 on 80/672; ThermoAgent used
  option 0 on 403/672 and option 7 on 268/672 (plus one option 6). Neither
  exercised bilateral negotiation options in this pilot. This policy collapse
  is a major limitation and means the large pilot performance difference is
  not yet evidence for sophisticated coalition negotiation.

## Exploratory outcome diagnostics (not confirmatory)

Every paired v1 seed favored ThermoAgent, but only three seeds were run and the
simulator mechanism was subsequently corrected. Mean lower-is-better primary
outcome improvements versus learned-no-entropy were 10.16 commercial service-
loss units in nominal, 9.96 under the single disruption, and 10.19 under
communication loss. Humanitarian weighted unmet-need improvements were 797.66,
621.73, and 639.77, respectively. Three-seed bootstrap intervals exclude zero,
but the exact sign-flip p-value is 0.333; these are pilot diagnostics only.

The magnitude is suspicious and mechanistically explained: ThermoAgent's
checkpoint frequently invoked emergency direct dispatch, while the non-entropy
checkpoint usually continued locally. In v1, source/route dynamics were too
permissive and did not yet include the audited route closure, facility outage,
or full lead-time constraints. These outcome effects will not be pooled with
the corrected stress pilot or cited as main evidence.

## Monitoring diagnostics

- Raw free energy was a poor one-sided disruption score (average precision
  0.429, ROC AUC 0.404) because it often moved downward after disruption.
- The pre-main absolute deviation from the nominal free-energy median was more
  useful (AP 0.618, ROC AUC 0.808). Operational entropy reached AP 0.713/AUC
  0.674; operational energy reached AP 0.719/AUC 0.688.
- Partitioned communication increased mean consensus RMSE from 0.00398 to
  0.02224, entropy absolute error from 0.00080 to 0.02229, and free-energy
  absolute error from 0.00049 to 0.01351. This supports the expected empirical
  relation between graph connectivity and estimator error.
- These monitoring rows repeat simulator states across methods and are not
  independent experimental units. Episode-level main analysis remains the
  basis for treatment claims.

The physical-route/lead-time completeness correction was made after v1 started.
V1 will be retained for planner throughput, invalid-action, monitoring, message,
and variance diagnostics. A separately named stress pilot will validate the
revised compound dynamics. Results from the two mechanisms will not be pooled.

The remaining pre-freeze gate is the separately named route/compound stress
pilot under the corrected mechanism, followed by the final matrix checksum.
The exact v1 config, sweep manifest, completion CSV/JSONL, raw event ledgers,
and original source checksum are archived and must remain unchanged.

## Corrected-shock v2 extension

V2 completed 90/90 cumulative rows with zero failures; 36 newly executed rows
covered the audited route-closure and compound mechanisms. The extension took
863.01 s (14.38 min), including model load and resuming v1 rows. Its source
checksum is `8719fb0f...` and its exact config/manifest/completion tables are
archived with `_v2` names.

All 12 v2 pairs again favored the old Thermo checkpoint, with mean primary
improvements versus learned-no-entropy ranging from 2.94 to 7.15 commercial
loss units and 324.31 to 544.54 humanitarian unmet-need units. These effects
remain non-confirmatory: the option-collapse audit still showed ThermoAgent on
continue/emergency for 462/480 decisions and no-entropy for 480/480. V2 tested
the shock implementation but not a semantically adequate learned coordination
policy. It will not be pooled with final-v3 or main effects.

V2 nevertheless validated the intended shocks: route closures, capacity loss,
lead inflation, demand surge, facility outage, coordinator loss, and partition
were present in the ledgers; all episodes conserved resources and completed.

## Paired-v5 interruption and paired-v6 qualification plan

Final-v3 was intentionally stopped after 19 new rows when the pre-freeze audit
found action-dependent RNG coupling. Those rows remain useful only for planner
behavior and throughput, and are excluded by a machine-readable prospective
registry. Two separately named paired-v5 conditions (nominal
mixed-information and compound partition) will use regenerated calibration and
checkpoints, event-triggered affected-agent decisions, terminating bargaining,
and route-enabling temporary coalitions. Paired-v5 was then intentionally
stopped after 16 published rows when a further privacy audit found
evaluator-only global consensus error and global interaction entropy in actor
features. No treatment outcomes were inspected; its IDs are machine-excluded
and retained.

The separately named paired-v6 rerun uses link-local neighbor residuals, each
agent's own interaction history, reliability-sampled gossip, completely zeroed
monitor controls, and newly matched checkpoints. Its gate is action diversity,
structured/tool validity, successful contracts/coalitions, and measured
throughput—not a requirement that ThermoAgent outperform.

## Paired-v6 interruption and paired-v7 qualification plan

Paired-v6 was interrupted after 13 published rows, before method outcomes were
compared. A continuing factorial audit found that the privacy factor changed
true marginal costs and agents' own local forecast noise, confounding the
planned information-regime response surface. The exact v6 configuration and
all raw artifacts are retained and excluded. In v7, scenario seeds hold the
economy and local observation quality fixed while privacy controls only what is
shared through the public-information interface. Calibration and policies are
rebuilt under this corrected environment before the same 84-row qualification.

## Paired-v7 interruption and paired-v8 qualification plan

Paired-v7 was interrupted after 10 published rows, before method outcomes were
compared. Ordinary messages used partition probabilities from period 0 whereas
the entropy-sketch graph switched at disruption onset. The exact v7 source,
configuration, and outputs are retained and excluded. V8 uses one time-varying
communication graph for both channels: reliable before onset, disconnected
across components afterward, and 0.85 reliable within each component. The PPO
pair is rebuilt under this corrected timing; nominal calibration is unchanged.

## Paired-v8 final qualification

Paired-v8 completed all 84 planned episodes (7 methods x 2 applications x 2
conditions x 3 paired scenario seeds) with no failed episode. The detached job
exited 0. Before source synchronization or outcome inspection, the original
remote snapshot replayed 84/84 ledgers with no tool-result or metric mismatch;
the maximum absolute material-conservation residual was `3.41e-13`. These are
the first comparative pilot rows eligible under all prospectively documented
RNG, privacy, local-monitor, partition-timing, and accounting boundaries.

Across all seven methods, paired-v8 recorded 3,586 LLM calls, 5,469,722 prompt
tokens, 210,166 generated tokens, and 3,131.0 seconds of summed episode wall
time. The full sweep, including one model load and orchestration, took about 54
minutes. ThermoAgent averaged 95.9 calls and 101.6 episode seconds; learned
coordination without entropy averaged 76.8 calls and 80.2 seconds.

Planner structure was reliable but tool execution remained imperfect:

- learned/no entropy: 99.89% weighted structured validity, 72.31% weighted
  valid-tool rate, 255 failed actions, and 71 later plan revisions;
- ThermoAgent: 99.83% weighted structured validity, 69.16% weighted valid-tool
  rate, 355 failed actions, and 119 later plan revisions.

The independent-agent mechanics were exercised in every eligible learned and
ThermoAgent episode. Raw ledgers show 139 successful no-entropy coalition
proposals and 69 independently accepted joins; ThermoAgent produced 283
successful proposals and 259 independently accepted joins. Every episode also
contained an offer and a plan revision. Counteroffers occurred in three episodes
per learned method. This establishes actual consent and replanning, but the high
coalition count is not itself evidence that those coalitions were useful.

### Exploratory paired outcomes

Lower primary outcomes are better. Against the parameter-matched learned policy
without entropy/free-energy inputs, ThermoAgent improved commercial nominal
service-loss AUC by a mean 0.717 across three seeds, tied exactly in commercial
compound disruption, and tied in mean humanitarian performance in both cells
(the nominal humanitarian seed differences were 0, +90, and -90). Thus the
pilot provides no stable treatment-effect evidence for entropy/free-energy
features.

The wider result was unfavorable to the proposed system and is retained:

- the legal centralized LLM beat ThermoAgent in all 12 paired application/cell
  seeds;
- the privileged full-information lookahead beat it in all 12;
- scripted independent agents beat it in 11 of 12;
- no-communication autonomous agents beat or tied it in all 12;
- fixed-period communication was mixed (two ThermoAgent wins, two ties, eight
  losses across the 12 pairs).

These are three-seed pilot diagnostics, and the central controllers are
strengthened prospectively by the already documented pre-outcome audit for the
main run. They nevertheless show that negotiation activity alone does not
justify autonomous coordination in this environment. The main experiment must
test whether private information/objective conflict creates a boundary where
that conclusion changes; no main setting or checkpoint was altered to make
ThermoAgent win.

### Monitoring and communication

Distributed estimates were accurate when connected and degraded under the
shared partition. Agent-local entropy MAE versus the evaluator was 0.00149
(commercial) and 0.000026 (humanitarian) on reliable graphs, versus 0.03283 and
0.03191 under partition. Free-energy MAE increased from 0.000929/0.000016 to
0.02051/0.01994. Consensus RMSE and entropy/free-energy error had Spearman
correlations near 0.94 under partition and 0.98 on reliable graphs. This
supports estimator convergence, but not control value.

ThermoAgent attempted 1,467 ordinary domain messages and incurred 26,168
mandatory directed sketch transmissions. The latter dominate communication
cost and must be shown separately and jointly. Because coalition fanout made
raw messages per decision exceed one, the final random control matches the
fraction of decision epochs with any validated policy-originated ordinary
message: 624/1,151 = `0.542137`. It does not claim exact byte/fanout matching.

### Frozen budget consequence

The conservative empirical-p90 projection for the planned 1,096 post-freeze
episodes is 20.269 GPU-hours; expected workload is 16.720 GPU-hours, about 3.98
million generated tokens expected (4.75 million p90), and about 136--189 MB of
raw artifacts. At the documented USD 0.34--0.69 hourly assumptions, the
projected range is USD 5.68--11.54 expected or USD 6.89--13.99 at p90. This is
below the 24-GPU-hour cutoff, so the full prospective matrix is retained.
