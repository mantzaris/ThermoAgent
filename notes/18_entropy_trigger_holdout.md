# DOET locked holdout record

Status: complete; outcome seal lifted only after the full run and replay gate.

## Independence from seen evidence

The original v1 holdout was examined during Phase A and is diagnostic-only. It
was not reused for v2 confirmation. New environment seeds 8101--8116, nominal
seeds 8201--8208, new LLM sampling seed 9101, and unseen topology
`tri_region_bridge_v2` were written before launch. Development, validation, and
holdout files occupy separate namespaces.

## Frozen design

- 144 matched base scenario panels.
- Per application: 16 seeds in each of isolated disruption, communication
  partition, correlated disruption, and compound OOD; 8 nominal seeds.
- 128 non-nominal and 16 nominal panels in total.
- 696 method episodes. Fixed always-on, learned non-entropic, DOET-rule, and
  DOET-RL ran on every applicable panel; secondary methods used the same frozen
  compute-priority subset.
- Five independently trained checkpoints for each learned method, balanced
  across matched panels.
- One complete episode is the experimental unit.

The frozen primary method was DOET-rule; the benchmark was always-on fixed
communication; relative non-inferiority margin was 2%; the practical
communication target was at least 20% reduction. Analysis, bootstrap seed and
replicate count, hypothesis rules, plots, and checkpoint selection were frozen
before the first episode.

## Provenance and freeze

- Branch: `entropy-triggered-communication`.
- Clean execution commit:
  `09ac91b72dd7fb5151fc6af2c28da9855653b2dc`.
- Execution source checksum:
  `655cb19264b51a33b47273c28c990f07eb85a0f9caa54da2b8ab4d96509e06c9`.
- Holdout freeze checksum:
  `34470a323c0b1adc868b8c1e67aded847a43631fb4feb507120c5efc77db18ec`.
- Frozen at: `2026-08-13T23:11:35.164930+00:00`.
- Locked input checksum opened by analysis:
  `f73608583849a5b063038dcfdec6e3bb3cd59969fcc7b5f969f60dd1167e9338`.

`notes/14_entropy_trigger_protocol.md` is deliberately left byte-exact because
its checksum is part of the holdout freeze. Its pre-launch status wording is
therefore historical. This note is the authoritative post-holdout record.

## Outcome-sealed execution

The detached RunPod job started at `2026-08-13T23:13:07.697032+00:00` and ended
at `2026-08-14T14:11:16.161732+00:00`. During execution, inspection was limited
to process health, completion counts, file/schema presence, finite values, and
catastrophic engineering failure. No partial method outcome was opened and no
threshold, prompt, checkpoint, seed, method, or analysis rule was changed.

The sweep completed 696/696 episodes with zero failed or timed-out rows. It
used 56,653 LLM calls, 100,908,718 prompt tokens, 3,745,964 generated tokens,
and 14.8746 summed episode GPU-hours. The including-load elapsed time was
53,888.5 seconds. The full v2 compute account, including profile, validation,
training, holdout, model loads, and later authorized controls, is 22.0623
single-GPU hours and approximately $7.50 at $0.34/hour.

The holdout artifacts were fetched only after job completion. The transferred
v2 archive SHA-256 was
`c4a7da92504c491ede7d0b3f78420d0829c97d99e85aa8bbd19907ce2a0527dc`.
Archive paths were validated before merging, and only
`results/entropy_triggered_v2/` was installed locally.

## Replay and outcome opening

The original locked holdout replay passed 696/696 before interpretation. With
validation and the later exploratory controls included, the final report passes
936/936 ledgers, has no public-metric or tool-result mismatch, and has maximum
absolute conservation residual `4.55e-13`.

The first remote postprocessing wrapper returned exit 1 after replay and
statistics succeeded because three diagnostic PDFs were intentionally absent
from the filtered remote copy. No experiment failed. The already-generated
diagnostic PDFs were fetched from the local namespace, and only derived figures
were rebuilt. No ledger, episode, prompt output, or statistical input changed.

The post-holdout exploratory-control wrapper also returned exit 1 because 12
alerting episodes exposed a replay event-order bug. Recorded tool results and
conservation already matched. Replay had applied `entropy_alert` protocol
messages after the same-period public metric snapshot, while live execution
applies them before that snapshot. The replay order and regression test were
corrected without rerunning any episode; all 96 control ledgers and the full
936-ledger set now replay exactly. This post-holdout correction is documented in
`reproducibility/post_holdout_presentation_changes.json`.

## Locked findings

DOET-rule meets the aggregate H1/H2 endpoints in both applications, but both
DOET variants remained quiet in all episodes. Maximum trigger statistic was
0.618 against `tau_on=1.2`. Consequently H4 and H5 fail, DOET-rule is Pareto-
dominated by no communication, and H3 fails. The formal H6 endpoint passes
because H1/H2 pass in both applications; it is not causal evidence for entropy.
Full numbers are in `notes/17_entropy_trigger_main_results.md` and
`results/entropy_triggered_v2/README.md`.

## Integrity statement

No holdout output was used to tune the trigger, change entropy direction,
select a checkpoint, alter the 2% margin, set a communication budget, remove a
seed, or selectively rerun an episode. The failure to activate and the invalid
exploratory label oracle are retained. Any revised trigger requires a new v3
validation stage and a genuinely unseen holdout.
