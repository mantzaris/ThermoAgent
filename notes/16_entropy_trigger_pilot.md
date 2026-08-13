# DOET development and validation log

The real-LLM validation completed 144/144 episodes and replayed 144/144 exactly.
The outcome seal was lifted only after completion and replay. The current
RunPod proxy remains reachable, and the launch source checksum is retained.

## Pre-outcome H4 evaluability audit

Before inspecting the active validation results, the timing-analysis code was
tested against the existing mock preflight and real-Qwen throughput-profile
episodes. The initial `pre-disruption mean + 0.10` service-loss rule classified
the disruption period itself as collapse in 12/12 non-nominal engineering
episodes, leaving no strict lead-time window. This was a structural measurement
failure. The rule is now a sustained severe threshold: service loss at least
0.90 for three consecutive post-disruption periods, with collapse timestamped
at the third period. It creates a two- or four-period window in all 12 audit
episodes. The primary DOET cases had zero activations, so the definition was not
selected to reward an observed trigger. The complete audit is
`results/entropy_triggered_v2/protocol/h4_evaluability_audit.json`.

Completed local work:

- 12 nominal monitor-only calibration episodes, seeds `5101`--`5106`;
- 18 disrupted monitor-only development episodes, seeds `5201`--`5203`;
- eight deterministic mock planner preflight episodes, all complete and exactly
  replayed;
- maximum absolute calibration conservation residual `6.82e-13`;
- maximum absolute preflight conservation residual below `1.14e-13`.

The monitor-only episodes set activation thresholds to unreachable values and
therefore cannot be used to infer treatment performance. Their only purposes
are nominal normalization and the prospectively declared direction diagnosis.

The final pre-validation fairness audit found that an earlier draft matched
random and periodic controls by active-state fraction and inadvertently made
inactive controls silent planners. No real validation outcome had been run.
Before validation, the design was corrected prospectively: inactive controls
plan locally every eight periods, intensive activation is evaluated on a
two-period opportunity grid, and validation converts DOET's fully counted
messages (including sketches) into matched random/periodic rates using the
fixed control's observed messages per active decision. The final analysis also
reports achieved budget mismatch.

The low-direction development leader and its weak transferred recall are
recorded in `notes/14_entropy_trigger_protocol.md`; all alternative direction
rows remain in `results/entropy_triggered_v2/calibration/direction_diagnostics.csv`.

Next stages on the connected Pod:

1. deploy the post-validation, source-equivalent training revision;
2. train five independent seeds for each learned method without seed removal;
3. generate the measured-runtime holdout design;
4. checksum-freeze source, protocol, thresholds, checkpoints, and analysis;
5. launch the genuinely unseen outcome-sealed holdout.

The automatic training handoff was stopped before validation completion after
a pre-training audit found that the DOET-RL trainer did not resolve the
validation-selected nominal-normalizer file. No training attempt existed and
no validation outcome was inspected. The active validation was left untouched.
The correction is covered by two fail-closed tests and will be deployed only
after validation artifacts are complete and retrieved.

## Completed real-LLM validation

The frozen selector chose `hysteresis_low`. Across 18 non-nominal paired panels,
commercial primary loss was 0.792% worse than fixed communication and
humanitarian primary loss tied exactly. The worst application/regime mean
degradation was 1.074%, under the frozen 2% cell limit. Fully counted message
reduction was 70.7%; byte, prompt-token, generated-token, LLM-call, and latency
reductions were 63.4%, 25.4%, 24.9%, 25.8%, and 25.3% respectively.

This is not yet evidence that entropy triggered useful coordination. Every one
of the four entropy candidates recorded zero trigger activations and zero active
agent-step fraction. The selected candidate won the fixed lexicographic screen
through the quiet-mode cadence and its slightly lower message use. The result is
therefore simultaneously eligible for the preregistered holdout and a negative
mechanistic result for event triggering. No threshold, direction, dwell,
cooldown, selection rule, or H4 definition will be changed in response.

The fixed control denominators were stable: commercial non-nominal loss ranged
from 13.041 to 14.358 and humanitarian loss from 3,708.64 to 7,379.59. The
default 2% relative non-inferiority margin remains frozen. Validation consumed
4.3054 reserved-Pod hours, 13,031 calls, 26,706,959 prompt tokens, and 974,801
generated tokens. All artifacts are in
`results/entropy_triggered_v2/validation/`, and the two replay reports each pass
144/144 episodes.

The restartable commands and outcome-seal boundary are now implemented. The
stale direct endpoint `213.173.109.33:19465` returned `Connection refused` on
all three attempts on 2026-08-13. The user-supplied `ssh.runpod.io` proxy then
connected successfully with forced PTY and the locally configured RSA key; the
requested `~/.ssh/id_ed25519` path did not exist. Read-only inspection found
the expected Pod, `/workspace/ThermoAgent`, RTX 4090 with 24,564 MiB VRAM,
CUDA-enabled PyTorch 2.8.0+cu128, 124 GiB RAM, and intact frozen-v1 counts
(944 main, 72 ablation, 80 holdout manifests and ten PDFs). The v1 freeze hash
remained `25141d7f9281320182af7256ea34815f3fe3b3a0b13d4589464b2224e7aa979e`.
No GPU process or v2 real-model call had started at this inspection point.

The first detached v2 setup job (`doet-setup`) retained exit status 2 after
dependency installation and the CUDA invariant check succeeded. It failed
before tests because `setup-doet-runpod.sh` passed the unsupported
`capture-env --output` option instead of `--results`. No experiment or model
inference ran. The failed status and log are retained; the setup script and the
health-only status script were corrected locally and reverified against all
129 tests before a separately named rerun.

The separately named `doet-setup-v2` rerun exited 0 after the CUDA invariant
and all 129 tests passed. The real-Qwen profile then completed 8/8 episodes and
replayed 8/8 exactly, with maximum absolute conservation residual
`1.14e-13`. Its sweep totals were 480 calls, 934,041 prompt tokens, 35,378
generated tokens, 0.1606 summed episode GPU-hours, and 714.10 seconds including
model load. The model smoke was 100% JSON/tool valid and added 77.59 seconds.

Agent-period scaling with a 15% buffer projected at least 92.57 hours for the
preferred 288-validation/1,296-holdout design before PPO training, so that
design was prospectively rejected under the 35-hour limit. No validation had
started. The authorized compute-priority reduction is 144 validation episodes
and a preferred 696-episode holdout, both at 16 periods. Its profile-based buffered
validation plus holdout estimate is 32.60 hours; profile/smoke and the setup
reserve bring the pre-training estimate to 32.92 hours. Before validation
outcomes, a deterministic runtime fallback was added: preserve all 576
priority-method episodes and all five RL seeds, then reduce only the common
secondary subset to produce 656 or 616 episodes if required. Actual validation
and all fifteen training times must still pass the fail-closed design gate.
