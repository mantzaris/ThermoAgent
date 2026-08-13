# DOET development and validation log

No real-LLM outcome pilot or validation result has yet been produced. The
current RunPod proxy is reachable and the committed v2 source was deployed by
a filtered archive with an exact source-checksum match. This file intentionally
contains no inferred or synthetic treatment result.

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

Pending when the same Pod becomes reachable:

1. synchronize the filtered branch snapshot;
2. run CUDA/model smoke and measure real Qwen throughput;
3. run the 288-episode real-LLM validation matrix;
4. apply the fixed selection rule without manual choice;
5. train five independent seeds for each learned method;
6. generate, inspect, checksum-freeze, and launch the genuinely unseen holdout.

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
