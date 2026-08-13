# DOET development and validation log

No real-LLM outcome pilot or validation result has been produced because the
existing RunPod endpoint is stopped/unreachable. This file intentionally
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
