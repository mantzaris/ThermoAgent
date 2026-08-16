# V7 compute projection and freeze readiness

## Measured CPU simulator throughput

The final retained pilot used six counterfactual probes per decision epoch
(five for its large panel). Median wall times were 5.97 seconds for small,
88.47 seconds for medium, and 275.76 seconds for large panels. Earlier
two-probe pilots measured roughly 2, 16, and 40 seconds. These measurements
were made locally on CPU and include dynamic paired branching, ledger writing,
and distributed-message simulation.

The frozen-candidate 100-panel reference design is projected at approximately
3.1 CPU wall-hours. Two cross-fitted dynamic controller runs per panel, with
two mechanistic probes per epoch, add approximately 1.0 hour. The 48-run
communication subset adds approximately 0.25 hour. Analysis, replay, tables,
and figures add a 0.5-hour allowance. Formal development is therefore
projected at approximately 4.9 CPU wall-hours and no GPU time.

If formal development passes, five PPO methods by five seeds and 80 training
plus 24 evaluation episodes per seed are conservatively projected at 29
single-GPU hours, although much of each episode is CPU simulator time. Model
loading and real-Qwen qualification are allocated 8 GPU-hours. A 15% safety
reserve gives 42.6 GPU-hours, below the 50-hour cap. At an illustrative
existing-Pod rate of USD 0.40 per hour, the projected GPU cost is about USD
17.04; the actual provider rate is not exposed locally and this is explicitly
an estimate. No new Pod or paid service is authorized.

The existing RunPod endpoint returned `connection refused` on the verified
SSH alias. This does not block CPU formal development. It will block PPO/CUDA
and real-Qwen execution if formal mechanism gates pass. The workflow will stop
at that point and report the exact reconnection requirement rather than
fabricating GPU evidence or creating a replacement Pod.

## Freeze readiness

The prospective feasibility gates passed on engineering and action-pool
criteria. They did not test or require a favorable entropy effect: the final
pilot coupling-by-fragmentation effect was negative. Before formal execution,
the complete source and candidate protocol must be committed cleanly, then the
freeze tool will hash that commit and generate untouched development,
validation, and holdout manifests. Validation and holdout remain locked.
