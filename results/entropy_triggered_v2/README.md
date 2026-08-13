# Distributed Operational Entropy Triggering study

Status: Phase A retrospective diagnostics, monitoring validation, DOET
implementation, nominal/development calibration, and deterministic preflight
complete. No v2 real-LLM performance claim or new holdout result exists yet.

This namespace preserves strict separation from the frozen v1 result tree. It
contains only derived v1 diagnostics and future v2 development, validation,
training, locked-holdout, analysis, and publication artifacts. Original files
under sibling result directories are inputs and are never rewritten.

The working research question is whether locally estimated distributed
operational entropy can trigger quiet, targeted, and crisis communication modes
that remain non-inferior to always-on communication while reducing all counted
communication and inference costs.

See `notes/13_entropy_trigger_diagnostics.md` through
`notes/19_entropy_trigger_paper_claims.md` for the contemporaneous research
record.

The frozen-v1 diagnosis found genuinely different policies but no demand-
reaching material consequence, explaining all 16 exact holdout ties. Global
ordinary KPIs subsume entropy for disruption classification; entropy adds about
0.10 AP only when the comparator is restricted to one agent's private local
KPIs. This boundary is now an explicit design constraint. See
`diagnostics/README.md` and `monitoring/README.md`.

## Current evidence boundary

- All 16 original ThermoAgent/no-entropy holdout pairs were exactly equal as
  raw floats, not merely after rounding. Policies nevertheless diverged at
  58.2% of common commercial and 46.2% of common humanitarian option epochs.
  The exact tie arose because none of 57 learned-policy material actions
  delivered material to demand; 54 failed and three ThermoAgent successes
  stopped at intermediate nodes.
- Full/global ordinary KPI models subsume entropy in the synthetic v1 data.
  Under private local information, adding distributed entropy improved
  development average precision by about 0.097--0.098 and ROC AUC by about
  0.169--0.171. This supports testing entropy as a compressed distributed
  statistic, not claiming universal predictive superiority.
- Low-direction distributed entropy led the new development comparison, but
  its nominal-threshold recall was only 0.178 commercial and 0.155
  humanitarian. Seven trigger candidates remain prospectively registered for
  real validation.
- Eight mock-planner preflight episodes completed and replayed exactly with a
  maximum absolute material residual below `1.14e-13`. Mock results are
  engineering evidence only.

## Frozen prospective design

Real validation uses four new seeds in both applications across nominal,
isolated, correlated, and compound-partition regimes (288 episodes). A fixed
lexicographic rule selects the trigger and derives random/periodic budget
matches from fully counted validation traffic without holdout access. Inactive
budget controls retain quiet local planning, and achieved message mismatch is
reported. The private-KPI comparator uses its own nominal normalizers and a
validation-only residual scaling set from the DOET/KPI counted-message ratio;
its achieved mismatch is likewise reported. Each learned method then receives five
independent 192-episode training runs (seeds 7301--7305), with the final-budget
checkpoint retained regardless of outcome.

After measured throughput and precision checks, the holdout generator proposes
144 genuinely new matched panels and 1,296 method episodes: 16 seeds per
application for each of four non-nominal regimes, eight nominal seeds per
application, nine core methods, LLM seed 9101, and unseen topology
`tri_region_bridge_v2`. The generator fails closed if the measured real-Qwen
profile/model smoke and validation/training time, a 0.1-hour unmeasured setup
reserve, and projected holdout resource time exceed 35
single-GPU hours. Its fixed 20,000-replicate stratified Monte Carlo precision
analysis draws the planned 16 panels per non-nominal regime from validation
paired-degradation distributions. CPU-bound PPO time is counted because the
paid Pod remains reserved.

## Reproduction sequence

```bash
./scripts/run-doet-calibration.sh
./scripts/runpod-smoke-test.sh
./scripts/runpod-sync.sh
./scripts/runpod-sync-v2-controls.sh bootstrap
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-profile ./scripts/run-doet-profile.sh'
./scripts/runpod-fetch-v2-results.sh
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-validation ./scripts/run-doet-validation.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-training ./scripts/train-doet-multiseed.sh'
./scripts/runpod-fetch-v2-results.sh
./scripts/design-doet-holdout.sh
git status --short
# Commit the validation-selected protocol and generated holdout config here.
./scripts/runpod-sync.sh
./scripts/runpod-sync-v2-controls.sh resume
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/freeze-doet-holdout.sh'
ssh runpod-thermo 'cd /workspace/ThermoAgent && ./scripts/start-doet-job.sh doet-holdout ./scripts/run-doet-holdout.sh'
./scripts/runpod-fetch-v2-results.sh
./scripts/rebuild-doet-results.sh
```

Use the remote `./scripts/doet-job-status.sh` between detached stages. It shows
only process state and completion counts during the outcome-sealed holdout.
The holdout freeze refuses dirty source provenance, so the commented commit
boundary is mandatory rather than illustrative.

Filtered deployment excludes Git metadata, environment files, keys,
credentials, virtual environments, caches, and the general results tree. The
v2 control sync carries only checksum-addressed calibration/provenance inputs;
`runpod-fetch-v2-results.sh` retrieves only this namespace. The holdout cannot
start until source, protocol, thresholds, all 15 checkpoints, design, and
analysis code have a non-overwritable checksum freeze.

The declared experiment `llm_seed` is applied to PyTorch and CUDA when loading
the real planner. Qwen decoding remains deterministic (`do_sample=false`), so
the seed is a reproducibility field rather than an extra stochastic replicate.

## Current blocker

The existing endpoint at `213.173.109.33:19465` refused all three SSH attempts. No
replacement Pod or paid resource was created. Once the same Pod is started,
retry `./scripts/runpod-smoke-test.sh`. Additional v2 GPU use remains zero.
