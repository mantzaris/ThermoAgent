# V6 frozen execution log

All timestamps below are recorded during execution and before comparative
formal outcomes were inspected. The normal operator remains simulated; no
human participants are involved.

## Freeze provenance

- Scientific source commit: `33d975b8760c672d2850ee3273907fd94893e73f`.
- Source checksum: `7b860191b0f2b55c5e32dcfaf3f629bd692e67b49cdd00310a71a068154519da`.
- Protocol version: `v6.0.0`.
- Protocol checksum: `7a61b2ff03bce7e83a8d80c784d8a7d218dfdcbcdeb601ab6dd51aa4b99ccb10`.
- Validation and holdout manifests were sealed before formal development;
  their outputs were absent at freeze time.

## RunPod verification and profile

The filtered source bundle at `/workspace/ThermoAgent` reproduced the frozen
source checksum exactly. The Pod-side V6 suite passed 91 tests. The complete
local repository suite independently passed 329 tests with no failures,
errors, or skips.

The existing Pod reports an RTX 4090 with 24,564 MiB VRAM, NVIDIA driver
570.195.03, CUDA-enabled PyTorch 2.8.0+cu128, Transformers 4.55.4,
bitsandbytes 0.47.0, and Python 3.12.3. No package was installed or upgraded.

A minimal valid profile containing one episode from each application completed
54 independent Qwen decision records and 57 generation calls in 37.46 seconds
including model load. It used 64,805 prompt tokens and 4,752 generated tokens.
The pinned model was `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, NF4 with BF16 computation.
The conservative pre-formal projection of 11.50 single-GPU hours and USD 3.91
including reserve therefore remains below the authorized caps.

An earlier one-application profiling subset completed its episode but failed
while serializing the cross-application summary because absent applications
produced NaN summary fields. This was a profiling-interface limitation, not a
formal scientific run; it wrote only below `/workspace/.cache/thermoagent/`.
The full frozen qualification always includes all three applications and does
not take this path. The failure is retained and will be included in the final
failure registry.

## Formal execution

The reference development matrix began in one exclusive `tmux` writer on
2026-08-15 America/New_York. Permitted monitoring is limited to process health,
completion counts, schema/finiteness health, disk use, and catastrophic
engineering failure. Comparative outcome metrics are not inspected while a
stage is incomplete. The remote result namespace is exclusively
`results/generalized_entropic_consensus_v6/`; frozen V1-V5 results are not
written by V6 jobs.

## Completed non-scientific maintenance

A pre-outcome audit of `git diff --check c895235d..HEAD` found one extra blank
line at EOF in each of `thermoagent/v6_entropy.py`,
`tests/test_v6_entropy.py`, and
`results/generalized_entropic_consensus_v6/v5_reanalysis/README.md`. These were
not CRLF defects and did not change Python semantics or numerical evidence.
They were removed in pre-outcome amendment commit `2fa69d5e` while the initial
reference generator continued on its unchanged `v6.0.0` bundle. Both diff
checks were clean before training. Per-episode manifests distinguish the
initial reference checksum from the amended training checksum.

## Pre-outcome training amendment

Source review during the still-sealed reference run found that PPO advantages
were propagated through the interleaved list of different organizations. That
could bootstrap one agent's critic value from another agent's private-policy
trajectory. Before any training began and before comparative outcomes were
opened, protocol `v6.0.1` changed GAE grouping to persistent `agent_id` and
added a negative test that fails under cross-agent bootstrapping. This is a
training-only correctness repair: no scientific hypothesis, gate, threshold,
seed, environment, feature block, comparator, or sealed input changed. The
running reference generator does not import or execute this training path and
continues unchanged under its recorded `v6.0.0` source provenance.

The amended training source commit is
`2fa69d5e5a075db290904897b449ded87945ffc5`, its source checksum is
`3d4893616700c956dfed6ad7e77e58dbf11d3055c099bc9e330a1f92613e7ce1`,
and the amended protocol checksum is
`5a6e9e041db841ca98a95451760b82570af1786518e90ffe1cb0f16ebcf5a8fb`.
The complete transition record is
`results/generalized_entropic_consensus_v6/reproducibility/source_transition_v6_0_1.json`.

## Pre-training communication instrumentation

Before the still-sealed dynamic stage closed and before PPO training began, a
second outcome-blind audit found that the PPO evaluation rows retained total
messages and bytes but training trajectories did not persist their exact
operational-versus-sketch split. Commit
`be729f536a04573f42ad0548b746c072b2b81f87` adds accounting fields only. Its
source checksum is
`e9e698f458a0ce32a19390d26c6eae67fddeb8b1ec39554bfe0baaa626fcb3c5`.
The protocol remains `v6.0.1` with checksum
`5a6e9e041db841ca98a95451760b82570af1786518e90ffe1cb0f16ebcf5a8fb`;
no observation, reward, policy, action, threshold, seed, environment, or gate
changed. The machine-readable record is
`results/generalized_entropic_consensus_v6/reproducibility/source_transition_pretraining_communication_instrumentation.json`.

## Reference and analysis execution

The frozen reference generator completed 1,260 formal reference panels and
720 communication-sketch panels with zero failed episodes. All 1,980 episode
manifests retained the initial `v6.0.0` source checksum. The detached wrapper
itself exited zero; its first status marker contained the literal suffix `n`
because of shell quoting, so that marker alone was normalized to `0\n` after
the tmux session had ended. No scientific artifact or episode was rerun.

After reference closure, the `v6.0.1` amended training/analysis bundle was
synced and its source and protocol checksums were verified before launch.
Grouped cross-fitting, entropy-family ablations, the 200-refit permutation
test in each primary application, and the supervised learnability ceiling
completed atomically before full dynamic execution. Comparative result values
were not opened while the composite analysis stage remained active.

## Dynamic development closure

The matched dynamic stage completed all 2,520 planned episodes with zero
generator failures. Together with the 1,260 formal reference and 720 sketch
episodes, frozen formal development contains 4,500 event-sourced episodes.
Only after the dynamic wrapper exited zero were paired effects opened. Gates 5
and 7 failed their frozen practical criteria, and Gate 6 also failed coverage
and escalation-burden subconditions. These failures permanently lock V6
validation and holdout.

A post-outcome audit found that the separate pooled supervised-learnability
diagnostic reused numeric environment seeds across applications even though it
isolated application-specific topology and scenario families. The primary
selective-risk cross-fitting isolated all three required axes. The pooled
ceiling is retained and labeled methodologically compromised; it cannot rescue
or unlock the study.

## Training closure

Sequential decentralized PPO began only after the outcome-blind training
amendments above. All five methods and all seeds `66201`--`66205` completed:
25 runs, 6,500 train/evaluation episodes, 156,000 decentralized decision
epochs, and zero selectively removed or collapsed seeds. The matrix was kept
sealed until every run closed. The combined controller exceeded the frozen
predictive-uncertainty comparator in mean reward, but its between-seed harm SD
was `0.09382`, above the frozen `0.08` maximum. Gate 9 therefore fails.

The real-Qwen qualification was launched in a separate exclusive writer only
after the training exit code was zero and the GPU was clear. It uses the
unchanged formal execution source commit and protocol checksum. Monitoring is
limited to atomic episode counts, process health, disk use, and GPU memory
until all 150 episodes close.
