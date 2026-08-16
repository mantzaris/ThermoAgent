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

## Pending non-scientific maintenance

A pre-outcome audit of `git diff --check c895235d..HEAD` found one extra blank
line at EOF in each of `thermoagent/v6_entropy.py`,
`tests/test_v6_entropy.py`, and
`results/generalized_entropic_consensus_v6/v5_reanalysis/README.md`. These are
not CRLF defects and do not change Python semantics or numerical evidence. To
avoid mixing source checksums within the active formal stage, the frozen
execution copy is not being changed. After all frozen execution ends, the
three trailing blank lines will be removed in a documented formatting-only
maintenance change; the scientific execution commit and checksum above remain
the authoritative provenance for formal results.

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
