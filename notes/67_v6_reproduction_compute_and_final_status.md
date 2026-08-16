# V6 reproduction, compute, and final status

## Frozen provenance

- Parent V5 snapshot: `c895235d02dd05ccc9315621d818def9345a398c`.
- Scientific source commit: `33d975b8760c672d2850ee3273907fd94893e73f`.
- Formal execution commit: `be729f536a04573f42ad0548b746c072b2b81f87`.
- Formal execution source checksum:
  `e9e698f458a0ce32a19390d26c6eae67fddeb8b1ec39554bfe0baaa626fcb3c5`.
- Protocol: `v6.0.1`.
- Protocol checksum:
  `5a6e9e041db841ca98a95451760b82570af1786518e90ffe1cb0f16ebcf5a8fb`.
- Post-outcome presentation changes are separately mapped in
  `results/generalized_entropic_consensus_v6/reproducibility/source_transition_post_outcome_*.json`;
  they do not change formal outcomes or frozen gate decisions.

## Episode and compute accounting

V6 retains 1,308 pilot/reference-iteration episodes, 1,260 formal reference
episodes, 720 sketch-policy episodes, 2,520 full-horizon dynamic episodes, 6,500
PPO train/evaluation episodes, and 150 Qwen episodes: 12,458 total. Validation
and holdout contain explicit not-run dispositions and no numerical evidence.

The existing RTX 4090 Pod used 1.69234 reserved wall-clock GPU hours, of which
1.14227 hours were GPU-active. The model performed 2,812 calls, consuming
3,161,730 prompt and 236,987 generated tokens. V6 recorded 1,737,561 messages
and 203,207,336 communicated bytes across all retained stages. Estimated Pod
cost is USD 0.575 at USD 0.34/hour. No dependency was installed or upgraded.

## Reproduction sequence

From the repository root with the documented environment:

```bash
./scripts/run-v6-tests.sh
./scripts/replay-v6-results.sh
./scripts/evaluate-v6-gates.sh
./scripts/build-v6-report.sh
./scripts/generate-v6-figures.sh
./scripts/validate-v6-pdfs.sh
./scripts/index-v6-artifacts.sh
```

The complete development execution commands, locks, seeds, and remote workflow
are in the V6 README, protocol, manifests, and logs. Validation and holdout
commands remain guarded by the stage-disposition file and must not be run for
this protocol.

## Final operational status

All formal writers finished and all retrieved manifests have terminal status.
The last successful read-only RunPod check found no tmux session, Python
experiment, or CUDA compute process. A final connection attempt on 2026-08-16
was refused at the established SSH endpoint, consistent with a stopped or
otherwise unreachable Pod. No new remote process was launched after the clean
audit. It remains safe to leave the Pod stopped; it should not be deleted.

The branch is intentionally not pushed by Codex. After final local verification
the user may publish it with:

```bash
git push -u origin generalized-entropic-consensus-v6
```
