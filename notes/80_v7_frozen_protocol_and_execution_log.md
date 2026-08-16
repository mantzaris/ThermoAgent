# V7 frozen protocol and execution log

## Freeze

The V7 protocol was frozen at `2026-08-16T15:10:17.832291+00:00`, before any
formal-development episode. The execution source is local commit
`39b463ca3607f5e55679c551ce4e5d0034b5b7b8` on branch
`complexity-entropic-coordination-v7`.

- Frozen protocol version: `v7-protocol-candidate-1.0` (status inside the
  frozen file is `frozen_before_formal_development`).
- Protocol SHA-256:
  `760e9d019140dc0a1edf16af76f0d0a393e09d3680a3ece2499e84a8b4d0fff5`.
- Source checksum:
  `ded1b83c41513ba6b052f2874f89a3d294999ed9af3fd28917d1ee4465043840`.
- Development manifest: 100 panels.
- Sealed validation manifest: 32 panels, SHA-256
  `a53cef9720d1c551817859d67256c9fbe431b612c43ce8d92ece004391954e50`.
- Sealed holdout manifest: 40 panels, SHA-256
  `e22c82a7cd419e539edb903460d4b9591d2e8a09cde3adc3f7df8737e20e4643`.

No validation or holdout seed, graph, or scenario was executed before the
freeze. Their manifests are provenance objects only and both stages remain
locked. The freeze command is fail-closed against overwrite or formal raw
outputs.

## Formal execution policy

The formal CPU batch will run reference panels, grouped cross-fitted dynamic
controller panels, the fully counted communication ablation, exact replay,
and prospective gate evaluation in that order. Episode outputs are atomic and
restartable; exclusive stage locks prevent two writers. Failures are retained
and cause the engineering progression condition to fail.

The output metrics may be inspected only after each complete formal stage;
they will not be used to change source, thresholds, feature blocks, panel
manifests, or success criteria. PPO and real-Qwen stages remain locked unless
all formal co-primary conditions pass. If formal development fails, V7 stops
before those stages and before validation or holdout.
