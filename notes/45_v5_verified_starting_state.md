# V5 verified starting state

Date: 2026-08-15 America/New_York

- Local branch and fetched remote branch `thermodynamic-human-oversight-v4`
  both resolve to the immutable V4 result snapshot
  `8ccd27df248940fc0cbb55c43a30949de3370533`.
- The worktree was clean before maintenance. V3 remains
  `3f844966930b1cfb5a43bdf3a4d3e744391d1018`; the V2 and V1 history and result
  namespaces remain intact.
- The existing RunPod Pod was reachable at `/workspace/ThermoAgent`. It had no
  tmux session, ThermoAgent/Qwen Python job, CUDA compute process, or partial,
  lock, or temporary result file.
- Hardware remained an RTX 4090 with 24,564 MiB and driver 570.195.03. The
  isolated environment reported PyTorch 2.8.0+cu128, CUDA 12.8, Transformers
  4.55.4, and bitsandbytes 0.47.0. No package change was made.
- `/workspace` reported 397 TB available on the shared filesystem. The SSH
  audit session was closed after inspection.
- Two top-level smoke CSVs had committed CRLF endings despite the LF policy.
  This is a clone-hygiene defect, not a scientific-result discrepancy.
- The V4 dashboard's zero-minute/applied-action combination is a timing-display
  issue: the authorized view records cumulative minutes immediately before the
  selected intervention is registered. The maintenance label will state this
  convention; recorded outcomes and operator accounting will not change.

V5 work may begin only after the non-destructive V4 maintenance commit and a
clean temporary-clone verification.
