# V6 verified starting state

Verified 2026-08-15 America/New_York before any V6 scientific work.

- Local V5 branch and `origin/thermodynamic-human-oversight-v5` both resolved
  to immutable commit `c895235d02dd05ccc9315621d818def9345a398c`.
- The V5 worktree was clean and its 10,174-entry artifact index and frozen
  protocol checksum verified with zero failures.
- V4 maintenance commit `d39eb2eefefa54259a2bafc6dcd6e9b0dbde2ffe`
  is an ancestor of V5. The remote V4 branch still points to the original V4
  result snapshot `8ccd27df248940fc0cbb55c43a30949de3370533`;
  this is documented state, not repaired by rewriting history.
- No local or remote V6 branch existed. New branch
  `generalized-entropic-consensus-v6` was created exactly from the immutable
  V5 commit. V5 validation and holdout remain locked and untouched.
- The existing RunPod project `/workspace/ThermoAgent` was reachable. Its RTX
  4090 reported 0% utilization and 1 MiB use; there were no CUDA compute jobs,
  tmux sessions, ThermoAgent/Qwen/training processes, or V6 result files.
- Remote software remained compatible: NVIDIA driver 570.195.03, CUDA 12.8,
  PyTorch 2.8.0+cu128, Transformers 4.55.4, and bitsandbytes 0.47.0. The
  Hugging Face cache existed outside Git tracking. No package change is needed
  at the start of V6.
- No Pod was created, stopped, deleted, or modified during this audit.

V6 is scientifically separate from the immutable V5 development no-go. Its
primary question concerns selective autonomy and unsafe-decision detection,
not direct intervention ranking.
