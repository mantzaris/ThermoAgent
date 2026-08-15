# ThermoHITL v4 final status

- Phase: complete development-stage no-go package.
- Branch: `thermodynamic-human-oversight-v4` from immutable v3 commit
  `3f844966930b1cfb5a43bdf3a4d3e744391d1018`.
- Decision: stop before validation because prospective Gate 3 failed.
- Completed: repository repair; utility simulator; independent agents;
  simulated operator; causal branching; formal development; Qwen qualification;
  cluster-aware analysis; dashboard exports; figures; documentation; indexing.
- Not run: validation, RL training, holdout, real-human study.
- Scientific classification: development mechanism and boundary evidence;
  insufficient for AIJ submission.
- Final integrity: 213/213 tests passed; 1,590/1,590 ledgers replayed
  exactly; replay mismatches and maximum conservation residual were both zero.
  All 21 vector PDFs opened, exposed fonts, rendered at 240 DPI, and passed
  original-resolution visual inspection. The 5,020-row artifact index has no
  missing files or checksum mismatches.
- Repository hygiene: v3/v4 maintained text has no CRLF; the 47 normalized v3
  artifacts match both the immutable v3 byte hashes and parsed CSV semantics;
  both required diff checks pass; no credential signature or file over 50 MB
  was found.
- RunPod final audit: the existing RTX 4090 Pod was reachable through the
  established proxy. `/workspace/ThermoAgent` exists; the filtered execution
  copy intentionally has no Git metadata. There were zero tmux sessions, zero
  Python/Qwen serving processes, and zero CUDA compute processes. SSH was
  closed and the Pod is safe to stop, but not delete.
- Push: not authorized; user receives the exact command after final commit.

The v4 result must remain intact in any successor. A future study needs a new
version and fresh protocol rather than weakening the 5% coordination threshold
or reusing the planned validation/holdout as development data.
