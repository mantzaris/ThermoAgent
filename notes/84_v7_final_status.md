# V7 final status

V7 ends as a formal-development no-go for the proposed positive selective-
safety mechanism. H1 and H2 failed; H3 supports only communication-efficient
distributed monitoring. Validation and holdout did not run.

The Git-facing package retains 45 smoke/pilot episodes and 348 formal episodes.
All 393 canonical episode payloads and event ledgers are compressed losslessly.
Before per-run candidate CSV duplicates were removed, every row and field was
semantically compared with the canonical episode payload. The compaction
manifest retains original and compressed checksums and sizes.

The final package distinguishes:

- formal development evidence from retained pilots;
- deterministic independent agents from unrun PPO and Qwen stages;
- simulated operator escalation from any real-human evidence;
- a monitoring-cost benefit from a causal selective-safety benefit;
- abstract defensive cyber-physical simulation from operational utility
  validation.

Final integrity checks passed: 379/379 repository tests, 41/41 V7-focused
tests, 393/393 exact ledger replays, zero privacy failures, maximum independently
reconstructed conservation residual `1.3500311979441904e-12`, and mechanical
plus manual QA for all 18 paper-facing vector PDFs. The indexed result package
contains 1,422 artifacts; no maintained V7 text has CRLF endings and no file
exceeds 50 MiB. The existing RunPod SSH endpoint still refuses connections;
the scientifically gated GPU stages did not run and no local experiment
process remains active.

The evidence is adequate for an engineering demonstration and potentially a
workshop boundary/negative-results paper. It is not sufficient for a positive
AIJ submission because the primary mechanism failed before validation and no
locked holdout was run.
