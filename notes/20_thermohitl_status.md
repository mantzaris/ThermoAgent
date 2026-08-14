# ThermoHITL status and provenance

## Provenance

- Study: ThermoHITL (v3), completed development-stage no-go.
- Created: 2026-08-14 America/New_York.
- Branch: `thermodynamic-human-oversight`.
- Parent: `c0aa6fe6c98cbce0cdd5e40a0f720a98f5facbe6`.
- Verified pushed antecedent:
  `origin/entropy-triggered-communication` resolves to the same commit.
- Frozen v1 tag: `thermoagent-v1-frozen`.
- Frozen v2 namespace: `results/entropy_triggered_v2/`.

The v3 study is additive. It does not reinterpret, regenerate, or replace v1
or v2 data. In particular, the v2 zero-activation finding and the dominance of
no communication over DOET-rule remain unchanged.

## Current phase

Development is complete. Gates 1, 2, 3, 4, and 6 passed; Gate 5 failed, so no
validation, learned-policy training, or holdout outcome was opened.

## Active jobs

None.

## Next actions

1. Complete repository hygiene and the local v3 commit.
2. Push only after separate authorization.
3. Treat any scientific continuation as v4 with new development and unseen
   validation/holdout seeds; do not unlock the v3 guarded stages.

## Stop rules

- A failed required development gate blocks validation and holdout.
- A trigger with zero activation is ineligible.
- More than 40 additional single-GPU hours requires user approval before the
  expensive run.
- No actual human participants will be recruited or studied.
- No branch will be pushed without explicit authorization.

## Final gate disposition

- Gate 1: passed (183 tests; 817 exact replays).
- Gate 2: passed after a retained Qwen v8 failure and versioned v9 retry.
- Gate 3: passed in both applications and all important regimes.
- Gate 4: passed with bounded causal interventions in both applications.
- Gate 5: failed cross-application; commercial failed, humanitarian passed.
- Gate 6: passed on final development candidates.

The stop rule was applied. Zero validation episodes, zero RL seeds, and zero
holdout episodes exist.
