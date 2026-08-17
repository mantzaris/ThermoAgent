# V8 reproduction, compute, and final status

Date: 2026-08-17
Branch: `entropy-triggered-belief-monitoring-v8`
Immutable parent: `e46b6738231883e92b9b525ab1c3c190e38391e7`

## Scientific disposition

V8 stopped at the prospectively declared trigger-feasibility pilot. The final
high-excursion hysteresis repair made the generalized trigger genuinely
information driven, but neither candidate met the fixed nominal-traffic gate
in both applications. At `tau_on=0.115`, pre-disruption noninitial transmission
rates were 82.69% humanitarian and 71.15% utility restoration; at
`tau_on=0.11`, they were 85.58% and 75.96%. The unchanged maximum was 10%.

Replacement formal development, five-seed decentralized training, validation,
locked holdout, and formal ablations were therefore not run. H1, H2, and H3
remain formally untested. The six-panel-per-application results below are
development diagnostics only.

## Final diagnostic estimates

For `generalized_0115_u8` versus always-on exchange:

- Humanitarian sketch-wire byte reduction: 0.2396, 95% panel bootstrap CI
  [0.1497, 0.3343]; fully counted byte reduction 0.2247 [0.1447, 0.3115];
  distributed-state error change -0.000338 [-0.000704, 0.000018].
- Utility-restoration sketch-wire byte reduction: 0.2775 [0.2042, 0.3525];
  fully counted byte reduction 0.2293 [0.1822, 0.2932]; distributed-state error
  change 0.000334 [0.000049, 0.000618].
- Rule-policy downstream diagnostics were not retention evidence. Service-loss
  differences were +117.85 humanitarian and -6.58 utility; harmful-action
  count differences were -0.33 and +0.33; reward degradations were +0.0855 and
  -0.00268, respectively.

Exact six-panel traffic totals were:

| Application | Scheduler | Transmitted sketches | Delivered | Dropped | Sketch bytes | Fully counted bytes |
|---|---:|---:|---:|---:|---:|---:|
| Humanitarian | always on | 438,662 | 399,286 | 60,937 | 14,914,508 | 15,471,277 |
| Humanitarian | generalized 0.115 | 374,440 | 340,957 | 51,905 | 12,730,960 | 13,287,574 |
| Utility restoration | always on | 231,063 | 210,199 | 35,721 | 7,856,142 | 8,219,191 |
| Utility restoration | generalized 0.115 | 187,949 | 170,873 | 28,862 | 6,390,266 | 6,750,950 |

The generalized-trigger activation rates were 0.7603 humanitarian and 0.7158
utility restoration. These high rates are the reason the communication point
estimates cannot be promoted to a supported monitoring claim.

## Autonomous-agent and execution boundary

Completed scientific pilots used persistent independent V7 agents with the
deterministic decentralized rule policy. A separate engineering-only
sequential-IPPO pilot used seed 88101 for six episodes and 1,320 temporally
linked transitions. It exercised all four delegation actions: 173 autonomous
executions, 859 deferrals, 146 abstentions, and 142 escalations. The frozen
five-seed set 88201--88205 was never trained. Qwen was not used in V8.

Completed retained stages contain 1,292 episodes: 196 `pilots`, 196
`pilots_v2`, 88 `pilots_v3`, 8 routing-repair pilots, three 60-episode
hysteresis-repair pilots, and 624 development episodes. Two invalidated stages
retain 396 additional complete episodes plus two partial run directories.

## Protocol and integrity

- Development closeout version: `v8-development-3.0-no-go`.
- Development protocol SHA-256:
  `32b519cd7c282fbc8eecc60757c511bc74322ddf26919a9af08752d74969684f`.
- Source checksum:
  `0ba9dbb58a6970600f7749bb442762064fcbe593946a2cc976cdc1454583af69`.
- Tests: 422 complete-repository tests and 43 focused V8 tests; zero failures,
  errors, or skips.
- Replay: 1,688 complete ledgers and 77,316,070 events; zero mismatches and
  zero privacy failures. Two partial invalidated runs are retained but are not
  promoted to completed ledgers.
- Maximum independently recorded conservation residual:
  `1.7337242752546445e-12`.
- Ten packed-stage reports and all 155 archives pass member-level round-trip
  checks. The largest archive is 51,464,716 bytes, below 50 MiB.
- All eight paper-facing PDFs open, contain embedded fonts, render at 240 DPI,
  and passed original-resolution inspection.

The V8 namespace is approximately 1.6 GiB, above the aspirational 150 MiB
target. This is an explicit auditability tradeoff: all 1,688 complete ledgers,
including 396 invalidated completed runs, and two partial invalidated runs are retained
losslessly, without unpacked duplication. No individual artifact exceeds the
Git-facing limit.

## Compute and environment

Manifest-accounted completed-episode CPU time is 2.8847 hours. Invalidated
partial-run wall time is retained but not reconstructed as completed-episode
CPU time. V8 used zero GPU hours, zero LLM calls, zero prompt or generated
tokens, and zero incremental cloud cost. No package was added. The established
RunPod SSH endpoint returned connection refused during the final read-only
check, consistent with a stopped or unreachable existing Pod; no replacement
Pod was created.

## Reproduction order

```bash
./scripts/run-v8-tests.sh
./scripts/run-v8-hysteresis-repair-pilot.sh
./scripts/run-v8-hysteresis-repair-pilot-v2.sh
./scripts/run-v8-hysteresis-repair-pilot-v3.sh
./scripts/analyze-v8-no-go.sh
./scripts/compact-v8-artifacts.sh
./scripts/replay-v8-results.sh
./scripts/close-v8-development-no-go.sh
./scripts/build-v8-report.sh
./scripts/generate-v8-figures.sh
./scripts/validate-v8-pdfs.sh
./scripts/record-v8-manual-pdf-qa.sh
./scripts/index-v8-artifacts.sh
```

The historical pilot/development scripts are retained for full reconstruction;
the compacted repository itself can be audited without rerunning outcomes by
starting at replay. Formal-stage launchers fail closed because no frozen V8
protocol or locked outcome manifest exists.

## Readiness

V8 is a strong reproducible engineering demonstration and an informative
trigger-design no-go. It is potentially useful as a negative methods or
workshop case study together with the earlier versions, but it is insufficient
for a positive conference paper, journal manuscript, or Artificial
Intelligence submission. The exact supported claim is that actual wire
serialization, delivered-message-only estimation, and an auditable local
generalized trigger were implemented, while the corrected trigger failed the
prospective nominal-communication gate. No communication-efficiency,
entropy-superiority, frozen-agent-retention, human, or confirmatory claim is
supported.
