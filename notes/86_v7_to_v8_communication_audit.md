# V7-to-V8 communication audit

Date: 2026-08-16
V7 immutable parent: `e46b6738231883e92b9b525ab1c3c190e38391e7`
V8 branch: `entropy-triggered-belief-monitoring-v8`

This is a post-V7 design audit. It does not alter the V7 results, reopen a V7
gate, or convert the V7 development result into confirmatory evidence.

## Verified findings

1. **The V7 event trigger used only absolute local Shannon-entropy change.**
   `V7CoupledEnvironment._should_send_sketch` sent the first local reference and
   subsequently sent when
   `abs(shannon_entropy(current) - last_sent_entropy) >= 0.045`
   (`thermoagent/v7_base.py`, lines 334-346 at the parent commit). It had no
   Jensen-Shannon mode-drift term, generalized entropy spectrum, confidence
   change, maximum-age deadline, cooldown, or hysteresis.

2. **The transmitted object was a full belief distribution.** The payload in
   `exchange_entropy_sketches` contained `belief_distribution`, confidence,
   asset, time, and hop count. The accurate V8 terminology for this object is an
   *entropy-triggered belief sketch*, not an entropy scalar.

3. **The FP16 formula was not the executed wire encoding.**
   `thermoagent/v7_entropy.py::encoded_sketch_bytes` returned a formula described
   as header plus FP16 belief, confidence, and age. However,
   `thermoagent/v7_base.py::send_message` independently charged
   `len(json.dumps(payload).encode("utf-8")) + 40`. No FP16 serializer or decoder
   was used. The frozen V7 byte result therefore represents its JSON simulation
   accounting, not an actual FP16 wire payload.

4. **The formal communication comparison was small and narrow.** The frozen
   paired table has 12 independent panels: 6 humanitarian and 6 utility. Every
   panel is medium complexity, high coupling, high fragmentation, high network
   disruption, and private-fragmented information.

5. **V7 H3 was an event-triggered-versus-always-on resource comparison under an
   always-act controller.** Every paired row compares `event_triggered` with
   `always_on` using `controller=always_act`. It established neither superiority
   of generalized-information scheduling over a matched-budget non-entropic
   scheduler nor retention under a frozen learned decentralized policy.

The frozen H3 estimates remain exactly as reported: message reductions of
0.3763 (humanitarian) and 0.4040 (utility), byte reductions of 0.3797 and
0.4055, and maximum reported event-trigger estimation MAE of 0.04564 and
0.04965. These are development-only V7 results with six panels per
application.

## Design consequences for V8

- Implement and count a real deterministic binary encoding.
- Detect equal-entropy belief-mode switches with Jensen-Shannon drift.
- Add hysteresis, cooldown, partition-recovery handling, and a maximum-silence
  deadline.
- Compare multiple schedulers over multiple actual byte budgets.
- Freeze the strongest matched-budget non-entropic comparator before validation.
- Evaluate the same frozen decentralized policy weights under every scheduler.
- Use at least 24/30/40 independent panels per application for development,
  validation, and locked holdout, subject to prospective power analysis.

## Frozen evidence checksums

| Artifact | SHA-256 |
|---|---|
| `thermoagent/v7_base.py` | `2894a2c72eb4acf87332ef6e326c6dd924c1bca25e7b3c6d0b7143f66a74790e` |
| `thermoagent/v7_entropy.py` | `6e2e013e6729cb77b516a1b0ebc533a78d6172ae7523851219fe3d4059564d07` |
| `thermoagent/v7_experiments.py` | `8b2bc23538cbfc9fbb84500007d31a74f7f3d523a4e979cc57f8384b617099c5` |
| V7 frozen protocol | `760e9d019140dc0a1edf16af76f0d0a393e09d3680a3ece2499e84a8b4d0fff5` |
| V7 communication analysis | `a23fadcfbc4e925fe5d5bc3f142779f6aac7ca9be87f9a985cf00d6b0e3dd74d` |
| V7 paired communication panels | `612e226a6ac0ff6bde03988096e27212d276b8ed48e7126d70930717aa7e2e4b` |
