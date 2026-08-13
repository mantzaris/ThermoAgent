# DOET implementation record

Status: local implementation and deterministic preflight complete; real-LLM
validation pending RunPod availability.

Planned implementations are `DOET-rule` (transparent stateful trigger and fixed
mode rules) and `DOET-RL` (the trigger gates expanded options while an independent
decentralized actor chooses among eligible options). Each agent retains its own
private observation, memory, utility, inbox, commitments, planning context, and
authority. The simulator validates actions but does not issue domain decisions.

All entropy sketches, alerts, operational messages, bytes, prompted/generated
tokens, LLM calls, communication-active epochs, latency, and estimated cost will
be included in accounting.

## Implemented components

- `thermoagent/doet.py`: independent per-agent CUSUM/simple-hysteresis state
  machines with direction fixed by development/validation, dwell, cooldown,
  separate on/off/crisis thresholds, confidence attenuation from local
  consensus disagreement, and bounded neighbor-alert evidence.
- Three modes: quiet (local planning and sparse sketching), targeted (bilateral
  information/negotiation), and crisis (coalition and accelerated planning).
- DOET-rule and DOET-RL. In DOET-RL the trigger masks expanded communication
  options; the actor still consumes exactly 24 private/local features and each
  LLM planner retains its own context and authority.
- Explicit `entropy_alert` messages use the ordinary lossy communication
  channel, consume the sender's budget, and are included in message/byte
  accounting. They carry only a coarse anomaly level and recommended mode, not
  an exact entropy value or true disruption label.
- Sparse privacy-preserving gossip retains a prior distributed estimate between
  exchanges and refreshes the agent's own sketch locally. Pairwise matchings
  bound per-round traffic. Every directed sketch is counted.
- Strong `fixed_always_on`, periodic, random budget-matched, private-local-KPI
  CUSUM, global-entropy oracle, and disruption-label oracle controls.
- Public-route/local-coalition action affordances. The planner sees public
  initial routes and its own known coalition state, never another agent's
  inventory/cost. Closed routes can still fail at execution, preserving genuine
  replanning. This repair addresses the v1 tie mechanism symmetrically across
  every v2 method.
- Multiple-checkpoint experiment matrices with balanced round-robin RL-seed
  assignment and unambiguous run IDs.
- Restartable three-variant multi-seed training. Each of no-entropy, ThermoAgent
  v1-style, and DOET-RL receives five independent initializations and an
  identical 192-episode budget; final checkpoints are selected by fixed budget,
  never outcome.
- Fail-closed holdout generation and protocol verification before every locked
  sweep; the generator checks checkpoint hashes, validation status, exact
  balance, new seed separation, episode count, and the 35-GPU-hour cap.
- Updated replay includes protocol messages in causal ledger order. All eight
  v2 mock preflight episodes replay exactly.

## Verification

The current complete suite is 120/120 passing. New tests cover trigger
validation, per-agent state isolation, no global trigger input, dwell/cooldown,
bounded alert propagation, mode cadence, route-information privacy, counted
sketches and alerts, strong fixed communication, DOET-RL actor inputs, unseen
topology connectivity, balanced five-seed assignment, and deterministic replay.

The eight-episode mock preflight completed with zero failures and maximum
absolute material residual below `1.14e-13`; all eight ledgers replayed exactly.
It is an engineering check only and supplies no research claim.
