# V7 independent-agent and action schema

## Four independent fields

V7 replaces the ambiguous V6 operational action field with:

1. `proposed_operational_action` and physical target/source/resource/quantity;
2. `information_action`;
3. `communication_action`;
4. `delegation_action`.

Verification and evidence requests are therefore never physical actions.
Physical acceptance and service-reaching rates use actionable physical
opportunities as their conditional denominator; unconditional rates remain
available separately.

## Independence contract

Every agent has a persistent identity, multi-asset scope, role authority,
private observation vault, private belief, memory, utility, commitments, inbox,
outbox, and seeded action process. A vault checks the requester identity.
Explicitly delivered peer evidence may update the recipient's belief; an
undelivered or partition-blocked message cannot.

The environment validates actions but does not substitute an oracle decision.
Counterfactual logic is evaluator-only. Domain rules and deterministic agents
are engineering controls; the main learned policy will use decentralized
execution after the pilot and protocol gates permit formal training.
