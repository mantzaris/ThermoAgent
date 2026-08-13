# Agent independence and architecture

Each `AutonomousAgent` owns a `PrivateStateVault` with capability checks. Its
private observation, working memory, episodic memory, beliefs, utility weights,
risk tolerance, trust estimates, inbox/outbox, commitment ledger, policy state,
communication budget, and RNG stream are distinct objects. A request naming a
different owner raises `PrivacyViolation`.

One shared frozen model can batch prompts, but `TransformersPlanner` constructs
one prompt per agent and returns one record per prompt. Prompts are never
concatenated across agents. The simulator validates actions and executes typed
tools; invalid model output recovers only to a non-mutating `no_op`. The
simulator does not substitute a supposedly sensible domain action.

The decision loop is:

1. deliver private observation and explicit messages;
2. retrieve private memories/commitments;
3. update beliefs and partner trust;
4. attach the local gossip-based entropy/free-energy estimate;
5. select one coordination option;
6. request concise JSON from the planner;
7. validate role, schema, bounds, and option compatibility;
8. execute the tool and log its result;
9. deliver messages/commitments explicitly;
10. reflect on success or failure in private memory.

Automated boundary tests cover cross-vault denial, event visibility, distinct
RNG/recurrent state, changed negotiation after agent removal, utility-dependent
offer decisions, refusal/countering/revision authority, and the absence of
simulator-authored decisions.

Dynamic validation now rejects unknown offer targets and malformed coalition
membership lists instead of silently filtering them. A coalition is counted as
formed only after an invited agent explicitly joins. Shipment status can be
verified after arrival, but only by its sender or recipient; only the
dispatching carrier/transport organization can reroute or expedite an active
shipment, and rerouting still requires a physical or joined-coalition route.
These checks close identifier-guessing and authority gaps without sharing any
private vault state.

When a scenario exposes shared or coarse information, the common dashboard is
an explicit `public_signal` ledger event and is identical for every eligible
recipient. Under strong privacy it contains identities only. This keeps shared
information distinct from hidden state or an undocumented central context.
The centralized-LLM control consumes this same legal interface and receives a
separate typed assignment for each reported demand organization. It is one
central actor, not one of the independent-agent treatments. The separate
full-information numerical controller is intentionally privileged and labeled
as an unattainable upper bound.
