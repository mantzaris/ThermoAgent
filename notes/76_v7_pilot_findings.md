# V7 retained feasibility pilots

## Execution status

The pilot matrix was specified in `configs/v7_pilot.yaml` before comparative
pilot outcomes were opened. The first command transports yielded without
showing that their child Python processes remained alive. Monitoring later
found three concurrent writers. Exact process ancestry identified two Codex
transport children and one tmux-supervised child. The two duplicate PIDs were
terminated, the tmux process was retained, and an exclusive nonblocking stage
lock was added. This is a retained pilot-stage protocol deviation, not formal
evidence. Every completed ledger will be replayed and checksummed; any corrupt
or ambiguous pilot artifact will be retained and excluded with a reason rather
than silently replaced.

## Preliminary engineering observations (not comparative results)

The two smoke episodes passed privacy and conservation, used nonzero
cross-community traffic, and produced both beneficial and harmful physical
actions. The humanitarian causal chains reached three recorded stages and the
utility chains reached three stages. These observations justify completing the
retained pilot but do not unlock development, validation, or holdout.

## Retained pilot iteration 1

Iteration 1 completed 24 episodes. It established exact replay and coupled
dynamics, but its counterfactual action pool was unsuitable for formal work:
utility restoration contained only 3 beneficial versus 113 harmful probes.
The apparent high-complexity humanitarian harm reduction was based on one
panel, the utility effect was zero, and the coupling-by-fragmentation
interaction was negative. These values are feasibility diagnostics, not
estimates of a scientific effect.

## Retained pilot iteration 2

The second iteration was configured before its outcomes were opened and then
ran once. It completed 9 episodes and 382 accepted counterfactual probes. The
humanitarian pool contained 30 beneficial and 62 harmful actions. The utility
pool remained severely imbalanced (10 beneficial, 280 harmful), largely
because broad default role authority produced 123 isolation and 74 relay
probes. The high-complexity humanitarian matched harm difference favored the
KPI controller by 0.0263; utility was tied on harm. The pilot
coupling-by-fragmentation coefficient was -0.0609. Neither result supports
progression.

The implementation audit found two domain-semantic issues before a third
pilot: physical service-edge failure could be classified as a communication
failure, and relay restoration operated on arbitrary global edges rather than
target-associated edges. It also found that four utility roles inherited the
entire utility action set. Those defects prevent a fair actionability test.

## Prospective iteration 3 changes

Before observing iteration 3 outcomes, the following fixed engineering
changes were recorded in `configs/v7_pilot_iteration3.yaml`:

- every utility role receives an explicit bounded action authority;
- communication and physical service-edge failure modes are separated;
- relay restoration prioritizes target-associated communication edges; and
- defensive isolation becomes a four-step bounded action instead of a
  permanent removal from service.

The belief model and all entropy calculations remain independent of action
outcomes. No formal thresholds, validation seeds, or holdout seeds have been
used. Iteration 3 is the final planned actionability feasibility iteration;
failure to produce a sufficiently mixed and competitive action pool will stop
the study before protocol freeze rather than trigger further outcome-driven
simulator tuning.
