# V8 hysteresis-repair pilot iteration 2

Recorded after pilot 1 failed and before executing fresh seeds 8812201--8812306.
Pilot 1 is retained unchanged. The 5% information-score traffic criterion is
not lowered. Its observed 2.07% humanitarian fraction showed that thresholds
0.125 and 0.13 left the scheduler dominated by maximum-silence refreshes.

Iteration 2 evaluates two coarser, prospectively declared on thresholds, 0.11
and 0.115, with the same weights, off threshold, cooldown, 30-step deadline,
uint8 encoding, and two-hop forwarding. The mechanism gate additionally makes
the existing nominal criterion explicit: mean pre-disruption non-initial
transmission rate must be at most 0.10 in each application. The selection rule
otherwise remains unchanged. No outcome-adaptive dense threshold search is
permitted after this iteration. If neither candidate passes, V8 stops before
formal development and protocol freeze.
