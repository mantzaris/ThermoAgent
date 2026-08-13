# RL coordination metapolicy

The metapolicy selects one of nine explicit coordination options: continue,
request information, disclose a summary, negotiate, respond to an offer,
propose a coalition, request reallocation, emergency coordination, or remain
silent.

The actor consumes exactly 24 locally available scalar features: local
operational state, selected private utility weights, commitments, mean partner
trust, communication budget, local surprisal, local distributed entropy and
free-energy signals, interaction entropy, consensus error, role, and prior
tool failure. Its API rejects any vector with extra fields; evaluator state
cannot enter execution.

The final-candidate training method is staged PPO with a 2 x 64 tanh
actor/value network, clipped objective, GAE, and the deterministic planner.
Both variants receive the same balanced behavior-cloning initialization from
32 scripted training episodes (256 rows per observed option), followed by 192
PPO episodes and a one-epoch imitation anchor after every update. Separate
checkpoints are trained with entropy fields present and zeroed. The frozen real
LLM is coupled during evaluation.

Execution applies private/local structural masks: respond-to-offer is available
only for a delivered pending offer; request-reallocation only follows a failed
tool; emergency requires local impairment/shortfall; coalition action requires
a delivered proposal or local/distributed stress; communication options close
when the private budget is exhausted. Continue and silence remain available.
The mask validates affordances, not which valid option is best.

The initial 96-episode jobs completed at RL seed 3001 after the original pre-episode
NameError was fixed. Rewards are assigned to the preceding decision through the
subsequent environment transition, and GAE/done boundaries are grouped by
agent, not by interleaved batch row. The critic is local rather than
centralized; no global feature enters either actor or value network. Checkpoint
metadata and every training episode remain under `results/checkpoints/` and
`results/logs/training/`.

The original checkpoints are retained as pilot-v1 artifacts. Their real-model
pilot action counts revealed collapse: ThermoAgent used continue/emergency on
671/672 decisions and learned-no-entropy did so on 672/672. A 384-episode
longer-training diagnostic made both actors choose request-reallocation almost
exclusively, so those candidate checkpoints were rejected. Intermediate
behavior-cloning trials and their negative outcomes are recorded in the
failure notes.

The final local qualification used unseen seeds 71--73, two applications, and
nominal/moderate/compound conditions. Both candidates exercised request,
negotiate, offer response, coalition, reallocation, emergency, continue and
silence; ThermoAgent also exercised disclosure. ThermoAgent did not dominate:
commercial mean loss was 13.85 versus 13.80 for no-entropy, and humanitarian
mean weighted unmet need was 1,651.55 versus 1,543.28. Checkpoint adoption is
therefore based on removing collapse and restoring action semantics, not
selecting a favorable treatment result. Real-Qwen v3 qualification remains the
last gate. Full-LLM weights remain frozen; no language-model fine-tuning occurs.

After the final actor-information audit, the matched pair was retrained for 192
episodes per policy with the same 2,304-row balanced initialization. The
privacy-corrected actor now receives a link-local consensus residual and its own
inbox/outbox interaction entropy; methods without monitoring receive zeros for
all monitor fields. This pair was archived after the information-factor audit.

The final pre-v8 rebuild again used 192 episodes per policy, the identical
2,304-row initialization, and RL seed 3001 under the corrected factor. Both
initialization accuracies were 0.6818. The active checkpoint checksums are
`edbe570a...7096` (no entropy) and `62d5a1c7...74d2` (ThermoAgent). Their
training outcomes are optimization diagnostics only and did not gate adoption.

## Final evaluation

The matched main comparison numerically favored the entropy-conditioned actor
by `0.187` commercial service-loss AUC and `106.51` humanitarian weighted-need
units. Hierarchical intervals excluded zero, but the prespecified Holm-adjusted
p-values were `0.0856` and `0.0584`; the inference is mixed rather than
confirmed. On the locked unseen topology/shock holdout, the actors tied exactly
in both applications. Four-seed ablations likewise did not isolate a monitor-
feature effect.

ThermoAgent selected more coalition/replanning activity and breached fewer
commitments, but its valid-tool rate was lower, failures and communication were
higher, and useful-coalition precision stayed below 6%. Staged PPO therefore
restored a diverse option policy but did not learn a broadly superior
coordination strategy. Multiple RL seeds, online LLM-coupled refinement, and a
new locked evaluation would be required before attributing the main numerical
difference to a stable learned mechanism.
