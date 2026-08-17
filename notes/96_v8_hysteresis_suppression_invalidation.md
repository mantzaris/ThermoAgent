# V8 hysteresis-suppression invalidation

This decision was made after the corrected 288-arm development batch completed
but before protocol freeze, training, validation, or holdout. The batch passed
its numerical resource/error screen, but a mandatory mechanism audit showed
that only one of 12,236 transmissions in the selected arm was caused by the
generalized-information score. Initialization, maximum silence, and partition
recovery accounted for the remainder.

Cause: the trigger entered an off latch after transmission, while release was
tested using a score containing message age. Age increases monotonically until
the next transmission, and routine one-epoch belief innovation also exceeded
the low off threshold. The state consequently remained latched until the
maximum-silence override.

The complete attempt is retained as
`development_final_hysteresis_suppression_invalidated`. It cannot contribute to
V8 claims or trigger selection. The prospective repair keeps age in the on
score but computes the off transition from sender-local JS, entropy-spectrum,
and confidence innovation only. This preserves a maximum-silence safety bound
without allowing age itself to prevent latch release. Fresh seed namespaces
will be used for repair pilots and the replacement formal development batch.
