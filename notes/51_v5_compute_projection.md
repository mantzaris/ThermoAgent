# V5 measured compute projection

Projection frozen before the V5 real-Qwen and multi-seed runs.

The retained V4 qualification generated six Qwen decisions in 6.879 inference
seconds, with 3,942 prompt and 328 generated tokens. V5 requires 108 initial
decisions in batches of 12; prompts are larger because they include private
beliefs and role-specific schemas. The conservative projection is:

- model load and CUDA smoke: 2-4 minutes;
- 108 initial calls plus up to 25% one-repair calls: 4-10 minutes;
- prompt tokens: 110,000-220,000;
- generated tokens: 6,000-15,000;
- Qwen GPU time including a 15% reserve: at most 0.35 single-GPU hours;
- additional model-cache disk: zero expected because the immutable revision is
  already cached; result artifacts under 100 MiB;
- Qwen cost at the project's illustrative USD 0.34/GPU-hour: at most USD 0.12.

The ten decentralized PPO-style runs use 30,000 decision epochs each. A local
256-step smoke run, including uncached dataset construction, completed in about
34 seconds. Dataset construction is cached per method for the formal run. The
projected CPU wall time is 6-15 minutes and GPU time is zero unless the remote
environment elects CUDA automatically. Even if executed on the RTX 4090, the
combined training projection is below 0.5 GPU hours.

The complete additional V5 projection is therefore below 1.0 single-GPU hour,
well under the 50-hour cap and below USD 0.35, with substantial margin under
the USD 40 cap. CPU-only refit permutation analysis may take 20-40 wall-clock
minutes but does not consume billed GPU inference time beyond keeping the
already-authorized Pod available.
