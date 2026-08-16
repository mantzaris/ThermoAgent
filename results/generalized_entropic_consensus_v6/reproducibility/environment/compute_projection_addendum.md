# Compute-projection accounting addendum

Status: documentation-only; recorded during frozen development execution and
before multi-seed training or the formal Qwen qualification. It changes no
hypothesis, gate, seed, model, threshold, or scientific output.

The frozen protocol projected 11.50 reserved single-GPU hours (including a 15%
reserve), 2,730 LLM calls, 2.73 million prompt tokens, 253,000 generated
tokens, 4.3 GiB of storage, and USD 3.91 at USD 0.34 per Pod hour. It did not
separately label two requested accounting fields:

- CPU-bound analysis still occupies the single-GPU Pod and therefore counts
  toward reserved-Pod wall time and monetary cost even when measured GPU
  utilization is zero. Final accounting reports reserved-Pod hours and
  GPU-active hours separately; retrospective CPU process time is labeled as
  measured rather than preregistered.
- The pre-run bandwidth bound is 4.3 GiB for one full result retrieval plus
  small SSH control traffic. Base-model download bandwidth is zero for V6
  because the pinned Qwen revision was already cached outside Git tracking.
  Final accounting reports the actual result namespace size transferred.
