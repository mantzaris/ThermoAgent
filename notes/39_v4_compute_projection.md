# V4 compute and cost projection

Projection frozen before the formal development matrix and before loading the
v4 Qwen qualification model.

The measured deterministic pilot sample contains 744 episodes with mean runner
time 0.248 seconds (range 0.155--0.405 seconds), excluding filesystem and
manifest overhead.  The 1,584-episode formal development design is projected
at 7--18 CPU wall-clock minutes and approximately 0.15 GB of new compressed
artifacts.  It uses no GPU.

The existing v3 real-Qwen measurement was 8 episodes in 714.10 seconds including
model load.  V4 qualification batches six independent contexts through the
already cached model.  The conservative qualification allocation is 0.25
single-GPU hours, 12 calls including the worst case of one repair per output,
20,000 prompt tokens, and 1,600 generated tokens.

If—and only if—all seven development gates pass, the provisional full v4
budget is:

| Component | GPU-hours | LLM calls | Prompt tokens | Generated tokens |
|---|---:|---:|---:|---:|
| real-Qwen qualification | 0.25 | 12 | 20,000 | 1,600 |
| one-shot validation | 3.50 | 1,500 | 4,500,000 | 192,000 |
| five-seed learned policies | 1.00 | 0 | 0 | 0 |
| locked holdout | 7.00 | 3,000 | 9,000,000 | 384,000 |
| robustness/export allowance | 0.50 | 100 | 300,000 | 12,800 |
| subtotal | 12.25 | 4,612 | 13,820,000 | 590,400 |
| 15% safety reserve | 1.84 | — | — | — |
| **cap-accounted projection** | **14.09** | **4,612** | **13,820,000** | **590,400** |

At the project's documented illustrative RTX 4090 rate of USD 0.34 per GPU
hour, this is approximately USD 4.79 including reserve.  The projection is
well below the 40-hour authorization cap.  Validation/training/holdout remain
locked until the prospective gate report permits them; a gate stop reduces the
actual GPU use to qualification only.

Model caches remain under `/workspace/.cache/huggingface`; the projected v4
Git-facing results are below 0.6 GB with every individual artifact below 50 MB.
