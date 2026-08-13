# RunPod setup and dependency record

## 2026-08-11 live audit

- Remote source path: `/workspace/ThermoAgent`
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB total, 24,090 MiB free at audit
- Driver: 570.195.03
- PyTorch: 2.8.0+cu128; CUDA available; compute capability 8.9
- CPU/RAM: 32 logical CPUs; 124 GiB RAM; no swap
- Storage: 50 GiB container overlay; persistent network `/workspace`
- Python/pip: 3.12.3 / pip 25.2
- Utilities: uv 0.9.0, tmux 3.4, rsync 3.2.7, Git 2.43.0, GCC 13.3
- Preinstalled research packages: NumPy 2.1.2, NetworkX 3.3, PyYAML 6.0.3
- Not installed at audit: Transformers, vLLM, SGLang, Accelerate,
  bitsandbytes, SciPy, pandas, matplotlib, pytest, pydantic, Gymnasium, and
  Stable-Baselines3.

## Isolated environment installation

The project environment was created at `/workspace/ThermoAgent/.venv` on
2026-08-11. Standard pip was deliberately used inside a
`--system-site-packages` venv so it recognized and retained the image's tested
PyTorch build. An initial `uv pip` attempt was stopped because its resolver
began downloading a redundant CUDA 13/PyTorch stack; the incomplete environment
was quarantined and is not used.

Installed additions: Transformers 4.55.4, Accelerate 1.10.1, bitsandbytes
0.47.0, huggingface-hub 0.34.4, safetensors 0.6.2, Pydantic 2.11.7, pytest
8.4.1, SciPy 1.16.1, pandas 2.3.1, matplotlib 3.10.5, NetworkX 3.5, and
PyYAML 6.0.2. The post-install invariant check reported PyTorch
2.8.0+cu128/CUDA 12.8 and all 26 tests passed in 7.40 seconds.

Primary model candidate: `Qwen/Qwen2.5-7B-Instruct`, immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`, with bitsandbytes NF4, double
quantization, and bfloat16 compute. Model/cache roots are
`/workspace/.cache/huggingface` and `/workspace/.cache/thermoagent`.

## Access caveat

An earlier direct endpoint became stale, while the operator gateway supported
only interactive PTY use. A refreshed direct TCP mapping restored the existing
non-interactive scripts. Endpoint and identity details remain in local operator
configuration; no private key, SSH configuration, token, or environment secret
is stored in this repository or copied to the Pod.
