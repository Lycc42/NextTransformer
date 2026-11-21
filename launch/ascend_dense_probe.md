# Ascend Dense component validation (RFC #75)

These tools validate a debug-sized model and the native optimizer and checkpoint components. They do not run the trainer CLI, reproduce the formal `dense_full` protocol, or validate distributed training or performance.

From the repository root in the pinned CPU environment (`uv sync --frozen --extra cpu`):

```sh
python -m launch.validate_dense_device --device cpu --steps 3
python -m launch.validate_dense_resume --device cpu
python -m pytest -q tests/test_dense_device_probe.py
```

In a separately provisioned TorchNPU environment, use exactly one device already admitted and mapped to container `npu:0`. Do not replace the image's Torch/TorchNPU pairing with the repository's CPU installation instructions. No probe allocates a physical card or installs packages. Before using the probes, check device visibility and a synchronized FP32 matrix multiplication.

```sh
python -m launch.validate_dense_device --device npu:0 --steps 3 --skip-inductor-tuning
python -m launch.validate_dense_resume --device npu:0
python -m launch.validate_dense_resume --device npu:0 --npu-rng-sidecar
```

The device probe disables compilation and forces FP32. SDPA output and Q/K/V gradients are compared with an explicit causal CPU reference; the probe checks nonzero parameter updates and evolving optimizer state for both Muon and AdamW on every step. The explicit device-probe tuning bypass does not change the parameter groups, scheduler or update functions. It skips the Inductor configuration context and places the zero-dimensional optimizer scalars on NPU. The NPU resume probe uses the same eager path.

The resume probe checks four completed updates in each arm, step-2 and step-4 checkpoints, distinct worker processes, losses, cursor and next tokens, Python/NumPy/CPU Torch RNG, and routing state. Each arm uses native checkpoint save/load. The optional RNG sidecar deliberately feeds NPU randomness into every update and is restored **only after** the native checkpoint RNG test. Do not use the sidecar as a production checkpoint format. The default temporary outputs are cleaned up; use `--output` with a new directory if local artifacts are needed. Only load checkpoints created by this tool in a trusted directory.

## Verified scope

| Check | Observed |
|---|---|
| Pinned Torch 2.9.1, macOS arm64 CPU, FP32 eager | Three Dense updates; tuning bypass and default agree; fresh-process 4 vs 2+2 agrees |
| Single 910B4, Python 3.10.19, Torch 2.7.1+cpu, TorchNPU 2.7.1.post2, CANN 8.5.0, FP32 eager | Synchronized FP32 matrix multiplication; causal SDPA output and Q/K/V gradients (max error 4.77e-7); three original Muon and AdamW updates with the probe-only tuning/scalar changes |
| Same NPU stack, native checkpoint, fixed synthetic data | Both arms complete updates 1–4, identical losses, step-2 and step-4 model/optimizer tensors with max error 0, cursor and next tokens match |
| Same NPU stack, NPU-random inputs | TorchNPU RNG API replay passes; native checkpoint NPU RNG replay fails; probe-only sidecar replay and four-update comparison pass with max checkpoint tensor error 0 |

The NPU update, default resume and RNG-consuming resume measurements were taken at commits `db2fa77`, `e99d924`, and `c73c892`, respectively, on the probe branch. The 2.7.1 stack is not the repository's default 2.9.1. BF16, compilation, the actual trainer entry point, and HCCL are not covered.

## Proposed backend integration (not implemented)

- Agree on an opt-in single-device runtime with maintainers before editing shared code. Keep CPU/CUDA defaults unchanged; reject unsupported device counts, distributed NPU execution, and formal protocol mode initially.
- Make synchronization and reported software identity device-aware. Use the original SDPA path and optimizer update kernels; handle compiler configuration and scalar placement explicitly for a supported eager NPU stack. Do not change optimizer math or silently claim compiled execution.
- Add NPU RNG to the native rank checkpoint payload, using the existing atomic commit and checksum mechanism. Define compatibility for old CPU/CUDA checkpoints and fail clearly if an NPU run requires an RNG field that is missing. Test native restore without any separate sidecar and with training that consumes NPU RNG.
- Before claiming trainer support, verify the real entry point with a debug-sized research configuration and valid resources, then rerun the single-device fresh-process comparison through that entry point. `--smoke-steps` only limits updates, not model size. Leave BF16, compile, HCCL, and formal admission to separately validated work.
