# Single-card Ascend Dense research smoke

The reference trainer has an explicit `--npu` research-smoke mode. It requires one visible NPU, one rank, a bounded debug `dense_full` configuration, and eager execution. It does not enable formal admission, change the registered protocol, or start a process group. Default CPU/CUDA selection is unchanged.

Use an independently provisioned, admitted single-card environment. The initial hardware stack used Python 3.10.19, Torch 2.7.1+cpu, TorchNPU 2.7.1.post2, and CANN 8.5.0. Keep these packages paired; do not run the repository's CPU dependency sync in that image. The historical image needed RustBPE 0.1.0 added using a verified wheel in an isolated `--target` directory (`pip install --no-index --no-deps --target <dir> <wheel>` and prepend the directory to `PYTHONPATH`). Import TorchNPU and verify a synchronized FP32 MatMul on the single visible device before launching.

From the source checkout, choose a new scratch directory on a persistent filesystem. Generate the fixture in the same Torch environment used for the run. This builds actual synthetic Parquet resources, tokenizer and proofs, then applies the regular resource and configuration review. The debug configuration is deliberately not admitted for formal results.

```sh
export NANOCHAT_DTYPE=float32 TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1
RUN_ROOT=/path/to/new/scratch
mkdir -p "$RUN_ROOT"
python -m launch.generate_dense_smoke_fixture --output "$RUN_ROOT/fixture"
run() {
  python -m nexttransformer.train --npu \
    --config "$RUN_ROOT/fixture/dense_full.json" \
    --resources "$RUN_ROOT/fixture/resources" --runtime-dir "$RUN_ROOT/runtime" \
    --micro-batch 1 --smoke-accumulation 1 --smoke-steps 4 \
    --wandb-mode disabled --log-every 1 --rolling-every 1 --rolling-keep 4 \
    --deterministic-attention "$@"
}
run --output "$RUN_ROOT/continuous"
run --output "$RUN_ROOT/segmented" --smoke-stop-at 2
run --output "$RUN_ROOT/segmented" --resume auto
python -m launch.compare_dense_trainer_resume \
  --continuous "$RUN_ROOT/continuous" --segmented "$RUN_ROOT/segmented" \
  --resources "$RUN_ROOT/fixture/resources"
python -m launch.validate_dense_resume --device npu:0 --require-native-npu-rng
```

`--smoke-stop-at 2` checkpoints after update 2 without changing the target update count or evaluation schedule. `--smoke-steps` does not resize models. The comparison reads locally generated trusted checkpoints only. It compares the first continued update and final tensors, the packed input reconstructed from each boundary cursor, the logged losses and gradient norms, and both optimizer kinds' parameter and state changes. The separate stochastic test uses native checkpoint restore and does not change the trainer's model: it consumes NPU RNG in a test stream and verifies that the next draw would differ without restoration.

NPU RNG lives in the checksummed rank payload. Legacy CPU/CUDA payloads keep their original schema and restore without an NPU field. An NPU training resume requires that field; loading model weights without an optimizer is separate from training resume. A changed code/runtime identity still follows existing trainer resume checks.

## Evidence and audit scope

- At `a0b82e3`, the real trainer completed three FP32 updates and both smoke validations on the stack above, using generated resources and current-build cost proof. At `90e3032`, real trainer runs completed 4 vs fresh-process 2+2: all logged losses, gradient norms and learning rate multipliers matched; step-2 and step-4 checkpoint tensor differences were at most `6.71e-8`. Both optimizer kinds advanced. A separate stochastic native checkpoint test matched exactly and detected the omitted-RNG negative control; no RNG sidecar was used. The extended comparator that also checks the first continued checkpoint and reconstructs the next packed input was separately verified on the real pinned CPU trainer.
- The default CPU suite on Torch 2.9.1 passed 675 tests (205 skipped). The change check flags the trainer, process, checkpoint and optimizer wrapper as training-contract changes. The registered cost fingerprint check passed at this revision; that is not a claim that source identity is unchanged or that historical runs cover the new runtime. Generate NPU research proofs on the actual Torch stack and new run provenance/checkpoint evidence for the new backend. Do not rewrite published results, audits or evidence. No CUDA execution was tested.
- BF16, runs beyond 20 updates, HCCL and compilation are not covered by these FP32 results.

The enhanced hardware comparison at `8b06a85` requests deterministic algorithms via `--deterministic-attention` on all three trainer invocations. Both 4 vs 2+2 and 20 vs 10+10 passed on one 910B4: all requested updates completed; reconstructed packed inputs, losses, gradient norms, and optimizer/data state matched; maximum absolute checkpoint tensor difference was `0.0` at steps 2/3/4 and 10/11/20, respectively. The independent stochastic native RNG check again passed with no sidecar and detected omitted restoration. The pinned CPU golden check (`--base 1a0828a6b0eb8a078380ef00eb7250e0b237bf1c`) reported every registered method bitwise identical on Torch 2.9.1, without TorchNPU installed. Without that option, the 910B stack exhibited about `1.01e-6` parameter drift even between two uninterrupted runs, so default-mode restarts are not claimed to be bitwise. The comparator uses `rtol=1e-5`, `atol=1e-7` for floating checkpoint tensors and exact equality for counters and packed inputs. For the long check, use `--smoke-steps 20 --rolling-every 1 --rolling-keep 20`, stop the first segment with `--smoke-stop-at 10`, and pass `--steps 20 --split 10` to the comparator; this retains checkpoints 10, 11 and 20.

The trainer uses the existing causal SDPA path, parameter grouping, learning rate schedule and Muon/AdamW kernels. On NPU only, eager mode bypasses compile-specific optimizer configuration and co-locates FP32 scalar tensors with parameters. FP64 loss accounting remains on the host for NPU smoke. The opt-in mode is not a benchmark; host metric copies affect timing.
