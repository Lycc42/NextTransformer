# Synthetic Dense research smoke fixture

This fixture exercises `nexttransformer.train` with actual Parquet, a serialized RustBPE/tiktoken tokenizer, the native resource manifest, and native cost auditing. It does not supply official data or claim protocol admission. It is distinct from the 64-token component probe: the byte tokenizer requires 256 byte tokens plus 9 special tokens, so this debug config uses a vocabulary of 265, with width 48, two layers, and a 12-token context.

From the repository root, in an existing supported CPU environment:

```sh
export CUDA_VISIBLE_DEVICES= TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1
fixture=.agent-work/dense-fixture
python -m launch.generate_dense_smoke_fixture --output "$fixture"
run_smoke() {
    python -m nexttransformer.train --research \
        --config "$fixture/dense_full.json" --resources "$fixture/resources" \
        --output "$fixture/$1" --runtime-dir "$fixture/runtime" \
        --micro-batch 1 --smoke-accumulation 1 --smoke-val-tokens 12 \
        --smoke-core-per-task 0 --smoke-steps "$2" --log-every 1 --wandb-mode disabled
}
run_smoke continuous 4
run_smoke segmented 2
run_smoke segmented 4
```

Use a new fixture directory for each generation. The second invocation for `segmented` automatically resumes its existing checkpoint, up to absolute step 4. The segmented arm may additionally perform smoke validation at step 2. `--smoke-steps` does not shrink the model; the generated config supplies the debug shape. Nothing here selects or reserves a physical NPU, adjusts the trainer's device selection, or supplies native NPU RNG checkpoint support.

The generator runs `Auditor.aggregate` for the debug config and the canonical Dense anchor using the active Torch build (cost auditing is on meta tensors), then calls the normal `review_config` and `verify_resources`. Expect findings for lack of formal preflight admission and budget mismatch. The fixture's manifest explicitly marks synthetic input; it retains the required revision field for this manifest schema, but neither the synthetic shards nor tokenizer are the pinned protocol artifacts. Generate and validate with the target interpreter, not by editing the Torch version in proof metadata. If the target Torch cannot audit the graph, stop rather than substitute unverifiable cost evidence.

For a historical Python 3.10/Torch 2.7 image, keep the installed Torch/TorchNPU pairing. A no-device metadata check of the CANN 8.5 image's `llm_test` environment found Python 3.10.19, Torch 2.7.1, TorchNPU 2.7.1.post2, PyArrow 23.0.0, tiktoken 0.12.0 and NumPy 1.26.0, but **no `rustbpe`**. Provision a compatible ARM64 Python 3.10 `rustbpe` in a separate overlay, verify imports with `python -m nexttransformer.train --help`, and generate the fixture in that environment. Existing image versions may differ from the default lock. Do not run a dependency resolver that replaces the Torch pair. Target-stack auditing and training were not run by the no-device metadata check. No network or data downloads are performed by the fixture itself.
