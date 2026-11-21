"""Fresh-process 4 vs 2+2 probe using the native checkpoint, not the trainer CLI.

Run from the repository root: python -m launch.validate_dense_resume --device cpu
On an admitted single NPU, use --device npu:0 in the prepared environment.
The default token stream does not consume NPU RNG. Use --require-native-npu-rng
for NPU-random updates with native restoration. The historical sidecar mode is
an independent comparison, not the trainer checkpoint format.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from launch.validate_dense_device import build_probe_training, require, train_updates


def worker(args):
    import random

    import numpy as np
    import torch

    if args.device == "npu:0":
        import torch_npu  # noqa: F401 - registers the NPU

        require(torch.npu.device_count() == 1, "expected exactly one NPU")
        torch.npu.set_device(0)
    from nexttransformer.runtime.checkpoint import load_checkpoint, save_checkpoint

    require(torch._dynamo.config.disable, "TorchDynamo must be disabled")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    consumes_npu_rng = args.npu_rng_sidecar or args.require_native_npu_rng
    components = build_probe_training(torch, device, skip_inductor_tuning=device.type == "npu")
    config, train, model, optimizer = components

    class Stream:
        def __init__(self):
            self.rows = 0
            self.generator = torch.Generator().manual_seed(43)

        def batch(self):
            self.rows += 1
            tokens = torch.randint(config.vocab_size, (1, config.sequence_len + 1), generator=self.generator)
            if consumes_npu_rng:
                # Consume the device default generator in every training update.
                shift = torch.randint(config.vocab_size, (), device=device).item()
                tokens = (tokens + shift) % config.vocab_size
            return tokens

        def state_dict(self):
            return {"rows": self.rows, "generator": self.generator.get_state()}

        def load_state_dict(self, state):
            self.rows = state["rows"]
            self.generator.set_state(state["generator"])

    def sample_npu():
        return torch.rand(32, device=device).cpu()

    def sample_host():
        return [random.random(), float(np.random.random()), *torch.rand(8).tolist()]

    def preview_tokens():
        state = stream.state_dict()
        device_state = torch.npu.get_rng_state(device) if consumes_npu_rng else None
        try:
            return stream.batch().tolist()
        finally:
            stream.load_state_dict(state)
            if device_state is not None:
                torch.npu.set_rng_state(device_state, device)

    def checkpoint(step):
        state = {"step": step, "input_tokens": stream.rows * config.sequence_len, "next_global_sequence": stream.rows}
        return save_checkpoint(args.output, model, optimizer, stream, config, train, state, rank=0, world=1)

    stream = Stream()
    random.seed(42)
    np.random.seed(42)
    if device.type == "npu":
        torch.npu.manual_seed(42)
    report = {"pid": os.getpid(), "updates": [], "npu_rng_api_replay": None,
              "native_checkpoint_npu_rng_replay": None, "sidecar_npu_rng_replay": None}
    if args.phase != "resume":
        report["updates"] += train_updates(torch, device, components, stream.batch, 0, 2)
        if device.type == "npu":
            # Draw before saving so the expected next state differs from the initial seed.
            sample_npu()
        boundary = checkpoint(2)
        report["next_tokens"] = preview_tokens()
        report["host_rng"] = sample_host()
        if device.type == "npu":
            saved_rng = torch.npu.get_rng_state(device)
            expected = sample_npu()
            sample_npu()
            torch.npu.set_rng_state(saved_rng, device)
            report["npu_rng_api_replay"] = bool(torch.equal(expected, sample_npu()))
            require(report["npu_rng_api_replay"], "TorchNPU RNG API replay failed")
            torch.save(expected, args.output / "expected-npu-draw.pt")
            if args.npu_rng_sidecar:
                torch.save(saved_rng, args.output / "probe-npu-rng-step2.pt")
            if consumes_npu_rng:
                torch.npu.set_rng_state(saved_rng, device)
    else:
        require(args.output.joinpath("checkpoint_00000002/complete.json").is_file(), "missing committed step 2")
        boundary = args.output / "checkpoint_00000002"
        if device.type == "npu":
            # Deliberately different state: a successful restore must overwrite it.
            torch.npu.manual_seed(1234)
        saved = load_checkpoint(boundary, model, optimizer, stream, rank=0, world=1)
        require(saved["state"]["step"] == stream.rows == 2, "step-2 cursor mismatch")
        report["host_rng"] = sample_host()
        if device.type == "npu":
            expected = torch.load(args.output / "expected-npu-draw.pt", map_location="cpu", weights_only=True)
            loaded_rng = torch.npu.get_rng_state(device)
            actual = sample_npu()
            report["native_checkpoint_npu_rng_replay"] = bool(torch.equal(expected, actual))
            if args.require_native_npu_rng:
                torch.npu.manual_seed(1234)
                report["omitted_npu_rng_restore_detected"] = not torch.equal(expected, sample_npu())
                require(report["omitted_npu_rng_restore_detected"], "negative RNG control unexpectedly matched")
                require(report["native_checkpoint_npu_rng_replay"], "native NPU RNG restore did not replay the next draw")
                torch.npu.set_rng_state(loaded_rng, device)
            if args.npu_rng_sidecar:
                saved_rng = torch.load(args.output / "probe-npu-rng-step2.pt", map_location="cpu", weights_only=True)
                torch.npu.set_rng_state(saved_rng, device)
                report["sidecar_npu_rng_replay"] = bool(torch.equal(expected, sample_npu()))
                require(report["sidecar_npu_rng_replay"], "RNG sidecar replay failed")
                torch.npu.set_rng_state(saved_rng, device)
        report["next_tokens"] = preview_tokens()
    if args.phase != "first":
        report["updates"] += train_updates(torch, device, components, stream.batch, 2, 4)
        checkpoint(4)
    report["rows"] = stream.rows
    args.output.joinpath(args.phase + ".json").write_text(json.dumps(report), encoding="utf-8")


def compare(a, b, torch, path, errors):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, f"tensor mismatch: {path}")
        if a.is_floating_point() and a.numel():
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7, msg=lambda message: f"{path}: {message}")
            errors.append((a - b).abs().max().item())
        else:
            require(torch.equal(a, b), f"tensor mismatch: {path}")
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"keys differ: {path}")
        for key in a:
            compare(a[key], b[key], torch, f"{path}.{key}", errors)
    elif isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), f"sequence mismatch: {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, torch, f"{path}.{i}", errors)
    else:
        require(a == b, f"value mismatch: {path}")


def orchestrate(device, root, npu_rng_sidecar, require_native_npu_rng):
    import torch

    continuous, segmented = root / "continuous", root / "segmented"
    for directory in (continuous, segmented):
        directory.mkdir()

    def run(phase, directory):
        command = [sys.executable, "-m", "launch.validate_dense_resume", "--device", device,
                   "--phase", phase, "--output", str(directory)]
        if npu_rng_sidecar:
            command.append("--npu-rng-sidecar")
        if require_native_npu_rng:
            command.append("--require-native-npu-rng")
        subprocess.run(command, check=True)
        return json.loads(directory.joinpath(phase + ".json").read_text(encoding="utf-8"))

    a, first, resumed = run("continuous", continuous), run("first", segmented), run("resume", segmented)
    require(len({a["pid"], first["pid"], resumed["pid"]}) == 3, "workers did not use distinct processes")
    b = first["updates"] + resumed["updates"]
    require([x["step"] for x in a["updates"]] == [x["step"] for x in b] == [1, 2, 3, 4], "missing updates")
    require(first["rows"] == 2 and a["rows"] == resumed["rows"] == 4, "wrong cursor")
    require(a["next_tokens"] == first["next_tokens"] == resumed["next_tokens"], "next tokens differ at step 2")
    require(a["host_rng"] == first["host_rng"] == resumed["host_rng"], "host RNG differs after checkpoint load")
    errors = []
    for step in (2, 4):
        left, right = (out / f"checkpoint_{step:08d}" for out in (continuous, segmented))
        for name, keys in (("model.pt", ("model", "state")), ("rank_0000.pt", ("optimizer", "data", "routing_rng"))):
            x, y = (torch.load(p / name, map_location="cpu", weights_only=False) for p in (left, right))
            for key in keys:
                compare(x[key], y[key], torch, f"step{step}.{key}", errors)
    losses_a, losses_b = ([x["loss"] for x in records] for records in (a["updates"], b))
    torch.testing.assert_close(torch.tensor(losses_a), torch.tensor(losses_b), rtol=1e-5, atol=1e-7)
    return {"scope": "standalone_native_checkpoint_components", "device": device,
            "status": "passed", "continuous_pids": [a["pid"]], "segmented_pids": [first["pid"], resumed["pid"]],
            "continuous_steps": [x["step"] for x in a["updates"]], "segmented_steps": [x["step"] for x in b],
            "continuous_losses": losses_a, "segmented_losses": losses_b,
            "max_abs_checkpoint_tensor_error": max(errors, default=0), "cursor_and_next_tokens_match": True,
            "host_rng_replay": True, "npu_rng_consumed_by_training": npu_rng_sidecar or require_native_npu_rng,
            "omitted_npu_rng_restore_detected": resumed.get("omitted_npu_rng_restore_detected"),
            "npu_rng_api_replay": first["npu_rng_api_replay"],
            "native_checkpoint_npu_rng_replay": resumed["native_checkpoint_npu_rng_replay"],
            "sidecar_npu_rng_replay": resumed["sidecar_npu_rng_replay"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    parser.add_argument("--phase", choices=("continuous", "first", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="new directory to retain trusted local checkpoint artifacts")
    rng_mode = parser.add_mutually_exclusive_group()
    rng_mode.add_argument("--npu-rng-sidecar", action="store_true", help="test NPU-random training with a separate RNG file")
    rng_mode.add_argument("--require-native-npu-rng", action="store_true", help="require native replay for NPU-random training")
    args = parser.parse_args()
    if (args.npu_rng_sidecar or args.require_native_npu_rng) and args.device != "npu:0":
        parser.error(f"{'--npu-rng-sidecar' if args.npu_rng_sidecar else '--require-native-npu-rng'} requires --device npu:0")
    os.environ.update(TORCH_COMPILE_DISABLE="1", NANOCHAT_DTYPE="float32")
    if args.phase:
        require(args.output is not None, "worker requires output directory")
        worker(args)
    elif args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(orchestrate(args.device, args.output, args.npu_rng_sidecar, args.require_native_npu_rng), indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix="dense-resume-") as path:
            print(json.dumps(orchestrate(args.device, Path(path), args.npu_rng_sidecar, args.require_native_npu_rng), indent=2))


if __name__ == "__main__":
    main()
