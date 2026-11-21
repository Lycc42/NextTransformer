"""Standalone eager FP32 component probe; not a trainer, benchmark, or resume test.

Run from the repository root:
    python -m launch.validate_dense_device --device cpu
    python -m launch.validate_dense_device --device npu:0

On Ascend, use the preconfigured Torch/TorchNPU environment with exactly one
admitted, isolated device. This probe does not select a physical card, install
dependencies, exercise the trainer, or verify checkpoint/RNG restoration.
Redirect stdout to retain the JSON result; stderr carries any traceback.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def attention_check(torch, flash_attention, device, rng):
    # The repository wrapper takes (batch, time, heads, head_dim).
    cpu = [torch.randn((1, 12, 2, 24), generator=rng, requires_grad=True) for _ in range(3)]
    q, k, v = (x.transpose(1, 2) for x in cpu)
    mask = torch.ones((12, 12), dtype=torch.bool).tril()
    weights = ((q @ k.transpose(-2, -1)) / (24 ** 0.5)).masked_fill(~mask, -torch.inf)
    expected = (weights.softmax(dim=-1) @ v).transpose(1, 2)
    upstream = torch.randn(expected.shape, generator=rng)
    expected.backward(upstream)

    actual_inputs = [x.detach().to(device).requires_grad_() for x in cpu]
    actual = flash_attention.flash_attn_func(*actual_inputs, causal=True, window_size=(12, 0))
    require(actual.device == device and actual.dtype == torch.float32, "SDPA did not produce FP32 on requested device")
    actual.backward(upstream.to(device))
    result = {}
    pairs = [("output", actual, expected), *zip(("dq", "dk", "dv"),
                                                (x.grad for x in actual_inputs), (x.grad for x in cpu))]
    for name, got, ref in pairs:
        require(got is not None and got.device == device, f"SDPA {name} missing on requested device")
        got = got.detach().cpu()
        torch.testing.assert_close(got, ref.detach(), rtol=1e-4, atol=1e-5, msg=name)
        result[name + "_max_abs_error"] = (got - ref.detach()).abs().max().item()
    with torch.no_grad():
        changed_v = actual_inputs[2].detach().clone()
        changed_v[:, 6:] += 10
        perturbed = flash_attention.flash_attn_func(*actual_inputs[:2], changed_v, causal=True, window_size=(12, 0))
        torch.testing.assert_close(actual[:, :6], perturbed[:, :6], rtol=1e-4, atol=1e-5)
        result["prefix_max_abs_error"] = (actual[:, :6] - perturbed[:, :6]).abs().max().item()
    return result


def build_probe_training(torch, device, skip_inductor_tuning=False):
    from nexttransformer.models import build_model
    from nexttransformer.models.config import debug_config
    from nexttransformer.models.native.config import NativeTrainConfig
    from nexttransformer.training.optimizer import build_optimizer

    config = debug_config("dense_full")
    train = NativeTrainConfig()
    # Initialize on CPU so both devices start from the same seed and weights.
    model = build_model(config, "cpu", seed=42).to(device).train()
    require(all(p.device == device and p.dtype == torch.float32 for p in model.parameters()),
            "model parameters are not all FP32 on the requested device")
    if skip_inductor_tuning:
        from nanochat.optim import MuonAdamW
        from nexttransformer.training.optimizer import parameter_groups
        from nexttransformer.training.sharded import ExperimentOptimizer

        class EagerProbeOptimizer(ExperimentOptimizer):
            # Skip only the compiler configuration context; keep the same
            # parameter groups, reduction methods, and update kernels.
            def step(self):
                return MuonAdamW.step(self)

        optimizer = EagerProbeOptimizer(parameter_groups(model, train))
        if device.type == "npu":
            # The eager NPU lerp overload requires its tensor weight on NPU.
            # Preserve FP32 zero-dimensional scalars and the original kernels.
            scalar_names = (
                "_adamw_step_t", "_adamw_lr_t", "_adamw_beta1_t", "_adamw_beta2_t",
                "_adamw_eps_t", "_adamw_wd_t", "_muon_momentum_t", "_muon_lr_t",
                "_muon_wd_t", "_muon_beta2_t",
            )
            for name in scalar_names:
                setattr(optimizer, name, getattr(optimizer, name).to(device))
    else:
        optimizer = build_optimizer(model, train)
    require({g["kind"] for g in optimizer.param_groups} == {"adamw", "muon"}, "missing optimizer kind")
    return config, train, model, optimizer


def train_updates(torch, device, components, next_tokens, start, stop):
    from nexttransformer.training.optimizer import set_schedule

    _, train, model, optimizer = components
    records = []
    for step in range(start, stop):
        model.set_step(step)
        model.set_row_offset(step)
        tokens = next_tokens().to(device)
        optimizer.zero_grad(set_to_none=True)
        set_schedule(optimizer, step, train)
        loss = model(tokens[:, :-1], tokens[:, 1:])
        require(loss.device == device and torch.isfinite(loss).item(), "loss missing on device or nonfinite")
        loss.backward()
        previous = {}
        active = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            require(param.grad is not None and param.grad.device == device and torch.isfinite(param.grad).all().item(),
                    f"invalid gradient: {name}")
            previous[name] = param.detach().clone()
            if torch.count_nonzero(param.grad).item():
                active.add(name)

        def state_tensors():
            for group in optimizer.param_groups:
                kind = group["kind"]
                keys = ("exp_avg", "exp_avg_sq") if kind == "adamw" else ("momentum_buffer", "second_momentum_buffer")
                owners = group["params"] if kind == "adamw" else group["params"][:1]
                for param in owners:
                    state = optimizer.state.get(param, {})
                    yield kind, param, state, keys

        old_state = {(param, key): state[key].detach().clone()
                     for _, param, state, keys in state_tensors() for key in keys if key in state}
        optimizer.step()
        if device.type == "npu":
            torch.npu.synchronize()
        summary = {kind: {"updated": [], "max_abs_update": 0.0, "changed_state": {}}
                   for kind in ("adamw", "muon")}
        for group in optimizer.param_groups:
            item = summary[group["kind"]]
            for name, param in zip(group["parameter_names"], group["params"], strict=True):
                require(torch.isfinite(param).all().item(), f"nonfinite parameter: {name}")
                delta = (param.detach() - previous[name]).abs().max().item()
                if name in active and delta > 0:
                    item["updated"].append(name)
                    item["max_abs_update"] = max(item["max_abs_update"], delta)
        for kind, param, state, keys in state_tensors():
            if kind == "adamw":
                require(state.get("step") == step + 1, "AdamW step counter did not advance")
            for key in keys:
                value = state.get(key)
                require(value is not None and value.device == device and torch.isfinite(value).all().item(),
                        f"missing or invalid {kind} state: {key}")
                old = old_state.get((param, key))
                changed = bool(torch.count_nonzero(value).item()) if old is None else not torch.equal(value, old)
                counts = summary[kind]["changed_state"]
                counts[key] = counts.get(key, 0) + int(changed)
        for kind, item in summary.items():
            require(item["updated"] and all(item["changed_state"].values()), f"{kind} did not update at step {step + 1}")
        records.append({"step": step + 1, "loss": loss.item(), "loss_dtype": str(loss.dtype), "groups": summary})
    return records


def train_check(torch, device, rng, steps, skip_inductor_tuning=False):
    components = build_probe_training(torch, device, skip_inductor_tuning)
    config = components[0]

    def next_tokens():
        return torch.randint(config.vocab_size, (1, config.sequence_len + 1), generator=rng)

    records = train_updates(torch, device, components, next_tokens, 0, steps)
    return {"shape": {key: getattr(config, key) for key in ("width", "heads", "layers", "sequence_len", "vocab_size")},
            "updates": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--skip-inductor-tuning", action="store_true",
                        help="probe-only: bypass the compiled optimizer tuning context, not its updates")
    args = parser.parse_args()
    require("torch" not in sys.modules, "run the probe in a fresh Python process")
    os.environ.update(TORCH_COMPILE_DISABLE="1", NANOCHAT_DTYPE="float32")
    import torch

    npu_version = None
    if args.device == "npu:0":
        import torch_npu  # noqa: F401 - registers the NPU device

        npu_version = importlib.metadata.version("torch-npu")
        require(torch.npu.device_count() == 1, "expected exactly one visible NPU")
        torch.npu.set_device(args.device)
    from nanochat import flash_attention
    from nanochat.common import COMPUTE_DTYPE

    require(not flash_attention.USE_FA3 and COMPUTE_DTYPE == torch.float32, "expected FP32 SDPA")
    require(torch._dynamo.config.disable, "TorchDynamo compile is not disabled")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=False)
    report = {"scope": "standalone_component_probe", "commit": git.stdout.strip() or None,
              "python": platform.python_version(), "platform": platform.system(), "arch": platform.machine(),
              "torch": torch.__version__, "torch_npu": npu_version, "device": str(device), "seed": 42,
              "compute_dtype": str(COMPUTE_DTYPE), "mode": "eager", "torch_compile_disabled": True,
              "inductor_tuning_skipped": args.skip_inductor_tuning,
              "optimizer_scalar_device": str(device) if args.skip_inductor_tuning else "cpu"}
    phase = "attention"
    try:
        report["attention"] = attention_check(torch, flash_attention, device, torch.Generator().manual_seed(42))
        phase = "training"
        report["training"] = train_check(torch, device, torch.Generator().manual_seed(43), args.steps,
                                         skip_inductor_tuning=args.skip_inductor_tuning)
    except Exception as exc:
        report.update(status="failed", phase=phase, error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(report, indent=2), flush=True)
        raise
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
