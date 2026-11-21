"""Compare trusted local outputs of two real single-device Dense smoke runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from launch.validate_dense_device import require
from launch.validate_dense_resume import compare


RTOL, ATOL = 1e-5, 1e-7


def check(left, right, resources, split_step, total, *, uninterrupted_control=False):
    import torch

    from nanochat.tokenizer import RustBPETokenizer
    from nexttransformer.runtime.data import BestFitStream

    if not uninterrupted_control:
        require((right / f"resume_{split_step:08d}.json").is_file(), "missing resumed segment provenance")

    def updates(root):
        rows = map(json.loads, (root / "metrics.jsonl").read_text().splitlines())
        return [row for row in rows if row["event"] == "train"]

    a, b = map(updates, (left, right))
    expected = list(range(1, total + 1))
    require([event["step"] for event in a] == [event["step"] for event in b] == expected, "missing actual train events")
    keys = ("step", "input_tokens", "loss", "grad_norm", "lr_multiplier")
    require([[e[k] for k in keys] for e in a] == [[e[k] for k in keys] for e in b], "update metrics differ")
    require(all(math.isfinite(e[k]) for e in a for k in ("loss", "grad_norm")), "nonfinite training metrics")

    def load(root, step, name):
        return torch.load(root / f"checkpoint_{step:08d}" / name, map_location="cpu", weights_only=False)

    errors = []
    for step in sorted({split_step, split_step + 1, total}):
        x, y = (load(p, step, "model.pt") for p in (left, right))
        compare(x["model"], y["model"], torch, f"step{step}.model", errors)
        for key in ("step", "input_tokens", "next_global_sequence"):
            require(x["state"][key] == y["state"][key], f"counter differs: {key}")
        x, y = (load(p, step, "rank_0000.pt") for p in (left, right))
        for key in ("data", "optimizer", "routing_rng"):
            compare(x[key], y[key], torch, f"step{step}.{key}", errors)
        if "npu" in x["rng"] or "npu" in y["rng"]:
            require(torch.equal(x["rng"]["npu"], y["rng"]["npu"]), "saved NPU RNG differs")

    token_dir = resources / "tokenizer"
    tokenizer = RustBPETokenizer.from_directory(token_dir)
    boundary_model = load(left, split_step, "model.pt")
    seq = boundary_model["model_config"]["sequence_len"]

    def next_batch(root):
        stream = BestFitStream(resources / "base_data_climbmix", tokenizer, "train", sequence_len=seq)
        stream.load_state_dict(load(root, split_step, "rank_0000.pt")["data"])
        return [tensor.tolist() for tensor in stream.batch(1, torch.device("cpu"))]

    tokens = next_batch(left)
    require(tokens == next_batch(right), "next packed input differs at resume boundary")
    m1, m2 = (load(left, step, "model.pt")["model"] for step in (split_step, total))
    o1, o2 = (load(left, step, "rank_0000.pt")["optimizer"] for step in (split_step, total))
    changes = {}
    for group in o2["param_groups"]:
        kind = group["kind"]
        fields = ("exp_avg", "exp_avg_sq") if kind == "adamw" else ("momentum_buffer", "second_momentum_buffer")
        summary = changes.setdefault(kind, {"moved": 0, "state_changed": dict.fromkeys(fields, 0)})
        summary["moved"] += sum(not torch.equal(m1[name], m2[name]) for name in group["parameter_names"])
        for owner in group["params"] if kind == "adamw" else group["params"][:1]:
            old, new = (state["state"][owner] for state in (o1, o2))
            for field in fields:
                require(torch.isfinite(new[field]).all().item(), f"nonfinite {kind} state: {field}")
                summary["state_changed"][field] += int(not torch.equal(old[field], new[field]))
            if kind == "adamw":
                require(old["step"] == split_step and new["step"] == total, "AdamW counter mismatch")
    require(set(changes) == {"adamw", "muon"} and all(c["moved"] and all(c["state_changed"].values())
            for c in changes.values()), "missing parameter/state updates")
    return {"status": "passed", "comparison": "uninterrupted_control" if uninterrupted_control else "resume",
            "steps": expected, "split_step": split_step, "losses": [e["loss"] for e in a],
            "next_packed_batch_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "max_checkpoint_abs_error": max(errors, default=0), "rtol": RTOL, "atol": ATOL,
            "compared_checkpoint_steps": sorted({split_step, split_step + 1, total}), "actual_updates": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--segmented", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--split", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--uninterrupted-control", action="store_true", help="compare two uninterrupted runs")
    args = parser.parse_args()
    if not 0 < args.split < args.steps:
        parser.error("split must be between 0 and steps")
    print(json.dumps(check(args.continuous, args.segmented, args.resources, args.split, args.steps,
                           uninterrupted_control=args.uninterrupted_control), indent=2))


if __name__ == "__main__":
    main()
