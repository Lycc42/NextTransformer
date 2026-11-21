"""CPU tests for opt-in selection and native checkpoint RNG compatibility."""

from types import SimpleNamespace

import pytest
import torch

from nexttransformer import train
from nexttransformer.runtime import process


BASE = ["--config", "debug.json", "--output", "unused", "--micro-batch", "1"]


def test_npu_requires_research_smoke():
    assert not train.parse(BASE).npu
    assert train.parse(BASE + ["--npu", "--smoke-steps", "4", "--smoke-stop-at", "2"]).npu
    for options in (["--npu"], ["--npu", "--smoke-steps", "4", "--formal"],
                    ["--smoke-steps", "4", "--smoke-stop-at", "4"]):
        with pytest.raises(SystemExit):
            train.parse(BASE + options)


def test_npu_refuses_multirank_before_import(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="single local rank"):
        process.initialize(42, npu=True)


def test_rng_restores_device_state_and_rejects_missing_field(monkeypatch):
    calls = []
    device = SimpleNamespace(type="npu")
    sample = torch.tensor([7], dtype=torch.uint8)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_rng_state=lambda d: sample.clone(),
                                                    set_rng_state=lambda state, d: calls.append((state, d))), raising=False)
    legacy = process.rng_state(torch.device("cpu"))
    assert "npu" not in legacy
    process.restore_rng(legacy, torch.device("cpu"))
    with pytest.raises(ValueError, match="requires NPU RNG state"):
        process.restore_rng(legacy, device)
    state = process.rng_state(device)
    process.restore_rng(state, device)
    assert len(calls) == 1 and calls[0][1] is device and torch.equal(calls[0][0], sample)
    with pytest.raises(ValueError, match="requires an NPU destination"):
        process.restore_rng(state, torch.device("cpu"))
