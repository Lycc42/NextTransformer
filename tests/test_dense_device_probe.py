"""CPU CI coverage for the standalone Ascend RFC component probes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def probe(module, *args):
    return subprocess.run([sys.executable, "-m", "launch." + module, *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_dense_eager_paths_keep_the_same_updates():
    reports = []
    for flag in ((), ("--skip-inductor-tuning",)):
        result = probe("validate_dense_device", "--device", "cpu", "--steps", "3", *flag)
        assert result.returncode == 0, result.stderr
        reports.append(json.loads(result.stdout))
    default, isolated = reports
    assert default.pop("inductor_tuning_skipped") is False
    assert isolated.pop("inductor_tuning_skipped") is True
    assert default == isolated
    assert default["status"] == "passed" and default["torch_compile_disabled"]
    assert default["attention"]["prefix_max_abs_error"] == 0
    assert [step["step"] for step in default["training"]["updates"]] == [1, 2, 3]
    for step in default["training"]["updates"]:
        assert set(step["groups"]) == {"adamw", "muon"}
        assert all(group["updated"] and all(group["changed_state"].values()) for group in step["groups"].values())


def test_dense_resume_replays_four_updates_in_fresh_processes(tmp_path):
    output = tmp_path / "artifacts"
    result = probe("validate_dense_resume", "--device", "cpu", "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["continuous_steps"] == report["segmented_steps"] == [1, 2, 3, 4]
    assert len(set(report["continuous_pids"] + report["segmented_pids"])) == 3
    assert report["max_abs_checkpoint_tensor_error"] == 0
    assert report["cursor_and_next_tokens_match"] and report["host_rng_replay"]
    assert report["npu_rng_api_replay"] is None and report["native_checkpoint_npu_rng_replay"] is None
    for arm in ("continuous", "segmented"):
        assert json.loads((output / arm / "latest.json").read_text())["step"] == 4
    repeated = probe("validate_dense_resume", "--device", "cpu", "--output", str(output))
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr


def test_sidecar_requires_npu():
    result = probe("validate_dense_resume", "--device", "cpu", "--npu-rng-sidecar")
    assert result.returncode == 2
    assert "requires --device npu:0" in result.stderr
