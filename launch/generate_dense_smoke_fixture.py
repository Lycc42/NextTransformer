"""Generate isolated synthetic resources for the real Dense research-smoke CLI.

Run from the repository root with the target Torch and non-Torch dependencies.
The fixture is not formal data, a formal tokenizer, or an admitted configuration.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path


def build(output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
    from nexttransformer.models.config import candidate_config, debug_config
    from nexttransformer.protocol.admission import review_config
    from nexttransformer.protocol.calibrate import Auditor
    from nexttransformer.protocol.spec import write_config
    from nexttransformer.runtime.data import (BestFitStream, data_inventory, inventory_fingerprint,
                                              tokenizer_fingerprint, verify_resources)
    from nexttransformer.runtime.files import DATA_REVISION, VALIDATION_SHARD, atomic_json, sha256
    from nexttransformer.runtime.resources import verify_upstream_sources

    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    resources = output / "resources"
    data_dir = resources / "base_data_climbmix"
    data_dir.mkdir(parents=True)
    train = [f"synthetic train document {i:04d} with words" for i in range(1024)]
    val = [f"synthetic validation example {i:04d}" for i in range(128)]
    for name, rows in (("shard_00000.parquet", train), (VALIDATION_SHARD, val)):
        pq.write_table(pa.table({"text": rows}), data_dir / name, compression="NONE", row_group_size=128)

    # RustBPE needs a full byte alphabet before the special tokens. Do not
    # force the standalone tensor probe's synthetic integer vocabulary of 64.
    tokenizer = RustBPETokenizer.train_from_iterator(iter(train), 256 + len(SPECIAL_TOKENS))
    tokenizer_dir = resources / "tokenizer"
    tokenizer.save(tokenizer_dir)
    special = {tokenizer.encode_special(token) for token in tokenizer.get_special_tokens()}
    sizes = [0 if i in special else len(tokenizer.decode_single_token_bytes(i))
             for i in range(tokenizer.get_vocab_size())]
    torch.save(torch.tensor(sizes, dtype=torch.int32), tokenizer_dir / "token_bytes.pt")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    pinned = json.loads((root / "configs/native/upstream_sources.json").read_text())
    manifest = dict(fixture="synthetic-research-smoke-v1", data_revision=DATA_REVISION,
                    tokenizer_sha256=tokenizer_fingerprint(resources),
                    data_inventory_sha256=inventory_fingerprint(data_inventory(data_dir)),
                    validation_sha256=sha256(data_dir / VALIDATION_SHARD),
                    upstream_sources=verify_upstream_sources(root, pinned["files"]))
    atomic_json(resources / "native_manifest.json", manifest)
    verify_resources(resources)

    config = replace(debug_config("dense_full"), vocab_size=tokenizer.get_vocab_size())
    auditor = Auditor(output / "audit-cache")
    proof = auditor.aggregate(config)
    anchor = auditor.aggregate(candidate_config("dense"))
    atomic_json(output / "dense_full-cost.json", proof)
    atomic_json(output / "dense.json", anchor)
    path = output / "dense_full.json"
    write_config(path, config, dict(status="synthetic_research_smoke", formula_version=proof["formula_version"],
                                    source_hash=proof["source_hash"], torch=proof["torch"],
                                    method_source_hash=proof["method_source"]["hash"],
                                    flops_per_token=proof["flops_per_token"],
                                    dense_flops_per_token=anchor["flops_per_token"],
                                    cost_record="dense_full-cost.json"))
    reviewed, _, findings = review_config(path)
    if reviewed != config:
        raise RuntimeError("reviewed fixture configuration differs from the generated model")
    for split in ("train", "val"):
        x, y = BestFitStream(data_dir, tokenizer, split, sequence_len=config.sequence_len).batch(1, torch.device("cpu"))
        if x.shape != (1, config.sequence_len) or x[:, 1:].tolist() != y[:, :-1].tolist():
            raise RuntimeError(f"fixture {split} stream did not produce shifted token pairs")
    print(json.dumps(dict(config=str(path), resources=str(resources), torch=torch.__version__,
                          vocab_size=config.vocab_size, research_findings=findings), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory, e.g. .agent-work/dense-smoke")
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
