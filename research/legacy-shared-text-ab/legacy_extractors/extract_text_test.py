"""Encode locked CH-SIMS test transcripts with the pinned frozen encoder.

The RoBERTa weight directory has no usable vocabulary. Use the pinned
bert-base-chinese vocab (same 21,128 ID space) instead of a 5-token fallback.
Only test raw_text is processed; official precomputed embeddings and labels
are never read. Run only after train/valid checkpoint selection is locked.
"""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import torch_musa  # noqa: F401
from transformers import BertModel, BertTokenizer

DATA_SHA256 = "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"
MODEL_SHA256 = "e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8"
VOCAB_SHA256 = "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c"
COUNTS = {"test": 457}
MAX_LENGTH = 64


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if torch.musa.device_count() != 1:
        raise RuntimeError("Select exactly one physical GPU at launch")
    if digest(args.source) != DATA_SHA256 or digest(args.model / "pytorch_model.bin") != MODEL_SHA256:
        raise ValueError("Unexpected source data or encoder weights")
    vocab_sha = digest(args.vocab)
    if vocab_sha != VOCAB_SHA256:
        raise ValueError("Unexpected Chinese tokenizer vocabulary hash")
    # Transformers on this host ignores the direct vocab_file constructor;
    # from_pretrained loads the actual 21,128-entry local vocabulary.
    tokenizer = BertTokenizer.from_pretrained(str(args.vocab.parent), local_files_only=True)
    if tokenizer.vocab_size != 21128 or tokenizer.cls_token_id != 101 or tokenizer.sep_token_id != 102:
        raise ValueError("Incompatible Chinese tokenizer vocabulary")
    with args.source.open("rb") as stream:
        payload = pickle.load(stream)
    for split, n in COUNTS.items():
        source = payload[split]
        if len(source["id"]) != n or len(source["raw_text"]) != n or len(set(source["id"])) != n:
            raise ValueError(f"{split}: unexpected ID/text count")
        lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in source["raw_text"]]
        if min(lengths) < 2 or max(lengths) > MAX_LENGTH:
            raise ValueError(f"{split}: transcript would be truncated: {min(lengths)}..{max(lengths)}")
    test_ids = set(payload["test"]["id"])
    if test_ids & (set(payload["train"]["id"]) | set(payload["valid"]["id"])):
        raise ValueError("Test clip IDs overlap train/valid")
    config = json.loads((args.model / "config.json").read_text())
    hidden = int(config["hidden_size"])
    if int(config["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    expected = sum(n * MAX_LENGTH * hidden * 2 for n in COUNTS.values())
    if shutil.disk_usage(args.output_dir.parent).free < expected + 2 * 1024**3:
        raise OSError("Insufficient disk for frozen text embeddings plus 2 GiB reserve")
    model = BertModel.from_pretrained(str(args.model), local_files_only=True,
                                      add_pooling_layer=False, attn_implementation="eager").eval().to("musa:0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"encoder": "chinese-roberta-wwm-ext-large-frozen",
                "source": "official raw_text (not official text embeddings)",
                "dataset_sha256": DATA_SHA256, "model_sha256": MODEL_SHA256,
                "vocab_sha256": vocab_sha, "vocab_size": tokenizer.vocab_size,
                "length": MAX_LENGTH, "hidden_size": hidden,
                "splits": ["test"], "test_exported": True,
                "id_sha256": {}, "file_sha256": {}, "max_actual_tokens": {}}
    for split, n in COUNTS.items():
        source = payload[split]
        texts, ids = source["raw_text"], source["id"]
        manifest["id_sha256"][split] = hashlib.sha256(
            json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest()
        manifest["max_actual_tokens"][split] = max(
            len(tokenizer.encode(text, add_special_tokens=True)) for text in texts)
        features = np.lib.format.open_memmap(args.output_dir / f"{split}_features.npy",
                                             mode="w+", dtype=np.float16,
                                             shape=(n, MAX_LENGTH, hidden))
        masks = np.lib.format.open_memmap(args.output_dir / f"{split}_mask.npy",
                                          mode="w+", dtype=np.bool_, shape=(n, MAX_LENGTH))
        token_ids = np.lib.format.open_memmap(args.output_dir / f"{split}_token_ids.npy",
                                              mode="w+", dtype=np.int32, shape=(n, MAX_LENGTH))
        for i in range(0, n, 8):
            batch = tokenizer(texts[i:i + 8], padding="max_length", max_length=MAX_LENGTH,
                              truncation=False, return_tensors="pt")
            with torch.inference_mode():
                encoded = model(**{key: value.to("musa:0") for key, value in batch.items()}).last_hidden_state
            array = encoded.float().cpu().numpy().astype(np.float16)
            mask = batch["attention_mask"].numpy().astype(np.bool_)
            array[~mask] = 0
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite text encoding: {split}:{i}")
            last = i + len(mask)
            features[i:last] = array
            masks[i:last] = mask
            token_ids[i:last] = batch["input_ids"].numpy().astype(np.int32)
            if last % 100 < 8 or last == n:
                print(json.dumps({"split": split, "encoded": last, "total": n}), flush=True)
        features.flush(); masks.flush(); token_ids.flush()
        with (args.output_dir / f"{split}_ids.json").open("x", encoding="utf-8") as stream:
            json.dump(ids, stream, ensure_ascii=False)
        manifest["file_sha256"][split] = {}
        for suffix in ("features.npy", "mask.npy", "token_ids.npy", "ids.json"):
            file = args.output_dir / f"{split}_{suffix}"
            manifest["file_sha256"][split][suffix] = digest(file)
        stored = np.load(args.output_dir / f"{split}_features.npy", mmap_mode="r")
        mask = np.load(args.output_dir / f"{split}_mask.npy", mmap_mode="r")
        if not np.isfinite(stored).all() or np.any(stored[~mask]):
            raise ValueError(f"Invalid stored text features: {split}")
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"completed": manifest}), flush=True)


if __name__ == "__main__":
    main()
