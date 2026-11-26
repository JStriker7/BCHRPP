"""Verify locked CH-SIMS test caches without reading labels or refitting stats."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


SOURCE_SHA = "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"
TRAIN_AUDIT_SHA = "202ae819690d5f781ef6a9fa5288474b9f1b2d3e4abde20c9ea594c999c06ecb"
SHAPES = {"text": (64, 1024), "audio": (128, 1024), "vision": (8, 1024)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "raw_dir", "train_audit", "text_dir", "audio_dir",
                 "vision_dir", "text_train", "audio_train", "vision_train", "report"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if digest(args.source) != SOURCE_SHA or digest(args.train_audit) != TRAIN_AUDIT_SHA:
        raise ValueError("Locked source or train/valid audit hash changed")
    train_audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    raw_manifest = json.loads((args.raw_dir / "manifest.json").read_text(encoding="utf-8"))
    if (raw_manifest["dataset_sha256"] != SOURCE_SHA or
            raw_manifest["archive_sha256"] != train_audit["raw_archive_sha256_from_unpack_manifest"]):
        raise ValueError("Raw archive provenance mismatch")
    with args.source.open("rb") as stream:
        payload = pickle.load(stream)
    ids = payload["test"]["id"]
    if (len(ids) != 457 or len(set(ids)) != 457 or
            set(ids) & (set(payload["train"]["id"]) | set(payload["valid"]["id"]))):
        raise ValueError("Unexpected test ID count or overlap")
    id_hash = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()
    report = {"source_sha256": SOURCE_SHA, "train_audit_sha256": TRAIN_AUDIT_SHA,
              "split": "test", "count": len(ids), "id_sha256": id_hash,
              "test_labels_accessed": False, "normalization_refit": False,
              "cache_sha256": {}, "valid_steps": {}, "raw_clip_sha256": {}}
    for mod in SHAPES:
        folder = getattr(args, mod + "_dir")
        train = json.loads((getattr(args, mod + "_train") / "manifest.json").read_text())
        current = json.loads((folder / "manifest.json").read_text())
        if (current["dataset_sha256"] != SOURCE_SHA or current["splits"] != ["test"] or
                current["id_sha256"]["test"] != id_hash or
                current["model_sha256"] != train["model_sha256"] or
                current["length"] != train["length"] or
                current["hidden_size"] != train["hidden_size"]):
            raise ValueError(f"{mod}: test/train feature protocol differs")
        if mod != "text" and current["raw_archive_sha256"] != raw_manifest["archive_sha256"]:
            raise ValueError(f"{mod}: test media provenance differs")
        if mod == "text" and (current["vocab_sha256"] != train["vocab_sha256"] or
                              current["vocab_size"] != 21128):
            raise ValueError("Text tokenizer differs")
        values = np.load(folder / "test_features.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(folder / "test_mask.npy", mmap_mode="r", allow_pickle=False)
        if (values.shape != (457, *SHAPES[mod]) or values.dtype != np.float16 or
                mask.shape != values.shape[:2] or mask.dtype != np.bool_ or
                not mask.any(axis=1).all() or not np.isfinite(values).all() or
                np.any(values[~mask] != 0)):
            raise ValueError(f"{mod}: corrupt shape/mask/values/padding")
        report["valid_steps"][mod] = int(mask.sum())
        report["cache_sha256"][mod] = {}
        files = ("features.npy", "mask.npy", "token_ids.npy", "ids.json") if mod == "text" else ("features.npy", "mask.npy")
        for suffix in files:
            actual = digest(folder / f"test_{suffix}")
            report["cache_sha256"][mod][f"test_{suffix}"] = actual
            if mod == "text" and current["file_sha256"]["test"][suffix] != actual:
                raise ValueError(f"text:{suffix} checksum mismatch")
        if mod == "text":
            tokens = np.load(folder / "test_token_ids.npy", mmap_mode="r", allow_pickle=False)
            if tokens.shape != mask.shape or tokens.dtype != np.int32:
                raise ValueError("Invalid test token IDs")
            if json.loads((folder / "test_ids.json").read_text(encoding="utf-8")) != ids:
                raise ValueError("Text ID file order differs")
    for i, clip_id in enumerate(ids, 1):
        path = args.raw_dir / "Raw" / (clip_id.replace("$_$", "/") + ".mp4")
        if not path.is_file():
            raise FileNotFoundError(path)
        report["raw_clip_sha256"][clip_id] = digest(path)
        if i % 100 == 0:
            print(json.dumps({"clips_hashed": i}), flush=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"completed": str(args.report), "sha256": digest(args.report)}), flush=True)


if __name__ == "__main__":
    main()
