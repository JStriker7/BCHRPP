"""Audit fresh CH-SIMS raw-input caches and fit train-only normalization.

No model selection, training, or test access. The archive itself is not
present; the Raw.zip hash in the verified unpack manifest is provenance,
not a claim of re-hashing the missing archive.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

DATA_SHA256 = "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"
ARCHIVE_SHA256 = "c1745724802dae37709b2d9babe5c6a1cef4f806a4bca3134a87e2fab2ee211a"
COUNTS = {"train": 1368, "valid": 456}
GRID = {"text": (64, 1024), "audio": (128, 1024), "vision": (8, 1024)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "raw_dir", "text_dir", "audio_dir", "vision_dir", "report"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if digest(args.source) != DATA_SHA256:
        raise ValueError("Source hash mismatch")
    with args.source.open("rb") as f:
        payload = pickle.load(f)
    unpack = json.loads((args.raw_dir / "manifest.json").read_text())
    if unpack["archive_sha256"] != ARCHIVE_SHA256 or unpack["dataset_sha256"] != DATA_SHA256:
        raise ValueError("Unpack provenance mismatch")
    dirs = {"text": args.text_dir, "audio": args.audio_dir, "vision": args.vision_dir}
    report = {"source_sha256": DATA_SHA256, "raw_archive_sha256_from_unpack_manifest": ARCHIVE_SHA256,
              "splits_audited": ["train", "valid"], "test_accessed": False,
              "checksums": {}, "steps": {}, "stats_fit_on_train_only": {},
              "raw_clip_sha256": {}, "source_video_group_overlap_count": 0}
    groups = {s: {x.split("$_$")[0] for x in payload[s]["id"]} for s in COUNTS}
    report["source_video_group_overlap_count"] = len(groups["train"] & groups["valid"])
    for mod, folder in dirs.items():
        manifest = json.loads((folder / "manifest.json").read_text())
        if (manifest["dataset_sha256"] != DATA_SHA256 or
                manifest["splits"] != ["train", "valid"] or
                manifest.get("test_exported", False) or
                (mod != "text" and (manifest["encoder"] != mod or
                 manifest["raw_archive_sha256"] != ARCHIVE_SHA256))):
            raise ValueError(f"{mod}: cache provenance or split mismatch")
        report["checksums"][mod], report["steps"][mod] = {}, {}
        for split, n in COUNTS.items():
            ids = payload[split]["id"]
            expected_ids = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest()
            if manifest["id_sha256"][split] != expected_ids or len(ids) != n:
                raise ValueError(f"{mod}:{split}: ID order mismatch")
            values = np.load(folder / f"{split}_features.npy", mmap_mode="r", allow_pickle=False)
            mask = np.load(folder / f"{split}_mask.npy", mmap_mode="r", allow_pickle=False)
            if values.shape != (n, *GRID[mod]) or values.dtype != np.float16:
                raise ValueError(f"{mod}:{split}: unexpected feature shape/dtype")
            if mask.shape != values.shape[:2] or mask.dtype != np.bool_ or not mask.any(axis=1).all():
                raise ValueError(f"{mod}:{split}: invalid mask")
            if not np.isfinite(values).all() or np.any(values[~mask] != 0):
                raise ValueError(f"{mod}:{split}: nonfinite features or nonzero padding")
            report["steps"][mod][split] = {"valid": int(mask.sum()),
                                            "padded": int((~mask).sum())}
            suffixes = ["features.npy", "mask.npy"]
            if mod == "text":
                token_ids = np.load(folder / f"{split}_token_ids.npy", mmap_mode="r", allow_pickle=False)
                if token_ids.shape != mask.shape or token_ids.dtype != np.int32:
                    raise ValueError(f"{mod}:{split}: token ID mismatch")
                suffixes += ["token_ids.npy", "ids.json"]
            for suffix in suffixes:
                name = f"{split}_{suffix}"
                h = digest(folder / name)
                report["checksums"][mod][name] = h
                if mod == "text" and manifest["file_sha256"][split][suffix] != h:
                    raise ValueError(f"{mod}:{split}: checksum mismatch")
        if mod != "text":
            train = np.load(folder / "train_features.npy", mmap_mode="r", allow_pickle=False)
            mask = np.load(folder / "train_mask.npy", mmap_mode="r", allow_pickle=False)
            valid_frames = train[mask].astype(np.float32)
            mean = valid_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = valid_frames.std(axis=0, dtype=np.float64).astype(np.float32)
            report["stats_fit_on_train_only"][mod] = {
                "mean": mean.tolist(), "std": np.where(std > 1e-3, std, 1.0).tolist(),
                "zero_variance_channels": int((std <= 1e-3).sum()),
                "train_frames": len(valid_frames)}
    for split in COUNTS:
        report["raw_clip_sha256"][split] = {}
        for clip_id in payload[split]["id"]:
            path = args.raw_dir / "Raw" / (clip_id.replace("$_$", "/") + ".mp4")
            if not path.is_file():
                raise FileNotFoundError(path)
            report["raw_clip_sha256"][split][clip_id] = digest(path)
        print(json.dumps({"raw_clips_hashed": split,
                          "count": len(report["raw_clip_sha256"][split])}), flush=True)
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"completed": str(args.report), "sha256": digest(args.report)}), flush=True)


if __name__ == "__main__":
    main()
