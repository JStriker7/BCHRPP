"""Load raw re-encoded CH-SIMS train/valid caches, never official embeddings."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SHA256 = "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"
COUNTS = {"train": 1368, "valid": 456}
SHAPES = {"text": (64, 1024), "audio": (128, 1024), "vision": (8, 1024)}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class SIMSSplit(Dataset):
    def __init__(self, source, name):
        n = COUNTS[name]
        ids = source["id"]
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"{name}: invalid sample IDs/count")
        labels = np.asarray(source["regression_labels"], dtype=np.float32).reshape(n, 1)
        unimodal = np.stack([np.asarray(source["regression_labels_" + m], dtype=np.float32)
                             for m in "TAV"], axis=1).reshape(n, 3, 1)
        if not np.isfinite(labels).all() or not np.isfinite(unimodal).all():
            raise ValueError(f"{name}: nonfinite labels")
        if max(np.abs(labels).max(), np.abs(unimodal).max()) > 1.00001:
            raise ValueError(f"{name}: labels outside [-1,1]")
        self.id_sha256 = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels))
        self.unimodal = torch.from_numpy(np.ascontiguousarray(unimodal))
        self.summary = {"name": name, "samples": n, "dims": (1024, 1024, 1024),
                        "lengths": (64, 128, 8), "zero_labels": int((labels == 0).sum())}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.text[index], self.audio[index], self.vision[index],
                *(mask[index] for mask in self.masks), self.labels[index], self.unimodal[index])


def load_splits(path):
    sha = digest(path)
    if sha != EXPECTED_SHA256:
        raise ValueError("Unexpected CH-SIMS source hash; refusing to unpickle")
    with path.open("rb") as stream:
        source = pickle.load(stream)
    splits = {name: SIMSSplit(source[name], name) for name in COUNTS}
    if set(source["train"]["id"]) & set(source["valid"]["id"]):
        raise ValueError("Train/valid clip IDs overlap")
    return splits, sha


def attach_reencoded_caches(splits, dataset_sha, cache_dirs, audit_path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["source_sha256"] != dataset_sha or
            audit["splits_audited"] != ["train", "valid"] or audit["test_accessed"]):
        raise ValueError("Raw re-encoding audit source/split mismatch")
    manifests = {}
    for index, name in enumerate(("text", "audio", "vision")):
        folder = cache_dirs[name]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["dataset_sha256"] != dataset_sha or
                manifest["splits"] != ["train", "valid"] or manifest.get("test_exported", False)):
            raise ValueError(f"{name}: unexpected cache provenance")
        if name != "text" and manifest["raw_archive_sha256"] != audit["raw_archive_sha256_from_unpack_manifest"]:
            raise ValueError(f"{name}: media archive provenance mismatch")
        if manifest["length"] != SHAPES[name][0] or manifest["hidden_size"] != SHAPES[name][1]:
            raise ValueError(f"{name}: feature dimensions mismatch")
        for split_name, split in splits.items():
            if manifest["id_sha256"][split_name] != split.id_sha256:
                raise ValueError(f"{name}/{split_name}: ID order mismatch")
            for suffix in ("features.npy", "mask.npy"):
                filename = f"{split_name}_{suffix}"
                if digest(folder / filename) != audit["checksums"][name][filename]:
                    raise ValueError(f"{name}/{split_name}: cache hash mismatch")
            values = np.load(folder / f"{split_name}_features.npy", mmap_mode="r", allow_pickle=False)
            mask = np.load(folder / f"{split_name}_mask.npy", mmap_mode="r", allow_pickle=False)
            if (values.shape != (len(split), *SHAPES[name]) or values.dtype != np.float16 or
                    mask.shape != values.shape[:2] or mask.dtype != np.bool_ or
                    not mask.any(axis=1).all()):
                raise ValueError(f"{name}/{split_name}: invalid shape/mask")
            converted = values.astype(np.float32)
            if name != "text":
                stats = audit["stats_fit_on_train_only"][name]
                converted = (converted - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(
                    stats["std"], dtype=np.float32)
            converted[~mask] = 0.0
            if not np.isfinite(converted).all():
                raise ValueError(f"{name}/{split_name}: nonfinite feature")
            setattr(split, name, torch.from_numpy(np.ascontiguousarray(converted)))
            if index == 0:
                split.masks = [None, None, None]
            split.masks[index] = torch.from_numpy(np.array(mask, copy=True))
        manifests[name] = manifest
    for split in splits.values():
        split.masks = tuple(split.masks)
    return manifests
