"""One-pass test evaluation of three validation-locked BCHR++ checkpoints."""

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch_musa  # noqa: F401
from torch.utils.data import DataLoader, TensorDataset

from metrics.metricsTop import MetricsTop
from online import OnlineBCHR
from sims_data import EXPECTED_SHA256, SHAPES, digest
from train_sims import source_hash


TRAIN_AUDIT_SHA = "202ae819690d5f781ef6a9fa5288474b9f1b2d3e4abde20c9ea594c999c06ecb"
SOURCE_CODE_SHA = "3546e3066d4856691161e3da9d7a59191dfce5b307baba4d1cae34d7cffbe2b1"
LOCKED_EPOCHS = {222: 3, 128: 3, 2026: 6}
CHECKPOINT_SHA256 = {
    222: "ca5f519e65e48bef12b02b3d38381c5ef132853dae259998842d782b647eeca4",
    128: "03a29c7d70ad3644510ad21e85ffd7b60ae61deddb22d3f004bd93a93c114fd0",
    2026: "fabb94295e79411c5c0b01a49cb09d50ee673dc42a9a701de0787463370a1413",
}
THRESHOLDS = {"MAE": 0.378, "Corr": 0.754, "Mult_acc_3": 0.7637,
              "Mult_acc_2": 0.8600, "F1_score": 0.8568}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "train_audit", "test_audit", "text_train", "audio_train",
                "vision_train", "text_test", "audio_test", "vision_test", "runs", "output_dir"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if torch.musa.device_count() != 1 or source_hash() != SOURCE_CODE_SHA:
        raise RuntimeError("Locked training code changed or GPU not uniquely selected")
    if (digest(args.source) != EXPECTED_SHA256 or
            digest(args.train_audit) != TRAIN_AUDIT_SHA):
        raise ValueError("Source or train-only normalization audit changed")
    train_audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    test_audit_hash = digest(args.test_audit)
    test_audit = json.loads(args.test_audit.read_text(encoding="utf-8"))
    if (test_audit["source_sha256"] != EXPECTED_SHA256 or
            test_audit["train_audit_sha256"] != TRAIN_AUDIT_SHA or
            test_audit["test_labels_accessed"] or test_audit["normalization_refit"] or
            test_audit["count"] != 457):
        raise ValueError("Invalid locked test extraction audit")
    with args.source.open("rb") as stream:
        raw = pickle.load(stream)
    ids = raw["test"]["id"]
    id_hash = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()
    if len(ids) != 457 or id_hash != test_audit["id_sha256"]:
        raise ValueError("Official test ID order differs")
    labels = np.asarray(raw["test"]["regression_labels"], dtype=np.float32).reshape(457, 1)
    if not np.isfinite(labels).all() or np.abs(labels).max() > 1.00001:
        raise ValueError("Invalid test labels")
    columns = []
    masks = []
    cache_hashes = {}
    for mod in ("text", "audio", "vision"):
        train_dir = getattr(args, mod + "_train")
        test_dir = getattr(args, mod + "_test")
        train_manifest = json.loads((train_dir / "manifest.json").read_text())
        manifest = json.loads((test_dir / "manifest.json").read_text())
        if (manifest["id_sha256"]["test"] != id_hash or
                manifest["model_sha256"] != train_manifest["model_sha256"] or
                manifest["length"] != SHAPES[mod][0] or
                manifest["hidden_size"] != SHAPES[mod][1]):
            raise ValueError(f"{mod}: test/train protocol mismatch")
        for suffix in ("features.npy", "mask.npy"):
            filename = f"test_{suffix}"
            if digest(test_dir / filename) != test_audit["cache_sha256"][mod][filename]:
                raise ValueError(f"{mod}: checksum mismatch")
        values = np.load(test_dir / "test_features.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(test_dir / "test_mask.npy", mmap_mode="r", allow_pickle=False)
        if (values.shape != (457, *SHAPES[mod]) or values.dtype != np.float16 or
                mask.shape != values.shape[:2] or mask.dtype != np.bool_):
            raise ValueError(f"{mod}: invalid test cache")
        values = values.astype(np.float32)
        if mod != "text":
            stats = train_audit["stats_fit_on_train_only"][mod]
            values = (values - np.asarray(stats["mean"], dtype=np.float32)) / np.asarray(
                stats["std"], dtype=np.float32)
        values[~mask] = 0
        if not np.isfinite(values).all():
            raise ValueError(f"{mod}: nonfinite normalized test features")
        columns.append(torch.from_numpy(np.ascontiguousarray(values)))
        masks.append(torch.from_numpy(np.array(mask, copy=True)))
        cache_hashes[mod] = digest(test_dir / "manifest.json")
    dataset = TensorDataset(*columns, *masks, torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    metric = MetricsTop("regression").getMetics("SIMS")
    output = {"split": "test", "source_sha256": EXPECTED_SHA256,
              "training_source_sha256": SOURCE_CODE_SHA, "train_audit_sha256": TRAIN_AUDIT_SHA,
              "test_audit_sha256": test_audit_hash, "test_cache_manifest_sha256": cache_hashes,
              "checkpoint_selection": "validation MAE; no test feedback",
              "feature_protocol": "re-encoded raw RoBERTa/HuBERT/CLIP; differs from CICA official features",
              "per_seed": {}, "thresholds": THRESHOLDS}
    for seed, epoch in LOCKED_EPOCHS.items():
        run = args.runs / f"sims_raw_reencode_screen_{seed}_20261103_v2"
        ckpt_path = run / "best.pt"
        if digest(ckpt_path) != CHECKPOINT_SHA256[seed]:
            raise ValueError(f"Seed {seed}: locked checkpoint hash mismatch")
        result = json.loads((run / "result.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        if (result["mode"] != "screen" or result["best_epoch"] != epoch or
                result["source_sha256"] != SOURCE_CODE_SHA or
                manifest["source_sha256"] != SOURCE_CODE_SHA or
                manifest["audit_sha256"] != TRAIN_AUDIT_SHA or
                manifest["test_accessed"]):
            raise ValueError(f"Seed {seed}: validation lock mismatch")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (checkpoint["epoch"] != epoch or checkpoint["manifest"] != manifest or
                checkpoint["validation"] != result["validation"]):
            raise ValueError(f"Seed {seed}: checkpoint/result mismatch")
        model = OnlineBCHR(manifest["model_config"]).float().to("musa:0")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for text, audio, vision, tm, am, vm, _ in loader:
                pred = model(text.to("musa:0"), audio.to("musa:0"), vision.to("musa:0"),
                             masks={"text": tm.to("musa:0"), "audio": am.to("musa:0"),
                                    "vision": vm.to("musa:0")})
                predictions.append(pred.float().cpu())
        metrics = metric(torch.cat(predictions), torch.from_numpy(labels))
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError(f"Seed {seed}: invalid metrics")
        output["per_seed"][str(seed)] = {"checkpoint_sha256": CHECKPOINT_SHA256[seed],
                                        "best_epoch": epoch, "validation": result["validation"],
                                        "test": metrics}
        print(json.dumps({"seed": seed, "test": metrics}), flush=True)
        del model, checkpoint
        torch.musa.empty_cache()
    keys = next(iter(output["per_seed"].values()))["test"]
    output["test_mean"] = {key: round(sum(row["test"][key] for row in output["per_seed"].values()) / 3, 6)
                           for key in keys}
    output["strict_thresholds_met"] = {key: (output["test_mean"][key] < value if key == "MAE"
                                            else output["test_mean"][key] > value)
                                       for key, value in THRESHOLDS.items()}
    output["all_five_met"] = all(output["strict_thresholds_met"].values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"completed": str(args.output_dir / 'result.json'),
                      "test_mean": output["test_mean"], "all_five_met": output["all_five_met"]}), flush=True)


if __name__ == "__main__":
    main()
