"""Fixed MOSI follow-up grids with input provenance and safe per-seed reuse."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_SPLITS = {"train": 1284, "valid": 229, "test": 686}
EXPECTED_FINGERPRINT = "6296d258a3255962bce41154e11bc9e41498e6c2a8e201b725c678a9f432c84e"
METRIC_PROTOCOL = "metricsTop.py:regression/MOSI"
METRIC_KEYS = ("MAE", "Corr", "Mult_acc_7", "Mult_acc_5", "Has0_acc_2",
               "Has0_F1_score", "Non0_acc_2", "Non0_F1_score")
SEEDS = (222, 128, 2026)
SETTINGS = {"epochs": 60, "patience": 15, "batch_size": 24, "hidden_dim": 128,
            "queries": 4, "topk": 4, "fusion_lr": 3e-4, "amp": True}
SOURCE_FILES = (
    "scripts/mosi_followup.py", "scripts/train_mosi_extract.py",
    "scripts/check_mosi_text_parity.py", "datasets/mosi_extract.py",
    "datasets/mosi_text_online.py", "bchrpp/mosi_online_text.py",
    "bchrpp/full_model.py", "losses/bchrpp_loss.py", "metrics/metricsTop.py",
)


@dataclass(frozen=True)
class Variant:
    name: str
    text_mode: str
    text_lr: float
    modalities: str = "full"
    tokenizer_prefix_space: bool = False


TRIALS = {
    "modality": (
        Variant("full", "unfrozen", 2e-5),
        Variant("no_audio", "unfrozen", 2e-5, "no_audio"),
        Variant("no_vision", "unfrozen", 2e-5, "no_vision"),
        Variant("text_only", "unfrozen", 2e-5, "text_only"),
    ),
    "text_lr": (
        Variant("lr_1e-5", "unfrozen", 1e-5),
        Variant("lr_2e-5", "unfrozen", 2e-5),
        Variant("lr_3e-5", "unfrozen", 3e-5),
    ),
    "roberta": (
        Variant("frozen", "frozen", 2e-5, tokenizer_prefix_space=True),
        Variant("unfrozen", "unfrozen", 2e-5, tokenizer_prefix_space=True),
    ),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", required=True, choices=tuple(TRIALS))
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--expected-text-model", required=True, type=Path)
    p.add_argument("--reference-features", type=Path,
                   help="Required for RoBERTa: verify CLAP/CLIP against the original .pt")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    return p


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_label_fingerprint(splits):
    digest = hashlib.sha256()
    for split in EXPECTED_SPLITS:
        for sample in sorted(splits[split].samples, key=lambda item: item["id"]):
            digest.update(
                f"{split}\t{sample['id']}\t{float(sample['label']):.9g}\n".encode()
            )
    return digest.hexdigest()


def check_gpu():
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 for physical GPU 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Physical GPU 0 is not available to PyTorch")
    torch.empty(1, device="cuda:0")


def check_original_modalities(candidate, reference):
    import numpy as np

    for split in EXPECTED_SPLITS:
        originals = {sample["id"]: sample for sample in reference[split].samples}
        if len(originals) != len(candidate[split].samples):
            raise ValueError(f"{split}: original IDs do not match the RoBERTa features")
        for sample in candidate[split].samples:
            previous = originals.get(sample["id"])
            if previous is None or sample["words"] != previous["words"] or \
                    float(sample["label"]) != float(previous["label"]):
                raise ValueError(f"{sample['id']}: text alignment or label has changed")
            for key in ("audio", "visual", "audio_valid", "visual_valid"):
                if not np.array_equal(np.asarray(sample[key]), np.asarray(previous[key])):
                    raise ValueError(f"{sample['id']}: original {key} was changed")


def validate_inputs(args):
    import numpy as np
    from datasets.mosi_extract import load_mosi_extract

    check_gpu()
    if not args.expected_text_model.is_dir():
        raise FileNotFoundError(f"Text model is missing: {args.expected_text_model}")
    splits, meta = load_mosi_extract(args.features)
    if meta["splits"] != EXPECTED_SPLITS:
        raise ValueError(f"MOSI split sizes differ: {meta['splits']}")
    fingerprint = sample_label_fingerprint(splits)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"MOSI sample/label fingerprint differs: {fingerprint}")
    text_model = meta["feature_models"]["text"]
    if Path(text_model).resolve() != args.expected_text_model.resolve():
        raise ValueError(f"Feature text model differs from expected: {text_model}")
    for split, dataset in splits.items():
        for sample in dataset.samples:
            if sample.get("alignment") != "uniform":
                raise ValueError(f"{sample['id']}: expected uniform word alignment")
            words = sample["words"]
            if any(np.asarray(sample[key]).shape[0] != len(words)
                   for key in ("text", "audio", "visual")):
                raise ValueError(f"{sample['id']}: modality sequence is not word-aligned")
    if args.trial == "roberta":
        if args.reference_features is None:
            raise ValueError("RoBERTa trial requires --reference-features")
        reference, reference_meta = load_mosi_extract(args.reference_features)
        if reference_meta["splits"] != EXPECTED_SPLITS or \
                sample_label_fingerprint(reference) != fingerprint:
            raise ValueError("Original features have different MOSI samples or labels")
        check_original_modalities(splits, reference)
    elif args.reference_features is not None:
        raise ValueError("--reference-features is only used for the RoBERTa trial")
    return {
        "features": str(args.features.resolve()),
        "features_sha256": sha256_file(args.features),
        "reference_features_sha256": (sha256_file(args.reference_features)
                                      if args.reference_features else None),
        "text_model": text_model,
        "feature_models": meta["feature_models"],
        "splits": meta["splits"],
        "sample_label_fingerprint": fingerprint,
    }


def make_manifest(args, inputs):
    return {
        "schema_version": 1,
        "trial": args.trial,
        **inputs,
        "variants": [asdict(variant) for variant in TRIALS[args.trial]],
        "seeds": list(SEEDS),
        "settings": SETTINGS,
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
    }


def variant_config(manifest, variant):
    raw = {"manifest": manifest, "variant": asdict(variant)}
    signature = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return {
        "signature": signature,
        "features_sha256": manifest["features_sha256"],
        "text_model": manifest["text_model"],
        "text_mode": variant.text_mode,
        "text_lr": variant.text_lr,
        "modalities": variant.modalities,
        "tokenizer_prefix_space": variant.tokenizer_prefix_space,
    }


def ensure_manifest(output_root, manifest):
    path = output_root / "manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf8")) != manifest:
            raise ValueError(f"Existing experiment manifest differs: {path}")
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Nonempty result directory has no manifest: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_completed_run(result_path, checkpoint, manifest, variant, seed):
    import torch

    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf8"))
    expected_config = variant_config(manifest, variant)
    allowed = (checkpoint, checkpoint.with_name(f"{checkpoint.stem}_recovered.pt"))
    saved = Path(result.get("checkpoint", ""))
    if saved not in allowed or not saved.is_file():
        raise ValueError(f"Missing or unexpected checkpoint for {result_path}")
    if (result.get("dataset") != "MOSI" or result.get("seed") != seed or
            result.get("metric_protocol") != METRIC_PROTOCOL or
            result.get("features") != manifest["features"] or
            result.get("trial_config") != expected_config):
        raise ValueError(f"Result configuration differs from this experiment: {result_path}")
    ckpt = torch.load(saved, map_location="cpu", weights_only=True)
    if (ckpt.get("version") != "bchrpp_rcse_lite_online_text_v1" or
            ckpt.get("dataset") != "MOSI" or ckpt.get("seed") != seed or
            ckpt.get("features") != manifest["features"] or
            ckpt.get("metric_protocol") != METRIC_PROTOCOL or
            ckpt.get("trial_config") != expected_config or
            not isinstance(ckpt.get("model_state"), dict) or not ckpt["model_state"]):
        raise ValueError(f"Checkpoint configuration differs from this experiment: {saved}")
    model_config = ckpt.get("model_config", {})
    if (model_config.get("text_mode") != variant.text_mode or
            model_config.get("text_model") != manifest["text_model"] or
            model_config.get("text_lr") != variant.text_lr or
            model_config.get("modalities") != variant.modalities or
            model_config.get("tokenizer_prefix_space") != variant.tokenizer_prefix_space):
        raise ValueError(f"Checkpoint model differs from this experiment: {saved}")
    if result.get("best_epoch") != ckpt.get("epoch") or not isinstance(result.get("last_epoch"), int):
        raise ValueError(f"Checkpoint epoch differs from the result: {result_path}")
    if not math.isfinite(float(ckpt.get("selection_val_mae", float("nan")))):
        raise ValueError(f"Nonfinite selection MAE: {saved}")
    for split in ("validation", "test"):
        metrics = result.get(split)
        if not isinstance(metrics, dict) or any(
                key not in metrics or not math.isfinite(float(metrics[key]))
                for key in METRIC_KEYS):
            raise ValueError(f"Missing or nonfinite {split} metrics: {result_path}")
    return result


def fields_for_runs():
    return ["trial", "variant", "seed", "best_epoch", "last_epoch", "text_mode",
            "text_lr", "modalities", "tokenizer_prefix_space"] + [
        f"{split}_{key}" for split in ("val", "test") for key in METRIC_KEYS
    ]


def row_for_run(trial, variant, result):
    row = {"trial": trial, "variant": variant.name, "seed": result["seed"],
           "best_epoch": result["best_epoch"], "last_epoch": result["last_epoch"],
           "text_mode": variant.text_mode, "text_lr": variant.text_lr,
           "modalities": variant.modalities,
           "tokenizer_prefix_space": variant.tokenizer_prefix_space}
    for prefix, split in (("val", "validation"), ("test", "test")):
        for key in METRIC_KEYS:
            row[f"{prefix}_{key}"] = result[split][key]
    return row


def summary_for_variant(trial, variant, rows):
    if len(rows) != len(SEEDS) or {row["seed"] for row in rows} != set(SEEDS):
        raise ValueError(f"Incomplete seed set for {variant.name}")
    summary = {"trial": trial, "variant": variant.name, "n_seeds": len(rows),
               "seeds": ",".join(str(seed) for seed in SEEDS),
               "text_mode": variant.text_mode, "text_lr": variant.text_lr,
               "modalities": variant.modalities,
               "tokenizer_prefix_space": variant.tokenizer_prefix_space}
    for prefix in ("val", "test"):
        for key in METRIC_KEYS:
            values = [float(row[f"{prefix}_{key}"]) for row in rows]
            summary[f"{prefix}_{key}_mean"] = statistics.mean(values)
            summary[f"{prefix}_{key}_std"] = statistics.stdev(values)
    return summary


def summary_fields():
    fields = ["trial", "variant", "n_seeds", "seeds", "text_mode", "text_lr",
              "modalities", "tokenizer_prefix_space"]
    for prefix in ("val", "test"):
        for key in METRIC_KEYS:
            fields.extend((f"{prefix}_{key}_mean", f"{prefix}_{key}_std"))
    return fields


def write_rows(path, fields, rows, immutable=False):
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue()
    if path.exists():
        existing = path.read_text(encoding="utf8")
        if existing == content:
            return
        if immutable:
            raise ValueError(f"Existing complete CSV differs: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_command(manifest, variant, seed, checkpoint):
    config = variant_config(manifest, variant)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_mosi_extract.py"),
           "--features", manifest["features"], "--checkpoint", str(checkpoint),
           "--expected-text-model", manifest["text_model"],
           "--trial-signature", config["signature"],
           "--features-sha256", config["features_sha256"],
           "--text-mode", variant.text_mode, "--text-lr", str(variant.text_lr),
           "--modalities", variant.modalities, "--seed", str(seed),
           "--epochs", str(SETTINGS["epochs"]), "--patience", str(SETTINGS["patience"]),
           "--batch-size", str(SETTINGS["batch_size"]),
           "--hidden-dim", str(SETTINGS["hidden_dim"]),
           "--queries", str(SETTINGS["queries"]), "--topk", str(SETTINGS["topk"]),
           "--lr", str(SETTINGS["fusion_lr"])]
    if variant.tokenizer_prefix_space:
        cmd.append("--tokenizer-prefix-space")
    if SETTINGS["amp"]:
        cmd.append("--amp")
    return cmd


def run_trial(args, manifest):
    ensure_manifest(args.output_root, manifest)
    summaries = []
    for variant in TRIALS[args.trial]:
        directory = args.output_root / variant.name
        directory.mkdir(exist_ok=True)
        rows = []
        for seed in SEEDS:
            checkpoint = directory / f"mosi_lite_seed{seed}.pt"
            result_path = checkpoint.with_suffix(".json")
            if result_path.exists():
                print(f"[reuse] {variant.name} seed={seed}", flush=True)
            else:
                if checkpoint.exists() or checkpoint.with_name(
                        f"{checkpoint.stem}_recovered.pt").exists():
                    raise FileExistsError(
                        f"Incomplete checkpoint; recover it manually or use a new output root: {checkpoint}"
                    )
                print(f"[train] {variant.name} seed={seed}", flush=True)
                subprocess.run(train_command(manifest, variant, seed, checkpoint),
                               cwd=ROOT, check=True)
            result = validate_completed_run(result_path, checkpoint, manifest, variant, seed)
            rows.append(row_for_run(args.trial, variant, result))
            write_rows(directory / "results.csv", fields_for_runs(), rows)
        summary = summary_for_variant(args.trial, variant, rows)
        write_rows(directory / "summary.csv", summary_fields(), [summary], immutable=True)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    write_rows(args.output_root / "comparison.csv", summary_fields(), summaries, immutable=True)
    print(f"[done] {args.output_root / 'comparison.csv'}", flush=True)


def main():
    args = parser().parse_args()
    inputs = validate_inputs(args)
    manifest = make_manifest(args, inputs)
    print(json.dumps({"trial": args.trial, "splits": inputs["splits"],
                      "fingerprint": inputs["sample_label_fingerprint"],
                      "features_sha256": inputs["features_sha256"],
                      "model": inputs["text_model"], "preflight": args.preflight_only},
                     ensure_ascii=False), flush=True)
    if not args.preflight_only:
        run_trial(args, manifest)


if __name__ == "__main__":
    main()
