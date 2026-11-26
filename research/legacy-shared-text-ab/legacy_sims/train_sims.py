"""Train BCHR++ only on freshly re-encoded CH-SIMS raw-input features.

Only train/valid caches and labels are loaded. No test feature is extracted,
read or scored; choose the validation-MAE checkpoint in a fresh directory.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_musa  # noqa: F401 - register the MUSA device
from torch.utils.data import DataLoader

from losses.bchrpp_loss import compute_loss
from metrics.metricsTop import MetricsTop
from online import OnlineBCHR
from sims_data import attach_reencoded_caches, load_splits


SEEDS = (222, 128, 2026)
METRIC = MetricsTop("regression").getMetics("SIMS")
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--audio-cache", type=Path, required=True)
    p.add_argument("--vision-cache", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--mode", choices=("smoke", "screen"), required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--sign-weight", type=float, default=0.0)
    p.add_argument("--sign-temperature", type=float, default=0.25)
    p.add_argument("--dry-steps", type=int, default=10)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    paths = ("train_sims.py", "sims_data.py", "online.py", "bchrpp/full_model.py",
             "losses/bchrpp_loss.py", "metrics/metricsTop.py")
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode("utf-8"))
        digest.update((Path(__file__).parent / relative).read_bytes())
    return digest.hexdigest()


def loader(split, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(split, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False, generator=generator)


def to_device(batch, device):
    tokens, audio, vision, tm, am, vm, label, unimodal = batch
    return (tokens.to(device), audio.to(device), vision.to(device),
            {"text": tm.to(device), "audio": am.to(device), "vision": vm.to(device)},
            label.to(device), unimodal.to(device))


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    predictions, labels = [], []
    for raw in data:
        text, audio, vision, masks, label, _ = to_device(raw, device)
        pred = model(text, audio, vision, masks=masks)
        predictions.append(pred.float().cpu())
        labels.append(label.float().cpu())
    pred = torch.cat(predictions)
    true = torch.cat(labels)
    metrics = METRIC(pred, true)
    if not all(math.isfinite(float(x)) for x in metrics.values()):
        raise RuntimeError("Nonfinite CH-SIMS evaluation")
    return metrics, float(metrics["MAE"])


def train_epoch(model, data, optimizer, device, args, *, limit=None, pretrain=False):
    model.train()
    weighted_loss, count = 0.0, 0
    for number, raw in enumerate(data, 1):
        text, audio, vision, masks, label, unimodal = to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, aux = model(text, audio, vision, masks=masks, return_aux=True)
        if pretrain:
            present = aux["modality_available"].to(unimodal).unsqueeze(-1)
            per_modality = F.smooth_l1_loss(
                aux["unimodal_predictions"], unimodal, beta=.5, reduction="none")
            loss = (per_modality * present).sum() / present.sum().clamp_min(1.)
        else:
            loss, _ = compute_loss(prediction, label, aux, modality_targets=unimodal)
        if args.sign_weight and not pretrain:
            sign = torch.where(label.reshape(-1) > 0, 1.0, -1.0)
            loss = loss + args.sign_weight * F.softplus(
                -sign * prediction.reshape(-1) / args.sign_temperature).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at batch {number}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        size = label.shape[0]
        weighted_loss += float(loss.detach()) * size
        count += size
        if limit is not None and number >= limit:
            break
    return weighted_loss / count, count


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    args = parse_args()
    if (min(args.batch_size, args.epochs, args.patience, args.dry_steps,
            args.hidden_dim) < 1 or
            args.pretrain_epochs < 0 or
            not math.isfinite(args.lr) or args.lr <= 0 or
            not math.isfinite(args.sign_weight) or args.sign_weight < 0 or
            not math.isfinite(args.sign_temperature) or args.sign_temperature <= 0):
        raise ValueError("Invalid training hyperparameters")
    if torch.musa.device_count() != 1:
        raise RuntimeError("Select one physical GPU with MUSA_VISIBLE_DEVICES before launch")
    device = torch.device("musa:0")
    seed_everything(args.seed)
    splits, file_sha256 = load_splits(args.source)
    caches = {"text": args.text_cache, "audio": args.audio_cache, "vision": args.vision_cache}
    raw_manifests = attach_reencoded_caches(splits, file_sha256, caches, args.audit)
    print(json.dumps({"data": {key: value.summary for key, value in splits.items()},
                      "sha256": file_sha256, "mode": args.mode, "seed": args.seed}), flush=True)
    dims = splits["train"].summary["dims"]
    model_config = dict(d=args.hidden_dim, output_dim=1, queries=4,
                        text_dim=1024, audio_dim=dims[1], vision_dim=dims[2],
                        dropout=0.1, topk=4, modality_dropout=0.1)
    model = OnlineBCHR(model_config).float().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.fusion.parameters(), "lr": args.lr},
        {"params": model.text_skip.parameters(), "lr": args.lr},
    ], weight_decay=0.01)
    train_data = loader(splits["train"], args.batch_size, True, args.seed)
    if args.mode == "smoke":
        pretrain_loss, _ = train_epoch(model, train_data, optimizer, device, args,
                                       limit=args.dry_steps, pretrain=True)
        loss, seen = train_epoch(model, train_data, optimizer, device, args, limit=args.dry_steps)
        print(json.dumps({"smoke_passed": True, "steps": args.dry_steps,
                          "samples": seen, "pretrain_loss": pretrain_loss,
                          "loss": loss}), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset": "CH-SIMS", "features_sha256": file_sha256,
                "text_model_sha256": raw_manifests['text']['model_sha256'],
                "audio_model_sha256": raw_manifests['audio']['model_sha256'],
                "vision_model_sha256": raw_manifests['vision']['model_sha256'],
                "raw_archive_sha256": raw_manifests['audio']['raw_archive_sha256'],
                "text_cache_manifest_sha256": hash_file(args.text_cache / 'manifest.json'),
                "audio_cache_manifest_sha256": hash_file(args.audio_cache / 'manifest.json'),
                "vision_cache_manifest_sha256": hash_file(args.vision_cache / 'manifest.json'),
                "audit_sha256": hash_file(args.audit),
                "source_sha256": source_hash(), "metric_protocol": "metricsTop.py:regression/SIMS",
                "mode": args.mode, "seed": args.seed, "model_config": model_config,
                "batch_size": args.batch_size, "epochs": args.epochs,
                "pretrain_epochs": args.pretrain_epochs,
                "patience": args.patience, "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "text_skip": "zero-initialized CLS residual on BCHR++ output",
                "sign_weight": args.sign_weight, "sign_temperature": args.sign_temperature,
                "checkpoint_selection": "validation clipped SIMS MAE; test never used for selection",
                "nontext_normalization": "train-only HuBERT/CLIP mean/std from audited cache, padding zero",
                "feature_protocol": "new frozen RoBERTa/HuBERT/CLIP raw-input encoding; not CICA features",
                "auxiliary_supervision": "train split regression_labels_T/A/V", "test_accessed": False}
    write_json_new(args.output_dir / "manifest.json", manifest)
    for pretrain_epoch in range(1, args.pretrain_epochs + 1):
        start = time.monotonic()
        train_loss, _ = train_epoch(model, train_data, optimizer, device, args,
                                    pretrain=True)
        print(json.dumps({"seed": args.seed, "pretrain_epoch": pretrain_epoch,
                          "unimodal_train_loss": train_loss,
                          "seconds": round(time.monotonic() - start, 1)}), flush=True)
    valid_data = loader(splits["valid"], args.batch_size, False, args.seed)
    best_mae, best_epoch, best_state, best_metrics = math.inf, -1, None, None
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        train_loss, _ = train_epoch(model, train_data, optimizer, device, args)
        metrics, mae = evaluate(model, valid_data, device)
        if mae < best_mae - 1e-6:
            best_mae, best_epoch, best_metrics = mae, epoch, metrics
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        print(json.dumps({"seed": args.seed, "epoch": epoch, "train_loss": train_loss,
                          "validation": metrics, "best_epoch": best_epoch,
                          "seconds": round(time.monotonic() - start, 1)}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = args.output_dir / "best.pt"
    with checkpoint.open("xb") as stream:
        torch.save({"manifest": manifest, "epoch": best_epoch,
                    "validation": best_metrics, "state_dict": best_state}, stream)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    selected_valid, selected_mae = evaluate(model, valid_data, device)
    if abs(selected_mae - best_mae) > 1e-4:
        raise RuntimeError("Checkpoint validation changed after reload")
    result = {"seed": args.seed, "mode": args.mode, "best_epoch": best_epoch,
              "validation": selected_valid, "features_sha256": file_sha256,
              "source_sha256": manifest["source_sha256"]}
    write_json_new(args.output_dir / "result.json", result)
    print(json.dumps({"completed": result}), flush=True)


if __name__ == "__main__":
    main()
