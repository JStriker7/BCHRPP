"""Train/evaluate BCHR++ RCSE-Lite on ``extract/extract_mosi_features.py`` output.

This is the bridge between the offline extractor and the RCSE-Lite model: it
loads the ``bhgm_sent_crc_mosi_v1`` ``.pt`` produced by
``extract/extract_mosi_features.py``, trains ``bchrpp.full_model.BCHRPP`` with
``losses.bchrpp_loss.compute_loss``, and reports
the teacher's ``metricsTop.py`` MOSI regression metrics. Its values use the
MMSA naming and four-decimal reporting convention.

Validation MAE selects the checkpoint; the test split is evaluated exactly once
and its metrics must not be used for model or hyperparameter selection.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bchrpp.full_model import BCHRPP
from bchrpp.mosi_online_text import MOSIOnlineText
from datasets.mosi_text_online import MOSIOnlineTextSplit
from losses.bchrpp_loss import compute_loss
from metrics.metricsTop import MetricsTop
from datasets.mosi_extract import FEATURE_DIMS, collate_mosi_extract, load_mosi_extract

VERSION = "bchrpp_rcse_lite_online_text_v1"
METRIC_PROTOCOL = "metricsTop.py:regression/MOSI"
MOSI_METRIC = MetricsTop("regression").getMetics("MOSI")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True,
                   help=".pt produced by extract/extract_mosi_features.py")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "checkpoints" / "mosi_extract_rcse_lite_best.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modality-dropout", type=float, default=0.1)
    p.add_argument("--routing-temperature", type=float, default=1.0)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--confidence-weight", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=222)
    p.add_argument("--text-mode", choices=("frozen", "unfrozen"), default="frozen")
    p.add_argument("--text-lr", type=float, default=2e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--label-reweight", action="store_true",
                   help="TRAIN labels only; optional")
    p.add_argument("--no-reliability", action="store_true")
    p.add_argument("--no-complementarity", action="store_true")
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--no-sparse-graph", action="store_true")
    p.add_argument("--no-refinement", action="store_true")
    p.add_argument("--check-features", action="store_true",
                   help="Validate the feature file and exit")
    p.add_argument("--evaluate", action="store_true",
                   help="Load --checkpoint and rescore validation and test without writing")
    p.add_argument("--recover-trusted-checkpoint", action="store_true",
                   help="Finalize only a checkpoint produced locally by this training script; "
                        "loads it with pickle, writes a separate safe checkpoint and run JSON")
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data, batch_size, shuffle, workers):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=data.collate,
                      pin_memory=torch.cuda.is_available())


def batch_to_device(raw, device):
    return {
        "text": raw["text"].to(device),
        "audio": raw["audio"].to(device),
        "vision": raw["vision"].to(device),
        "label": raw["label"].to(device),
        "masks": {m: raw[f"{m}_mask"].to(device) for m in ("text", "audio", "vision")},
        "text_tokens": ({key: raw[f'text_{key}'].to(device) for key in
                          ('input_ids', 'attention_mask', 'token_type_ids')
                          if f'text_{key}' in raw}, raw['text_word_ids'].to(device)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for raw in loader:
        batch = batch_to_device(raw, device)
        p = model(batch["text"], batch["audio"], batch["vision"], masks=batch["masks"],
                  text_tokens=batch["text_tokens"])
        predictions.append(p.cpu().reshape(-1).numpy())
        labels.append(batch["label"].cpu().reshape(-1).numpy())
    p = np.concatenate(predictions)
    y = np.concatenate(labels)
    # metricsTop.py can return NumPy scalars; keep its values but serialize only
    # Python primitives so checkpoints remain readable with weights_only=True.
    official = {key: float(value) for key, value in
                MOSI_METRIC(torch.from_numpy(p), torch.from_numpy(y)).items()}
    # Select checkpoints at full precision, not from the four-decimal report.
    selection_mae = float(np.mean(np.abs(p.astype(np.float64) - y.astype(np.float64))))
    return official, selection_mae


def build_model(args):
    cfg = dict(
        d=args.hidden_dim, output_dim=1, queries=args.queries,
        text_dim=FEATURE_DIMS["text"], audio_dim=FEATURE_DIMS["audio"],
        vision_dim=FEATURE_DIMS["vision"], dropout=args.dropout, topk=args.topk,
        temporal=not args.no_temporal, sparse_graph=not args.no_sparse_graph,
        reliability=not args.no_reliability,
        complementarity=not args.no_complementarity,
        refinement=not args.no_refinement,
        modality_dropout=args.modality_dropout,
        routing_temperature=args.routing_temperature,
    )
    config = {"fusion": cfg, "text_model": args.text_model, "text_mode": args.text_mode,
              "text_lr": args.text_lr}
    return model_from_config(config), config


def model_from_config(config):
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(config["text_model"], local_files_only=True)
    if int(encoder.config.hidden_size) != FEATURE_DIMS["text"]:
        raise ValueError("Online text model must produce 1024-dimensional features")
    return MOSIOnlineText(encoder, BCHRPP(**config["fusion"]),
                          train_text=config["text_mode"] == "unfrozen")


def label_weight_lookup(train, device):
    y = np.asarray(train.labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(-3.5, 3.5, 8)
    bins = np.clip(np.digitize(y, edges[1:-1], right=True), 0, len(edges) - 2)
    freq = np.bincount(bins, minlength=len(edges) - 1)
    weights = 1.0 / np.sqrt(np.maximum(freq, 1))
    weights = np.clip(weights / (weights[bins].mean() + 1e-12), 0.5, 2.0)
    return (torch.tensor(edges[1:-1], device=device, dtype=torch.float32),
            torch.tensor(weights, device=device, dtype=torch.float32))


def fit(args, splits, device):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size and patience must be positive")
    seed_everything(args.seed)
    loaders = {s: make_loader(splits[s], args.batch_size, s == "train", args.num_workers)
               for s in ("train", "valid", "test")}
    model, cfg = build_model(args)
    model = model.to(device)
    groups = [{"params": model.fusion.parameters(), "lr": args.lr}]
    if args.text_mode == "unfrozen":
        groups.append({"params": model.encoder.parameters(), "lr": args.text_lr})
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = max(args.epochs * len(loaders["train"]), 1)
    warmup = int(total * args.warmup_ratio)

    def lr_scale(step):
        if step < warmup:
            return max((step + 1) / max(warmup, 1), 0.001)
        t = min(1.0, (step - warmup) / max(total - warmup, 1))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_scale)
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    edges, weights = (label_weight_lookup(splits["train"], device)
                      if args.label_reweight else (None, None))
    best_mae, best_epoch, last_epoch = float("inf"), 0, 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "dataset": "MOSI (extract)", "device": str(device),
        "sizes": {k: len(v) for k, v in splits.items()},
        "config": cfg,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }, ensure_ascii=False), flush=True)

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        running, seen = 0.0, 0
        for raw in loaders["train"]:
            batch = batch_to_device(raw, device)
            sample_weights = (weights[torch.bucketize(batch["label"].reshape(-1), edges, right=True)]
                              if edges is not None else None)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction, aux = model(batch["text"], batch["audio"], batch["vision"],
                                        masks=batch["masks"], text_tokens=batch["text_tokens"],
                                        return_aux=True)
                loss, _ = compute_loss(prediction, batch["label"], aux,
                                       sample_weights=sample_weights,
                                       aux_weight=args.aux_weight,
                                       confidence_weight=args.confidence_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                           args.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            n = batch["label"].shape[0]
            running += loss.detach().item() * n
            seen += n
        val, selection_mae = evaluate(model, loaders["valid"], device)
        print(json.dumps({"epoch": epoch, "train_loss": round(running / seen, 6),
                          "valid": val, "lr": optim.param_groups[0]["lr"]}), flush=True)
        if selection_mae < best_mae - 1e-6:
            best_mae, best_epoch = selection_mae, epoch
            torch.save({
                "version": VERSION, "model_state": model.state_dict(),
                "model_config": cfg, "epoch": epoch, "valid_metrics": val,
                "metric_protocol": METRIC_PROTOCOL,
                "selection_val_mae": selection_mae,
                "dataset": "MOSI", "seed": args.seed,
                "text_mode": args.text_mode,
                "features": str(args.features.resolve()),
            }, args.checkpoint)
        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test, _ = evaluate(model, loaders["test"], device)
    result = {
        "dataset": "MOSI", "features": str(args.features.resolve()),
        "metric_protocol": METRIC_PROTOCOL,
        "text_mode": args.text_mode,
        "seed": args.seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
        "validation": ckpt["valid_metrics"], "test": test,
        "checkpoint": str(args.checkpoint),
        "note": "Test metrics must not be used for model or hyperparameter selection.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    args.checkpoint.with_suffix(".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    return result


def evaluate_checkpoint(args, splits, device):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("version") != VERSION:
        raise ValueError(f"unknown checkpoint version {ckpt.get('version')!r}")
    if (ckpt.get("features") != str(args.features.resolve()) or
            ckpt["model_config"].get("text_mode") != args.text_mode or
            ckpt["model_config"].get("text_model") != args.text_model):
        raise ValueError("Checkpoint features, text model, or mode do not match this run")
    model = model_from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    validation, _ = evaluate(
        model, make_loader(splits["valid"], args.batch_size, False, args.num_workers), device
    )
    test, _ = evaluate(
        model, make_loader(splits["test"], args.batch_size, False, args.num_workers), device
    )
    result = {
        "checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"),
        "metric_protocol": METRIC_PROTOCOL,
        "text_mode": args.text_mode,
        "checkpoint_metric_protocol": ckpt.get("metric_protocol", "legacy_mosi_metrics"),
        "saved_validation": ckpt.get("valid_metrics"),
        "validation": validation, "test": test,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def recover_trusted_checkpoint(args, splits, device):
    """Finalize a local run interrupted after training by unsafe metric scalars."""
    original = args.checkpoint
    recovered = original.with_name(f"{original.stem}_recovered.pt")
    result_path = original.with_suffix(".json")
    if not original.is_file():
        raise FileNotFoundError(f"checkpoint not found: {original}")
    if recovered.exists() or result_path.exists():
        raise FileExistsError("Recovered checkpoint or result JSON already exists; refusing to overwrite")
    # This explicit mode is only for a checkpoint made by our own local run.
    # Unrestricted pickle loading must never be used for an untrusted file.
    ckpt = torch.load(original, map_location="cpu", weights_only=False)
    if (ckpt.get("version") != VERSION or ckpt.get("dataset") != "MOSI" or
            ckpt.get("metric_protocol") != METRIC_PROTOCOL or
            ckpt.get("features") != str(args.features.resolve()) or
            ckpt.get("seed") != args.seed):
        raise ValueError("Checkpoint does not match this MOSI feature file, metric protocol, or seed")
    best_epoch = ckpt.get("epoch")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= args.epochs:
        raise ValueError("Invalid saved best epoch or --epochs")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    if (ckpt["model_config"].get("text_mode") != args.text_mode or
            ckpt["model_config"].get("text_model") != args.text_model):
        raise ValueError("Checkpoint text model or mode does not match this run")
    model = model_from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    validation, selection_mae = evaluate(
        model, make_loader(splits["valid"], args.batch_size, False, args.num_workers), device
    )
    if not math.isclose(selection_mae, float(ckpt["selection_val_mae"]), rel_tol=0, abs_tol=1e-4):
        raise ValueError("Saved validation MAE differs from re-evaluation; refusing recovery")
    test, _ = evaluate(
        model, make_loader(splits["test"], args.batch_size, False, args.num_workers), device
    )
    safe = dict(ckpt)
    safe["valid_metrics"] = validation
    safe["selection_val_mae"] = float(ckpt["selection_val_mae"])
    temporary = recovered.with_name(f".{recovered.name}.{os.getpid()}.tmp")
    try:
        torch.save(safe, temporary)
        torch.load(temporary, map_location="cpu", weights_only=True)
        temporary.replace(recovered)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "dataset": "MOSI", "features": str(args.features.resolve()),
        "metric_protocol": METRIC_PROTOCOL, "seed": args.seed,
        "text_mode": args.text_mode,
        "best_epoch": best_epoch,
        "last_epoch": min(best_epoch + args.patience, args.epochs),
        "validation": validation, "test": test,
        "checkpoint": str(recovered), "recovered_from": str(original),
        "note": "Last epoch inferred from the run's epochs/patience; test was evaluated once after recovery.",
    }
    result_temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    try:
        result_temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
        result_temporary.replace(result_path)
    finally:
        result_temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main():
    args = parser().parse_args()
    splits, meta = load_mosi_extract(args.features)
    print(json.dumps({"features": str(args.features.resolve()), **meta},
                     ensure_ascii=False), flush=True)
    if args.check_features:
        return
    if args.text_lr <= 0:
        raise ValueError("Text learning rate must be positive")
    from transformers import AutoTokenizer

    args.text_model = meta["feature_models"]["text"]
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("Online word alignment requires a fast tokenizer")
    splits = {name: MOSIOnlineTextSplit(split.samples, tokenizer) for name, split in splits.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.recover_trusted_checkpoint:
        if args.evaluate:
            raise ValueError("--evaluate and --recover-trusted-checkpoint are mutually exclusive")
        return recover_trusted_checkpoint(args, splits, device)
    if args.evaluate:
        return evaluate_checkpoint(args, splits, device)
    return fit(args, splits, device)


if __name__ == "__main__":
    main()
