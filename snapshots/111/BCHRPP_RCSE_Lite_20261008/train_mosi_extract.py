"""Train/evaluate BCHR++ RCSE-Lite on ``extract/extract_mosi_features.py`` output.

This is the bridge between the offline extractor and the RCSE-Lite model: it
loads the ``bhgm_sent_crc_mosi_v1`` ``.pt`` produced by
``extract/extract_mosi_features.py``, trains ``bchrpp.full_model.BCHRPP`` with
``losses.bchrpp_loss.compute_loss``, and reports
``metrics.evaluation.mosi_metrics`` (acc7 / mae / mse / corr / acc2 / f1).

Validation MAE selects the checkpoint; the test split is evaluated exactly once
and its metrics must not be used for model or hyperparameter selection.
"""
import argparse
import json
import math
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
from losses.bchrpp_loss import compute_loss
from metrics.evaluation import mosi_metrics
from datasets.mosi_extract import FEATURE_DIMS, collate_mosi_extract, load_mosi_extract

VERSION = "bchrpp_rcse_lite_v1"


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
                   help="Load --checkpoint and evaluate the test split only")
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data, batch_size, shuffle, workers):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_mosi_extract,
                      pin_memory=torch.cuda.is_available())


def batch_to_device(raw, device):
    return {
        "text": raw["text"].to(device),
        "audio": raw["audio"].to(device),
        "vision": raw["vision"].to(device),
        "label": raw["label"].to(device),
        "masks": {m: raw[f"{m}_mask"].to(device) for m in ("text", "audio", "vision")},
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for raw in loader:
        batch = batch_to_device(raw, device)
        p = model(batch["text"], batch["audio"], batch["vision"], masks=batch["masks"])
        predictions.append(p.cpu().reshape(-1).numpy())
        labels.append(batch["label"].cpu().reshape(-1).numpy())
    return mosi_metrics(np.concatenate(predictions), np.concatenate(labels))


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
    return BCHRPP(**cfg), cfg


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
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                                        masks=batch["masks"], return_aux=True)
                loss, _ = compute_loss(prediction, batch["label"], aux,
                                       sample_weights=sample_weights,
                                       aux_weight=args.aux_weight,
                                       confidence_weight=args.confidence_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            n = batch["label"].shape[0]
            running += loss.detach().item() * n
            seen += n
        val = evaluate(model, loaders["valid"], device)
        print(json.dumps({"epoch": epoch, "train_loss": round(running / seen, 6),
                          "valid": val, "lr": optim.param_groups[0]["lr"]}), flush=True)
        if val["mae"] < best_mae - 1e-6:
            best_mae, best_epoch = val["mae"], epoch
            torch.save({
                "version": VERSION, "model_state": model.state_dict(),
                "model_config": cfg, "epoch": epoch, "valid_metrics": val,
                "dataset": "MOSI", "seed": args.seed,
                "features": str(args.features.resolve()),
            }, args.checkpoint)
        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, loaders["test"], device)
    result = {
        "dataset": "MOSI", "features": str(args.features.resolve()),
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
    model = BCHRPP(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    test = evaluate(model, make_loader(splits["test"], args.batch_size, False, args.num_workers), device)
    result = {
        "checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"),
        "validation": ckpt.get("valid_metrics"), "test": test,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    args = parser().parse_args()
    splits, meta = load_mosi_extract(args.features)
    print(json.dumps({"features": str(args.features.resolve()), **meta},
                     ensure_ascii=False), flush=True)
    if args.check_features:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.evaluate:
        return evaluate_checkpoint(args, splits, device)
    return fit(args, splits, device)


if __name__ == "__main__":
    main()
