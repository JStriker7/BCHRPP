"""Train/evaluate the teacher's BCHRPP skeleton on offline MOSI features."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from bchrpp.mosi_model import MOSIRegressor
from datasets.mosi_features import collate_mosi, load_mosi_features
from metrics.evaluation import mosi_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MOSI",), default="MOSI")
    parser.add_argument("--config", type=Path, default=Path("configs/bchrpp_base.yaml"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-features", action="store_true")
    parser.add_argument("--evaluate", type=Path, help="Evaluate an existing checkpoint on test only.")
    args = parser.parse_args()
    if not args.check_features and not args.evaluate and args.run_dir is None:
        parser.error("--run-dir is required for training")
    return args


def load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    for name in ("hidden_dim", "batch_size", "epochs", "patience"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if config["hidden_dim"] % 4:
        raise ValueError("hidden_dim must be divisible by 4 transformer heads")
    for name in ("lr", "grad_clip", "huber_beta"):
        if not isinstance(config.get(name), (int, float)) or config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not isinstance(config.get("weight_decay"), (int, float)) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if type(config.get("amp")) is not bool:
        raise ValueError("amp must be a boolean")
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            predictions.extend(model(batch).float().cpu().tolist())
            labels.extend(batch["label"].float().cpu().tolist())
    return mosi_metrics(labels, predictions)


def train_epoch(model, loader, optimizer, scaler, device, config):
    model.train()
    total_loss, count = 0.0, 0
    use_amp = bool(config["amp"] and device.type == "cuda")
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(batch)
            loss = F.smooth_l1_loss(
                predictions, batch["label"], beta=config["huber_beta"]
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect features and learning rate")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        size = len(batch["label"])
        total_loss += loss.item() * size
        count += size
    return total_loss / count


def make_loader(split, config, shuffle, device, generator=None):
    return DataLoader(
        split, batch_size=config["batch_size"], shuffle=shuffle,
        collate_fn=collate_mosi, pin_memory=device.type == "cuda",
        num_workers=0, generator=generator,
    )


def save_checkpoint(path, model, config, epoch, validation, feature_summary):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "config": config, "epoch": epoch,
        "validation": validation, "feature_models": feature_summary["feature_models"],
    }, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    splits, feature_summary = load_mosi_features(args.features)
    print("features:", json.dumps(feature_summary, ensure_ascii=False))
    if args.check_features:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else "cpu" if args.device == "auto" else args.device)

    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        if checkpoint["feature_models"] != feature_summary["feature_models"]:
            raise ValueError("Checkpoint and feature extraction model identities differ")
        config = checkpoint["config"]
        model = MOSIRegressor(config["hidden_dim"]).to(device)
        model.load_state_dict(checkpoint["model"])
        result = predict(model, make_loader(splits["test"], config, False, device), device)
        print("test:", json.dumps(result, ensure_ascii=False))
        return

    config = load_config(args.config)
    set_seed(config["seed"])
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = make_loader(splits["train"], config, True, device, generator)
    valid_loader = make_loader(splits["valid"], config, False, device)
    test_loader = make_loader(splits["test"], config, False, device)
    model = MOSIRegressor(config["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"] and device.type == "cuda")
    best_mae, best_epoch, stale = float("inf"), 0, 0
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, config)
        validation = predict(model, valid_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.6f} "
              f"valid_MAE={validation['MAE']:.6f} valid_Acc7={validation['Acc-7']:.4f}", flush=True)
        if validation["MAE"] < best_mae:
            best_mae, best_epoch, stale = validation["MAE"], epoch, 0
            save_checkpoint(checkpoint_path, model, config, epoch, validation, feature_summary)
        else:
            stale += 1
            if stale >= config["patience"]:
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "dataset": "MOSI", "features": str(args.features.resolve()),
        "feature_summary": feature_summary, "config": config,
        "best_epoch": best_epoch, "validation": checkpoint["validation"],
        "test": predict(model, test_loader, device),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("test:", json.dumps(result["test"], ensure_ascii=False))
    print(f"checkpoint={checkpoint_path} result={run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
