"""Multi-seed + ablation runner for RCSE-Lite on extract MOSI features.

For every (mode, seed) pair this launches ``scripts/train_mosi_extract.py`` as
a subprocess, reads back the per-run ``.json``, and writes two CSVs:

    results.csv   one row per (mode, seed), full val/test metrics
    summary.csv   per-mode mean +/- std across seeds (key test metrics)

Run from the project root, e.g.:

    python scripts/run_mosi_extract_ablations.py \
        --features /path/to/mosi_deberta_clip_clap.pt \
        --output-dir /path/to/results/ablations \
        --seeds 222,128,2026 \
        --modes lite,no_reliability,no_complementarity,no_temporal,no_sparse_graph,no_refinement \
        --epochs 60 --batch-size 24 --hidden-dim 128 --queries 4 --topk 4 --lr 0.0003 --amp
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Each mode maps to the exact CLI flags passed to train_mosi_extract.py.
MODES = {
    "lite": [],
    "no_reliability": ["--no-reliability"],
    "no_complementarity": ["--no-complementarity"],
    "no_temporal": ["--no-temporal"],
    "no_sparse_graph": ["--no-sparse-graph"],
    "no_refinement": ["--no-refinement"],
    "no_modality_dropout": ["--modality-dropout", "0"],
    "no_quality_aux": ["--aux-weight", "0", "--confidence-weight", "0"],
}

METRIC_KEYS = ["mae", "mse", "corr", "acc2", "f1", "acc7"]
TEST_KEYS = ["mae", "corr", "acc2", "f1", "acc7"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True,
                   help=".pt produced by extract/extract_mosi_features.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", default="222,128,2026")
    p.add_argument("--modes",
                   default="lite,no_reliability,no_complementarity,no_temporal,"
                           "no_sparse_graph,no_refinement")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--amp", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.csv"

    fieldnames = ["mode", "seed", "best_epoch", "last_epoch"] + \
                 [f"val_{k}" for k in METRIC_KEYS] + \
                 [f"test_{k}" for k in METRIC_KEYS]

    rows = []
    for mode in modes:
        for seed in seeds:
            tag = f"{mode}_seed{seed}"
            checkpoint = args.output_dir / f"mosi_{tag}.pt"
            cmd = [sys.executable, str(Path(__file__).parent / "train_mosi_extract.py"),
                   "--features", str(args.features),
                   "--checkpoint", str(checkpoint),
                   "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size),
                   "--hidden-dim", str(args.hidden_dim),
                   "--queries", str(args.queries),
                   "--topk", str(args.topk),
                   "--lr", str(args.lr),
                   "--seed", str(seed),
                   *MODES[mode]]
            if args.amp:
                cmd.append("--amp")
            print(f"== RUN mode={mode} seed={seed} ==", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            summary = json.loads(
                checkpoint.with_suffix(".json").read_text(encoding="utf8")
            )
            row = {"mode": mode, "seed": seed,
                   "best_epoch": summary["best_epoch"],
                   "last_epoch": summary["last_epoch"]}
            for k in METRIC_KEYS:
                row[f"val_{k}"] = summary["validation"][k]
                row[f"test_{k}"] = summary["test"][k]
            rows.append(row)
            # Incremental write so partial progress survives an interruption.
            with csv_path.open("w", newline="", encoding="utf8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    # ---- per-mode summary (mean +/- std across seeds) ----
    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary_fieldnames = ["mode", "n_seeds", "seeds",
                          "val_mae_mean", "val_mae_std"]
    for k in TEST_KEYS:
        summary_fieldnames += [f"test_{k}_mean", f"test_{k}_std"]

    summary_rows = []
    for mode in sorted(by_mode):
        runs = by_mode[mode]
        run_seeds = [r["seed"] for r in runs]
        entry = {"mode": mode, "n_seeds": len(runs),
                 "seeds": ",".join(map(str, run_seeds))}

        def agg(prefix, key):
            values = [r[f"{prefix}_{key}"] for r in runs]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean, std

        entry["val_mae_mean"], entry["val_mae_std"] = agg("val", "mae")
        for k in TEST_KEYS:
            entry[f"test_{k}_mean"], entry[f"test_{k}_std"] = agg("test", k)
        summary_rows.append(entry)

    with summary_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n== done {len(rows)} runs ==", flush=True)
    print(f"== per-run csv: {csv_path} ==", flush=True)
    print(f"== summary csv: {summary_path} ==", flush=True)
    for entry in summary_rows:
        print(json.dumps(entry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
