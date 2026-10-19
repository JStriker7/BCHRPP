"""Re-extract only MOSI CLIP vision features after face cropping.

The original trusted .pt supplies splits, labels, words, text/audio features,
validity masks, and source paths. The original extractor supplies frame loading,
audio duration, and uniform intervals. No training or evaluation happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.face_crop_ops import crop_first_face, pool_frame_features


SPLITS = ("train", "valid", "test")
EXTRACTOR_SHA256 = "73101f9a57fff62ef54cfe089700121063eb9f69d045f376d56cb05db68c163b"
CACHE_VERSION = 1
PARITY_ATOL = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True, help="Trusted original MOSI .pt")
    p.add_argument("--output", type=Path, help="New .pt; required outside dry-run")
    p.add_argument("--cache-dir", type=Path, help="Resumable cache; required outside dry-run")
    p.add_argument("--extractor", type=Path, default=ROOT.parent / "extract" / "extract_mosi_features.py")
    p.add_argument("--face-model", type=Path,
                   default=Path("/mnt/data26t/lxt/GraphMamba/models/blaze_face_short_range.tflite"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--video-sample-fps", type=float, default=4.0)
    p.add_argument("--media-batch-size", type=int, default=32)
    p.add_argument("--min-score", type=float, default=0.8)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--dry-run-samples", type=int, default=0,
                   help="Process this many samples per split, verify parity, write nothing")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--overwrite-cache", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_original_extractor(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Original extractor not found: {path}")
    actual = sha256_file(path)
    if actual != EXTRACTOR_SHA256:
        raise ValueError(f"Extractor SHA256 differs from the reviewed original: {actual}; expected {EXTRACTOR_SHA256}")
    spec = importlib.util.spec_from_file_location("original_mosi_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_stamp(path: Path) -> str:
    """Invalidate a cached sample if its original audio or frame source changes."""
    if not path.exists():
        raise FileNotFoundError(f"Saved source is unavailable on this server: {path}")
    digest = hashlib.sha256(str(path.resolve()).encode())
    items = sorted(path.iterdir()) if path.is_dir() else [path]
    for item in items:
        if path.is_dir() and item.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        stat = item.stat()
        digest.update(f"{item.name}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def validate_input(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Expected the original MOSI dictionary")
    metadata = data.get("_metadata", {})
    if (metadata.get("format") != "bhgm_sent_crc_mosi_v1" or
            metadata.get("dimensions") != {"text": 1024, "audio": 512, "visual": 768}):
        raise ValueError("Unexpected original feature format or dimensions")
    if "face_crop" in metadata:
        raise ValueError("Input is already face-cropped; use the original whole-frame .pt")
    if not metadata.get("models", {}).get("visual"):
        raise ValueError("Original CLIP model identifier is missing")
    seen = set()
    for split in SPLITS:
        samples = data.get(split)
        if not isinstance(samples, list) or not samples or metadata.get("splits", {}).get(split) != len(samples):
            raise ValueError(f"Invalid {split} samples/count")
        for sample in samples:
            uid = sample.get("id")
            words = sample.get("words")
            if not isinstance(uid, str) or uid in seen or not isinstance(words, (list, tuple)) or not words:
                raise ValueError(f"Invalid or duplicate sample id/words: {uid!r}")
            seen.add(uid)
            if sample.get("alignment") != "uniform":
                raise ValueError(f"{uid}: timestamp alignment cannot be reconstructed from this .pt")
            if np.asarray(sample["visual"]).shape != (len(words), 768):
                raise ValueError(f"{uid}: unexpected original visual shape")
            if sample.get("feature_models") != metadata["models"]:
                raise ValueError(f"{uid}: feature model identifiers differ from metadata")
            if not np.array_equal(np.asarray(sample["visual_valid"]), np.ones(len(words))):
                raise ValueError(f"{uid}: expected fully valid original visual positions")
            if not sample.get("sources", {}).get("audio") or not sample.get("sources", {}).get("visual"):
                raise ValueError(f"{uid}: original audio/visual source path is missing")
    return metadata


def encode_frames(frames, processor, model, device, batch_size: int) -> np.ndarray:
    import torch

    with torch.inference_mode():
        chunks = []
        for start in range(0, len(frames), batch_size):
            inputs = processor(images=list(frames[start:start + batch_size]), return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"].to(device))
            chunks.append(outputs.image_embeds.float().cpu())
        return torch.cat(chunks).numpy()


def reextract_one(sample, original, detector, processor, model, device, args, verify: bool):
    import cv2
    import mediapipe as mp

    uid = sample["id"]
    audio_path = Path(sample["sources"]["audio"])
    visual_path = Path(sample["sources"]["visual"])
    waveform, sr = original.load_audio(audio_path)
    frames, frame_times, video_duration = original.load_visual(visual_path, args.video_sample_fps)
    if waveform is None or frames is None or frame_times is None or len(frames) == 0:
        raise ValueError(f"{uid}: source audio or visual frames could not be decoded")
    audio_duration = len(waveform) / sr
    if visual_path.is_dir() or visual_path.suffix.lower() in {".npy", ".npz"}:
        frame_times = np.linspace(0.0, audio_duration, len(frames), endpoint=False, dtype=np.float32)
        video_duration = audio_duration
    intervals = original.uniform_intervals(len(sample["words"]), max(audio_duration, video_duration))

    parity_delta = None
    if verify:
        uncropped = encode_frames(frames, processor, model, device, args.media_batch_size)
        baseline = pool_frame_features(uncropped, frame_times, intervals)
        parity_delta = float(np.max(np.abs(baseline - np.asarray(sample["visual"], dtype=np.float32))))
        if not np.isfinite(parity_delta) or parity_delta > PARITY_ATOL:
            raise ValueError(
                f"{uid}: uncropped CLIP does not reproduce the original .pt "
                f"(max_abs_diff={parity_delta:.6g}, allowed={PARITY_ATOL}); "
                "check model, source, FPS, and original extractor before running the trial"
            )

    cropped = []
    detected = 0
    area_fraction = 0.0
    for frame in frames:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = detector.detect(mp_image)
        face, found = crop_first_face(frame, result.detections, args.min_score)
        if found:
            detected += 1
            area_fraction += face.shape[0] * face.shape[1] / (frame.shape[0] * frame.shape[1])
            cropped.append(cv2.resize(face, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR))
        else:
            cropped.append(frame)

    features = encode_frames(cropped, processor, model, device, args.media_batch_size)
    pooled = pool_frame_features(features, frame_times, intervals)
    if not np.isfinite(pooled).all() or pooled.shape != np.asarray(sample["visual"]).shape:
        raise ValueError(f"{uid}: invalid new visual features")
    stats = {"frames": len(frames), "face_frames": detected,
             "fallback_frames": len(frames) - detected,
             "mean_face_area_fraction": area_fraction / detected if detected else 0.0,
             "parity_max_abs_diff": parity_delta}
    return pooled, stats


def main() -> None:
    args = parse_args()
    if args.media_batch_size < 1 or args.video_sample_fps <= 0 or args.crop_size < 1 or args.progress_every < 1:
        raise ValueError("Batch size, sample FPS, crop size, and progress interval must be positive")
    if not 0 <= args.min_score <= 1 or args.dry_run_samples < 0:
        raise ValueError("Score must be in [0,1] and dry-run count must be nonnegative")
    dry_run = args.dry_run_samples > 0
    if not dry_run and (args.output is None or args.cache_dir is None):
        raise ValueError("Full extraction requires --output and --cache-dir")
    if args.output is not None:
        if args.output.resolve() == args.features.resolve():
            raise ValueError("Output must not overwrite the original .pt")
        if not dry_run and args.output.exists():
            raise FileExistsError(f"Output already exists; refusing to overwrite: {args.output}")
    if not args.features.is_file() or not args.face_model.is_file():
        raise FileNotFoundError("Original features or face detector model not found")

    import torch

    original = load_original_extractor(args.extractor)
    # The NumPy arrays in this .pt require weights_only=False; only load a trusted local file.
    data = torch.load(args.features, map_location="cpu", weights_only=False)
    metadata = validate_input(data)
    original_hash = sha256_file(args.features)
    face_hash = sha256_file(args.face_model)
    clip_id = metadata["models"]["visual"]
    if not Path(clip_id).is_dir():
        raise FileNotFoundError(f"Original CLIP model directory not available locally: {clip_id}")
    import cv2
    import mediapipe as mp
    import PIL
    import transformers

    settings = {"cache_version": CACHE_VERSION, "input_sha256": original_hash,
                "script_sha256": sha256_file(Path(__file__)),
                "crop_ops_sha256": sha256_file(ROOT / "scripts" / "face_crop_ops.py"),
                "extractor_sha256": EXTRACTOR_SHA256, "face_model_sha256": face_hash,
                "clip_model": clip_id, "video_sample_fps": args.video_sample_fps,
                "media_batch_size": args.media_batch_size, "device": args.device,
                "min_score": args.min_score, "crop_size": args.crop_size,
                "fallback": "uncropped_original_frame",
                "runtime_versions": {"numpy": np.__version__, "torch": str(torch.__version__),
                                     "opencv": cv2.__version__, "pillow": PIL.__version__,
                                     "mediapipe": mp.__version__,
                                     "transformers": transformers.__version__}}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from transformers import AutoProcessor, CLIPVisionModelWithProjection

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(clip_id, local_files_only=True)
    model = CLIPVisionModelWithProjection.from_pretrained(clip_id, local_files_only=True).to(device).eval()
    if int(model.config.projection_dim) != 768:
        raise ValueError("Original CLIP projection must be 768-dimensional")
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.face_model.resolve())))
    print(json.dumps({"mode": "dry-run" if dry_run else "full", "device": str(device),
                      "clip_model": clip_id, "settings": settings,
                      "split_sizes": {s: len(data[s]) for s in SPLITS}}, ensure_ascii=False), flush=True)

    reports = {}
    new_data = {"_metadata": dict(metadata)} if not dry_run else None
    with vision.FaceDetector.create_from_options(options) as detector:
        for split in SPLITS:
            samples = data[split][:args.dry_run_samples] if dry_run else data[split]
            new_samples = []
            report = {"samples": 0, "cache_hits": 0, "frames": 0, "face_frames": 0,
                      "fallback_frames": 0, "parity_max_abs_diff": 0.0}
            for index, sample in enumerate(samples):
                uid = sample["id"]
                sources = sample["sources"]
                stamps = {key: source_stamp(Path(sources[key])) for key in ("audio", "visual")}
                cache_file = None if dry_run else args.cache_dir / split / f"{hashlib.sha256(uid.encode()).hexdigest()}.pt"
                verify = dry_run or index == 0
                saved = None
                if cache_file is not None and cache_file.is_file() and not args.overwrite_cache and not verify:
                    candidate = torch.load(cache_file, map_location="cpu", weights_only=False)
                    if (candidate.get("signature") == signature and candidate.get("uid") == uid and
                            candidate.get("stamps") == stamps and
                            np.asarray(candidate.get("visual")).shape == np.asarray(sample["visual"]).shape):
                        saved = candidate
                        report["cache_hits"] += 1
                if saved is None:
                    visual, stats = reextract_one(sample, original, detector, processor, model, device, args, verify)
                    saved = {"signature": signature, "uid": uid, "stamps": stamps,
                             "visual": visual, "stats": stats}
                    if cache_file is not None:
                        original.atomic_torch_save(saved, cache_file)
                stats = saved["stats"]
                report["samples"] += 1
                for key in ("frames", "face_frames", "fallback_frames"):
                    report[key] += stats[key]
                if stats["parity_max_abs_diff"] is not None:
                    report["parity_max_abs_diff"] = max(report["parity_max_abs_diff"], stats["parity_max_abs_diff"])
                if dry_run:
                    print(json.dumps({"split": split, "id": uid, **stats}, ensure_ascii=False), flush=True)
                else:
                    changed = dict(sample)
                    changed["visual"] = saved["visual"]
                    new_samples.append(changed)
                    if (index + 1) % args.progress_every == 0 or index + 1 == len(samples):
                        print(f"[faces] {split} {index + 1}/{len(samples)}", flush=True)
            reports[split] = report
            if not dry_run:
                new_data[split] = new_samples

    print(json.dumps({"report": reports}, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        print("[dry-run] no files written", flush=True)
        return
    if sum(report["face_frames"] for report in reports.values()) == 0:
        raise RuntimeError("No faces detected; refusing to save a no-op feature file")
    new_data["_metadata"]["face_crop"] = {**settings, "coverage": reports}
    original.atomic_torch_save(new_data, args.output)
    print(f"[done] saved={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
