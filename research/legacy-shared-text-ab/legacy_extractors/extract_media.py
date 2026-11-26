"""Extract pinned HuBERT audio or CLIP video features from CH-SIMS clips.

Screen mode constructs only train/valid tensors. Test needs an explicit later
mode, after model selection has been locked without test feedback.
"""

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch_musa  # noqa: F401
from transformers import CLIPImageProcessor, CLIPModel, HubertModel, Wav2Vec2FeatureExtractor


DATA_SHA256 = 'c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f'
HUBERT_SHA256 = '482c19cd60877832995d790ed10dd1203b1f8c86ed7a69b4bd27f470fc27ee43'
CLIP_SHA256 = 'a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb'
COUNTS = {'train': 1368, 'valid': 456, 'test': 457}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audio_from_clip(path):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
               '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1']
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode or not process.stdout:
        raise ValueError(f'Audio decode failed: {path}: {process.stderr[:200]!r}')
    return np.frombuffer(process.stdout, dtype=np.float32).copy()


def frames_from_clip(path, count=8):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Video open failed: {path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f'No video frames: {path}')
    desired = np.rint(np.linspace(0, total - 1, count)).astype(int)
    frames = []
    frame_index = 0
    target = 0
    while target < count:
        success, frame = cap.read()
        if not success:
            break
        while target < count and desired[target] == frame_index:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target += 1
        frame_index += 1
    cap.release()
    if not frames:
        raise ValueError(f'Video decode produced no frames: {path}')
    while len(frames) < count:
        frames.append(frames[-1])
    return frames


def reduce_audio_time(output, length=128):
    indices = torch.linspace(0, output.shape[0] - 1, min(output.shape[0], length),
                             device=output.device).round().long()
    return output.index_select(0, indices)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--encoder', choices=('audio', 'vision'), required=True)
    p.add_argument('--mode', choices=('smoke', 'screen', 'test'), required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if torch.musa.device_count() != 1:
        raise RuntimeError('Select one physical GPU with MUSA_VISIBLE_DEVICES')
    if digest(args.features) != DATA_SHA256:
        raise ValueError('Official CH-SIMS features hash mismatch')
    raw_manifest = json.loads((args.raw_dir / 'manifest.json').read_text())
    if raw_manifest['dataset_sha256'] != DATA_SHA256:
        raise ValueError('Raw video source mismatch')
    model_file = args.model / ('pytorch_model.bin' if args.encoder == 'audio' else 'model.safetensors')
    expected = HUBERT_SHA256 if args.encoder == 'audio' else CLIP_SHA256
    if digest(model_file) != expected:
        raise ValueError('Encoder weight SHA-256 mismatch')
    with args.features.open('rb') as stream:
        data = pickle.load(stream)
    splits = ('train',) if args.mode == 'smoke' else (
        ('test',) if args.mode == 'test' else ('train', 'valid'))
    if args.encoder == 'audio':
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
        model = HubertModel.from_pretrained(str(args.model), local_files_only=True,
                                           use_safetensors=False,
                                           attn_implementation='eager').eval().to('musa:0')
        length = 128
    else:
        processor = CLIPImageProcessor.from_pretrained(str(args.model), local_files_only=True)
        model = CLIPModel.from_pretrained(str(args.model), local_files_only=True,
                                          attn_implementation='eager').eval().to('musa:0')
        length = 8
    hidden = model.config.hidden_size if args.encoder == 'audio' else model.config.vision_config.hidden_size
    if args.mode != 'smoke':
        args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'encoder': args.encoder, 'mode': args.mode, 'splits': splits,
                'dataset_sha256': DATA_SHA256, 'raw_archive_sha256': raw_manifest['archive_sha256'],
                'model_sha256': expected, 'length': length, 'hidden_size': hidden,
                'id_sha256': {}}
    for split in splits:
        ids = data[split]['id'][:2] if args.mode == 'smoke' else data[split]['id']
        if args.mode != 'smoke' and len(ids) != COUNTS[split]:
            raise ValueError(f'Unexpected {split} split length')
        if args.mode != 'smoke':
            values = np.lib.format.open_memmap(args.output_dir / f'{split}_features.npy',
                                               mode='w+', dtype=np.float16,
                                               shape=(len(ids), length, hidden))
            masks = np.lib.format.open_memmap(args.output_dir / f'{split}_mask.npy',
                                              mode='w+', dtype=np.bool_, shape=(len(ids), length))
        for index, key in enumerate(ids):
            video = args.raw_dir / 'Raw' / (key.replace('$_$', '/') + '.mp4')
            if not video.is_file():
                raise FileNotFoundError(video)
            with torch.inference_mode():
                if args.encoder == 'audio':
                    wave = audio_from_clip(video)
                    batch = processor(wave, sampling_rate=16000, return_tensors='pt')
                    encoded = model(**{k: v.to('musa:0') for k, v in batch.items()}).last_hidden_state[0]
                    encoded = reduce_audio_time(encoded, length)
                else:
                    frames = frames_from_clip(video, count=length)
                    pixels = processor(images=frames, return_tensors='pt').pixel_values.to('musa:0')
                    encoded = model.vision_model(pixel_values=pixels).pooler_output
            array = encoded.float().cpu().numpy().astype(np.float16)
            if array.shape[1] != hidden or not np.isfinite(array).all():
                raise ValueError(f'Invalid {split}:{key} output')
            if args.mode != 'smoke':
                values[index] = 0
                values[index, :len(array)] = array
                masks[index] = False
                masks[index, :len(array)] = True
            if (index + 1) % 100 == 0 or index + 1 == len(ids):
                print(json.dumps({'split': split, 'encoded': index + 1, 'total': len(ids)}), flush=True)
        if args.mode != 'smoke':
            values.flush(); masks.flush()
            manifest['id_sha256'][split] = hashlib.sha256(
                json.dumps(ids, ensure_ascii=False).encode('utf-8')).hexdigest()
    if args.mode != 'smoke':
        with (args.output_dir / 'manifest.json').open('x') as stream:
            json.dump(manifest, stream, indent=2)
    print(json.dumps({'completed': manifest}), flush=True)


if __name__ == '__main__':
    main()
