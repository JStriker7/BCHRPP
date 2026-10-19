from __future__ import annotations

"""Extract word-aligned MOSI features for BHGM-Sent-CRC.

Text uses DeBERTa-v3, visual frames use CLIP ViT-L/14@336, and audio
windows use the CLAP HTSAT encoder. The resulting .pt file can be passed
directly to train.py.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('/mnt/data8tb/wxr/MMML/data/MOSI/processed'),
                        help='MOSI root, or its processed/ directory.')
    parser.add_argument('--output', type=Path,
                        default=Path('/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/data/MOSI/processed/mosi_deberta_clip_clap.pt'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/data/MOSI/processed/bhgm_feature_cache'))
    parser.add_argument('--alignment-json', type=Path,
                        help='JSON mapping uid to [{word,start,end}, ...].')
    parser.add_argument('--allow-uniform-alignment', action='store_true', default=True,
                        help='Fallback to equal-duration word windows when timestamps are unavailable.')
    parser.add_argument('--require-word-timestamps', dest='allow_uniform_alignment', action='store_false')
    parser.add_argument('--deberta-model',
                        default='/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/model/deberta-v3-large')
    parser.add_argument('--clip-model', default='/mnt/data26t/model/clip-vit-large-patch14')
    parser.add_argument('--clap-model',
                        default='/mnt/data8tb/wxr/MMML/checkpoints/630k-audioset-best.pt')
    parser.add_argument('--clap-bert-model', default='/mnt/data26t/model/bert-base-uncased')
    parser.add_argument('--clap-roberta-model', default='/mnt/data26t/model/roberta-base')
    parser.add_argument('--clap-roberta-tokenizer', default='/mnt/data26t/model/roberta-large')
    parser.add_argument('--clap-bart-model', default='/mnt/data26t/model/bart-base')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--video-sample-fps', type=float, default=4.0)
    parser.add_argument('--media-batch-size', type=int, default=32)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--check-layout', action='store_true',
                        help='Report resolved labels/audio/visual files without loading any models.')
    return parser.parse_args()


def configure_quiet_output() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module=r'(transformers|timm|huggingface_hub).*')
    logging.basicConfig(level=logging.ERROR)
    for name in ('transformers', 'huggingface_hub', 'timm', 'laion_clap'):
        logging.getLogger(name).setLevel(logging.ERROR)


def canonical_split(value: str) -> str:
    value = value.strip().lower()
    return {'validation': 'valid', 'val': 'valid', 'dev': 'valid'}.get(value, value)


def sample_uid(row: Dict[str, str]) -> str:
    return f"{row['video_id'].strip()}_{row['clip_id'].strip()}"


class MOSILayout:
    """Resolve both raw MOSI and MMML-style processed directory layouts."""

    def __init__(self, data_dir: Path) -> None:
        supplied = data_dir.expanduser()
        if supplied.name.lower() == 'processed' and (supplied.parent / 'label.csv').exists():
            self.root = supplied.parent
            self.processed = supplied
        else:
            self.root = supplied
            self.processed = supplied / 'processed'
        self.label_path = self.root / 'label.csv'
        if not self.label_path.exists() and (supplied / 'label.csv').exists():
            self.label_path = supplied / 'label.csv'
        if not self.label_path.exists():
            raise FileNotFoundError(
                f'Cannot find label.csv from {supplied}. Expected {supplied / "label.csv"} '
                f'or {supplied.parent / "label.csv"}.'
            )

    @staticmethod
    def split_names(row: Dict[str, str]) -> List[str]:
        split = canonical_split(row['mode'])
        return ['val', 'valid', 'dev'] if split == 'valid' else [split]

    @staticmethod
    def first_existing(candidates: Sequence[Path]) -> Path | None:
        return next((path for path in candidates if path.exists()), None)

    def audio_path(self, row: Dict[str, str]) -> Path | None:
        video_id, clip_id, uid = row['video_id'].strip(), row['clip_id'].strip(), sample_uid(row)
        candidates: List[Path] = [
            self.root / 'wav' / video_id / f'{clip_id}.wav',
            self.root / 'wav' / f'{uid}.wav',
        ]
        for split in self.split_names(row):
            base = self.processed / 'audio' / split
            candidates.extend([
                base / video_id / f'{clip_id}.wav', base / f'{uid}.wav',
                base / f'{video_id}[{clip_id}].wav', base / video_id / f'{uid}.wav',
            ])
        return self.first_existing(candidates)

    def visual_path(self, row: Dict[str, str]) -> Path | None:
        video_id, clip_id, uid = row['video_id'].strip(), row['clip_id'].strip(), sample_uid(row)
        candidates: List[Path] = [
            self.root / 'Raw' / video_id / f'{clip_id}.mp4',
            self.root / 'Raw' / f'{uid}.mp4',
        ]
        for split in self.split_names(row):
            base = self.processed / 'frames' / split
            candidates.extend([
                base / video_id / clip_id, base / uid, base / f'{video_id}[{clip_id}]',
                base / video_id / f'{clip_id}.npy', base / f'{uid}.npy',
                base / video_id / f'{clip_id}.npz', base / f'{uid}.npz',
                base / video_id / f'{clip_id}.mp4', base / f'{uid}.mp4',
            ])
        frame_root = self.processed / 'frames'
        candidates.extend([
            frame_root / video_id / clip_id, frame_root / uid,
            frame_root / video_id / f'{clip_id}.npy', frame_root / f'{uid}.npy',
        ])
        return self.first_existing(candidates)


def read_labels(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        preview = handle.read(16384)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(preview, delimiters=',\t;|')
        except csv.Error:
            dialect = csv.excel
        raw_rows = list(csv.DictReader(handle, dialect=dialect))
    if not raw_rows:
        raise ValueError(f'{path} contains no label rows.')

    def clean_name(value: str) -> str:
        return re.sub(r'_+', '_', value.strip().lower().replace('-', '_').replace(' ', '_'))

    aliases = {
        'video_id': ('video_id', 'videoid', 'video', 'vid'),
        'clip_id': ('clip_id', 'clipid', 'clip', 'segment_index', 'utterance_index'),
        'text': ('text', 'transcript', 'sentence', 'utterance', 'words'),
        'label': ('label', 'sentiment', 'sentiment_label', 'sentiment_score', 'score'),
        'mode': ('mode', 'split', 'partition', 'set', 'subset'),
        'uid': ('uid', 'id', 'segment_id', 'utterance_id', 'key'),
    }
    detected = {clean_name(name): name for name in (raw_rows[0].keys() if raw_rows else []) if name}

    def source_column(target: str) -> str | None:
        return next((detected[name] for name in aliases[target] if name in detected), None)

    columns = {target: source_column(target) for target in aliases}
    missing = [name for name in ('text', 'label', 'mode') if columns[name] is None]
    has_separate_id = columns['video_id'] is not None and columns['clip_id'] is not None
    if missing or (not has_separate_id and columns['uid'] is None):
        raise ValueError(
            f'{path} has unsupported columns. Detected={list(raw_rows[0].keys())}; '
            f'missing={missing or ["video_id+clip_id (or a combined uid)"]}.'
        )

    def split_uid(value: str) -> Tuple[str, str]:
        value = value.strip()
        bracket = re.fullmatch(r'(.+)\[(\d+)\]', value)
        if bracket:
            return bracket.group(1), bracket.group(2)
        for separator in ('_', '-'):
            if separator in value:
                left, right = value.rsplit(separator, 1)
                if right.isdigit():
                    return left, right
        raise ValueError(f'Cannot split combined uid {value!r} into video_id and clip_id.')

    rows: List[Dict[str, Any]] = []
    for line_number, raw in enumerate(raw_rows, 2):
        if has_separate_id:
            video_id = str(raw[columns['video_id']]).strip()
            clip_id = str(raw[columns['clip_id']]).strip()
        else:
            try:
                video_id, clip_id = split_uid(str(raw[columns['uid']]))
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
        row = dict(raw)
        row.update({
            'video_id': video_id,
            'clip_id': clip_id,
            'text': str(raw[columns['text']]).strip(),
            'label': str(raw[columns['label']]).strip(),
            'mode': str(raw[columns['mode']]).strip(),
        })
        rows.append(row)
    print(f'[labels] loaded={len(rows)}, delimiter={repr(dialect.delimiter)}')
    return rows


def load_alignment_file(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError('Alignment JSON must be an object keyed by MOSI uid.')
    return data


def normalize_alignment(value: Any, expected_words: Sequence[str]) -> List[Tuple[float, float]] | None:
    if not isinstance(value, list) or len(value) != len(expected_words):
        return None
    intervals: List[Tuple[float, float]] = []
    for entry in value:
        if isinstance(entry, dict):
            start, end = entry.get('start'), entry.get('end')
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            start, end = entry[-2], entry[-1]
        else:
            return None
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(start_f) or not math.isfinite(end_f) or end_f <= start_f:
            return None
        intervals.append((start_f, end_f))
    return intervals


def row_alignment(row: Dict[str, Any], alignments: Dict[str, Any],
                  words: Sequence[str]) -> List[Tuple[float, float]] | None:
    uid = sample_uid(row)
    value = alignments.get(uid)
    if value is None and row.get('word_timestamps'):
        try:
            value = json.loads(row['word_timestamps'])
        except json.JSONDecodeError:
            value = None
    return normalize_alignment(value, words)


def load_audio(path: Path | None) -> Tuple[np.ndarray | None, int]:
    if path is None:
        return None, 16000
    import librosa
    waveform, sr = librosa.load(path, sr=48000, mono=True)
    return waveform.astype(np.float32), int(sr)


def load_visual(path: Path | None, sample_fps: float) -> Tuple[np.ndarray | None, np.ndarray | None, float]:
    if path is None:
        return None, None, 0.0
    if path.is_dir():
        from PIL import Image
        def natural_key(item: Path):
            return [int(part) if part.isdigit() else part.lower()
                    for part in re.split(r'(\d+)', item.name)]
        image_paths = sorted(
            (item for item in path.iterdir() if item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}),
            key=natural_key,
        )
        if not image_paths:
            return None, None, 0.0
        frames = np.stack([np.asarray(Image.open(item).convert('RGB')) for item in image_paths])
        times = np.arange(len(frames), dtype=np.float32) / max(sample_fps, 1e-3)
        return frames, times, len(frames) / max(sample_fps, 1e-3)
    if path.suffix.lower() in {'.npy', '.npz'}:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = 'frames' if 'frames' in loaded.files else loaded.files[0]
            frames = loaded[key]
        else:
            frames = loaded
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError(f'Frame array {path} must have shape [T,H,W,C], got {frames.shape}.')
        frames = frames[..., :3].astype(np.uint8, copy=False)
        times = np.arange(len(frames), dtype=np.float32) / max(sample_fps, 1e-3)
        return frames, times, len(frames) / max(sample_fps, 1e-3)
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if math.isfinite(fps) and fps > 0 else 25.0
    stride = max(1, round(fps / max(sample_fps, 1e-3)))
    frames, times = [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(index / fps)
        index += 1
    cap.release()
    if not frames:
        return None, None, 0.0
    return np.stack(frames), np.asarray(times, dtype=np.float32), index / fps


def uniform_intervals(num_words: int, duration: float) -> List[Tuple[float, float]]:
    duration = max(float(duration), 0.1 * num_words)
    edges = np.linspace(0.0, duration, num_words + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(num_words)]


def resolve_hf_resource(root: Path, required_groups: Sequence[Sequence[str]], label: str) -> Path:
    """Resolve a HF directory, including cache snapshots nested below it."""
    root = root.expanduser()
    candidates = [root] if root.is_dir() else []
    if root.is_dir():
        candidates.extend(path.parent for path in root.rglob('config.json'))
        candidates.extend(path.parent for path in root.rglob('tokenizer_config.json'))
        candidates.extend(path.parent for path in root.rglob('vocab.json'))
        candidates.extend(path.parent for path in root.rglob('vocab.txt'))
    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if all(any((candidate / filename).exists() for filename in group)
               for group in required_groups):
            return candidate
    expected = [' or '.join(group) for group in required_groups]
    raise FileNotFoundError(
        f'{label} is incomplete under {root}. Required in one directory: {expected}'
    )


def load_laion_clap_local(checkpoint: Path, device: torch.device,
                          bert_path: Path, roberta_path: Path,
                          roberta_tokenizer_path: Path, bart_path: Path):
    """Load full LAION-CLAP strictly from local model directories."""
    from transformers import BartTokenizer, BertTokenizer, RobertaModel, RobertaTokenizer

    if not checkpoint.is_file():
        raise FileNotFoundError(f'CLAP checkpoint not found: {checkpoint}')
    bert_dir = resolve_hf_resource(bert_path, (('vocab.txt',),), 'BERT tokenizer')
    roberta_model_dir = resolve_hf_resource(
        roberta_path,
        (('config.json',), ('model.safetensors', 'pytorch_model.bin',
                            'model.safetensors.index.json', 'pytorch_model.bin.index.json')),
        'RoBERTa model',
    )
    try:
        roberta_tokenizer_dir = resolve_hf_resource(
            roberta_path, (('vocab.json',), ('merges.txt',)),
            'RoBERTa tokenizer',
        )
    except FileNotFoundError:
        roberta_tokenizer_dir = resolve_hf_resource(
            roberta_tokenizer_path,
            (('vocab.json',), ('merges.txt',)),
            'RoBERTa tokenizer fallback',
        )
        print(f'[models] roberta-base tokenizer incomplete; using tokenizer from {roberta_tokenizer_dir}')
    bart_dir = resolve_hf_resource(
        bart_path, (('vocab.json',), ('merges.txt',)),
        'BART tokenizer',
    )

    sentinel = object()
    patched = []

    def patch_class(cls, name: str, value) -> None:
        previous = cls.__dict__.get(name, sentinel)
        patched.append((cls, name, previous))
        setattr(cls, name, value)

    def redirect_loader(original_loader, local_path: Path):
        def load(cls, model_id, *args, **kwargs):
            kwargs['local_files_only'] = True
            return original_loader(str(local_path), *args, **kwargs)
        return classmethod(load)

    resources = (
        (BertTokenizer, bert_dir),
        (RobertaTokenizer, roberta_tokenizer_dir),
        (BartTokenizer, bart_dir),
        (RobertaModel, roberta_model_dir),
    )
    for cls, local_path in resources:
        patch_class(cls, 'from_pretrained', redirect_loader(cls.from_pretrained, local_path))
    try:
        import laion_clap
        try:
            model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        except TypeError:  # compatibility with older laion-clap releases
            model = laion_clap.CLAP_Module(enable_fusion=False)
            model.model.to(device)
    finally:
        for cls, name, previous in reversed(patched):
            if previous is sentinel:
                delattr(cls, name)
            else:
                setattr(cls, name, previous)
    model.load_ckpt(str(checkpoint), verbose=False)
    return model


class Extractors:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection
        self.device = torch.device(args.device)
        self.batch_size = args.media_batch_size
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.deberta_model, use_fast=True)
        if not self.text_tokenizer.is_fast:
            raise ValueError('DeBERTa extraction requires a fast tokenizer for word_ids().')
        self.text_model = AutoModel.from_pretrained(args.deberta_model).to(self.device).eval()
        self.clip_processor = AutoProcessor.from_pretrained(args.clip_model)
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model).to(self.device).eval()
        self.text_dim = int(self.text_model.config.hidden_size)
        self.visual_dim = int(self.clip_model.config.projection_dim)
        clap_path = Path(args.clap_model).expanduser()
        if clap_path.is_file():
            import importlib.util
            if importlib.util.find_spec('laion_clap') is None:
                raise ImportError(
                    'A .pt CLAP checkpoint requires the laion-clap package: pip install laion-clap'
                )
            self.clap_backend = 'laion_clap'
            self.clap_model = load_laion_clap_local(
                clap_path, self.device,
                Path(args.clap_bert_model),
                Path(args.clap_roberta_model),
                Path(args.clap_roberta_tokenizer),
                Path(args.clap_bart_model),
            )
            probe = np.zeros((1, 48000), dtype=np.float32)
            probe_embedding = self.clap_model.get_audio_embedding_from_data(x=probe, use_tensor=False)
            self.audio_dim = int(probe_embedding.shape[-1])
            self.clap_processor = None
        else:
            from transformers import ClapModel
            self.clap_backend = 'transformers'
            self.clap_processor = AutoProcessor.from_pretrained(args.clap_model)
            self.clap_model = ClapModel.from_pretrained(args.clap_model).to(self.device).eval()
            self.audio_dim = int(self.clap_model.config.audio_config.hidden_size)
        self.model_ids = {
            'text': args.deberta_model,
            'audio': args.clap_model,
            'visual': args.clip_model,
            'clap_bert': args.clap_bert_model,
            'clap_roberta': args.clap_roberta_model,
            'clap_roberta_tokenizer': args.clap_roberta_tokenizer,
            'clap_bart': args.clap_bart_model,
        }
        expected = {'text': 1024, 'audio': 512, 'visual': 768}
        actual = {'text': self.text_dim, 'audio': self.audio_dim, 'visual': self.visual_dim}
        if actual != expected:
            raise ValueError(f'Unexpected feature dimensions: expected={expected}, actual={actual}')
        print('[models] loaded locally; dimensions: text=1024, audio=512, visual=768')

    @torch.inference_mode()
    def text(self, words: Sequence[str]) -> np.ndarray:
        encoded = self.text_tokenizer(
            list(words), is_split_into_words=True, return_tensors='pt',
            truncation=True, max_length=512,
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.text_model(**model_inputs).last_hidden_state[0].float().cpu()
        result = torch.zeros(len(words), hidden.shape[-1])
        counts = torch.zeros(len(words), 1)
        for token_index, word_index in enumerate(word_ids):
            if word_index is not None and word_index < len(words):
                result[word_index] += hidden[token_index]
                counts[word_index] += 1
        return (result / counts.clamp_min(1)).numpy().astype(np.float32)

    @torch.inference_mode()
    def visual(self, frames: np.ndarray | None, frame_times: np.ndarray | None,
               intervals: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        if frames is None or frame_times is None:
            return np.zeros((len(intervals), self.visual_dim), np.float32), np.zeros(len(intervals), np.float32)
        chunks = []
        for start in range(0, len(frames), self.batch_size):
            inputs = self.clip_processor(images=list(frames[start:start + self.batch_size]), return_tensors='pt')
            outputs = self.clip_model(pixel_values=inputs['pixel_values'].to(self.device))
            chunks.append(outputs.image_embeds.float().cpu())
        frame_features = torch.cat(chunks).numpy()
        result = np.zeros((len(intervals), frame_features.shape[-1]), np.float32)
        valid = np.ones(len(intervals), np.float32)
        for i, (start, end) in enumerate(intervals):
            selected = np.flatnonzero((frame_times >= start) & (frame_times < end))
            if selected.size == 0:
                selected = np.asarray([int(np.abs(frame_times - (start + end) / 2).argmin())])
            result[i] = frame_features[selected].mean(axis=0)
        return result, valid

    @torch.inference_mode()
    def audio(self, waveform: np.ndarray | None, sr: int,
              intervals: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        if waveform is None:
            return np.zeros((len(intervals), self.audio_dim), np.float32), np.zeros(len(intervals), np.float32)
        windows = []
        valid = np.ones(len(intervals), np.float32)
        for start, end in intervals:
            lo, hi = max(0, int(start * sr)), min(len(waveform), int(end * sr))
            if hi <= lo:
                windows.append(np.zeros(max(1, int(0.1 * sr)), np.float32))
                valid[len(windows) - 1] = 0
            else:
                windows.append(waveform[lo:hi])
        chunks = []
        for start in range(0, len(windows), self.batch_size):
            window_batch = windows[start:start + self.batch_size]
            if self.clap_backend == 'laion_clap':
                target_length = max(len(window) for window in window_batch)
                repeated = []
                for window in window_batch:
                    repeats = max(1, math.ceil(target_length / max(len(window), 1)))
                    repeated.append(np.tile(window, repeats)[:target_length])
                audio_array = np.stack(repeated).astype(np.float32, copy=False)
                embedding = self.clap_model.get_audio_embedding_from_data(
                    x=audio_array, use_tensor=False,
                )
                chunks.append(np.asarray(embedding, dtype=np.float32))
            else:
                inputs = self.clap_processor(
                    audios=window_batch, sampling_rate=sr,
                    return_tensors='pt', padding=True,
                )
                model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.clap_model.audio_model(**model_inputs)
                chunks.append(output.pooler_output.float().cpu().numpy())
        result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        result[valid == 0] = 0
        return result, valid


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def build_sample(row: Dict[str, Any], args: argparse.Namespace, layout: MOSILayout, extractors: Extractors,
                 alignments: Dict[str, Any]) -> Dict[str, Any]:
    uid = sample_uid(row)
    words = row['text'].strip().split()
    if not words:
        raise ValueError(f'{uid} has empty text.')
    audio_path = layout.audio_path(row)
    visual_path = layout.visual_path(row)
    waveform, sr = load_audio(audio_path)
    frames, frame_times, video_duration = load_visual(visual_path, args.video_sample_fps)
    audio_duration = 0.0 if waveform is None else len(waveform) / sr
    if (visual_path is not None and frames is not None and
            (visual_path.is_dir() or visual_path.suffix.lower() in {'.npy', '.npz'}) and
            audio_duration > 0):
        frame_times = np.linspace(0.0, audio_duration, len(frames), endpoint=False, dtype=np.float32)
        video_duration = audio_duration
    intervals = row_alignment(row, alignments, words)
    if intervals is None:
        if not args.allow_uniform_alignment:
            raise ValueError(
                f'{uid} has no valid word timestamps. Provide --alignment-json or explicitly '
                'enable --allow-uniform-alignment.'
            )
        intervals = uniform_intervals(len(words), max(audio_duration, video_duration))
    text_features = extractors.text(words)
    visual_features, visual_valid = extractors.visual(frames, frame_times, intervals)
    audio_features, audio_valid = extractors.audio(waveform, sr, intervals)
    return {
        'id': uid,
        'words': words,
        'text': text_features,
        'audio': audio_features,
        'visual': visual_features,
        'text_valid': np.ones(len(words), np.float32),
        'audio_valid': audio_valid,
        'visual_valid': visual_valid,
        'label': np.float32(row['label']),
        'alignment': 'timestamps' if row_alignment(row, alignments, words) is not None else 'uniform',
        'feature_models': extractors.model_ids,
        'sources': {
            'audio': None if audio_path is None else str(audio_path),
            'visual': None if visual_path is None else str(visual_path),
        },
    }


def main() -> None:
    configure_quiet_output()
    args = parse_args()
    layout = MOSILayout(args.data_dir)
    rows = read_labels(layout.label_path)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    alignments = load_alignment_file(args.alignment_json)
    if args.check_layout:
        audio_paths = [layout.audio_path(row) for row in rows]
        visual_paths = [layout.visual_path(row) for row in rows]
        timestamp_count = sum(
            row_alignment(row, alignments, row['text'].strip().split()) is not None for row in rows
        )
        summary = {
            'data_root': str(layout.root), 'processed_dir': str(layout.processed),
            'label_csv': str(layout.label_path), 'samples': len(rows),
            'audio_found': sum(path is not None for path in audio_paths),
            'visual_found': sum(path is not None for path in visual_paths),
            'word_timestamps_found': timestamp_count,
            'first_audio': next((str(path) for path in audio_paths if path is not None), None),
            'first_visual': next((str(path) for path in visual_paths if path is not None), None),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    extractors = Extractors(args)
    dataset: Dict[str, List[Dict[str, Any]]] = {'train': [], 'valid': [], 'test': []}
    failures = []
    for index, row in enumerate(rows, 1):
        uid = sample_uid(row)
        cache_path = args.cache_dir / f'{uid}.pt'
        try:
            use_cache = cache_path.exists() and not args.overwrite_cache
            if use_cache:
                sample = torch.load(cache_path, map_location='cpu', weights_only=False)
                timestamp_now_available = row_alignment(row, alignments, row['text'].strip().split()) is not None
                cache_matches = sample.get('feature_models') == extractors.model_ids
                cache_matches = cache_matches and not (timestamp_now_available and sample.get('alignment') != 'timestamps')
                use_cache = cache_matches
            if not use_cache:
                sample = build_sample(row, args, layout, extractors, alignments)
                atomic_torch_save(sample, cache_path)
            split = canonical_split(row['mode'])
            if split not in dataset:
                raise ValueError(f'Unknown split {row["mode"]!r}.')
            dataset[split].append(sample)
            if index == 1 or index % max(args.progress_every, 1) == 0 or index == len(rows):
                print(f'[extract] {index}/{len(rows)}')
        except Exception as exc:
            failures.append((uid, str(exc)))
            print(f'[{index}/{len(rows)}] ERROR {uid}: {exc}')
    if failures:
        preview = '\n'.join(f'  {uid}: {message}' for uid, message in failures[:10])
        raise RuntimeError(f'Feature extraction failed for {len(failures)} samples:\n{preview}')
    metadata = {
        'format': 'bhgm_sent_crc_mosi_v1',
        'models': extractors.model_ids,
        'dimensions': {'text': extractors.text_dim, 'audio': extractors.audio_dim,
                       'visual': extractors.visual_dim},
        'splits': {key: len(value) for key, value in dataset.items()},
        'data_root': str(layout.root),
        'processed_dir': str(layout.processed),
    }
    dataset['_metadata'] = metadata  # ignored by MOSIDataset
    atomic_torch_save(dataset, args.output)
    print(f'[done] splits={metadata["splits"]}; dimensions={metadata["dimensions"]}')
    print(f'[done] saved={args.output.resolve()}')


if __name__ == '__main__':
    main()
