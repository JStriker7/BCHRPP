"""Extract CLAP + CLIP features for CFIv2 and UDIVA personality data.

CFIv2 standard annotation/transcription pickle files are discovered
automatically.  UDIVA is session/participant/view structured, so a normalized
CSV manifest is preferred to prevent assigning one participant's labels to the
other participant or camera view.

Normalized manifest columns:
  id, split, video_path, audio_path(optional), text(optional),
  openness, conscientiousness, extraversion, agreeableness, neuroticism
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import torch

from extract_mosi_features import atomic_torch_save, configure_quiet_output
from extract_mosi_reference_av import ReferenceAVExtractor


SOURCE_ROOT = Path('/mnt/data8tb/wxr/MMML/data')
OUTPUT_ROOT = Path('/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/data')
TRAITS = (
    'openness', 'conscientiousness', 'extraversion',
    'agreeableness', 'neuroticism',
)
TRAIT_ALIASES = {
    'openness': ('openness', 'open', 'o'),
    'conscientiousness': ('conscientiousness', 'conscientious', 'c'),
    'extraversion': ('extraversion', 'extroversion', 'extravert', 'e'),
    'agreeableness': ('agreeableness', 'agreeable', 'a'),
    # CFIv2's field is historically named neuroticism although the released
    # score is commonly described as emotional stability.  Values are kept in
    # the dataset's original direction; no silent inversion is performed.
    'neuroticism': (
        'neuroticism', 'neurotisicm', 'emotional_stability',
        'emotion_stability', 'stability', 'n',
    ),
}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset', type=str.lower,
                        choices=('cfiv2', 'udiva'), required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--manifest', type=Path,
                        help='Optional normalized CSV manifest; recommended for UDIVA.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--clip-model', type=Path,
                        default=Path('/mnt/data26t/model/ViT-L-14-336px.pt'))
    parser.add_argument('--clap-model', type=Path,
                        default=Path('/mnt/data26t/model/clap-htsat-fused'))
    parser.add_argument('--audio-backbone', choices=('clap', 'wav2vec2', 'hubert'),
                        default='clap')
    parser.add_argument('--wav2vec2-model', type=Path,
                        default=Path('/mnt/data26t/model/wav2vec2-large-960h'))
    parser.add_argument('--hubert-model', type=Path,
                        default=Path('/mnt/data26t/model/hubert-large-ll60k'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frame-batch-size', type=int, default=64)
    parser.add_argument('--max-frames', type=int, default=32,
                        help='Uniformly sampled video frames before CLIP extraction.')
    parser.add_argument('--max-audio-steps', type=int, default=64)
    parser.add_argument('--max-visual-steps', type=int, default=32)
    parser.add_argument(
        '--affect-preserving', action='store_true',
        help=(
            'Use concatenated full-frame and detected-face CLIP features in '
            'a separate output/cache; legacy personality features stay intact.'
        ),
    )
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--download-missing', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--progress-every', type=int, default=25)
    parser.add_argument('--check-layout', action='store_true')
    args = parser.parse_args()

    name = 'CFIv2' if args.dataset == 'cfiv2' else 'UDIVA'
    args.data_dir = args.data_dir or SOURCE_ROOT / name
    output_name = (
        f'{args.dataset}_personality_affect_fullface_av.pt'
        if args.affect_preserving
        else f'{args.dataset}_personality_av.pt'
    )
    cache_name = (
        'personality_affect_fullface_cache'
        if args.affect_preserving else 'personality_av_cache'
    )
    args.output = args.output or OUTPUT_ROOT / name / 'processed' / output_name
    args.cache_dir = args.cache_dir or OUTPUT_ROOT / name / 'processed' / cache_name
    if args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    # Required by the shared reference extractor.
    args.independent_clocks = True
    args.reuse_visual_data = None
    args.visual_view = 'full-face' if args.affect_preserving else 'full'
    args.face_padding = 0.20
    args.preserve_native_frames = True
    return args


def canonical_split(value: str) -> str:
    text = str(value).strip().lower()
    if text in {'train', 'training', 'trn'}:
        return 'train'
    if text in {'valid', 'validation', 'val', 'dev', 'development'}:
        return 'valid'
    if text in {'test', 'testing', 'tst'}:
        return 'test'
    raise ValueError(f'Unknown split name: {value!r}')


def infer_split(path: Path) -> str:
    lowered = '/'.join(part.lower() for part in path.parts)
    for token, result in (
        ('training', 'train'), ('train', 'train'),
        ('validation', 'valid'), ('valid', 'valid'), ('dev', 'valid'),
        ('testing', 'test'), ('test', 'test'),
    ):
        if token in lowered:
            return result
    raise ValueError(f'Cannot infer train/valid/test split from {path}')


def load_pickle(path: Path) -> Any:
    with path.open('rb') as handle:
        try:
            return pickle.load(handle, encoding='latin1')
        except TypeError:
            return pickle.load(handle)


def normalized_keys(row: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in row:
        decoded = (
            key.decode('utf-8', errors='ignore')
            if isinstance(key, bytes) else str(key)
        )
        result[decoded.strip().lower()] = key
    return result


def first_column(keys: Mapping[str, Any], aliases: Iterable[str]) -> Any | None:
    return next((keys[name] for name in aliases if name in keys), None)


def build_media_index(root: Path, extensions: set[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
    return index


def resolve_media(value: str | None, root: Path, sample_id: str,
                  index: Mapping[str, Path]) -> Path | None:
    if value:
        supplied = Path(value)
        candidate = supplied if supplied.is_absolute() else root / supplied
        if candidate.is_file():
            return candidate
    for key in (sample_id.lower(), Path(sample_id).name.lower(),
                Path(sample_id).stem.lower()):
        if key in index:
            return index[key]
    return None


def read_text_value(value: Any, root: Path) -> str:
    text = '' if value is None else str(value).strip()
    if text:
        candidate = Path(text)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file() and candidate.suffix.lower() in {'.txt', '.text'}:
            return candidate.read_text(encoding='utf-8', errors='ignore').strip()
    return text


def read_manifest(path: Path, root: Path) -> List[Dict[str, Any]]:
    video_index = build_media_index(root, VIDEO_EXTENSIONS)
    audio_index = build_media_index(root, AUDIO_EXTENSIONS)
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'{path}: missing CSV header')
        for line_number, raw in enumerate(reader, 2):
            keys = normalized_keys(raw)
            id_col = first_column(keys, ('id', 'uid', 'video_id', 'clip_id',
                                         'sample_id', 'filename', 'file'))
            split_col = first_column(keys, ('split', 'mode', 'partition', 'set'))
            if id_col is None or split_col is None:
                raise ValueError(
                    f'{path}: requires id and split columns; found {reader.fieldnames}'
                )
            sample_id = str(raw[id_col]).strip()
            video_col = first_column(keys, ('video_path', 'video', 'path', 'file_path'))
            audio_col = first_column(keys, ('audio_path', 'audio', 'wav_path'))
            text_col = first_column(keys, ('text', 'transcript', 'transcription', 'sentence'))
            labels = []
            for trait in TRAITS:
                column = first_column(keys, TRAIT_ALIASES[trait])
                if column is None or str(raw[column]).strip() == '':
                    raise ValueError(
                        f'{path}:{line_number}: missing {trait}; columns={reader.fieldnames}'
                    )
                labels.append(float(raw[column]))
            interview_col = first_column(keys, ('interview', 'invite', 'job_interview'))
            if interview_col is not None and str(raw[interview_col]).strip() != '':
                labels.append(float(raw[interview_col]))
            video = resolve_media(
                raw.get(video_col) if video_col else None, root, sample_id,
                video_index,
            )
            audio = resolve_media(
                raw.get(audio_col) if audio_col else None, root, sample_id,
                audio_index,
            )
            rows.append({
                'id': sample_id,
                'split': canonical_split(raw[split_col]),
                'video_path': video,
                'audio_path': audio,
                'text': read_text_value(raw.get(text_col) if text_col else '', root),
                'labels': np.asarray(labels, dtype=np.float32),
            })
    return rows


def normalize_transcriptions(obj: Any) -> Dict[str, str]:
    if not isinstance(obj, Mapping):
        return {}
    for key in ('transcription', 'transcriptions', 'text', 'texts'):
        if key in obj and isinstance(obj[key], Mapping):
            obj = obj[key]
            break
    result = {}
    for key, value in obj.items():
        if isinstance(key, bytes):
            key = key.decode('utf-8', errors='ignore')
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        if isinstance(value, str):
            result[str(key)] = value
    return result


def normalize_cfi_annotations(obj: Any) -> Dict[str, np.ndarray]:
    if not isinstance(obj, Mapping):
        raise ValueError('CFIv2 annotation pickle must contain a dictionary.')
    keys = normalized_keys(obj)
    trait_maps: Dict[str, Mapping[str, Any]] = {}
    for trait in TRAITS:
        source = first_column(keys, TRAIT_ALIASES[trait])
        if source is None or not isinstance(obj[source], Mapping):
            raise ValueError(f'CFIv2 annotations missing mapping for {trait}.')
        trait_maps[trait] = {
            (key.decode('utf-8', errors='ignore') if isinstance(key, bytes)
             else str(key)): value
            for key, value in obj[source].items()
        }
    interview_source = first_column(keys, ('interview', 'invite', 'job_interview'))
    raw_interview_map = (
        obj[interview_source]
        if interview_source is not None and isinstance(obj[interview_source], Mapping)
        else None
    )
    interview_map = None if raw_interview_map is None else {
        (key.decode('utf-8', errors='ignore') if isinstance(key, bytes)
         else str(key)): value
        for key, value in raw_interview_map.items()
    }
    ids = set.intersection(*(set(values) for values in trait_maps.values()))
    # Keep a rectangular target matrix.  If interview labels are present,
    # retain only samples carrying that sixth target rather than producing a
    # mixture of five- and six-dimensional labels.
    if interview_map is not None:
        ids &= set(interview_map)
    result = {}
    for sample_id in ids:
        label = [float(trait_maps[trait][sample_id]) for trait in TRAITS]
        if interview_map is not None:
            label.append(float(interview_map[sample_id]))
        result[sample_id] = np.asarray(label, dtype=np.float32)
    return result


def discover_cfiv2(root: Path) -> List[Dict[str, Any]]:
    annotation_files = [
        path for path in root.rglob('*.pkl')
        if 'annot' in path.name.lower() and 'pair' not in path.name.lower()
    ]
    if not annotation_files:
        raise FileNotFoundError(
            f'No CFIv2 annotation *.pkl found under {root}. '
            'Pass --manifest with a normalized CSV if your copy was converted.'
        )
    transcription_files = [
        path for path in root.rglob('*.pkl')
        if 'transcript' in path.name.lower()
    ]
    transcripts_by_split: Dict[str, Dict[str, str]] = {}
    for path in transcription_files:
        try:
            transcripts_by_split.setdefault(infer_split(path), {}).update(
                normalize_transcriptions(load_pickle(path))
            )
        except ValueError:
            continue
    video_index = build_media_index(root, VIDEO_EXTENSIONS)
    rows: List[Dict[str, Any]] = []
    for path in sorted(annotation_files):
        split = infer_split(path)
        annotations = normalize_cfi_annotations(load_pickle(path))
        transcripts = transcripts_by_split.get(split, {})
        for sample_id, labels in annotations.items():
            video = resolve_media(None, root, sample_id, video_index)
            text = transcripts.get(sample_id, transcripts.get(Path(sample_id).name, ''))
            rows.append({
                'id': sample_id, 'split': split, 'video_path': video,
                'audio_path': None, 'text': text, 'labels': labels,
            })
    return rows


def discover_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], str]:
    if args.manifest is not None:
        if not args.manifest.is_file():
            raise FileNotFoundError(f'Manifest not found: {args.manifest}')
        return read_manifest(args.manifest, args.data_dir), str(args.manifest)
    candidates = []
    for name in ('personality_manifest.csv', 'manifest.csv', 'labels.csv', 'label.csv'):
        candidates.extend(args.data_dir.rglob(name))
    for path in candidates:
        try:
            return read_manifest(path, args.data_dir), str(path)
        except (ValueError, KeyError):
            continue
    if args.dataset == 'cfiv2':
        return discover_cfiv2(args.data_dir), 'standard CFIv2 annotation/transcription pickles'
    raise FileNotFoundError(
        'UDIVA requires a participant-safe normalized manifest because labels '
        'are participant-level while recordings contain multiple people/views. '
        'Create personality_manifest.csv under the dataset root with columns: '
        'id,split,video_path,audio_path(optional),text(optional),openness,'
        'conscientiousness,extraversion,agreeableness,neuroticism; then rerun.'
    )


def load_sampled_frames(video_path: Path, max_frames: int) -> np.ndarray:
    try:
        from decord import VideoReader
        from PIL import Image
        reader = VideoReader(str(video_path))
        if len(reader) == 0:
            raise ValueError(f'{video_path}: zero frames')
        indices = np.unique(np.linspace(
            0, len(reader) - 1, min(max_frames, len(reader)), dtype=np.int64,
        ))
        frames = reader.get_batch(indices.tolist()).asnumpy()
        return np.stack([
            np.asarray(Image.fromarray(frame).resize((224, 224), Image.BILINEAR))
            for frame in frames
        ])
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            cap.release()
            raise ValueError(f'{video_path}: zero frames')
        indices = np.unique(np.linspace(
            0, count - 1, min(max_frames, count), dtype=np.int64,
        ))
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (224, 224)))
        cap.release()
        if not frames:
            raise ValueError(f'{video_path}: failed to decode sampled frames')
        return np.stack(frames)


def safe_cache_name(sample_id: str) -> str:
    readable = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(sample_id).stem)[:80]
    digest = hashlib.sha1(sample_id.encode('utf-8')).hexdigest()[:10]
    return f'{readable}_{digest}'


def ensure_wav(row: Mapping[str, Any], args: argparse.Namespace) -> Path:
    audio = row.get('audio_path')
    if isinstance(audio, Path) and audio.is_file() and audio.suffix.lower() == '.wav':
        return audio
    source = audio if isinstance(audio, Path) and audio.is_file() else row['video_path']
    wav = args.cache_dir / 'audio' / f'{safe_cache_name(str(row["id"]))}.wav'
    if wav.is_file() and not args.overwrite_cache:
        return wav
    wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.ffmpeg, '-y', '-v', 'error', '-i', str(source), '-vn',
        '-ac', '1', '-ar', '16000', str(wav),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not wav.is_file():
        raise RuntimeError(
            f'ffmpeg audio extraction failed for {source}: {result.stderr.strip()}'
        )
    return wav


def layout_report(rows: List[Dict[str, Any]], source: str) -> None:
    if not rows:
        raise ValueError(f'No samples discovered from {source}')
    split_counts = {
        split: sum(row['split'] == split for row in rows)
        for split in ('train', 'valid', 'test')
    }
    missing_video = [row['id'] for row in rows if not isinstance(row['video_path'], Path)]
    with_text = sum(bool(str(row.get('text', '')).strip()) for row in rows)
    label_widths = sorted({int(np.asarray(row['labels']).size) for row in rows})
    print(f'[layout] source={source}')
    print(
        f'[layout] samples={len(rows)}, splits={split_counts}, '
        f'video={len(rows)-len(missing_video)}/{len(rows)}, '
        f'text={with_text}/{len(rows)}, label_widths={label_widths}'
    )
    if rows and isinstance(rows[0]['video_path'], Path):
        print(f'[layout] first_video={rows[0]["video_path"]}')
    for index, trait in enumerate(TRAITS):
        values = np.asarray([row['labels'][index] for row in rows], dtype=np.float32)
        print(
            f'[labels] {trait}: min={values.min():.4f}, '
            f'max={values.max():.4f}, mean={values.mean():.4f}'
        )
    if len(label_widths) != 1:
        raise ValueError(
            f'Inconsistent target widths={label_widths}; every sample must '
            'contain the same ordered personality targets.'
        )
    all_labels = np.concatenate([
        np.asarray(row['labels'], dtype=np.float32).reshape(-1) for row in rows
    ])
    if all_labels.min() < 0.0 or all_labels.max() > 1.0:
        print(
            '[warning] labels are outside [0,1]. rg_train.py defaults to a '
            'sigmoid output and ACC=1-MAE; normalize the labels first or pass '
            '--regression-output-activation none explicitly.'
        )
    if missing_video:
        raise FileNotFoundError(
            f'Missing videos={len(missing_video)}; first={missing_video[:10]}'
        )
    missing_splits = [name for name, count in split_counts.items() if count == 0]
    if missing_splits:
        raise ValueError(f'Missing required split(s): {missing_splits}')


def main() -> None:
    configure_quiet_output()
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {args.data_dir}')
    rows, source = discover_rows(args)
    rows.sort(key=lambda row: (row['split'], str(row['id'])))
    layout_report(rows, source)
    if args.check_layout:
        print(f'[layout] {args.dataset.upper()} personality layout check passed')
        return
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    extractor = ReferenceAVExtractor(args)
    signature = {
        'clip': str(args.clip_model),
        'audio_backbone': args.audio_backbone,
        'audio_model': str(
            args.clap_model if args.audio_backbone == 'clap'
            else args.wav2vec2_model if args.audio_backbone == 'wav2vec2'
            else args.hubert_model
        ),
        'alignment': 'independent_clocks_v1',
        'max_frames': args.max_frames,
        'max_audio_steps': args.max_audio_steps,
        'max_visual_steps': args.max_visual_steps,
        'traits': TRAITS,
    }
    if args.affect_preserving:
        signature = {
            **signature,
            'visual_preprocessing': 'affect_preserving_v1',
            'visual_view': args.visual_view,
            'face_padding': args.face_padding,
        }
    dataset: Dict[str, List[Dict[str, Any]]] = {
        'train': [], 'valid': [], 'test': [],
    }
    failures = []
    for index, row in enumerate(rows, 1):
        sample_id = str(row['id'])
        words = str(row.get('text', '')).strip().split()
        if not words:
            words = ['[NO_TRANSCRIPT]']
        cache_path = args.cache_dir / 'samples' / f'{safe_cache_name(sample_id)}.pt'
        try:
            sample = None
            if cache_path.is_file() and not args.overwrite_cache:
                cached = torch.load(cache_path, map_location='cpu', weights_only=False)
                if (cached.get('_signature') == signature
                        and cached.get('words') == words
                        and np.allclose(np.asarray(cached.get('label')),
                                        np.asarray(row['labels']))):
                    sample = cached
            if sample is None:
                wav = ensure_wav(row, args)
                audio = extractor.audio(wav, len(words))
                frames = load_sampled_frames(row['video_path'], args.max_frames)
                visual = extractor.visual(frames, len(words))
                sample = {
                    'id': sample_id,
                    'words': words,
                    'text': np.zeros((len(words), 1), dtype=np.float32),
                    'audio': audio.astype(np.float32),
                    'visual': visual.astype(np.float32),
                    'text_valid': np.ones(len(words), dtype=np.float32),
                    'audio_valid': np.ones(audio.shape[0], dtype=np.float32),
                    'visual_valid': np.ones(visual.shape[0], dtype=np.float32),
                    'text_time': (
                        np.arange(len(words), dtype=np.float32) + 0.5
                    ) / len(words),
                    'audio_time': (
                        np.arange(audio.shape[0], dtype=np.float32) + 0.5
                    ) / audio.shape[0],
                    'visual_time': (
                        np.arange(visual.shape[0], dtype=np.float32) + 0.5
                    ) / visual.shape[0],
                    'label': np.asarray(row['labels'], dtype=np.float32),
                    '_signature': signature,
                }
                atomic_torch_save(sample, cache_path)
            dataset[row['split']].append(sample)
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                print(f'[extract-{args.dataset}-personality] {index}/{len(rows)}')
        except Exception as exc:
            failures.append((sample_id, str(exc)))
            print(f'[error] {sample_id}: {exc}')
    if failures:
        raise RuntimeError(f'Failed samples={len(failures)}; first={failures[:5]}')

    metadata = {
        'format': 'bhgm_personality_av_online_deberta_v1',
        'dataset': args.dataset.upper(),
        'source': source,
        'traits': [*TRAITS] + (
            ['interview'] if any(len(row['labels']) > 5 for row in rows) else []
        ),
        'label_note': (
            'Values are preserved in the source direction. CFIv2 field '
            'neuroticism is often described as emotional stability.'
        ),
        'models': signature,
        'dimensions': {
            'text': 'online-deberta-1024',
            'audio': extractor.audio_dim,
            'visual': extractor.visual_dim,
        },
        'splits': {name: len(samples) for name, samples in dataset.items()},
    }
    dataset['_metadata'] = metadata
    atomic_torch_save(dataset, args.output)
    print(f'[done] {json.dumps(metadata, ensure_ascii=False)}')
    print(f'[done] saved={args.output}')


if __name__ == '__main__':
    main()
