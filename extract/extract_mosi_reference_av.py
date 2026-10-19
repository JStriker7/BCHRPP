from __future__ import annotations

"""Reproduce the reference CLIP/CLAP word-level AV extraction pipeline.

Text is intentionally not embedded here. Raw words are saved and DeBERTa is
run/fine-tuned online by train.py --online-deberta.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import torch

from extract_mosi_features import (MOSILayout, atomic_torch_save, canonical_split,
                                   configure_quiet_output, read_labels,
                                   sample_uid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-name', default='MOSI',
                        help='Name stored in metadata/logs; extraction method is unchanged.')
    parser.add_argument('--data-dir', type=Path,
                        default=Path('/mnt/data8tb/wxr/MMML/data/MOSI/processed'))
    parser.add_argument('--output', type=Path,
                        default=Path('/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/data/MOSI/processed/mosi_reference_av.pt'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/data/MOSI/processed/reference_av_cache'))
    parser.add_argument('--clip-model', type=Path,
                        default=Path('/mnt/data26t/model/ViT-L-14-336px.pt'),
                        help='Exact OpenAI CLIP ViT-L/14@336 .pt used by the reference extractor.')
    parser.add_argument('--clap-model', type=Path,
                        default=Path('/mnt/data26t/model/clap-htsat-fused'),
                        help='Local Hugging Face laion/clap-htsat-fused directory.')
    parser.add_argument('--audio-backbone', choices=('clap', 'wav2vec2', 'hubert'), default='clap')
    parser.add_argument('--wav2vec2-model', type=Path,
                        default=Path('/mnt/data26t/model/wav2vec2-large-960h'),
                        help='Local Hugging Face Wav2Vec2 model directory.')
    parser.add_argument('--hubert-model', type=Path,
                        default=Path('/mnt/data26t/model/hubert-large-ll60k'),
                        help='Local Hugging Face HuBERT model directory.')
    parser.add_argument('--reuse-visual-data', type=Path,
                        help='Reuse visual/word features from an existing .pt and skip CLIP.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frame-batch-size', type=int, default=32)
    parser.add_argument('--independent-clocks', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Keep native audio/video sequences instead of resampling them to text units.')
    parser.add_argument('--max-audio-steps', type=int, default=64)
    parser.add_argument('--max-visual-steps', type=int, default=32)
    parser.add_argument(
        '--preserve-native-frames',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Keep decoded frame resolution/aspect ratio and let CLIP perform '
            'its own resize/crop; legacy default first forces 224x224.'
        ),
    )
    parser.add_argument(
        '--visual-view', choices=('full', 'face', 'full-face'), default='full',
        help=(
            'CLIP view: full frame, detected face, or concatenated full+face '
            'features. Face detection falls back to the full frame.'
        ),
    )
    parser.add_argument('--face-padding', type=float, default=0.20)
    parser.add_argument(
        '--text-units', choices=('whitespace', 'char'), default='whitespace',
        help=(
            'Units used to align audio/video features. English reference data '
            'uses whitespace words; Chinese data can use non-whitespace chars.'
        ),
    )
    parser.add_argument(
        '--text-feature-description', default='online-deberta-1024',
        help='Text encoder description stored in output metadata.',
    )
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--download-missing', action=argparse.BooleanOptionalAction, default=True,
                        help='Download the two exact reference checkpoints when absent.')
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--check-layout', action='store_true',
                        help='Validate labels/audio/raw video paths, then exit before loading models.')
    args = parser.parse_args()
    if not 0.0 <= args.face_padding <= 1.0:
        parser.error('--face-padding must be in [0, 1]')
    return args


def distribute_frames(features: np.ndarray, num_words: int) -> np.ndarray:
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f'Expected nonempty [T,D] frame features, got {features.shape}.')
    result = np.zeros((num_words, features.shape[1]), dtype=np.float32)
    counts = np.zeros(num_words, dtype=np.int32)
    for frame_index, feature in enumerate(features):
        word_index = min(int(frame_index * num_words / len(features)), num_words - 1)
        result[word_index] += feature
        counts[word_index] += 1
    result /= np.maximum(counts[:, None], 1)
    return result


def temporal_pool(features: np.ndarray, max_steps: int) -> np.ndarray:
    """Order-preserving segment means; does not align features to text."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f'Expected nonempty [T,D] features, got {features.shape}.')
    if max_steps <= 0 or len(features) <= max_steps:
        return features.astype(np.float32, copy=False)
    boundaries = np.linspace(0, len(features), max_steps + 1, dtype=np.int64)
    return np.stack([
        features[boundaries[i]:boundaries[i + 1]].mean(axis=0)
        for i in range(max_steps)
    ]).astype(np.float32)


class ReferenceAVExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(args.device)
        self.frame_batch_size = args.frame_batch_size
        self.audio_backbone = args.audio_backbone
        self.independent_clocks = args.independent_clocks
        self.max_audio_steps = args.max_audio_steps
        self.max_visual_steps = args.max_visual_steps
        self.visual_view = args.visual_view
        self.face_padding = float(args.face_padding)
        self.face_detector = None
        self.face_frames_total = 0
        self.face_frames_detected = 0
        self.clip = None
        self.clip_preprocess = None
        if args.reuse_visual_data is None:
            import clip
            if not callable(getattr(clip, 'load', None)):
                raise RuntimeError(
                    "The installed 'clip' package is not OpenAI CLIP. "
                    "PyPI clip==0.2.0 does not provide clip.load(). "
                    "Use the environment that extracted MOSI/MOSEI successfully, "
                    "or uninstall it and install https://github.com/openai/CLIP."
                )
            if args.clip_model.is_file():
                clip_source = str(args.clip_model)
            elif args.download_missing:
                args.clip_model.parent.mkdir(parents=True, exist_ok=True)
                print(f'[models] downloading OpenAI ViT-L/14@336px -> {args.clip_model.parent}')
                clip_source = 'ViT-L/14@336px'
            else:
                raise FileNotFoundError(
                    f'Exact OpenAI CLIP ViT-L/14@336px checkpoint is required: {args.clip_model}'
                )
            self.clip, self.clip_preprocess = clip.load(
                clip_source, device=self.device, download_root=str(args.clip_model.parent),
            )
            self.clip.eval()
            self.visual_dim = 1536 if self.visual_view == 'full-face' else 768
            if self.visual_view != 'full':
                import cv2
                cascade_path = (
                    Path(cv2.data.haarcascades)
                    / 'haarcascade_frontalface_default.xml'
                )
                detector = cv2.CascadeClassifier(str(cascade_path))
                if detector.empty():
                    raise RuntimeError(
                        f'Failed to load OpenCV face detector: {cascade_path}'
                    )
                self.face_detector = detector
        else:
            # Reused data are validated against their stored width later.
            self.visual_dim = 1536 if self.visual_view == 'full-face' else 768
        if self.audio_backbone == 'clap':
            from transformers import ClapModel, ClapProcessor
            if not args.clap_model.is_dir() and args.download_missing:
                from huggingface_hub import snapshot_download
                print(f'[models] downloading laion/clap-htsat-fused -> {args.clap_model}')
                snapshot_download(
                    repo_id='laion/clap-htsat-fused', local_dir=str(args.clap_model),
                )
            if not args.clap_model.is_dir():
                raise FileNotFoundError(
                    f'Exact Hugging Face laion/clap-htsat-fused is required: {args.clap_model}'
                )
            try:
                self.clap = ClapModel.from_pretrained(
                    str(args.clap_model), local_files_only=True,
                ).to(self.device).eval()
                self.clap_processor = ClapProcessor.from_pretrained(
                    str(args.clap_model), local_files_only=True,
                )
            except OSError:
                if not args.download_missing:
                    raise
                from huggingface_hub import snapshot_download
                print(f'[models] repairing incomplete clap-htsat-fused -> {args.clap_model}')
                snapshot_download(
                    repo_id='laion/clap-htsat-fused', local_dir=str(args.clap_model),
                )
                self.clap = ClapModel.from_pretrained(
                    str(args.clap_model), local_files_only=True,
                ).to(self.device).eval()
                self.clap_processor = ClapProcessor.from_pretrained(
                    str(args.clap_model), local_files_only=True,
                )
            self.audio_dim = 768
        elif self.audio_backbone == 'wav2vec2':
            from transformers import AutoFeatureExtractor, Wav2Vec2Model
            if not args.wav2vec2_model.is_dir():
                raise FileNotFoundError(
                    f'Local Wav2Vec2 model directory is required: {args.wav2vec2_model}'
                )
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                str(args.wav2vec2_model), local_files_only=True,
            ).to(self.device).eval()
            self.wav2vec2_processor = AutoFeatureExtractor.from_pretrained(
                str(args.wav2vec2_model), local_files_only=True,
            )
            self.audio_dim = int(self.wav2vec2.config.hidden_size)
        else:
            from transformers import AutoFeatureExtractor, HubertModel
            if not args.hubert_model.is_dir():
                raise FileNotFoundError(
                    f'Local HuBERT model directory is required: {args.hubert_model}'
                )
            self.hubert = HubertModel.from_pretrained(
                str(args.hubert_model), local_files_only=True,
            ).to(self.device).eval()
            self.hubert_processor = AutoFeatureExtractor.from_pretrained(
                str(args.hubert_model), local_files_only=True,
            )
            self.audio_dim = int(self.hubert.config.hidden_size)

    @torch.inference_mode()
    def _face_views(self, frames: np.ndarray) -> List[np.ndarray]:
        if self.face_detector is None:
            raise RuntimeError('Face view requested without a face detector.')
        import cv2

        results: List[np.ndarray] = []
        for frame in frames:
            frame = np.asarray(frame)
            self.face_frames_total += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            minimum = max(24, int(min(frame.shape[:2]) * 0.12))
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(minimum, minimum),
            )
            if len(faces) == 0:
                results.append(frame)
                continue
            x, y, width, height = max(
                faces, key=lambda item: int(item[2]) * int(item[3])
            )
            pad_x = int(width * self.face_padding)
            pad_y = int(height * self.face_padding)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(frame.shape[1], x + width + pad_x)
            y1 = min(frame.shape[0], y + height + pad_y)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                results.append(frame)
                continue
            self.face_frames_detected += 1
            results.append(crop)
        return results

    def _clip_views(self, frames: Any) -> np.ndarray:
        from PIL import Image
        processed = [
            self.clip_preprocess(Image.fromarray(np.asarray(frame)))
            for frame in frames
        ]
        chunks = []
        for start in range(0, len(processed), self.frame_batch_size):
            batch = torch.stack(
                processed[start:start + self.frame_batch_size]
            ).to(self.device)
            chunks.append(self.clip.encode_image(batch).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.inference_mode()
    def visual(self, frames: np.ndarray, num_words: int) -> np.ndarray:
        if self.clip is None or self.clip_preprocess is None:
            raise RuntimeError('CLIP is disabled because --reuse-visual-data was selected.')
        full_features = (
            self._clip_views(frames)
            if self.visual_view in {'full', 'full-face'} else None
        )
        face_features = (
            self._clip_views(self._face_views(frames))
            if self.visual_view in {'face', 'full-face'} else None
        )
        if self.visual_view == 'full':
            features = full_features
        elif self.visual_view == 'face':
            features = face_features
        else:
            features = np.concatenate((full_features, face_features), axis=-1)
        if features.shape[1] != self.visual_dim:
            raise ValueError(
                f'CLIP visual output must be {self.visual_dim}d, '
                f'got {features.shape}.'
            )
        return (temporal_pool(features, self.max_visual_steps)
                if self.independent_clocks else distribute_frames(features, num_words))

    @torch.inference_mode()
    def audio(self, audio_path: Path, num_words: int) -> np.ndarray:
        waveform_16k, _ = librosa.load(audio_path, sr=16000, mono=True)
        if self.audio_backbone == 'wav2vec2':
            inputs = self.wav2vec2_processor(
                waveform_16k.astype(np.float32), sampling_rate=16000,
                return_tensors='pt',
            )
            # This is one unpadded utterance. Passing only input_values also
            # matches the recommended behavior for group-norm 960h checkpoints.
            input_values = inputs['input_values'].to(self.device)
            hidden = self.wav2vec2(input_values=input_values).last_hidden_state
            features = hidden.squeeze(0).float().cpu().numpy()
            if features.shape[1] != self.audio_dim:
                raise ValueError(
                    f'Wav2Vec2 hidden feature must be {self.audio_dim}d, got {features.shape}.'
                )
            return (temporal_pool(features, self.max_audio_steps)
                    if self.independent_clocks else distribute_frames(features, num_words))
        if self.audio_backbone == 'hubert':
            inputs = self.hubert_processor(
                waveform_16k.astype(np.float32), sampling_rate=16000,
                return_tensors='pt',
            )
            input_values = inputs['input_values'].to(self.device)
            hidden = self.hubert(input_values=input_values).last_hidden_state
            features = hidden.squeeze(0).float().cpu().numpy()
            if features.shape[1] != self.audio_dim:
                raise ValueError(
                    f'HuBERT hidden feature must be {self.audio_dim}d, got {features.shape}.'
                )
            return (temporal_pool(features, self.max_audio_steps)
                    if self.independent_clocks else distribute_frames(features, num_words))
        waveform_48k = librosa.resample(
            y=waveform_16k.astype(np.float64), orig_sr=16000, target_sr=48000,
        ).astype(np.float32)
        inputs = self.clap_processor(
            audios=waveform_48k, sampling_rate=48000, return_tensors='pt',
        )
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.clap.audio_model(**model_inputs)
        hidden = output.last_hidden_state
        if hidden.dim() != 4:
            raise ValueError(f'Reference expects CLAP hidden [B,D,F,T], got {tuple(hidden.shape)}.')
        features = hidden.mean(dim=2).squeeze(0).permute(1, 0).float().cpu().numpy()
        if features.shape[1] != 768:
            raise ValueError(f'Reference CLAP hidden feature must be 768d, got {features.shape}.')
        return (temporal_pool(features, self.max_audio_steps)
                if self.independent_clocks else distribute_frames(features, num_words))


def resolve_reference_video(layout: MOSILayout, row: Dict[str, Any]) -> Path:
    """Resolve raw clips without changing the reference decoding procedure."""
    video_id = row['video_id'].strip()
    clip_id = row['clip_id'].strip()
    uid = sample_uid(row)
    raw_root = layout.root / 'Raw'
    split_names = layout.split_names(row)
    candidates = [
        raw_root / video_id / f'{clip_id}.mp4',
        raw_root / f'{uid}.mp4',
        raw_root / f'{video_id}[{clip_id}].mp4',
    ]
    for split in split_names:
        split_root = raw_root / split
        candidates.extend([
            split_root / video_id / f'{clip_id}.mp4',
            split_root / f'{uid}.mp4',
            split_root / f'{video_id}[{clip_id}].mp4',
            split_root / video_id / f'{uid}.mp4',
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = '\n  '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Raw clip not found; checked:\n  {checked}')


def load_reference_frames(layout: MOSILayout, row: Dict[str, Any],
                          preserve_native: bool = False) -> np.ndarray:
    """Mirror load_mosi_origin_dataset.py: all Raw MP4 frames resized to 224."""
    video_path = resolve_reference_video(layout, row)
    try:
        from decord import VideoReader
        from PIL import Image
        reader = VideoReader(str(video_path))
        if len(reader) == 0:
            raise ValueError(f'{video_path}: zero frames')
        frames = reader.get_batch(list(range(len(reader)))).asnumpy()
        if preserve_native:
            return frames
        return np.stack([
            np.asarray(Image.fromarray(frame).resize((224, 224), Image.BILINEAR))
            for frame in frames
        ])
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                frame if preserve_native else cv2.resize(frame, (224, 224))
            )
        cap.release()
        if not frames:
            raise ValueError(f'{video_path}: zero frames')
        return np.stack(frames)

def build_sample(row: Dict[str, Any], layout: MOSILayout,
                 extractor: ReferenceAVExtractor,
                 reused: Dict[str, Any] | None = None,
                 text_units: str = 'whitespace',
                 preserve_native_frames: bool = False) -> Dict[str, Any]:
    uid = sample_uid(row)
    sentence = row['text'].strip()
    if text_units == 'char':
        words = [character for character in sentence if not character.isspace()]
    else:
        words = sentence.split()
    if not words:
        raise ValueError(f'{uid}: empty text.')
    audio_path = layout.audio_path(row)
    if audio_path is None:
        raise FileNotFoundError(f'{uid}: audio not found')
    audio = extractor.audio(audio_path, len(words))
    if reused is None:
        frames = load_reference_frames(
            layout, row, preserve_native=preserve_native_frames,
        )
        visual = extractor.visual(frames, len(words))
    else:
        reused_words = list(reused.get('words', []))
        if reused_words != words:
            raise ValueError(
                f'{uid}: reused words differ from label.csv: '
                f'{len(reused_words)} vs {len(words)}.'
            )
        reused_visual = reused['visual']
        if isinstance(reused_visual, torch.Tensor):
            reused_visual = reused_visual.detach().cpu().numpy()
        visual = np.asarray(reused_visual, dtype=np.float32)
        expected_visual_steps = visual.shape[0] if extractor.independent_clocks else len(words)
        if visual.shape != (expected_visual_steps, extractor.visual_dim):
            raise ValueError(
                f'{uid}: reused visual feature dimension must be '
                f'{extractor.visual_dim}, got {visual.shape}.'
            )
    return {
        'id': uid,
        'words': words,
        'text': np.zeros((len(words), 1), dtype=np.float32),
        'audio': audio.astype(np.float32),
        'visual': visual.astype(np.float32),
        'text_valid': np.ones(len(words), dtype=np.float32),
        'audio_valid': np.ones(audio.shape[0], dtype=np.float32),
        'visual_valid': np.ones(visual.shape[0], dtype=np.float32),
        'text_time': (np.arange(len(words), dtype=np.float32) + 0.5) / len(words),
        'audio_time': (np.arange(audio.shape[0], dtype=np.float32) + 0.5) / audio.shape[0],
        'visual_time': (np.arange(visual.shape[0], dtype=np.float32) + 0.5) / visual.shape[0],
        'label': np.float32(row['label']),
    }


def main() -> None:
    configure_quiet_output()
    args = parse_args()
    layout = MOSILayout(args.data_dir)
    rows = read_labels(layout.label_path)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    reused_by_id: Dict[str, Dict[str, Any]] | None = None
    if args.reuse_visual_data is not None:
        if not args.reuse_visual_data.is_file():
            raise FileNotFoundError(f'Reused visual dataset not found: {args.reuse_visual_data}')
        reused_raw = torch.load(args.reuse_visual_data, map_location='cpu', weights_only=False)
        reused_by_id = {
            str(sample['id']): sample
            for split in ('train', 'valid', 'test')
            for sample in reused_raw.get(split, [])
        }
        print(f'[reuse] visual samples={len(reused_by_id)}, source={args.reuse_visual_data}')
    missing_video = []
    missing_audio = []
    first_video = None
    for row in rows:
        if reused_by_id is None:
            try:
                resolved = resolve_reference_video(layout, row)
                if first_video is None:
                    first_video = resolved
            except FileNotFoundError:
                missing_video.append(sample_uid(row))
        elif sample_uid(row) not in reused_by_id:
            missing_video.append(sample_uid(row))
        if layout.audio_path(row) is None:
            missing_audio.append(sample_uid(row))
    visual_label = 'reused_visual' if reused_by_id is not None else 'raw_video'
    print(f'[layout] {visual_label}={len(rows) - len(missing_video)}/{len(rows)}, '
          f'audio={len(rows) - len(missing_audio)}/{len(rows)}')
    if first_video is not None:
        print(f'[layout] first_raw_video={first_video}')
    if missing_video or missing_audio:
        raise FileNotFoundError(
            f'Incomplete media layout: missing_video={len(missing_video)} '
            f'(first={missing_video[:5]}), missing_audio={len(missing_audio)} '
            f'(first={missing_audio[:5]}).'
        )
    if args.check_layout:
        print(f'[layout] {args.dataset_name.upper()} layout check passed')
        return
    extractor = ReferenceAVExtractor(args)
    required_methods = ('audio',) if reused_by_id is not None else ('audio', 'visual')
    for method_name in required_methods:
        if not callable(getattr(extractor, method_name, None)):
            raise RuntimeError(f'Extractor self-check failed: missing method {method_name!r}.')
    print(f'[models] audio_backbone={args.audio_backbone}, audio_dim={extractor.audio_dim}')
    print('[models] extractor self-check passed')
    dataset: Dict[str, List[Dict[str, Any]]] = {'train': [], 'valid': [], 'test': []}
    failures = []
    if args.audio_backbone == 'clap':
        # Preserve the old signature so existing CLAP caches remain reusable.
        model_signature = {
            'clip': str(args.clip_model),
            'clap': str(args.clap_model),
            'method': 'reference_raw224_hidden_768_v2',
        }
    elif args.audio_backbone == 'wav2vec2':
        model_signature = {
            'clip': str(args.clip_model),
            'audio_backbone': 'wav2vec2',
            'audio_model': str(args.wav2vec2_model),
            'visual_source': str(args.reuse_visual_data) if args.reuse_visual_data else 'raw-video',
            'method': 'reference_raw224_wav2vec2_hidden_v1',
        }
    else:
        model_signature = {
            'clip': str(args.clip_model),
            'audio_backbone': 'hubert',
            'audio_model': str(args.hubert_model),
            'visual_source': str(args.reuse_visual_data) if args.reuse_visual_data else 'raw-video',
            'method': 'reference_raw224_hubert_hidden_v1',
        }
    # Keep the historical signature byte-for-byte compatible in default mode,
    # while ensuring enhanced visual caches can never be mistaken for it.
    if args.preserve_native_frames or args.visual_view != 'full':
        model_signature = {
            **model_signature,
            'visual_preprocessing': 'affect_preserving_v1',
            'preserve_native_frames': args.preserve_native_frames,
            'visual_view': args.visual_view,
            'face_padding': args.face_padding,
        }
    if args.independent_clocks:
        model_signature = {
            **model_signature,
            'alignment': 'independent_clocks_v1',
            'max_audio_steps': args.max_audio_steps,
            'max_visual_steps': args.max_visual_steps,
        }
    for index, row in enumerate(rows, 1):
        uid = sample_uid(row)
        cache_path = args.cache_dir / f'{uid}.pt'
        try:
            sample = None
            if cache_path.exists() and not args.overwrite_cache:
                cached = torch.load(cache_path, map_location='cpu', weights_only=False)
                if cached.get('_models') == model_signature:
                    sample = cached
            if sample is None:
                reused = reused_by_id.get(uid) if reused_by_id is not None else None
                sample = build_sample(
                    row, layout, extractor, reused=reused,
                    text_units=args.text_units,
                    preserve_native_frames=args.preserve_native_frames,
                )
                sample['_models'] = model_signature
                atomic_torch_save(sample, cache_path)
            dataset[canonical_split(row['mode'])].append(sample)
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                print(f'[extract-{args.dataset_name.lower()}-reference-av] {index}/{len(rows)}')
        except Exception as exc:
            failures.append((uid, str(exc)))
            print(f'[error] {uid}: {exc}')
    if failures:
        raise RuntimeError(f'Failed samples={len(failures)}; first={failures[:5]}')
    metadata = {
        'format': (
            'bhgm_reference_av_online_text_v1'
            if args.text_units == 'char'
            else 'bhgm_reference_av_online_deberta_v1'
        ),
        'dataset': args.dataset_name.upper(),
        'models': model_signature,
        'dimensions': {'text': args.text_feature_description,
                       'audio': extractor.audio_dim,
                       'visual': extractor.visual_dim},
        'visual_preprocessing': {
            'preserve_native_frames': args.preserve_native_frames,
            'visual_view': args.visual_view,
            'face_padding': args.face_padding,
            'face_frames_detected': extractor.face_frames_detected,
            'face_frames_total': extractor.face_frames_total,
        },
        'independent_clocks': args.independent_clocks,
        'splits': {key: len(value) for key, value in dataset.items()},
    }
    dataset['_metadata'] = metadata
    atomic_torch_save(dataset, args.output)
    print(f'[done] {json.dumps(metadata, ensure_ascii=False)}')
    print(f'[done] saved={args.output}')


if __name__ == '__main__':
    main()
