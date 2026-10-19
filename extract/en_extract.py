"""Unified MOSI/MOSEI feature extraction with DeBERTa + CLAP + CLIP."""

from __future__ import annotations

import argparse
import sys

import extract_mosi_reference_av


SOURCE_ROOT = '/mnt/data8tb/wxr/MMML/data'
OUTPUT_ROOT = '/mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/data'


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=True, allow_abbrev=False,
    )
    parser.add_argument('--dataset', type=str.lower, choices=('mosi', 'mosei'),
                        required=True)
    parser.add_argument(
        '--affect-preserving', action='store_true',
        help=(
            'Write a separate native-resolution, independent-clock '
            'full-frame+face CLIP feature set.'
        ),
    )
    args, remaining = parser.parse_known_args()
    name = args.dataset.upper()
    output_name = (
        f'{args.dataset}_affect_fullface_av.pt'
        if args.affect_preserving
        else f'{args.dataset}_reference_av.pt'
    )
    cache_name = (
        'affect_fullface_av_cache'
        if args.affect_preserving else 'reference_av_cache'
    )
    fixed = [
        '--dataset-name', name,
        '--data-dir', f'{SOURCE_ROOT}/{name}/processed',
        '--output', f'{OUTPUT_ROOT}/{name}/processed/{output_name}',
        '--cache-dir', f'{OUTPUT_ROOT}/{name}/processed/{cache_name}',
        '--clip-model', '/mnt/data26t/model/ViT-L-14-336px.pt',
        '--clap-model', '/mnt/data26t/model/clap-htsat-fused',
        '--audio-backbone', 'clap',
        '--text-units', 'whitespace',
        '--text-feature-description', 'online-deberta-1024',
    ]
    if args.affect_preserving:
        fixed.extend((
            '--independent-clocks',
            '--preserve-native-frames',
            '--visual-view', 'full-face',
            '--face-padding', '0.20',
        ))
    sys.argv = [sys.argv[0], *fixed, *remaining]
    extract_mosi_reference_av.main()


if __name__ == '__main__':
    main()
