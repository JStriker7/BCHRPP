"""Unified SIMS/SIMS2 extraction with raw Chinese text + CLAP + CLIP.

The established full-frame feature file remains the default.  Pass
``--affect-preserving`` to create a separate native-resolution full-frame +
face feature file matching the current MOSI visual preprocessing.  Keeping a
separate filename/cache prevents feature experiments from overwriting the
reproducible Chinese baseline.
"""

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
    parser.add_argument('--dataset', type=str.lower, choices=('sims', 'sims2'),
                        required=True)
    parser.add_argument(
        '--affect-preserving', action='store_true',
        help=(
            'Write a separate native-resolution, independent-clock '
            'full-frame+face CLIP feature set matching the MOSI visual view.'
        ),
    )
    args, remaining = parser.parse_known_args()
    name = args.dataset.upper()
    # Keep the legacy filename and cache untouched unless the new visual view
    # is requested. Text is raw characters and is encoded online in ch_train.
    output_name = (
        f'{args.dataset}_affect_fullface_av.pt'
        if args.affect_preserving
        else f'{args.dataset}_roberta_clap_clip_independent.pt'
    )
    cache_name = (
        'affect_fullface_av_cache'
        if args.affect_preserving
        else 'roberta_clap_clip_independent_cache'
    )
    fixed = [
        '--dataset-name', 'CH-SIMS' if args.dataset == 'sims' else 'CH-SIMS-V2',
        '--data-dir', f'{SOURCE_ROOT}/{name}/processed',
        '--output', f'{OUTPUT_ROOT}/{name}/processed/{output_name}',
        '--cache-dir', f'{OUTPUT_ROOT}/{name}/processed/{cache_name}',
        '--clip-model', '/mnt/data26t/model/ViT-L-14-336px.pt',
        '--clap-model', '/mnt/data26t/model/clap-htsat-fused',
        '--audio-backbone', 'clap',
        '--frame-batch-size', '64',
        '--text-units', 'char',
        '--text-feature-description', 'raw-chinese-text-online-encoder',
        '--independent-clocks',
    ]
    if args.affect_preserving:
        fixed.extend((
            '--preserve-native-frames',
            '--visual-view', 'full-face',
            '--face-padding', '0.20',
        ))
    sys.argv = [sys.argv[0], *fixed, *remaining]
    extract_mosi_reference_av.main()


if __name__ == '__main__':
    main()
