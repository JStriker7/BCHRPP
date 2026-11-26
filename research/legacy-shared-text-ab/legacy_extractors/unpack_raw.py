"""Validate and unpack the pinned CH-SIMS Raw.zip into a fresh directory."""

import argparse
import hashlib
import json
import pickle
import shutil
import zipfile
from pathlib import Path


EXPECTED_ZIP_SHA256 = 'c1745724802dae37709b2d9babe5c6a1cef4f806a4bca3134a87e2fab2ee211a'
EXPECTED_DATA_SHA256 = 'c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if digest(args.archive) != EXPECTED_ZIP_SHA256 or digest(args.features) != EXPECTED_DATA_SHA256:
        raise ValueError('Source SHA-256 mismatch')
    with args.features.open('rb') as stream:
        data = pickle.load(stream)
    with zipfile.ZipFile(args.archive) as archive:
        members = archive.infolist()
        names = {item.filename for item in members}
        if len(names) != len(members):
            raise ValueError('Archive has duplicate member names; refusing overwrite')
        for item in members:
            path = Path(item.filename)
            if not item.filename.startswith('Raw/') or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe archive member: {item.filename}')
        counts = {}
        for split in ('train', 'valid', 'test'):
            ids = data[split]['id']
            missing = [key for key in ids if f'Raw/{key.replace("$_$", "/")}.mp4' not in names]
            if missing:
                raise ValueError(f'{split}: {len(missing)} missing videos')
            counts[split] = len(ids)
        expanded = sum(item.file_size for item in members)
        free = shutil.disk_usage(args.output_dir.parent).free
        if free < expanded + 2 * 1024**3:
            raise OSError(f'Need 2 GiB spare after unpack: {free} free, {expanded} required')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for i, item in enumerate(members, 1):
            archive.extract(item, args.output_dir)
            if i % 400 == 0:
                print(json.dumps({'extracted': i, 'of': len(members)}), flush=True)
    manifest = {'archive_sha256': EXPECTED_ZIP_SHA256,
                'dataset_sha256': EXPECTED_DATA_SHA256,
                'archive_revision': 'b1c55d5b49497ebf53a39eac6d14e35fafc26ebe',
                'entries': len(members), 'expanded_bytes': expanded, 'split_counts': counts}
    with (args.output_dir / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({'completed': manifest}), flush=True)


if __name__ == '__main__':
    main()
