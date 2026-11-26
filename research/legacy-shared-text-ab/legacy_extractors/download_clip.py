"""Download and checksum the user's pinned ModelScope CLIP ViT-L/14 model."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests


REVISION = 'f14d10b70c22170418f57120d301a7017cf60012'
FILES = {
    'config.json': ('8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a', 4519),
    'preprocessor_config.json': ('910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd', 316),
    'model.safetensors': ('a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb', 1710540580),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required = sum(n for _, n in FILES.values()) + 2 * 1024**3
    if shutil.disk_usage(args.output_dir.parent).free < required:
        raise OSError('Insufficient disk space for CLIP weights plus 2 GiB headroom')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = 'https://www.modelscope.cn/models/AI-ModelScope/clip-vit-large-patch14/resolve/'
    session = requests.Session()
    for filename, (expected, length) in FILES.items():
        url = base + REVISION + '/' + filename
        sha = hashlib.sha256()
        count = 0
        with session.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with (args.output_dir / filename).open('xb') as stream:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        sha.update(chunk)
                        count += len(chunk)
        if sha.hexdigest() != expected or count != length:
            raise ValueError(f'{filename}: checksum or size mismatch')
        print(json.dumps({'verified': filename, 'bytes': count}), flush=True)
    manifest = {'model': 'AI-ModelScope/clip-vit-large-patch14',
                'revision': REVISION, 'sha256': {k: v[0] for k, v in FILES.items()}}
    with (args.output_dir / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({'completed': manifest}), flush=True)


if __name__ == '__main__':
    main()
