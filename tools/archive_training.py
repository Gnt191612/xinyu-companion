"""分包备份实验记录和训练资源；不删除本地文件，不包含登录凭据。"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [path for path in (root / '实验').rglob('*') if path.is_file()
             and '__pycache__' not in path.parts and path.suffix.lower() not in {'.zip', '.png', '.jpg', '.exe'}]
    files += [path for path in (root / '数据').glob('companion*/*') if path.is_file()]
    files += list((root / '交付材料').glob('*.md'))
    manifest = []
    archive, size, number = None, 0, 0
    try:
        for path in sorted(set(files)):
            length = path.stat().st_size
            if archive is None or size + length > 1200000000:
                if archive:
                    archive.close()
                number += 1
                archive_path = args.output / f'xinyu-training-history-{number:02d}.zip'
                archive = zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1)
                size = 0
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            manifest.append({'path': relative, 'bytes': length, 'sha256': digest(path), 'archive': archive_path.name})
            size += length
    finally:
        if archive:
            archive.close()
    for path in sorted(args.output.glob('xinyu-training-history-*.zip')):
        with zipfile.ZipFile(path) as check:
            if check.testzip() is not None:
                raise RuntimeError('备份压缩包完整性验证失败')
        print(path.name, path.stat().st_size, digest(path), flush=True)
    (args.output / 'training-history-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已备份文件数量：', len(manifest), flush=True)


if __name__ == '__main__':
    main()
