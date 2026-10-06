"""验证从私有仓库取回的备份，每个文件同时检查大小、摘要和ZIP读取完整性。"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.download / 'training-history-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    expected_archives = {
        'xinyu-training-history-01.zip': '194592bafcdc918e634d7bdeb849ab2c8c492f0554bf29cac614c8cefe7c0c2a',
        'xinyu-training-history-02.zip': '45f3265c24163079a130deb70741e06027d72db85b058eb3ddc5f58fe793ef2c',
        'xinyu-training-history-03.zip': 'f0ad754217066f0279c356eecb4aa7b3587d655407af1c632da4383cad41771a',
    }
    result = {'全部通过': False, '文件数量': 0, '原始文件字节': 0, '归档': []}
    with manifest_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == '8c0384e966ad0d0c68e6e5388b476a06e3d8965185205b38f1dcfe6507eccafa'
    for name, expected in expected_archives.items():
        path = args.download / name
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == expected, f'{name}压缩包摘要不符'
        with zipfile.ZipFile(path) as archive:
            entries = [item for item in manifest if item['archive'] == name]
            assert set(archive.namelist()) == {item['path'] for item in entries}
            for item in entries:
                relative = Path(item['path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                assert archive.getinfo(item['path']).file_size == item['bytes']
                with archive.open(item['path']) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256'], item['path']
                result['文件数量'] += 1
                result['原始文件字节'] += item['bytes']
        result['归档'].append({'文件': name, '字节': path.stat().st_size, 'sha256': actual, '文件数': len(entries)})
        print(name, '大小、SHA256、ZIP文件内容均通过', flush=True)
    result['全部通过'] = True
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
