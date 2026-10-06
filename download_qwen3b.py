"""通过curl下载发布者魔搭仓库文件，并核对发布摘要。"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((ROOT / '实验/Qwen3B魔搭仓库信息.json').read_text(encoding='utf-8'))
    assert metadata['Success']
    names = {'LICENSE', 'README.md', 'config.json', 'generation_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'merges.txt', 'vocab.json', 'model.safetensors.index.json',
             'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'}
    revision = '8f4992eda43eea7c770690ddc0de8f732da246f5'
    args.target.mkdir(parents=True, exist_ok=True)
    records = []
    for item in metadata['Data']['Files']:
        name = item['Path']
        if name not in names:
            continue
        target = args.target / name
        partial = target.with_suffix(target.suffix + '.part')
        url = 'https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-3B-Instruct/repo?' + urlencode({'Revision': revision, 'FilePath': name})
        if not target.exists():
            print(f'正在下载{name}，预期{item["Size"]}字节', flush=True)
            command = ['curl.exe', '--noproxy', '*', '--ssl-revoke-best-effort', '--fail', '--silent',
                       '--show-error', '-L', '--connect-timeout', '15', '--max-time', '600']
            if item['Size'] > 10 * 1024 * 1024:
                chunk_size = 16 * 1024 * 1024
                count = (item['Size'] + chunk_size - 1) // chunk_size
                def fetch(number):
                    first = number * chunk_size
                    last = min(item['Size'] - 1, first + chunk_size - 1)
                    segment = args.target / f'{name}.segment-{number:04d}'
                    if not segment.exists() or segment.stat().st_size != last - first + 1:
                        received = segment.stat().st_size if segment.exists() else 0
                        assert received <= last - first + 1
                        remainder = args.target / f'{name}.remainder-{number:04d}'
                        subprocess.run([*command, '--retry', '2', '-H', f'Range: bytes={first + received}-{last}',
                                        '-o', str(remainder), url], check=True)
                        assert remainder.stat().st_size == last - first + 1 - received
                        with segment.open('ab') as destination, remainder.open('rb') as source:
                            while block := source.read(8 * 1024 * 1024):
                                destination.write(block)
                    assert segment.stat().st_size == last - first + 1
                    return segment
                with ThreadPoolExecutor(max_workers=12) as pool:
                    futures = [pool.submit(fetch, number) for number in range(count)]
                    for completed, future in enumerate(as_completed(futures), 1):
                        future.result()
                        if completed % 8 == 0 or completed == count:
                            print(f'{name}分段完成{completed}/{count}', flush=True)
                with partial.open('wb') as destination:
                    for number in range(count):
                        with (args.target / f'{name}.segment-{number:04d}').open('rb') as source:
                            while block := source.read(8 * 1024 * 1024):
                                destination.write(block)
            else:
                subprocess.run([*command, '-C', '-', '-o', str(partial), url], check=True)
            assert partial.stat().st_size == item['Size'], f'{name}大小不符'
            with partial.open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == item['Sha256'], f'{name}摘要不符'
            partial.rename(target)
        assert target.stat().st_size == item['Size']
        with target.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        assert digest == item['Sha256']
        records.append({'文件': name, '字节': target.stat().st_size, 'sha256': digest})
        print(f'{name}大小和SHA256通过', flush=True)
    assert {item['文件'] for item in records} == names
    (args.target / '下载校验.json').write_text(json.dumps({'来源': 'https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct',
        'revision': revision, '目标目录': str(args.target.resolve()), '文件': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('3B基座下载及校验完成。', flush=True)


if __name__ == '__main__':
    main()
