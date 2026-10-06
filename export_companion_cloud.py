"""归档补充训练日志、适配器与验收输出，记录合并源的张量指纹。"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent


def tensor_digest(path):
    digest = hashlib.sha256()
    with safe_open(path, framework='pt', device='cpu') as file:
        for key in sorted(file.keys()):
            tensor = file.get_tensor(key)
            digest.update(key.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    baseline = Path('/root/participant/model')
    candidate = ROOT / 'model-candidate'
    logs = [json.loads(line) for line in (ROOT / 'experiment/train_log.jsonl').read_text().splitlines()]
    assert logs[-1]['step'] == json.loads((ROOT / 'experiment/train_config.json').read_text())['steps'], '训练尚未完成'
    assert json.loads((ROOT / 'regression/performance_report.json').read_text())['complete']
    info = {'训练位置': '咪咕西南一区RTX 4090', '实际完成steps': logs[-1]['step'],
            '基准张量sha256': tensor_digest(baseline / 'model.safetensors'),
            '候选张量sha256': tensor_digest(candidate / 'model.safetensors'),
            '候选文件sha256': hashlib.file_digest((candidate / 'model.safetensors').open('rb'), 'sha256').hexdigest(),
            '说明': '张量指纹独立于safetensors保存顺序；本机合并后必须对照指纹核验。数值训练记录与配置均保留，原512步权重未修改。'}
    (ROOT / '模型来源.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    files = [ROOT / '模型来源.json']
    if (ROOT / 'run.log').exists():
        files.append(ROOT / 'run.log')
    for folder in ('experiment', 'baseline', 'candidate', 'regression'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
    output = ROOT / '陪伴补充训练结果.zip'
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'结果包字节': output.stat().st_size, '结果包sha256': hashlib.file_digest(output.open('rb'), 'sha256').hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
