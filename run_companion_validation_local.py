"""对已通过陪伴验收的本机候选执行同输入比赛回归；不代替Linux验收。"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'EVALUATION_BACKEND': 'transformers',
           'EVALUATION_HARDWARE_LABEL': 'RTX 4060 Laptop GPU 8GB Windows'}
    inputs = ROOT / '数据/processed/val_100_inputs.jsonl'
    targets = ROOT / '数据/processed/val_100_targets.jsonl'
    for name, model in [('candidate', args.model_path), ('baseline', ROOT / 'participant/model')]:
        destination = args.output_dir / name
        subprocess.run([sys.executable, str(ROOT / 'participant/run_inference.py'),
                        str(inputs), str(destination)],
                       env={**env, 'EVALUATION_MODEL_PATH': str(model.resolve())}, check=True)
        subprocess.run([sys.executable, str(ROOT / 'evaluate_predictions.py'),
                        str(destination / 'submission.jsonl'), str(targets)], env=env, check=True)
    (args.output_dir / '验证说明.json').write_text(json.dumps({
        '官方验证子集题数': 100, '输入相同': True, '联网': False,
        '运行入口': 'participant/run_inference.py', 'Linux入口已验收': False,
        '说明': 'Windows离线回归，不代表正式Linux无人干预测试或官方成绩。'
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
