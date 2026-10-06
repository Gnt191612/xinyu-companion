"""按配置串行训练、开发回归和可选首次留出；自动化不代替质量评审。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--include-holdout', action='store_true', help='仅开发质量通过后使用；首次后转为回归')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf8'))
    if not args.evaluate_only and config.get('data_audit'):
        audit = json.loads((ROOT / config['data_audit']).read_text(encoding='utf8'))
        assert audit.get('允许开训') is True, '数据审计未放行，停止训练；请先完成审计报告中的待办'
    experiment = ROOT / config['experiment']
    experiment.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

    def execute(name, arguments):
        log = experiment / (name + '.log')
        assert not log.exists(), f'保留旧记录，请改用新实验目录：{log}'
        with log.open('w', encoding='utf8') as handle:
            result = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT,
                                    env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name}失败，详见{log}，未执行后续环节')

    model = config['merged_model']
    if not args.evaluate_only:
        extra = (['--initial-adapter', config['initial_adapter']] if config.get('initial_adapter') else []) + (['--adapter-only'] if config.get('adapter_only') else [])
        if config.get('keep_checkpoints'):
            extra += ['--keep-checkpoints']
        if config.get('replay_file'):
            extra += ['--replay-file', config['replay_file']]
        if config.get('validation_data'):
            extra += ['--validation-data', config['validation_data'], '--eval-every', str(config.get('eval_every', 20))]
        if config.get('training_extra_prompt'):
            extra += ['--training-extra-prompt', config['training_extra_prompt']]
        execute('train', ['train_companion.py', '--data-dir', config['data'],
            '--model-path', config['base_model'], '--competition-only', '--load-in-4bit',
            '--target-modules', 'all-linear', '--learning-rate', str(config['learning_rate']),
            '--steps', str(config['steps']), '--replay-count', str(config['replay_count']),
            '--output-dir', str(experiment / 'candidate'), '--merged-output', model, *extra])
        license_file = Path(config['base_model']) / 'LICENSE'
        if license_file.exists() and not config.get('adapter_only'):
            shutil.copy2(license_file, Path(model) / 'LICENSE')
    for name, data, split in [('dev', config['dev_data'], 'dev')] + (
            [('holdout', config['holdout_data'], 'final')] if args.include_holdout else []):
        assert not (experiment / name / 'responses.json').exists(), '不覆盖已生成的回复'
        eval_extra = ['--load-in-4bit', '--adapter-path', config.get('evaluation_adapter', str(experiment / 'candidate/adapter'))] if config.get('adapter_only') else []
        execute(name, ['evaluate_companion.py', '--data-dir', data, '--split', split,
            '--competition-format', '--formal-generation', '--model-path', config['base_model'] if config.get('adapter_only') else model,
            '--output-dir', str(experiment / name), *eval_extra])
    (experiment / '流程状态.json').write_text(json.dumps({
        '训练执行': not args.evaluate_only, '开发生成完成': True,
        '留出生成完成': args.include_holdout, '人工语义验收': '待逐题审阅',
        'Linux验收': False, '第二次提交': False,
        '说明': '脚本执行完成不代表质量合格；不自动部署、上传或提交。'
    }, ensure_ascii=False, indent=2), encoding='utf8')
    print('流程执行完成，请审阅responses.json；尚未部署或提交。')


if __name__ == '__main__':
    main()
