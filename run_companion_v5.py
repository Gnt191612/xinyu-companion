"""云端比较两种步数的all-linear原生陪伴候选，不读取最终验收答案。"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path('/root/companion_upgrade_v5')
BASE = Path('/root/模型/Qwen2.5-1.5B-Instruct')

def main():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / '256/train_log.jsonl').exists(), '已有训练记录，禁止重复覆盖'
    with ZipFile('/root/陪伴强化v5.zip') as archive:
        assert archive.testzip() is None
        assert all((ROOT / i.filename).resolve().is_relative_to(ROOT) for i in archive.infolist())
        archive.extractall(ROOT)
    baseline_hash = hashlib.file_digest((BASE/'model.safetensors').open('rb'),'sha256').hexdigest()
    assert baseline_hash == 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee', '云端原始基座与固定来源不符'
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}
    def run(args):
        print('执行：'+' '.join(args),flush=True)
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,check=True)
    run(['evaluate_companion.py','--data-dir','数据/companion_v5','--split','dev','--model-path',str(BASE),'--output-dir','baseline-dev'])
    for steps in (256,512):
        run(['train_companion.py','--data-dir','数据/companion_v5','--model-path',str(BASE),'--native-only',
             '--target-modules','all-linear','--learning-rate','0.00005','--steps',str(steps),
             '--output-dir',str(steps),'--merged-output','model-'+str(steps)])
        run(['evaluate_companion.py','--data-dir','数据/companion_v5','--split','dev',
             '--model-path','model-'+str(steps),'--output-dir',str(steps)+'/dev'])
    info = {'原始基座sha256':baseline_hash,'实际步数':[256,512],'训练位置':'RTX4090免费实例',
            '范围':'原生日常聊天；比赛已提交模型保留','最终验收已运行':False}
    (ROOT/'来源与状态.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[ROOT/'来源与状态.json']
    for folder in ('baseline-dev','256','512','数据/companion_v5'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    output=ROOT/'陪伴强化v5结果.zip'
    with ZipFile(output,'w',ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p,p.relative_to(ROOT).as_posix())
    with ZipFile(output) as archive:
        assert archive.testzip() is None
    print(json.dumps({'结果包字节':output.stat().st_size,'结果包sha256':hashlib.file_digest(output.open('rb'),'sha256').hexdigest()},ensure_ascii=False),flush=True)
    print('ALL_FINISHED',flush=True)

if __name__ == '__main__':
    main()
