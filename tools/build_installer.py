"""从已验收的开发环境制作可离线安装的 Windows 发行包。"""
import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--desktop', type=Path, required=True, help='包含心语.exe和_internal的便携启动器目录')
    args = parser.parse_args()
    project, stage = args.project.resolve(), args.stage.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    runtime = stage / '.venv'
    base_python = Path(sys.base_prefix)
    for name in ['python.exe', 'pythonw.exe', 'python311.dll', 'python3.dll', 'vcruntime140.dll', 'vcruntime140_1.dll']:
        source = base_python / name
        if source.exists():
            copy_file(source, runtime / 'Scripts' / name)
    shutil.copytree(base_python / 'Lib', runtime / 'Lib', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'test', 'tests', 'tkinter', 'idlelib', 'ensurepip'))
    shutil.copytree(base_python / 'DLLs', runtime / 'DLLs', dirs_exist_ok=True)
    copy_file(base_python / 'LICENSE.txt', runtime / 'LICENSE.txt')
    (runtime / 'Scripts' / 'python311._pth').write_text('.\n..\\DLLs\n..\\Lib\n..\\Lib\\site-packages\n..\\..\\体验\n..\\..\\participant\n', encoding='utf-8')

    pending = ['torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes', 'scikit-learn', 'joblib', 'numpy']
    distributions = {}
    while pending:
        name = pending.pop()
        distribution = metadata.distribution(name)
        canonical = distribution.metadata['Name'].lower().replace('_', '-')
        if canonical in distributions:
            continue
        distributions[canonical] = distribution
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    package_root = runtime / 'Lib' / 'site-packages'
    package_root.mkdir(parents=True, exist_ok=True)
    for name, distribution in sorted(distributions.items()):
        source_root = Path(distribution.locate_file('')).resolve()
        for relative in distribution.files or []:
            relative = Path(relative)
            if '..' in relative.parts or '__pycache__' in relative.parts or relative.suffix == '.pyc':
                continue
            source = source_root / relative
            if source.is_file():
                copy_file(source, package_root / relative)
    (stage / '运行环境版本.json').write_text(json.dumps({name: item.version for name, item in sorted(distributions.items())}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('运行依赖已复制，开始复制界面与指定权重。', flush=True)

    for name in ['chat_app.py', 'desktop_edge.py', 'story_intent.py', 'index.html', 'motion.js', '开场.html', '观点执行候选提示.txt']:
        copy_file(project / '体验' / name, stage / '体验' / name)
    shutil.copytree(project / '体验' / 'assets', stage / '体验' / 'assets', dirs_exist_ok=True)
    for name in ['run_inference.py', 'auxiliary.py']:
        copy_file(project / 'participant' / name, stage / 'participant' / name)
    shutil.copytree(project / 'participant' / 'classifiers', stage / 'participant' / 'classifiers', dirs_exist_ok=True)
    shutil.copytree(project / 'participant' / 'configs', stage / 'participant' / 'configs', dirs_exist_ok=True)
    shutil.copytree(project / '实验' / '陪伴强化本机_v24' / 'coverage_round' / 'candidate' / 'checkpoints' / 'step_000160', stage / 'participant' / 'adapter', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    experience = json.loads((project / '体验' / 'config.json').read_text(encoding='utf-8-sig'))
    experience.update(model_path='participant/model', adapter_path='participant/adapter')
    (stage / '体验' / 'config.json').write_text(json.dumps(experience, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copytree(args.desktop.resolve(), stage, dirs_exist_ok=True)
    source_repo = Path(__file__).resolve().parents[1]
    shutil.copytree(source_repo / 'licenses', stage / 'licenses', dirs_exist_ok=True)
    copy_file(source_repo / 'NOTICE', stage / 'NOTICE')
    (stage / '启动网页版.cmd').write_text('@echo off\ncd /d "%~dp0"\nstart "" http://127.0.0.1:7860\n".venv\\Scripts\\python.exe" "体验\\chat_app.py"\npause\n', encoding='utf-8')
    (stage / '安装说明.txt').write_text('心语共鸣 v24\n需要 Windows 64 位、可用的 NVIDIA 显卡驱动和足够显存。开发环境 RTX 4060 已验证，其他硬件未完成验收。\n安装后双击桌面“心语”即可体验，无需自行安装 Python。\n所有安装分卷与 exe 放在同一目录。安装后可离线运行。\n本发行包仅作比赛期间私有备份及研究体验；素材和官方派生资源的公开再分发许可尚待确认。\nBuilt with Qwen，基座受 Qwen 研究许可证约束。\n', encoding='utf-8')

    args.output.mkdir(parents=True, exist_ok=True)
    script = args.output / 'xinyu.iss'
    script.write_text(f'''[Setup]
AppId=XinyuCompanionResearchV24
AppName=心语共鸣
AppVersion=2.0.0
AppPublisher=心语共鸣项目
DefaultDirName={{localappdata}}\\Programs\\XinyuCompanion
DefaultGroupName=心语共鸣
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={args.output.resolve()}
OutputBaseFilename=Xinyu-Setup-v2.0.0
SetupIconFile={stage}\\体验\\assets\\心语.ico
UninstallDisplayIcon={{app}}\\心语.exe
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=1500000000
SlicesPerDisk=1
WizardStyle=modern
DisableProgramGroupPage=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\\ChineseSimplified.isl"

[Files]
Source: "{stage}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{args.base.resolve()}\\*"; DestDir: "{{app}}\\participant\\model"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.partial,*.incomplete,__pycache__\\*,.cache\\*"

[Icons]
Name: "{{userdesktop}}\\心语"; Filename: "{{app}}\\心语.exe"; WorkingDir: "{{app}}"
Name: "{{group}}\\心语"; Filename: "{{app}}\\心语.exe"; WorkingDir: "{{app}}"
Name: "{{group}}\\心语网页版"; Filename: "{{app}}\\启动网页版.cmd"; WorkingDir: "{{app}}"
''', encoding='utf-8-sig')
    print('开始编译离线安装包。', flush=True)
    subprocess.run([str(args.compiler), str(script)], check=True)


if __name__ == '__main__':
    main()
