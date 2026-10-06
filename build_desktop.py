"""只封装桌面入口，模型和CUDA环境使用现有项目目录，避免重复数GB权重。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile',
        '--windowed', '--name', '心语樱花版', '--distpath', str(ROOT),
        '--workpath', str(ROOT / 'tmp/desktop-build'), '--specpath', str(ROOT / 'tmp/desktop-build'),
        '--exclude-module', 'torch', '--exclude-module', 'transformers', '--exclude-module', 'numpy',
        '--exclude-module', 'pandas', '--exclude-module', 'matplotlib', '--exclude-module', 'scipy',
        '--exclude-module', 'sklearn', '--exclude-module', 'IPython', '--exclude-module', 'pytest',
        '--exclude-module', 'PyQt5', '--exclude-module', 'PyQt6', '--exclude-module', 'PySide6',
        '--exclude-module', 'tkinter', *(['--icon', str(ROOT / '体验/assets/心语.ico')] if (ROOT / '体验/assets/心语.ico').exists() else []), str(ROOT / '体验/desktop_app.py')], check=True, cwd=ROOT)
    print('已生成心语樱花版.exe，放在项目根目录使用。模型仍保存在项目目录。')
