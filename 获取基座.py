"""下载当前3B基座到明确位置，避免误用历史1.5B版本。"""
import argparse
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parent / 'participant/model')
    args = parser.parse_args()
    subprocess.run([sys.executable, str(Path(__file__).with_name('download_qwen3b.py')),
                    '--target', str(args.target.resolve())], check=True)
