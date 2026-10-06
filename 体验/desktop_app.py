"""桌面窗口复用本机模型服务；开场动画期间异步加载模型。"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:7860'
owned_server = None
closing = False


def ready():
    try:
        with urllib.request.urlopen(URL + '/health', timeout=2) as response:
            return json.load(response).get('service') == 'xinyu-local'
    except (OSError, ValueError):
        return False


def close_server():
    global closing
    closing = True
    # 只结束桌面窗口自己启动的服务，已经运行的网页服务继续保留。
    if owned_server is not None and owned_server.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(owned_server.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            return
        owned_server.terminate()
        try:
            owned_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owned_server.kill()


def load_chat(window):
    global owned_server
    started = time.monotonic()
    try:
        if not ready():
            python = ROOT / '.venv/Scripts/python.exe'
            if not python.exists():
                raise RuntimeError('缺少本机模型运行环境，请先按桌面说明安装环境。')
            logs = ROOT / 'tmp'
            logs.mkdir(exist_ok=True)
            environment = os.environ.copy()
            environment['PYTHONIOENCODING'] = 'utf-8'
            with (logs / '桌面模型服务.log').open('w', encoding='utf-8') as log:
                owned_server = subprocess.Popen([str(python), '-u', str(ROOT / '体验/chat_app.py')],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.monotonic() + 180
            while not ready():
                if closing:
                    return
                if owned_server.poll() is not None:
                    raise RuntimeError('模型服务未能启动，请查看tmp/桌面模型服务.log。')
                if time.monotonic() > deadline:
                    raise RuntimeError('模型加载超过三分钟，请检查运行日志和可用内存。')
                time.sleep(.4)
        # 已就绪时也展示完整的短开场，不延长实际模型加载时间。
        while not closing and time.monotonic() - started < 2.4:
            time.sleep(.1)
        if not closing:
            window.load_url(URL)
    except (OSError, RuntimeError) as exc:
        if not closing:
            text = str(exc)
            window.evaluate_js('document.getElementById("loading").textContent=' + json.dumps(text, ensure_ascii=False))
        close_server()


def main():
    opening = (ROOT / '体验/开场.html').read_text(encoding='utf-8')
    window = webview.create_window('心语 · 樱花陪伴', html=opening, width=1280, height=820,
        min_size=(760, 560), background_color='#f7eef5')
    window.events.closed += close_server
    webview.start(load_chat, window, gui='edgechromium', private_mode=True,
        icon=str(ROOT / '体验/assets/心语.ico') if (ROOT / '体验/assets/心语.ico').exists() else None)


if __name__ == '__main__':
    main()
