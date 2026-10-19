"""用系统 Edge 应用窗口显示离线心语，避免依赖 Python.NET。"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:' + os.environ.get('XINYU_PORT', '7860')
started = time.monotonic()
error = ''


def ready():
    try:
        with urllib.request.urlopen(URL + '/health', timeout=1) as response:
            return json.load(response).get('service') == 'xinyu-local'
    except (OSError, ValueError):
        return False


def start_model():
    global error
    try:
        if ready():
            return
        logs = ROOT / 'tmp'
        logs.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        with (logs / '桌面模型服务.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u', str(ROOT / '体验/chat_app.py')],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 180
        while not ready():
            if child.poll() is not None:
                raise RuntimeError('模型启动失败，请查看 tmp/桌面模型服务.log。')
            if time.monotonic() > deadline:
                raise RuntimeError('模型加载超时，请查看 tmp/桌面模型服务.log。')
            time.sleep(.5)
    except (OSError, RuntimeError) as exc:
        error = str(exc)


class OpeningHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            body = json.dumps({'ready': ready() and time.monotonic() - started >= 2.4,
                               'error': error}, ensure_ascii=False).encode('utf-8')
            content_type = 'application/json; charset=utf-8'
        else:
            html = (ROOT / '体验/开场.html').read_text(encoding='utf-8')
            script = """<script>setInterval(async()=>{try{const s=await fetch('/status').then(r=>r.json());
if(s.ready)location.replace('http://127.0.0.1:7860/');
else if(s.error)document.getElementById('loading').textContent=s.error;}catch(e){}},500);</script>"""
            body = html.replace('</body>', script.replace('http://127.0.0.1:7860/', URL + '/') + '</body>').encode('utf-8')
            content_type = 'text/html; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    candidates = [Path(os.environ.get(key, '')) / 'Microsoft/Edge/Application/msedge.exe'
                  for key in ('PROGRAMFILES(X86)', 'PROGRAMFILES', 'LOCALAPPDATA')]
    edge = next((path for path in candidates if path.is_file()), None)
    server = ThreadingHTTPServer(('127.0.0.1', 0), OpeningHandler)
    threading.Thread(target=start_model, daemon=True).start()
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 1
    opening_url = f'http://127.0.0.1:{server.server_port}/'
    if edge is not None:
        subprocess.Popen([str(edge), f'--app={opening_url}', '--window-size=1280,820'],
                         startupinfo=startup)
    else:
        webbrowser.open(opening_url)
    # 只关闭临时开场服务，模型服务继续保留，不结束其他 Edge 窗口。
    timer = threading.Timer(240, server.shutdown)
    timer.daemon = True
    timer.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        timer.cancel()


if __name__ == '__main__':
    main()

