"""在独立端口验证安装目录，只启动并清理本次验证自己创建的服务。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.install.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = 'http://127.0.0.1:7861'
    try:
        opener.open(url + '/health', timeout=1)
    except OSError:
        pass
    else:
        raise RuntimeError('测试端口已被占用，拒绝干扰现有服务')
    result = {'安装目录': str(root), '端口': 7861, '测试': []}
    environment = dict(os.environ, XINYU_PORT='7861', PYTHONIOENCODING='utf-8', PYTHONNOUSERSITE='1')
    environment.pop('XINYU_CONFIG', None)
    with (args.output / '安装服务.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(root / '.venv/Scripts/python.exe'), '-u', str(root / '体验/chat_app.py')],
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('安装目录服务启动失败，请查看安装服务.log')
                try:
                    health = json.load(opener.open(url + '/health', timeout=2))
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('安装目录服务未在三分钟内就绪')
            result['健康检查'] = health
            print(json.dumps(health, ensure_ascii=False), flush=True)
            for question in ['我爱你', '听着，请你说说你对大学生独自旅行的看法。', '今天有点低落，想和你说说话。']:
                request = urllib.request.Request(url + '/chat', data=json.dumps({'history': [{'role': 'user', 'content': question}]}).encode(), headers={'Content-Type': 'application/json'})
                started = time.monotonic()
                response = json.load(opener.open(request, timeout=120))
                result['测试'].append({'问题': question, '输出': response, '请求秒数': round(time.monotonic() - started, 3)})
                print(json.dumps(result['测试'][-1], ensure_ascii=False), flush=True)
            result['结果'] = '三条请求均返回非空回复' if all(item['输出'].get('response_text', '').strip() for item in result['测试']) else '存在空回复'
        finally:
            # process 是上面 Popen 的唯一子进程；不按程序名、端口或外部 PID 结束任何进程。
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
            (args.output / '安装验收.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
