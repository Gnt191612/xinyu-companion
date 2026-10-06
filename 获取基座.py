"""准备阶段从官方仓库下载固定基座并检查权重。"""
from pathlib import Path
import hashlib
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
target = root / '模型/Qwen2.5-1.5B-Instruct'
snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct',
    revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_dir=target,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'])
with (target / 'model.safetensors').open('rb') as handle:
    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
assert actual == 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee', '基座权重校验不符'
assert (target / 'tokenizer.json').exists(), '分词器缺失'
print('固定版本基座与分词器准备完成')
