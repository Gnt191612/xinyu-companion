"""核验云端基准后合并独立体验候选，不覆盖比赛模型。"""
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from export_companion_cloud import tensor_digest

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--local-revalidation', action='store_true', help='基准不同时仅生成待重新验收的本机候选')
    args = parser.parse_args()
    source = ROOT / 'participant/model'
    target = ROOT / '模型/情感陪伴_companion_v2'
    info = json.loads((args.archive / '模型来源.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    baseline_digest = tensor_digest(source / 'model.safetensors')
    matched = baseline_digest == info['基准张量sha256']
    assert matched or args.local_revalidation, '本机基准与云端不同，禁止冒充同一模型'
    assert not target.exists(), '候选目录已存在，避免覆盖'
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, torch_dtype=torch.bfloat16, device_map='cpu')
    model = PeftModel.from_pretrained(model, args.archive / 'experiment/adapter', local_files_only=True).merge_and_unload()
    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(target)
    actual = tensor_digest(target / 'model.safetensors')
    assert not matched or actual == info['候选张量sha256'], '合并张量与云端不同，请保留记录并重新验收'
    (target / '来源核验.json').write_text(json.dumps({**info, '本机基准张量sha256': baseline_digest,
        '本机合并张量sha256': actual, '与云端同一权重': matched and actual == info['候选张量sha256'],
        '需要本机独立验收': not matched}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('独立体验候选合并完成。' + ('张量与云端一致。' if matched else '基准不同，必须重新在本机验收；云端指标不可沿用。'))
