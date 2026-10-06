"""校验原始基座后在CPU合并自写陪伴适配器，保留来源记录。"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / '模型/Qwen2.5-1.5B-Instruct'
    digest = hashlib.file_digest((base/'model.safetensors').open('rb'), 'sha256').hexdigest()
    assert digest == 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee', '原始基座校验失败'
    assert not args.output.exists(), '禁止覆盖已有候选'
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(args.output)
    info = {'原始基座sha256':digest, '适配器sha256':hashlib.file_digest((args.adapter/'adapter_model.safetensors').open('rb'),'sha256').hexdigest(),
            '合并设备':'CPU', '说明':'需独立本机验收，不假定与云端合并的舍入结果完全相同。'}
    (args.output/'合并来源.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('本机独立候选合并完成。')

if __name__ == '__main__':
    main()
