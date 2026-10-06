"""保存真实留出题生成结果；自动检查不代替人工表达质量评审。"""
import argparse
import json
import time
import sys
import hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'participant'))
from run_inference import SYSTEM_PROMPT, EMOTIONS, PROFILE_LABELS, conversation_text, guard_listener_ack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / '数据/companion_v1')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'eval', 'final'), default='eval')
    parser.add_argument('--competition-format', action='store_true', help='使用正式入口JSON提示评估')
    parser.add_argument('--formal-generation', action='store_true', help='使用比赛入口的生成参数')
    parser.add_argument('--load-in-4bit', action='store_true', help='对照NF4推理，不合并权重')
    parser.add_argument('--adapter-path', type=Path, help='加载与所选基座匹配的适配器')
    parser.add_argument('--experience-prompt', type=Path, help='与网页一致的额外风格提示，用于相同入口对照')
    parser.add_argument('--prompt-sample-id', help='固定推理提示中的样本ID，保留结果记录原ID')
    args = parser.parse_args()
    generation = json.loads((ROOT / 'participant/configs/inference_config.json').read_text(encoding='utf8'))['generation'] if args.formal_generation else dict(do_sample=False, max_new_tokens=384 if args.competition_format else 192, repetition_penalty=1.08)
    if args.experience_prompt:
        assert args.competition_format and not args.formal_generation, '网页对照使用比赛格式加风格提示'
        generation['repetition_penalty'] = 1.05
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    load_options = {'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16), 'device_map': {'': 'cuda'}} if args.load_in_4bit else {'device_map': 'auto'}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, **load_options)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model.eval()
    folder = args.data_dir
    style = (folder / 'style.txt').read_text(encoding='utf-8').strip()
    records = []
    for line in (folder / (args.split + '.jsonl')).read_text(encoding='utf-8').splitlines():
        sample = json.loads(line)
        if args.competition_format:
            system = SYSTEM_PROMPT.format(emotions=', '.join(sorted(EMOTIONS)), personality=', '.join(sorted(PROFILE_LABELS['personality_traits'])), interests=', '.join(sorted(PROFILE_LABELS['interests'])), styles=', '.join(sorted(PROFILE_LABELS['style'])))
            if args.experience_prompt:
                system += '\n' + args.experience_prompt.read_text(encoding='utf-8')
            prompt_sample = {**sample, 'id': args.prompt_sample_id} if args.prompt_sample_id else sample
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': conversation_text(prompt_sample)}]
        else:
            messages = [{'role': 'system', 'content': style}, *sample['history']]
        inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True), return_tensors='pt').to(next(model.parameters()).device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        tokens = output[0, inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw = text
        valid_json = None
        if args.competition_format:
            try:
                prediction = json.loads(text)
                valid_json = set(prediction) == {'response_text', 'emotion_label', 'user_profile', 'memory_refs'} and prediction['emotion_label'] in EMOTIONS and isinstance(prediction['response_text'], str) and bool(prediction['response_text'].strip()) and prediction['memory_refs'] == [] and set(prediction['user_profile']) == set(PROFILE_LABELS) and all(isinstance(prediction['user_profile'][k], list) and all(v in PROFILE_LABELS[k] for v in prediction['user_profile'][k]) for k in PROFILE_LABELS)
                text = prediction['response_text']
            except (ValueError, KeyError, TypeError):
                valid_json = False
        protected, guard_applied = guard_listener_ack(sample['history'], text)
        records.append({**sample, 'response_text': text, '入口保护后回复': protected, '角色保护触发': guard_applied,
                        'seconds': round(time.perf_counter() - started, 3),
                        '原始输出': raw, '合法JSON': valid_json,
                        '生成token数': len(tokens), '达到长度上限': len(tokens) >= generation['max_new_tokens']})
        print(sample['id'] + ' 已完成', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'responses.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'不合法JSON': sum(r['合法JSON'] is False for r in records), '生成参数': generation, '题数': len(records), '空回复': sum(not r['response_text'] for r in records),
               '系统提示sha256': hashlib.sha256((system if args.competition_format else style).encode('utf-8')).hexdigest(),
               '达到长度上限': sum(r['达到长度上限'] for r in records),
               '平均秒数': sum(r['seconds'] for r in records) / len(records),
               '数据划分': args.split, 'NF4推理': args.load_in_4bit,
               '适配器': str(args.adapter_path) if args.adapter_path else None,
               '提示样本ID覆盖': args.prompt_sample_id,
               '说明': '相同提示、贪心生成；开发题用于诊断，最终留出题首次使用才作为独立验收。温柔与贴心需逐条阅读，非用户满意度分数。'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
