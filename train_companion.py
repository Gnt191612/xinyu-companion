"""补充陪伴表达，支持比赛格式回放或独立原生聊天训练，保存候选权重。"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'participant'))
from run_inference import SYSTEM_PROMPT, EMOTIONS, PROFILE_LABELS, conversation_text


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def system_prompt():
    return SYSTEM_PROMPT.format(emotions=', '.join(sorted(EMOTIONS)),
        personality=', '.join(sorted(PROFILE_LABELS['personality_traits'])),
        interests=', '.join(sorted(PROFILE_LABELS['interests'])), styles=', '.join(sorted(PROFILE_LABELS['style'])))


def main():
    parser = argparse.ArgumentParser(description='离线陪伴表达补充训练')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'participant/model')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '实验/companion_v1')
    parser.add_argument('--merged-output', type=Path, default=ROOT / '模型/情感陪伴_companion_v1')
    parser.add_argument('--data-dir', type=Path, default=ROOT / '数据/companion_v1')
    parser.add_argument('--steps', type=int, default=96)
    parser.add_argument('--keep-checkpoints', action='store_true', help='完整覆盖后保留各验证检查点，供独立语义复核')
    parser.add_argument('--max-length', type=int, default=896)
    parser.add_argument('--replay-count', type=int, default=48)
    parser.add_argument('--replay-file', type=Path, help='已经审查并冻结的官方训练回放')
    parser.add_argument('--validation-data', type=Path, help='独立开发监督JSONL，不加入训练')
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--training-extra-prompt', type=Path, help='冻结的额外系统提示，与指定体验入口一致')
    parser.add_argument('--learning-rate', type=float, default=0.00008)
    parser.add_argument('--native-only', action='store_true', help='独立日常体验训练，不训练比赛JSON格式')
    parser.add_argument('--initial-adapter', type=Path, help='在同一基座上继续已有适配器训练')
    parser.add_argument('--adapter-only', action='store_true', help='仅保存适配器，不执行完整合并')
    parser.add_argument('--competition-only', action='store_true', help='仅训练正式比赛JSON格式，与入口提示一致')
    parser.add_argument('--load-in-4bit', action='store_true', help='NF4低显存训练，保存适配器后从原始基座在CPU合并')
    parser.add_argument('--target-modules', default='q_proj,v_proj', help='逗号分隔的层名，或all-linear')
    args = parser.parse_args()
    if args.validation_data:
        assert args.competition_only and args.adapter_only and args.eval_every > 0, '周期验证仅用于比赛格式适配器候选'
    assert not (args.native_only and args.competition_only), '训练格式不能同时指定'
    target_modules = 'all-linear' if args.target_modules == 'all-linear' else args.target_modules.split(',')
    assert torch.cuda.is_available(), '需要CUDA训练实例'
    assert args.model_path.resolve() != args.merged_output.resolve(), '候选不能覆盖原模型'
    assert not (args.output_dir / 'train_log.jsonl').exists(), '日志已存在，请指定新的实验目录'
    random.seed(20261004)
    torch.manual_seed(20261004)
    torch.set_num_threads(4)
    folder = args.data_dir
    style = (folder / 'style.txt').read_text(encoding='utf-8').strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    examples = []
    sample_metadata = {}
    def encode(messages, target, sample_id, category):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target + tokenizer.eos_token, add_special_tokens=False)
        if len(prompt_ids) + len(target_ids) <= args.max_length:
            examples.append((prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids))
            sample_metadata[tuple(prompt_ids + target_ids)] = {"id": sample_id, "类别": category}
            return True
        return False
    rows = read(folder / 'train.jsonl')
    task_system = system_prompt()
    if args.training_extra_prompt:
        assert args.competition_only, '额外系统提示用于单一比赛格式对照'
        task_system += '\n' + args.training_extra_prompt.read_text(encoding='utf-8')
    for row in rows:
        if not args.competition_only:
            encode([{'role': 'system', 'content': style}, *row['history']], row['response_text'], row['id'], row.get('category', '原生表达'))
        if args.native_only:
            continue
        target = {key: row[key] for key in ('response_text', 'emotion_label')}
        target.update(user_profile={'personality_traits': [], 'interests': [], 'style': []}, memory_refs=[])
        encode([{'role': 'system', 'content': task_system if args.competition_only else task_system + '\n' + style},
                {'role': 'user', 'content': conversation_text(row)}], json.dumps(target, ensure_ascii=False), row['id'], row.get('category', '自写表达'))
    authored_count = len(examples)
    replay = [] if args.native_only or args.replay_count == 0 else read(args.replay_file or ROOT / '数据/processed/train_sft.jsonl')
    random.shuffle(replay)
    replay_used = 0
    for row in replay:
        target = {k: v for k, v in row['target'].items() if k != 'id'}
        if encode([{'role': 'system', 'content': task_system},
                   {'role': 'user', 'content': conversation_text(row['input'])}], json.dumps(target, ensure_ascii=False), row['input']['id'], '官方回放'):
            replay_used += 1
        if replay_used >= args.replay_count:
            break
    assert authored_count == len(rows) * (1 if args.native_only or args.competition_only else 2), '有自写样本超过长度限制，停止以免默默丢弃'
    random.shuffle(examples)
    validation = []
    if args.validation_data:
        training_examples = examples
        examples = []
        for row in read(args.validation_data):
            target = {key: row[key] for key in ('response_text', 'emotion_label')}
            target.update(user_profile={'personality_traits': [], 'interests': [], 'style': []}, memory_refs=[])
            assert encode([{'role': 'system', 'content': task_system}, {'role': 'user', 'content': conversation_text(row)}], json.dumps(target, ensure_ascii=False), row['id'], '开发'), '开发样本超长'
        validation = examples
        examples = training_examples
    required_examples = {tuple(ids) for ids, labels in examples}
    seen_examples = set()
    assert not validation or args.steps * 2 >= len(examples), "步数不足以完整覆盖训练集，先增加步数预算"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_files = {}
    for source in sorted(args.model_path.glob('*.safetensors')):
        with source.open('rb') as handle:
            source_files[source.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert source_files, '缺少基座权重文件'
    source_hash = next(iter(source_files.values())) if len(source_files) == 1 else hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    config = {'种子': 20261004, 'steps': args.steps, '梯度累积': 2, 'batch_size': 1,
              'learning_rate': args.learning_rate, 'max_length': args.max_length, '自写对话': len(rows),
              '格式自写样本': authored_count, '仅原生聊天': args.native_only,
              '仅比赛格式': args.competition_only,
              '官方训练回放样本': replay_used, '总训练样本': len(examples),
              '基准权重sha256': source_hash, 'rank': 16, 'alpha': 32,
              '基准权重文件sha256': source_files, '四位训练': args.load_in_4bit,
              'target_modules': target_modules, '验证集参与训练': False,
              '冻结回放文件': str(args.replay_file) if args.replay_file else None,
              '开发监督文件': str(args.validation_data) if args.validation_data else None,
              '开发验证间隔': args.eval_every if validation else None,
              '初始适配器': str(args.initial_adapter) if args.initial_adapter else None,
              '完整覆盖后才选轮': bool(validation), '保留检查点': args.keep_checkpoints,
              '仅保存适配器': args.adapter_only,
              '正式系统提示sha256': hashlib.sha256(task_system.encode('utf-8')).hexdigest(),
              '自写数据来源': 'AI辅助合成，不是真人访谈或用户聊天', '实际设备': torch.cuda.get_device_name(0)}
    (args.output_dir / 'train_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    load_options = {'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16), 'device_map': {'': 'cuda'}} if args.load_in_4bit else {}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', **load_options)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to('cuda')
    model.config.use_cache = False
    if args.initial_adapter:
        adapter_config = json.loads((args.initial_adapter / 'adapter_config.json').read_text(encoding='utf8'))
        assert Path(adapter_config['base_model_name_or_path']).resolve() == args.model_path.resolve(), '必须使用初始适配器的相同基座'
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=target_modules, task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    best_validation = float('inf')
    stale_evaluations = 0
    cursor = 0
    def validate(step):
        nonlocal best_validation, stale_evaluations
        model.eval()
        total, tokens = 0.0, 0
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for ids, labels in validation:
                count = sum(x != -100 for x in labels[1:])
                value = model(input_ids=torch.tensor([ids], device='cuda'), labels=torch.tensor([labels], device='cuda')).loss.item()
                total += value * count
                tokens += count
        mean = total / tokens
        eligible = seen_examples == required_examples
        improved = eligible and mean < best_validation
        if eligible and args.keep_checkpoints:
            checkpoint = args.output_dir / "checkpoints" / f"step_{step:06d}"
            model.save_pretrained(checkpoint)
            (checkpoint / "语义复核状态.json").write_text(json.dumps({"step": step, "总loss": mean, "角色验收": "待逐条生成复核", "可部署": False}, ensure_ascii=False, indent=2), encoding="utf8")
        if improved:
            best_validation = mean
            stale_evaluations = 0
            model.save_pretrained(args.output_dir / 'adapter')
        elif eligible:
            stale_evaluations += 1
        with (args.output_dir / 'validation_log.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'step': step, 'validation_loss': mean, 'supervised_tokens': tokens, 'saved_best': improved, '完整覆盖': eligible, '已覆盖样本': len(seen_examples), '总样本': len(required_examples)}) + '\n')
        model.train()
        return mean
    if validation:
        validate(0)
    model.train()
    started = time.perf_counter()
    with (args.output_dir / 'train_log.jsonl').open('w', encoding='utf-8') as log:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            batch_metadata = []
            for micro in range(2):
                if cursor == len(examples):
                    random.shuffle(examples)
                    cursor = 0
                ids, labels = examples[cursor]
                cursor += 1
                seen_examples.add(tuple(ids))
                batch_metadata.append(sample_metadata[tuple(ids)])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=torch.tensor([ids], device='cuda'),
                                 labels=torch.tensor([labels], device='cuda')).loss
                assert torch.isfinite(loss), '训练损失非有限值，停止'
                (loss / 2).backward()
                losses.append(loss.item())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(grad_norm), "梯度非有限值，停止"
            optimizer.step()
            record = {'step': step + 1, 'loss': sum(losses) / 2,
                      '微批样本': batch_metadata, '微批loss': losses,
                      '裁剪前梯度L2': float(grad_norm),
                      '裁剪系数': min(1.0, 1.0 / (float(grad_norm) + 1e-6)),
                      '已覆盖样本': len(seen_examples), '总样本': len(required_examples),
                      'elapsed_seconds': time.perf_counter() - started,
                      'peak_memory_mb': torch.cuda.max_memory_allocated() / 1048576}
            log.write(json.dumps(record) + '\n')
            log.flush()
            if (step + 1) % 4 == 0:
                print(json.dumps(record), flush=True)
            if validation and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
                print(json.dumps({'step': step + 1, 'validation_loss': validate(step + 1)}), flush=True)
                if stale_evaluations >= 3:
                    print('开发损失连续三次未改善，提前停止，保留开发最佳适配器。', flush=True)
                    break
    if not validation:
        model.save_pretrained(args.output_dir / 'adapter')
    if args.adapter_only:
        print('补充训练完成，已保存适配器；未合并或部署。', flush=True)
        return
    if args.load_in_4bit:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={'': 'cpu'})
        model = PeftModel.from_pretrained(base, args.output_dir / 'adapter').merge_and_unload()
    else:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(args.merged_output, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_output)
    print('补充训练完成，已保存独立候选。', flush=True)


if __name__ == '__main__':
    main()
