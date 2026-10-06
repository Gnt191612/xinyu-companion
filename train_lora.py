"""微调本地模型；仅目标 JSON 计算损失，记录配置与训练过程。"""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_data import read_rows

ROOT = Path(__file__).resolve().parent


def official_module():
    spec = importlib.util.spec_from_file_location("official_inference", ROOT / "participant" / "run_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_messages(sample, official):
    system = official.SYSTEM_PROMPT.format(emotions=", ".join(sorted(official.EMOTIONS)),
        personality=", ".join(sorted(official.PROFILE_LABELS["personality_traits"])),
        interests=", ".join(sorted(official.PROFILE_LABELS["interests"])),
        styles=", ".join(sorted(official.PROFILE_LABELS["style"])))
    return [{"role": "system", "content": system}, {"role": "user", "content": official.conversation_text(sample)}]


def main():
    parser = argparse.ArgumentParser(description="离线 LoRA 训练")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--sample-limit", type=int, default=2048)
    parser.add_argument("--model-path", type=Path, default=ROOT / "模型" / "Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "实验" / "lora")
    parser.add_argument("--merged-output", type=Path, default=ROOT / "participant" / "model")
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model_path = args.model_path.resolve()
    output = args.output_dir.resolve()
    if (output / "train_log.jsonl").exists():
        raise RuntimeError("训练日志已存在，请指定新输出目录，保留原实验记录")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    official = official_module()
    rows = read_rows(ROOT / "数据" / "processed" / "train_sft.jsonl")
    random.shuffle(rows)
    examples = []
    # 不截断目标 JSON，避免训练出不完整的结构化输出。
    for row in rows:
        target = {key: value for key, value in row["target"].items() if key != "id"}
        target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        target_ids = tokenizer.encode(target_text + tokenizer.eos_token, add_special_tokens=False)
        if len(target_ids) > 640:
            continue
        messages = make_messages(row["input"], official)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        allowance = args.max_length - len(target_ids)
        # 保留枚举提示和最近消息；完整验证仍按官方输入执行。
        if len(prompt_ids) > allowance:
            system_tokens = tokenizer.encode(tokenizer.apply_chat_template(messages[:1], tokenize=False,
                                                                          add_generation_prompt=False), add_special_tokens=False)
            if len(system_tokens) >= allowance - 128:
                continue
            prompt_ids = system_tokens + prompt_ids[-(allowance - len(system_tokens)):]
        examples.append((prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids))
        if len(examples) >= args.sample_limit:
            break
    config = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "seed": 42, "steps": args.steps,
              "max_length": args.max_length, "examples": len(examples), "batch_size": 1,
              "gradient_accumulation": 4, "learning_rate": args.learning_rate, "lora_rank": 8,
              "实际模型路径": str(model_path), "合并输出目录": str(args.merged_output.resolve()),
              "target_modules": ["q_proj", "v_proj"], "验证集参与训练": False,
              "长目标处理": "目标超过640 token的训练点不用于本轮；不截断目标JSON"}
    (output / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
    model.train()
    started = time.perf_counter()
    log = (output / "train_log.jsonl").open("w", encoding="utf-8")
    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(4):
                ids, labels = examples[(step * 4 + micro) % len(examples)]
                tensors = {"input_ids": torch.tensor([ids], device="cuda"),
                           "labels": torch.tensor([labels], device="cuda")}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**tensors).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("出现非有限训练损失，停止保存模型")
                (loss / 4).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record = {"step": step + 1, "loss": sum(losses) / 4,
                      "elapsed_seconds": time.perf_counter() - started,
                      "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (step + 1) % 5 == 0:
                print(json.dumps(record), flush=True)
        model.save_pretrained(output / "adapter")
        model = model.merge_and_unload()
        model.config.use_cache = True
        model.save_pretrained(args.merged_output, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_output)
        print("训练结束，已保存可离线加载的合并模型", flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    main()
