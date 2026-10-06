"""整理官方数据，严格截断到目标用户轮次，生成训练与独立验证样本。"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expand(rows, prefix):
    samples, targets, excluded = [], [], []
    for number, row in enumerate(rows, 1):
        turns = {turn["turn_id"]: turn for turn in row["turns"]}
        for point in row["prediction_points"]:
            stop = point["target_user_turn_id"]
            assert turns[stop]["role"] == "user"
            if point["assistant_response_turn_id"] <= stop:
                excluded.append({"conversation_id": row["conversation_id"], "target_user_turn_id": stop,
                                 "原因": "回复轮次不晚于目标用户轮次，会把目标回复泄漏到输入"})
                continue
            target = dict(point["target"])
            assert turns[point["assistant_response_turn_id"]]["content"] == target["response_text"]
            sample = {"id": f"{prefix}_{number:06d}_t{stop}", "conversation_id": row["conversation_id"],
                      "target_user_turn_id": stop,
                      "history": [turn for turn in row["turns"] if turn["turn_id"] <= stop]}
            assert sample["history"][-1]["turn_id"] == stop
            # 官方推理提示要求没有公开记忆库时输出空引用。
            target["memory_refs"] = []
            samples.append(sample)
            targets.append({"id": sample["id"], **target})
    return samples, targets, excluded


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    output = ROOT / "数据" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    report = {"切分方式": "使用官方训练集与验证集，不按预测轮次随机拆分", "splits": {}}
    conversations, fingerprints = {}, {}
    for split in ("train", "val"):
        path = ROOT / "数据" / split / f"{split}_public.jsonl"
        rows = read_rows(path)
        ids = [row["conversation_id"] for row in rows]
        assert len(set(ids)) == len(ids), "会话编号重复"
        conversations[split] = set(ids)
        fingerprints[split] = {hashlib.sha256(json.dumps(row["turns"], ensure_ascii=False, sort_keys=True).encode()).hexdigest() for row in rows}
        samples, targets, excluded = expand(rows, split)
        write_rows(output / f"{split}_excluded.jsonl", excluded)
        write_rows(output / f"{split}_inputs.jsonl", samples)
        write_rows(output / f"{split}_targets.jsonl", targets)
        write_rows(output / f"{split}_sft.jsonl", [{"input": sample, "target": target} for sample, target in zip(samples, targets)])
        report["splits"][split] = {"conversations": len(rows), "prediction_points": len(samples),
            "emotion_counts": dict(Counter(target["emotion_label"] for target in targets)),
            "profile_counts": {key: dict(Counter(label for target in targets for label in target["user_profile"][key])) for key in ("personality_traits", "interests", "style")},
            "excluded_prediction_points": len(excluded),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "max_history_characters": max(sum(len(turn["content"]) for turn in sample["history"]) for sample in samples)}
    overlap = fingerprints["train"] & fingerprints["val"]
    report["train_val_identical_conversations"] = len(overlap)
    assert not overlap, "训练验证集有完全相同的对话，需先处理泄漏"
    write_rows(output / "val_smoke_inputs.jsonl", read_rows(output / "val_inputs.jsonl")[:8])
    write_rows(output / "val_smoke_targets.jsonl", read_rows(output / "val_targets.jsonl")[:8])
    write_rows(output / "val_100_inputs.jsonl", read_rows(output / "val_inputs.jsonl")[:100])
    write_rows(output / "val_100_targets.jsonl", read_rows(output / "val_targets.jsonl")[:100])
    (ROOT / "数据检查.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
