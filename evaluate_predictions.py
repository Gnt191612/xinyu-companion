"""检查完整输出并报告本地指标；不伪造官方总分。"""
import argparse
import json
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

from prepare_data import read_rows


class CharacterTokenizer:
    def tokenize(self, text):
        return [character for character in text if not character.isspace()]


def main():
    parser = argparse.ArgumentParser(description="本地预测评估")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args()
    predictions = read_rows(args.predictions)
    targets = read_rows(args.targets)
    truth = {row["id"]: row for row in targets}
    assert len({row["id"] for row in predictions}) == len(predictions), "重复预测编号"
    assert {row["id"] for row in predictions} == set(truth), "预测和目标编号不一致"
    targets = [truth[row["id"]] for row in predictions]
    report = {"samples": len(predictions), "emotion_accuracy": accuracy_score(
        [row["emotion_label"] for row in targets], [row["emotion_label"] for row in predictions]),
        "emotion_macro_f1": f1_score([row["emotion_label"] for row in targets], [row["emotion_label"] for row in predictions], average="macro", zero_division=0), "profiles": {}}
    for key in ("personality_traits", "interests", "style"):
        matches = [set(pred["user_profile"][key]) == set(target["user_profile"][key]) for pred, target in zip(predictions, targets)]
        report["profiles"][key] = {"exact_match": float(np.mean(matches))}
    report["profile_mean_exact_match"] = float(np.mean([value["exact_match"] for value in report["profiles"].values()]))
    tokenize = CharacterTokenizer().tokenize
    references = [[tokenize(row["response_text"])] for row in targets]
    candidates = [tokenize(row["response_text"]) for row in predictions]
    report["character_bleu4"] = corpus_bleu(references, candidates, smoothing_function=SmoothingFunction().method1)
    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=CharacterTokenizer())
    report["character_rougeL_f1"] = float(np.mean([scorer.score(target["response_text"], pred["response_text"])["rougeL"].fmeasure for pred, target in zip(predictions, targets)]))
    report["empty_responses"] = sum(not row["response_text"].strip() for row in predictions)
    report["说明"] = "回复指标采用中文字符切分；画像为集合完全匹配。未计算BERTScore，未复现官方归一化，因此不能作为官方比赛分数。"
    path = args.predictions.parent / "local_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
