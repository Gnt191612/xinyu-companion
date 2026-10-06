"""字符 TF-IDF 监督基线；分类情感并推断三个多标签画像维度。"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from prepare_data import read_rows

ROOT = Path(__file__).resolve().parent


def sample_text(sample, mode):
    users = [turn["content"] for turn in sample["history"] if turn["role"] == "user"]
    return users[-1] if mode == "latest" else "\n".join(users[-8:])


def main():
    start = time.perf_counter()
    directory = ROOT / "数据" / "processed"
    train = read_rows(directory / "train_sft.jsonl")
    val = read_rows(directory / "val_sft.jsonl")
    output = ROOT / "实验" / "classifiers"
    output.mkdir(parents=True, exist_ok=True)
    report = {"说明": "本地验证指标；画像为集合完全匹配和micro F1，不能等同官网未公开细则的画像平均准确率。", "seed": 42, "runs": {}}
    for mode in ("latest", "history"):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3, max_features=60000,
                                     sublinear_tf=True, dtype=np.float32)
        train_x = vectorizer.fit_transform([sample_text(row["input"], mode) for row in train])
        val_x = vectorizer.transform([sample_text(row["input"], mode) for row in val])
        model = LogisticRegression(C=4, max_iter=400, random_state=42, solver="lbfgs")
        train_y = [row["target"]["emotion_label"] for row in train]
        val_y = [row["target"]["emotion_label"] for row in val]
        model.fit(train_x, train_y)
        predictions = model.predict(val_x)
        result = {"emotion_accuracy": accuracy_score(val_y, predictions),
                  "emotion_macro_f1": f1_score(val_y, predictions, average="macro", zero_division=0), "profiles": {}}
        profile_models, profile_labels = {}, {}
        profile_pred = {}
        for key in ("personality_traits", "interests", "style"):
            labels = MultiLabelBinarizer()
            y = labels.fit_transform([row["target"]["user_profile"][key] for row in train])
            truth = labels.transform([row["target"]["user_profile"][key] for row in val])
            classifier = OneVsRestClassifier(LogisticRegression(C=4, max_iter=300, random_state=42, solver="liblinear"), n_jobs=1)
            classifier.fit(train_x, y)
            predicted = classifier.predict(val_x)
            result["profiles"][key] = {"exact_match": accuracy_score(truth, predicted),
                                       "micro_f1": f1_score(truth, predicted, average="micro", zero_division=0)}
            profile_models[key] = classifier
            profile_labels[key] = labels
            profile_pred[key] = labels.inverse_transform(predicted)
        result["profile_mean_exact_match"] = float(np.mean([item["exact_match"] for item in result["profiles"].values()]))
        report["runs"][mode] = result
        joblib.dump({"vectorizer": vectorizer, "emotion": model, "profile_models": profile_models,
                     "profile_labels": profile_labels, "mode": mode}, output / f"{mode}.joblib", compress=3)
        errors = [{"id": row["input"]["id"], "text": sample_text(row["input"], mode), "truth": truth_label,
                   "prediction": pred} for row, truth_label, pred in zip(val, val_y, predictions) if truth_label != pred]
        (output / f"{mode}_emotion_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(mode, json.dumps(result, ensure_ascii=False), flush=True)
    report["majority_emotion_accuracy"] = val_y.count(max(set(train_y), key=train_y.count)) / len(val_y)
    report["elapsed_seconds"] = time.perf_counter() - start
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
