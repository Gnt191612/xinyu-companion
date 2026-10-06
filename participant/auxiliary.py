"""使用已训练的监督分类器补充生成模型的情感与画像输出。"""
from pathlib import Path

import joblib


class AuxiliaryPredictor:
    def __init__(self):
        directory = Path(__file__).resolve().parent / "classifiers"
        self.latest = joblib.load(directory / "latest.joblib")
        self.history = joblib.load(directory / "history.joblib")

    def predict(self, sample):
        users = [turn["content"] for turn in sample["history"] if turn["role"] == "user"]
        latest_x = self.latest["vectorizer"].transform([users[-1]])
        history_x = self.history["vectorizer"].transform(["\n".join(users[-8:])])
        emotion = self.latest["emotion"].predict(latest_x)[0]
        profile = {key: list(self.history["profile_labels"][key].inverse_transform(
            self.history["profile_models"][key].predict(history_x))[0])
            for key in ("personality_traits", "interests", "style")}
        return str(emotion), profile

    def emotion_evidence(self, sample):
        """供体验界面显示不确定性，分数不是经过校准的真实概率。"""
        latest_user = next(turn['content'] for turn in reversed(sample['history']) if turn['role'] == 'user')
        features = self.latest['vectorizer'].transform([latest_user])
        scores = sorted(self.latest['emotion'].predict_proba(features)[0], reverse=True)
        return {'emotion_score': float(scores[0]),
                'emotion_uncertain': float(scores[0]) < 0.6 or float(scores[0] - scores[1]) < 0.2}
