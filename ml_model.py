from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/email_priority.joblib")
DATA_PATH = Path("data/emails.csv")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def _combine(subject: str, body: str) -> str:
    return f"Subject: {subject.strip()}\nBody: {body.strip()}".strip()


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "body", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["subject", "body", "label"]).copy()
    df["text"] = df.apply(lambda row: _combine(str(row["subject"]), str(row["body"])), axis=1)
    df["label"] = df["label"].str.strip().str.lower()
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    strip_accents="unicode",
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    df = load_dataset(data_path)
    stratify = df["label"] if df["label"].nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, model_path)
    return metrics


def load_or_train_model() -> tuple[Pipeline, dict[str, Any]]:
    if not MODEL_PATH.exists():
        metrics = train_model()
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle.get("metrics", metrics if "metrics" in locals() else {})


def predict_priority(subject: str, body: str) -> Prediction:
    pipeline, _ = load_or_train_model()
    text = _combine(subject, body)
    label = str(pipeline.predict([text])[0])

    probabilities: dict[str, float] = {}
    confidence = 0.0
    if hasattr(pipeline, "predict_proba"):
        classes = [str(cls) for cls in pipeline.classes_]
        scores = pipeline.predict_proba([text])[0]
        probabilities = {cls: round(float(score), 4) for cls, score in zip(classes, scores)}
        confidence = probabilities.get(label, 0.0)

    return Prediction(label=label, confidence=confidence, probabilities=probabilities)
