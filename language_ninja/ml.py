from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

MODEL_VERSION = "2.2.1"


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def confidence_level(self) -> str:
        if self.confidence >= 0.75:
            return "HIGH"
        if self.confidence >= 0.55:
            return "MODERATE"
        return "LOW"


@dataclass(slots=True)
class TrainingReport:
    rows: int
    labels: dict[str, int]
    accuracy: float | None
    macro_f1: float | None
    report: str
    feature_count: int
    algorithm: str
    class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    label_order: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Dataset must contain text,label columns.")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError("Dataset contains no usable text/label rows.")
    if len(set(labels)) < 2:
        raise ValueError("Dataset needs at least two different labels.")
    return texts, labels


def build_vectorizer() -> FeatureUnion:
    # Word n-grams capture phrases such as "not good", "really love", and
    # contrast constructions. Character n-grams improve resilience to spelling,
    # contractions, punctuation, and unseen word forms.
    word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        max_df=0.995,
        max_features=30000,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=30000,
        strip_accents="unicode",
    )
    return FeatureUnion(
        [("word", word), ("char", char)],
        transformer_weights={"word": 1.4, "char": 0.7},
    )


def _fit_classifier(matrix, labels: list[str]):
    counts = {label: labels.count(label) for label in set(labels)}
    min_class = min(counts.values())

    if min_class >= 5:
        # Linear SVM gives a strong text decision boundary. Sigmoid calibration
        # converts its margins into probabilities that are more meaningful for
        # the TUI confidence display.
        base = LinearSVC(C=1.5, class_weight="balanced", random_state=42)
        classifier = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        classifier.fit(matrix, labels)
        return classifier, "Calibrated Linear SVM (5-fold sigmoid)"

    # Small custom datasets cannot support 5-fold calibration. Keep a reliable
    # probability-capable fallback so users can still train their own corpus.
    classifier = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(matrix, labels)
    return classifier, "Balanced Logistic Regression (small-data fallback)"


def train_model(texts: Iterable[str], labels: Iterable[str]):
    texts = list(texts)
    labels = list(labels)
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    classifier, algorithm = _fit_classifier(matrix, labels)
    return classifier, vectorizer, matrix.shape[1], algorithm


def train_and_evaluate(texts: Iterable[str], labels: Iterable[str]):
    texts = list(texts)
    labels = list(labels)
    label_order = sorted(set(labels))
    label_counts = {label: labels.count(label) for label in label_order}

    accuracy = None
    macro_f1 = None
    report_text = "Dataset is too small for a reliable holdout evaluation. Add more examples per label."
    class_metrics: dict[str, dict[str, float]] = {}
    matrix_rows: list[list[int]] = []

    min_class = min(label_counts.values())
    if len(texts) >= 30 and min_class >= 8:
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            random_state=31415,
            stratify=labels,
        )
        eval_clf, eval_vec, _, _ = train_model(x_train, y_train)
        transformed = eval_vec.transform(x_test)
        predictions = eval_clf.predict(transformed)
        accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
        report_text = classification_report(y_test, predictions, labels=label_order, zero_division=0)
        report_dict = classification_report(
            y_test, predictions, labels=label_order, output_dict=True, zero_division=0
        )
        class_metrics = {
            label: {
                "precision": float(report_dict[label]["precision"]),
                "recall": float(report_dict[label]["recall"]),
                "f1": float(report_dict[label]["f1-score"]),
                "support": float(report_dict[label]["support"]),
            }
            for label in label_order
        }
        matrix_rows = confusion_matrix(y_test, predictions, labels=label_order).tolist()

    classifier, vectorizer, feature_count, algorithm = train_model(texts, labels)
    return classifier, vectorizer, TrainingReport(
        rows=len(texts),
        labels=label_counts,
        accuracy=accuracy,
        macro_f1=macro_f1,
        report=report_text,
        feature_count=feature_count,
        algorithm=algorithm,
        class_metrics=class_metrics,
        confusion=matrix_rows,
        label_order=label_order,
    )


def save_model(path: str | Path, classifier, vectorizer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MODEL_VERSION,
        "classifier": classifier,
        "vectorizer": vectorizer,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_model(path: str | Path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    # Backward compatibility with v2.0/v2.1 tuple model files.
    if isinstance(payload, tuple) and len(payload) == 2:
        return payload
    if isinstance(payload, dict) and "classifier" in payload and "vectorizer" in payload:
        return payload["classifier"], payload["vectorizer"]
    raise ValueError("Unsupported model file format.")


def predict(text: str, classifier, vectorizer) -> Prediction:
    matrix = vectorizer.transform([text])
    label = str(classifier.predict(matrix)[0])
    probabilities: dict[str, float] = {}
    confidence = 1.0
    if hasattr(classifier, "predict_proba"):
        values = classifier.predict_proba(matrix)[0]
        probabilities = {
            str(name): float(value) for name, value in zip(classifier.classes_, values)
        }
        confidence = probabilities.get(label, max(probabilities.values(), default=1.0))
    return Prediction(label=label, confidence=confidence, probabilities=probabilities)
