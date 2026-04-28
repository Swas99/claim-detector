"""Train a TF-IDF + XGBoost baseline for claim detection.

This serves as a classical ML baseline to demonstrate the value
of transformer-based models. Trains in seconds, no GPU needed.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.config import settings
from src.data.dataset import load_claim_dataset


def train(output_dir: Path | None = None) -> dict:
    """Train TF-IDF + XGBoost and return metrics."""
    output_dir = output_dir or settings.model_dir / "tfidf-xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    ds = load_claim_dataset()
    train_texts = ds["train"]["text"]
    train_labels = np.array(ds["train"]["label"])
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])

    # TF-IDF vectorization
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train matrix: {X_train.shape}")

    # XGBoost classifier
    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, train_labels)

    # Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "tfidf-xgboost",
        "accuracy": float(accuracy_score(test_labels, y_pred)),
        "precision": float(precision_score(test_labels, y_pred)),
        "recall": float(recall_score(test_labels, y_pred)),
        "f1": float(f1_score(test_labels, y_pred)),
        "confusion_matrix": confusion_matrix(test_labels, y_pred).tolist(),
        "classification_report": classification_report(test_labels, y_pred, output_dict=True),
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"TF-IDF + XGBoost Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # Save artifacts
    print(f"\nSaving to {output_dir}...")
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    return metrics


if __name__ == "__main__":
    train()
