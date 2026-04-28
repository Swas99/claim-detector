"""Fine-tune a HuggingFace transformer model for claim detection.

Supports any sequence classification model: DistilBERT, BERT, DeBERTa, etc.
Uses HuggingFace Trainer with Apple Silicon MPS acceleration.

Usage:
    python -m src.training.train_transformer --model distilbert-base-uncased
    python -m src.training.train_transformer --model bert-base-uncased
    python -m src.training.train_transformer --model microsoft/deberta-v3-base
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import settings
from src.data.dataset import load_claim_dataset


def compute_metrics(eval_pred):
    """Compute metrics during training evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
    }


def train(
    model_name: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    output_dir: Path | None = None,
) -> dict:
    """Fine-tune a transformer model for claim detection."""

    # Determine output directory from model name
    short_name = model_name.split("/")[-1]
    output_dir = output_dir or settings.model_dir / short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    # Determine device
    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    print(f"Device: {device_str}")

    # Load data
    print("Loading dataset...")
    ds = load_claim_dataset()

    # Load tokenizer and model
    print(f"Loading {model_name}...")
    # DeBERTa-v3 needs use_fast=False due to tiktoken/spm compatibility
    use_fast = "deberta" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "not_claim", 1: "claim"},
        label2id={"not_claim": 0, "claim": 1},
    )

    # Tokenize
    print("Tokenizing...")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        fp16=False,  # MPS doesn't support fp16 well
        dataloader_num_workers=0,  # MPS compatibility
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\nTraining {model_name} for {epochs} epochs...")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")

    # Evaluate
    print("Evaluating best model...")
    eval_result = trainer.evaluate()

    # Get predictions for detailed metrics
    predictions = trainer.predict(tokenized["test"])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = np.array(ds["test"]["label"])
    y_prob = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

    metrics = {
        "model": model_name,
        "short_name": short_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "training_time_seconds": training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "device": device_str,
        "train_samples": len(ds["train"]),
        "test_samples": len(ds["test"]),
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "training_log": [
            {k: v for k, v in entry.items()}
            for entry in trainer.state.log_history
        ],
    }

    # Print results
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Train Loss: {metrics['train_loss']:.4f}")
    if metrics["eval_loss"] is not None:
        print(f"  Eval Loss:  {metrics['eval_loss']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # Save model, tokenizer, and metrics
    print(f"\nSaving to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Clean up checkpoints to save disk space
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    print("Done.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for claim detection")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
