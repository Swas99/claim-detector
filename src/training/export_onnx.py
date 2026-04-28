"""Export a trained transformer model to ONNX format.

Uses HuggingFace Optimum for reliable ONNX export across model architectures.
ONNX Runtime provides 2-3x faster CPU inference compared to PyTorch.

Usage:
    python -m src.training.export_onnx
    python -m src.training.export_onnx --model models/distilbert-base-uncased
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


def export_to_onnx(
    model_dir: Path,
    output_dir: Path,
) -> Path:
    """Export a trained PyTorch model to ONNX using Optimum."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    print(f"Exporting to {output_dir}...")

    # Optimum handles all model-specific export quirks
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, export=True
    )
    ort_model.save_pretrained(str(output_dir))

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(str(output_dir))

    # Copy source metrics
    metrics_src = model_dir / "metrics.json"
    if metrics_src.exists():
        import shutil
        shutil.copy2(metrics_src, output_dir / "source_metrics.json")

    onnx_path = output_dir / "model.onnx"
    if onnx_path.exists():
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model size: {file_size_mb:.1f} MB")

    return onnx_path


def benchmark(
    model_dir: Path,
    onnx_dir: Path,
    max_length: int = 128,
    n_runs: int = 100,
) -> dict:
    """Benchmark PyTorch vs ONNX Runtime inference speed."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    test_sentences = [
        "The Empire State Building is the tallest building in New York City.",
        "I think pizza is the best food ever.",
        "GDP grew 2.1% last quarter according to official statistics.",
        "What time does the meeting start?",
        "The unemployment rate fell to 3.5% in December 2023.",
    ]

    inputs = tokenizer(
        test_sentences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # PyTorch benchmark
    print("\nBenchmarking PyTorch...")
    import inspect
    import torch
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    pt_model.eval()

    # Filter inputs to only those the model accepts
    valid_params = set(inspect.signature(pt_model.forward).parameters.keys())
    pt_inputs = {k: v for k, v in inputs.items() if k in valid_params}

    with torch.no_grad():
        for _ in range(5):
            pt_model(**pt_inputs)

    pt_times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            pt_model(**pt_inputs)
            pt_times.append(time.perf_counter() - start)

    # ONNX Runtime benchmark
    print("Benchmarking ONNX Runtime...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))

    ort_inputs = tokenizer(
        test_sentences,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    for _ in range(5):
        ort_model(**tokenizer(test_sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length))

    ort_times = []
    for _ in range(n_runs):
        pt_inputs = tokenizer(test_sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        start = time.perf_counter()
        ort_model(**pt_inputs)
        ort_times.append(time.perf_counter() - start)

    # Verify outputs match
    with torch.no_grad():
        pt_logits = pt_model(**pt_inputs).logits.numpy()
    ort_out = ort_model(**pt_inputs)
    ort_logits = ort_out.logits.detach().numpy()
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))

    results = {
        "pytorch_ms": {
            "mean": float(np.mean(pt_times) * 1000),
            "p50": float(np.median(pt_times) * 1000),
            "p95": float(np.percentile(pt_times, 95) * 1000),
            "p99": float(np.percentile(pt_times, 99) * 1000),
        },
        "onnx_ms": {
            "mean": float(np.mean(ort_times) * 1000),
            "p50": float(np.median(ort_times) * 1000),
            "p95": float(np.percentile(ort_times, 95) * 1000),
            "p99": float(np.percentile(ort_times, 99) * 1000),
        },
        "speedup": float(np.mean(pt_times) / np.mean(ort_times)),
        "max_logit_diff": max_diff,
        "batch_size": len(test_sentences),
        "n_runs": n_runs,
    }

    print(f"\n{'='*50}")
    print(f"Benchmark Results ({len(test_sentences)} sentences, {n_runs} runs)")
    print(f"{'='*50}")
    print(f"  PyTorch:      {results['pytorch_ms']['mean']:.1f}ms mean, {results['pytorch_ms']['p95']:.1f}ms p95")
    print(f"  ONNX Runtime: {results['onnx_ms']['mean']:.1f}ms mean, {results['onnx_ms']['p95']:.1f}ms p95")
    print(f"  Speedup:      {results['speedup']:.2f}x")
    print(f"  Max logit diff: {max_diff:.6f} (should be < 0.001)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model dir")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model) if args.model else settings.model_dir / "distilbert-base-uncased"
    output_dir = Path(args.output) if args.output else settings.model_dir / "onnx"

    onnx_path = export_to_onnx(model_dir, output_dir)

    if not args.skip_benchmark:
        results = benchmark(model_dir, output_dir)
        with open(output_dir / "benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark saved to {output_dir / 'benchmark.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
