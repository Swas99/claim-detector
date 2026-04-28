"""Token-level attribution for claim detection predictions.

Uses attention weights from the transformer's last layer to approximate
which tokens contributed most to the prediction. This is a lightweight
proxy for full gradient-based attribution (e.g., integrated gradients),
trading some fidelity for much faster inference.

Production upgrade: use captum or shap for rigorous attribution.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class AttentionAttributor:
    """Extract token importance scores from transformer attention weights."""

    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def attribute(self, text: str, max_length: int = 128, top_k: int = 5) -> list[dict]:
        """Compute token attribution scores for a single sentence.

        Returns a list of {token, weight} dicts, sorted by weight descending.
        Special tokens ([CLS], [SEP], [PAD]) are excluded.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        # Filter inputs to match model signature
        import inspect
        valid_params = set(inspect.signature(self.model.forward).parameters.keys())
        filtered = {k: v for k, v in inputs.items() if k in valid_params}

        with torch.no_grad():
            outputs = self.model(**filtered, output_attentions=True)

        # Average attention across all heads in the last layer
        # Shape: (1, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
        last_layer_attention = outputs.attentions[-1][0].mean(dim=0)

        # CLS token's attention over all other tokens (row 0)
        cls_attention = last_layer_attention[0].numpy()

        # Map back to tokens
        token_ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Filter out special tokens, padding, and punctuation
        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>",
                          ".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"}
        attributions = []
        for token, weight in zip(tokens, cls_attention):
            if token in special_tokens:
                continue
            # Merge subword tokens (## prefix for BERT-family)
            if token.startswith("##") and attributions:
                attributions[-1]["token"] += token[2:]
                attributions[-1]["weight"] += float(weight)
            else:
                attributions.append({"token": token, "weight": float(weight)})

        # Normalize weights to sum to 1
        total = sum(a["weight"] for a in attributions)
        if total > 0:
            for a in attributions:
                a["weight"] = round(a["weight"] / total, 4)

        # Sort by weight descending, return top_k
        attributions.sort(key=lambda x: x["weight"], reverse=True)
        return attributions[:top_k]
