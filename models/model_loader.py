"""
Model loader for the fake-news detection backend.

The transformer path uses `hamzab/roberta-fake-news-classification`, a
RoBERTa model already fine-tuned on fake-news data — NOT an untrained base
model. This avoids the previous bug where a randomly-initialised DeBERTa-v3
was loaded and produced meaningless predictions.

Label convention for that model:
  FAKE  →  label 0
  TRUE  →  label 1  (the model uses "TRUE" for real news)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional


# ── Transformer model (HuggingFace pipeline) ────────────────────────────────

_TRANSFORMER_MODEL_ID = "hamzab/roberta-fake-news-classification"

_pipeline_instance: Optional[Any] = None


def get_transformer_pipeline():
    """
    Lazy-loads the fine-tuned RoBERTa pipeline. Cached after first call.
    Raises RuntimeError if the transformers package is not available.
    """
    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is not installed. "
            "Run: pip install transformers torch"
        ) from exc

    print(f"Loading transformer model: {_TRANSFORMER_MODEL_ID}")
    _pipeline_instance = hf_pipeline(
        "text-classification",
        model=_TRANSFORMER_MODEL_ID,
        device=-1,          # CPU; set to 0 for GPU
        truncation=True,
        max_length=512,
    )
    print("Transformer model loaded successfully.")
    return _pipeline_instance


def predict_with_transformer(text: str) -> Dict[str, Any]:
    """
    Run a single text through the fine-tuned RoBERTa model.

    Returns a dict compatible with what app.py expects:
        {
            "prediction":  "REAL" | "FAKE",
            "confidence":  float,          # confidence of the predicted class
            "model_used":  str,
        }
    """
    pipe = get_transformer_pipeline()
    truncated = text[:1500]
    result = pipe(truncated)[0]

    raw_label: str = result["label"]   # "FAKE" or "TRUE"
    score: float   = float(result["score"])

    # Normalise label to REAL / FAKE
    prediction = "FAKE" if raw_label == "FAKE" else "REAL"

    return {
        "prediction": prediction,
        "confidence": score,
        "model_used": f"Transformer ({_TRANSFORMER_MODEL_ID})",
    }


def predict_transformer_fake_prob(texts: List[str]) -> List[float]:
    """
    Batch helper used by the perturbation explainer.
    Returns a list of P(FAKE) values in [0, 1].
    """
    pipe = get_transformer_pipeline()
    truncated = [t[:1500] for t in texts]
    results = pipe(truncated)
    probs: List[float] = []
    for res in results:
        label: str  = res["label"]
        score: float = float(res["score"])
        # score is the confidence for the predicted label
        if label == "FAKE":
            probs.append(score)
        else:
            probs.append(1.0 - score)
    return probs


# ── Legacy class-based API (kept for backwards compatibility) ────────────────
# Code that imports FakeNewsModel or get_model_instance still works, but now
# delegates to the pipeline above instead of loading an untrained base model.

class FakeNewsModel:
    """Thin wrapper around the HuggingFace pipeline, kept for API compatibility."""

    def __init__(self, model_name: str = _TRANSFORMER_MODEL_ID):
        self.model_name = model_name
        self.label_map  = {0: "Fake", 1: "Real"}
        self._loaded    = False

    def load_model(self):
        # Triggers lazy load of the shared pipeline
        get_transformer_pipeline()
        self._loaded = True
        print("Model ready.")

    def predict(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Call load_model() first.")
        result = predict_with_transformer(text)
        pred_label = result["prediction"]
        confidence = result["confidence"]
        label_idx  = 1 if pred_label == "REAL" else 0
        return {
            "prediction": pred_label.capitalize(),   # "Real" / "Fake"
            "confidence": confidence,
            "label":      label_idx,
            "probabilities": {
                "Fake": 1.0 - confidence if pred_label == "REAL" else confidence,
                "Real": confidence        if pred_label == "REAL" else 1.0 - confidence,
            },
        }

    def predict_batch(self, texts: List[str], max_length: int = 512) -> List[Dict[str, Any]]:
        if not self._loaded:
            raise RuntimeError("Call load_model() first.")
        return [self.predict(t, max_length) for t in texts]


# Convenience factory & singleton — unchanged public API
def get_model(model_type: str = "roberta") -> FakeNewsModel:
    """Returns a FakeNewsModel instance (model_type is ignored; always RoBERTa)."""
    return FakeNewsModel()


_model_instance: Optional[FakeNewsModel] = None


def get_model_instance(model_type: str = "roberta") -> FakeNewsModel:
    """Singleton accessor."""
    global _model_instance
    if _model_instance is None:
        _model_instance = FakeNewsModel()
        _model_instance.load_model()
    return _model_instance
