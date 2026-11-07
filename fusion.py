"""Local post-processing: fuse per-phrase evidence from the lip-reading models into one calibrated distribution.

features[clip, phrase, k] is evidence source k's score for that phrase (e.g. log-prob from Gemini, from Muse Spark,
log prior from the user's phrase bank). The model is a conditional logit: one shared weight per source, softmax over
phrases. Few weights means it trains from tens of labeled clips, and the weights double as temperature calibration.

Pools of different sizes are padded into one tensor; `mask[clip, candidate]` is True for real candidates, and padded
entries get zero probability and contribute nothing to the fit. `mask=None` means every entry is real.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax


def _logits(features: np.ndarray, weights: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    logits = features @ weights
    return logits if mask is None else np.where(mask, logits, -np.inf)


def predict(features: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return np.exp(log_softmax(_logits(features, weights, mask), axis=1))


def fit(features: np.ndarray, labels: np.ndarray, l2: float = 1e-3, mask: np.ndarray | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None) -> np.ndarray:
    """`bounds`: per-weight (low, high) limits, None for no limit (scipy L-BFGS-B bounds)."""
    rows = np.arange(len(labels))

    def loss(weights):
        log_probs = log_softmax(_logits(features, weights, mask), axis=1)
        expected = np.einsum("np,npk->k", np.exp(log_probs), features)
        observed = features[rows, labels].sum(0)
        nll = -log_probs[rows, labels].mean() + l2 * weights @ weights
        return nll, (expected - observed) / len(labels) + 2 * l2 * weights

    return minimize(loss, np.ones(features.shape[2]), jac=True, method="L-BFGS-B", bounds=bounds).x


def gate_threshold(probs: np.ndarray, labels: np.ndarray, precision: float = 0.95) -> float:
    """Lowest top-1 confidence at which auto-sent picks still meet `precision`; inf means always ask the user."""
    confidence = probs.max(1)
    order = np.argsort(-confidence)
    correct = (probs.argmax(1) == labels)[order]
    running_precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    passing = np.flatnonzero(running_precision >= precision)
    return confidence[order][passing[-1]] if len(passing) else np.inf
