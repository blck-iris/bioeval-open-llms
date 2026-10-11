"""Pure-numpy evaluation metrics (no torch needed, so they are easy to unit test)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def expected_calibration_error(conf: Sequence[float], correct: Sequence[int], n_bins: int = 10):
    """ECE with equal-width bins. Returns (ece, per-bin rows)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc, avg_conf, frac = correct[mask].mean(), conf[mask].mean(), mask.mean()
        ece += frac * abs(acc - avg_conf)
        rows.append({"lo": lo, "hi": hi, "n": int(mask.sum()), "acc": float(acc), "conf": float(avg_conf)})
    return float(ece), rows


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two label vectors of equal length."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float((a == b).mean())
    pe = sum(float((a == l).mean() * (b == l).mean()) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def cyclic_permutation(n: int, shift: int) -> list[int]:
    """Presentation order: position j shows original option perm[j]."""
    return [(j + shift) % n for j in range(n)]


def softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
