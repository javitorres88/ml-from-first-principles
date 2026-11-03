"""Module 01 — evaluation: metrics, splits and cross-validation."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "confusion_matrix_scratch",
    "metrics_scratch",
    "stratified_holdout_scratch",
    "k_fold_indices_scratch",
    "confusion_matrix",
    "metrics",
    "stratified_split",
    "standardize",
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "k_fold_indices",
]


# --------------------------------------------------------------- pure python
def confusion_matrix_scratch(
    y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1
) -> dict[str, int]:
    """Count the four cells of a binary confusion matrix."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive:
            tp += t == positive
            fp += t != positive
        else:
            fn += t == positive
            tn += t != positive
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def metrics_scratch(cm: dict[str, int]) -> dict[str, float]:
    """Accuracy, sensitivity, specificity, precision, F1 and balanced accuracy."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n = tp + fp + tn + fn
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
    return {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": (sens + spec) / 2,
        "prevalence": (tp + fn) / n if n else float("nan"),
    }


def stratified_holdout_scratch(y: Sequence[int], frac: float, seed: int = 0) -> list[int]:
    """Indices of a hold-out set that preserves each class's prevalence."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)
    held: list[int] = []
    for idx in by_class.values():
        rng.shuffle(idx)
        held.extend(idx[: max(1, round(len(idx) * frac))])
    return sorted(held)


def k_fold_indices_scratch(n: int, k: int = 5, seed: int = 0) -> list[list[int]]:
    """Return k lists of indices covering range(n) exactly once."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


# -------------------------------------------------------------------- numpy
def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> np.ndarray:
    """Return ``[[TN, FP], [FN, TP]]`` — the scikit-learn convention."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]])


def metrics(cm: np.ndarray) -> dict[str, float]:
    """Metrics from a 2x2 confusion matrix in scikit-learn layout."""
    tn, fp, fn, tp = np.asarray(cm).ravel().astype(float)
    total = tn + fp + fn + tp
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
    return {
        "accuracy": (tp + tn) / total if total else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": (sens + spec) / 2,
        "prevalence": (tp + fn) / total if total else float("nan"),
    }


def stratified_split(y: Sequence[int], frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Boolean mask selecting a stratified hold-out."""
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y_arr), dtype=bool)
    for cls in np.unique(y_arr):
        idx = np.flatnonzero(y_arr == cls)
        rng.shuffle(idx)
        mask[idx[: max(1, int(round(len(idx) * frac)))]] = True
    return mask


def standardize(
    X: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise standardisation; fit on training data and reuse the stats."""
    X = np.asarray(X, dtype=float)
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


def mse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean((a - b) ** 2))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(a - b)))


def r2_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def k_fold_indices(n: int, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Shuffled k-fold index arrays."""
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]
