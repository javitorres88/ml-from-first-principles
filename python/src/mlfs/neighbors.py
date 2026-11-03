"""Module 02 — k-nearest neighbours and distance metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "euclidean",
    "manhattan",
    "cosine_distance",
    "mahalanobis",
    "knn_predict_scratch",
    "KNNScratch",
    "pairwise_distances",
    "knn_predict",
    "KNNNumPy",
]


# --------------------------------------------------------------- pure python
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def mahalanobis(a: Sequence[float], b: Sequence[float], cov_inv: np.ndarray) -> float:
    """Mahalanobis distance; pass the *inverse* covariance matrix."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(d @ cov_inv @ d))


def knn_predict_scratch(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    x_query: Sequence[float],
    k: int = 3,
    dist: Callable[[Sequence[float], Sequence[float]], float] = euclidean,
    weighted: bool = False,
) -> tuple[int, list[tuple[float, int]]]:
    """Majority (or distance-weighted) vote among the k nearest neighbours."""
    pairs = sorted(((dist(x_query, x), label) for x, label in zip(X_train, y_train)))
    neighbours = pairs[:k]
    if weighted:
        votes: dict[int, float] = {}
        for d, label in neighbours:
            votes[label] = votes.get(label, 0.0) + 1.0 / (d + 1e-12)
        return max(votes, key=votes.get), neighbours
    counts = Counter(label for _, label in neighbours)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return best, neighbours


class KNNScratch:
    """A minimal k-NN classifier in pure Python."""

    def __init__(self, k: int = 3, dist=euclidean, weighted: bool = False):
        self.k = k
        self.dist = dist
        self.weighted = weighted

    def fit(self, X, y):
        self.X_, self.y_ = list(X), list(y)
        return self

    def predict_one(self, x) -> int:
        return knn_predict_scratch(self.X_, self.y_, x, self.k, self.dist, self.weighted)[0]

    def predict(self, X) -> list[int]:
        return [self.predict_one(x) for x in X]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(list(y))


# -------------------------------------------------------------------- numpy
def pairwise_distances(X: np.ndarray, Q: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distances between every row of Q (n_q, d) and every row of X (n, d)."""
    X = np.asarray(X, float)
    Q = np.asarray(Q, float)
    if metric == "euclidean":
        # ||q - x||^2 = ||q||^2 - 2 q.x + ||x||^2
        d2 = (Q**2).sum(1)[:, None] - 2 * Q @ X.T + (X**2).sum(1)[None, :]
        return np.sqrt(np.clip(d2, 0.0, None))
    if metric == "manhattan":
        return np.abs(Q[:, None, :] - X[None, :, :]).sum(axis=2)
    if metric == "cosine":
        Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - Qn @ Xn.T
    if metric == "mahalanobis":
        cov = np.cov(X, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        diff = Q[:, None, :] - X[None, :, :]
        return np.sqrt(np.einsum("ijk,kl,ijl->ij", diff, cov_inv, diff))
    raise ValueError(f"unknown metric {metric!r}")


def knn_predict(
    X: np.ndarray,
    y: np.ndarray,
    Q: np.ndarray,
    k: int = 3,
    metric: str = "euclidean",
    weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, neighbour indices) for every query row of Q."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    Q = np.atleast_2d(np.asarray(Q, float))
    D = pairwise_distances(X, Q, metric)
    idx = np.argsort(D, axis=1)[:, :k]
    nb_labels = y[idx]
    classes = np.unique(y)
    if weighted:
        w = 1.0 / (np.take_along_axis(D, idx, axis=1) + 1e-12)
        scores = np.stack([(w * (nb_labels == c)).sum(1) for c in classes], axis=1)
    else:
        scores = np.stack([(nb_labels == c).sum(1) for c in classes], axis=1)
    return classes[scores.argmax(1)], idx


class KNNNumPy:
    """Vectorised k-NN classifier."""

    def __init__(self, k: int = 3, metric: str = "euclidean", weighted: bool = False):
        self.k = k
        self.metric = metric
        self.weighted = weighted

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X) -> np.ndarray:
        return knn_predict(self.X_, self.y_, X, self.k, self.metric, self.weighted)[0]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
