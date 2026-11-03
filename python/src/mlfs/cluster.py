"""Module 09 — clustering: k-means, k-means++, DBSCAN, agglomerative."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "kmeans_scratch",
    "kmeans_pp_scratch",
    "dbscan_scratch",
    "kmeans",
    "kmeans_pp",
    "inertia",
    "silhouette_score",
    "dbscan",
    "agglomerative",
    "linkage_distance",
    "dendrogram_merges",
]


# --------------------------------------------------------------- pure python
def _d2(a, b):
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def kmeans_scratch(X: Sequence[Sequence[float]], k: int, iters: int = 100, seed: int = 0):
    """Lloyd's algorithm with random initialisation. Returns (centers, labels, inertia)."""
    import random

    rng = random.Random(seed)
    X = [list(map(float, row)) for row in X]
    centers = [list(p) for p in rng.sample(X, k)]
    labels = [-1] * len(X)
    for _ in range(iters):
        new_labels = [min(range(k), key=lambda j: _d2(x, centers[j])) for x in X]
        if new_labels == labels:
            break
        labels = new_labels
        for j in range(k):
            members = [x for x, l in zip(X, labels) if l == j]
            if members:
                centers[j] = [sum(c) / len(members) for c in zip(*members)]
    total = sum(_d2(x, centers[l]) for x, l in zip(X, labels))
    return centers, labels, total


def kmeans_pp_scratch(X, k: int, iters: int = 100, seed: int = 0):
    """k-means++ seeding followed by Lloyd iterations."""
    import random

    rng = random.Random(seed)
    X = [list(map(float, row)) for row in X]
    centers = [list(X[rng.randrange(len(X))])]
    while len(centers) < k:
        d2 = [min(_d2(x, c) for c in centers) for x in X]
        total = sum(d2)
        r = rng.random() * total
        acc = 0.0
        pick = 0
        for i, v in enumerate(d2):
            acc += v
            if acc >= r:
                pick = i
                break
        centers.append(list(X[pick]))
    labels = [-1] * len(X)
    for _ in range(iters):
        new_labels = [min(range(k), key=lambda j: _d2(x, centers[j])) for x in X]
        new_centers = []
        for j in range(k):
            members = [x for x, l in zip(X, new_labels) if l == j]
            new_centers.append([sum(c) / len(members) for c in zip(*members)] if members else centers[j])
        if all(_d2(a, b) < 1e-12 for a, b in zip(new_centers, centers)):
            centers, labels = new_centers, new_labels
            break
        centers, labels = new_centers, new_labels
    total = sum(_d2(x, centers[l]) for x, l in zip(X, labels))
    return centers, labels, total


def dbscan_scratch(X, eps: float, min_pts: int):
    """DBSCAN. Returns (labels, n_clusters); label -1 means noise."""
    X = [list(map(float, row)) for row in X]
    n = len(X)
    neighbours = [
        [j for j in range(n) if j != i and math.dist(X[i], X[j]) <= eps] for i in range(n)
    ]
    core = [len(nb) + 1 >= min_pts for nb in neighbours]
    labels = [-1] * n
    cluster = 0
    for i in range(n):
        if not core[i] or labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cluster
        while stack:
            cur = stack.pop()
            if not core[cur]:
                continue
            for j in neighbours[cur]:
                if labels[j] == -1:
                    labels[j] = cluster
                    stack.append(j)
        cluster += 1
    return labels, cluster


# -------------------------------------------------------------------- numpy
def kmeans(X, k: int, iters: int = 300, seed: int = 0, init: str = "++"):
    """Vectorised k-means with k-means++ or random initialisation."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    n = len(X)
    if init == "++":
        centers = [X[rng.integers(n)]]
        for _ in range(k - 1):
            d2 = ((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(1)
            probs = d2 / d2.sum() if d2.sum() else np.full(n, 1 / n)
            centers.append(X[rng.choice(n, p=probs)])
        C = np.array(centers)
    else:
        C = X[rng.choice(n, k, replace=False)]
    labels = np.full(n, -1)
    for _ in range(iters):
        D = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = D.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                C[j] = X[labels == j].mean(0)
    return C, labels, inertia(X, C, labels)


def kmeans_pp(X, k: int, iters: int = 300, seed: int = 0):
    return kmeans(X, k, iters, seed, init="++")


def inertia(X, centers, labels) -> float:
    X = np.asarray(X, float)
    centers = np.asarray(centers, float)
    return float(((X - centers[np.asarray(labels)]) ** 2).sum())


def silhouette_score(X, labels) -> float:
    """Mean silhouette over all points; labels must be a 1-D integer array."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    scores = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        other = [D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i]]
        if not other:
            continue
        b = min(other)
        denom = max(a, b)
        scores.append((b - a) / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def dbscan(X, eps: float, min_samples: int = 5):
    """Vectorised DBSCAN."""
    X = np.asarray(X, float)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    n = len(X)
    neighbours = [np.flatnonzero((D[i] <= eps) & (np.arange(n) != i)) for i in range(n)]
    core = np.array([len(nb) + 1 >= min_samples for nb in neighbours])
    labels = np.full(n, -1)
    cluster = 0
    for i in range(n):
        if not core[i] or labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cluster
        while stack:
            cur = stack.pop()
            if not core[cur]:
                continue
            for j in neighbours[cur]:
                if labels[j] == -1:
                    labels[j] = cluster
                    stack.append(j)
        cluster += 1
    return labels, cluster


def linkage_distance(A: np.ndarray, B: np.ndarray, method: str = "average") -> float:
    """Distance between two clusters given their point sets."""
    D = np.sqrt(((A[:, None, :] - B[None]) ** 2).sum(-1))
    if method == "single":
        return float(D.min())
    if method == "complete":
        return float(D.max())
    if method == "average":
        return float(D.mean())
    if method == "ward":
        ca, cb = A.mean(0), B.mean(0)
        return float(len(A) * len(B) / (len(A) + len(B)) * ((ca - cb) ** 2).sum())
    raise ValueError(f"unknown linkage {method!r}")


def dendrogram_merges(X, method: str = "average") -> list[tuple[int, int, float]]:
    """Agglomerative clustering; returns the merge list (i, j, height)."""
    X = np.asarray(X, float)
    clusters = {i: X[i : i + 1] for i in range(len(X))}
    merges: list[tuple[int, int, float]] = []
    next_id = len(X)
    while len(clusters) > 1:
        keys = list(clusters)
        best = None
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                d = linkage_distance(clusters[keys[a]], clusters[keys[b]], method)
                if best is None or d < best[0]:
                    best = (d, keys[a], keys[b])
        d, ka, kb = best
        merges.append((ka, kb, d))
        clusters[next_id] = np.vstack([clusters.pop(ka), clusters.pop(kb)])
        next_id += 1
    return merges


def agglomerative(X, n_clusters: int = 2, method: str = "ward") -> np.ndarray:
    """Agglomerative clustering returning integer labels."""
    X = np.asarray(X, float)
    clusters = {i: [i] for i in range(len(X))}
    points = {i: X[i : i + 1] for i in range(len(X))}
    next_id = len(X)
    while len(clusters) > n_clusters:
        keys = list(clusters)
        best = None
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                d = linkage_distance(points[keys[a]], points[keys[b]], method)
                if best is None or d < best[0]:
                    best = (d, keys[a], keys[b])
        _, ka, kb = best
        clusters[next_id] = clusters.pop(ka) + clusters.pop(kb)
        points[next_id] = np.vstack([points.pop(ka), points.pop(kb)])
        next_id += 1
    labels = np.empty(len(X), int)
    for label, (_, members) in enumerate(clusters.items()):
        labels[members] = label
    return labels
