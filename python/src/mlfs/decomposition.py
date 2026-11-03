"""Module 13 — decomposition: SVD, PCA, low-rank approximation, autoencoders."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "power_iteration_scratch",
    "pca_scratch",
    "svd_components",
    "pca_svd",
    "transform",
    "inverse_transform",
    "explained_variance_ratio",
    "low_rank",
    "reconstruction_error",
    "linear_autoencoder",
    "PCA",
]


# --------------------------------------------------------------- pure python
def _matvec(M, v):
    return [sum(m * x for m, x in zip(row, v)) for row in M]


def _norm(v):
    return sum(x * x for x in v) ** 0.5


def power_iteration_scratch(M: Sequence[Sequence[float]], iters: int = 200):
    """Leading eigenvector and eigenvalue of a symmetric PSD matrix."""
    d = len(M)
    v = [1.0 / d**0.5] * d
    lam = 0.0
    for _ in range(iters):
        w = _matvec(M, v)
        n = _norm(w) or 1.0
        v = [x / n for x in w]
        lam = n
    return v, lam


def pca_scratch(X: Sequence[Sequence[float]], k: int = 2):
    """PCA by deflated power iteration. Returns (components, variances, mean)."""
    X = [list(map(float, row)) for row in X]
    n, d = len(X), len(X[0])
    mu = [sum(row[j] for row in X) / n for j in range(d)]
    Xc = [[row[j] - mu[j] for j in range(d)] for row in X]
    S = [
        [sum(Xc[i][a] * Xc[i][b] for i in range(n)) / (n - 1) for b in range(d)]
        for a in range(d)
    ]
    comps, variances = [], []
    for _ in range(k):
        v, lam = power_iteration_scratch(S)
        comps.append(v)
        variances.append(lam)
        S = [[S[a][b] - lam * v[a] * v[b] for b in range(d)] for a in range(d)]
    return comps, variances, mu


# -------------------------------------------------------------------- numpy
def svd_components(X, k: int | None = None):
    """Thin SVD with the singular values sorted descending."""
    U, S, Vt = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    if k is not None:
        U, S, Vt = U[:, :k], S[:k], Vt[:k]
    return U, S, Vt


def pca_svd(X, k: int | None = None) -> dict:
    """PCA via SVD of the centred data matrix."""
    X = np.asarray(X, float)
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S**2 / (len(X) - 1)
    k = k or len(S)
    return {
        "components": Vt[:k],
        "singular_values": S[:k],
        "explained_variance": var[:k],
        "explained_ratio": var[:k] / var.sum(),
        "mean": mu,
        "scores": U[:, :k] * S[:k],
    }


def transform(model: dict, X) -> np.ndarray:
    return (np.asarray(X, float) - model["mean"]) @ model["components"].T


def inverse_transform(model: dict, Z) -> np.ndarray:
    return np.asarray(Z, float) @ model["components"] + model["mean"]


def explained_variance_ratio(X) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    S = np.linalg.svd(Xc, compute_uv=False)
    var = S**2
    return var / var.sum()


def low_rank(A, k: int) -> np.ndarray:
    """Best rank-k approximation (Eckart-Young)."""
    U, S, Vt = np.linalg.svd(np.asarray(A, float), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def reconstruction_error(A, k: int) -> float:
    """Relative Frobenius error of the rank-k approximation."""
    A = np.asarray(A, float)
    rec = low_rank(A, k)
    return float(np.linalg.norm(A - rec) / (np.linalg.norm(A) or 1.0))


def linear_autoencoder(X, k: int = 1, lr: float = 0.05, steps: int = 2000, seed: int = 0):
    """Gradient descent on ||X - X W V||^2 with W: d->k and V: k->d."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (d, k))
    V = rng.normal(0, 0.1, (k, d))
    losses = []
    for _ in range(steps):
        H = X @ W
        R = H @ V - X
        gW = X.T @ (R @ V.T) / n
        gV = H.T @ R / n
        W -= lr * gW
        V -= lr * gV
        losses.append(float((R**2).mean()))
    return W, V, losses


class PCA:
    """Scikit-learn-style PCA fitted by SVD."""

    def __init__(self, n_components: int | None = None, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, float)
        model = pca_svd(X, self.n_components)
        self.mean_ = model["mean"]
        self.components_ = model["components"]
        self.singular_values_ = model["singular_values"]
        self.explained_variance_ = model["explained_variance"]
        self.explained_variance_ratio_ = model["explained_ratio"]
        return self

    def transform(self, X) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_)
        return Z

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        if self.whiten:
            Z = np.asarray(Z, float) * np.sqrt(self.explained_variance_)
        return np.asarray(Z, float) @ self.components_ + self.mean_

    def score(self, X) -> float:
        """Mean reconstruction error (negative MSE, scikit-learn convention)."""
        rec = self.inverse_transform(self.transform(X))
        return float(-np.mean((np.asarray(X, float) - rec) ** 2))
