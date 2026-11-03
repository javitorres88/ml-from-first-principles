"""Module 14 — support vector machines and kernels."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "margin_width",
    "signed_distance",
    "hinge_loss",
    "SVMHingeScratch",
    "svm_hinge_sgd",
    "linear_kernel",
    "poly_kernel",
    "rbf_kernel",
    "gram_matrix",
    "is_psd",
    "kernel_perceptron",
    "kernel_predict",
    "SVMNumPy",
]


def margin_width(w) -> float:
    """Width of the margin between w.x + b = +-1: 2/||w||."""
    w = np.asarray(w, float)
    return float(2.0 / np.linalg.norm(w))


def signed_distance(w, b: float, x) -> float:
    w = np.asarray(w, float)
    x = np.asarray(x, float)
    return float((w @ x + b) / np.linalg.norm(w))


def hinge_loss(X, y, w, b, C: float = 1.0) -> float:
    """0.5||w||^2 + C * sum_i max(0, 1 - y_i (w.x_i + b))."""
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float)
    margins = 1 - y * (X @ np.asarray(w, float) + b)
    return float(0.5 * np.sum(np.asarray(w, float) ** 2) + C * np.maximum(0, margins).sum())


class SVMHingeScratch:
    """Soft-margin SVM trained by subgradient descent on the hinge loss (pure Python)."""

    def __init__(self, C: float = 1.0, lr: float = 0.01, epochs: int = 2000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = [list(map(float, row)) for row in X]
        y = [float(v) for v in y]
        d = len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        n = len(X)
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                margin = yi * (sum(w * x for w, x in zip(self.w, xi)) + self.b)
                if margin < 1:
                    self.w = [w - self.lr * (w - self.C * yi * x) for w, x in zip(self.w, xi)]
                    self.b += self.lr * self.C * yi
                else:
                    self.w = [w - self.lr * w for w in self.w]
        return self

    def decision_function(self, X):
        return [sum(w * x for w, x in zip(self.w, row)) + self.b for row in np.atleast_2d(X)]

    def predict(self, X):
        return np.array([1 if v >= 0 else -1 for v in self.decision_function(X)])

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


def svm_hinge_sgd(X, y, C: float = 1.0, lr: float = 0.01, epochs: int = 500, seed: int = 0):
    """Vectorised subgradient descent. Returns (w, b)."""
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(n):
            if y[i] * (X[i] @ w + b) < 1:
                w -= lr * (w - C * y[i] * X[i])
                b += lr * C * y[i]
            else:
                w -= lr * w
    return w, b


# ------------------------------------------------------------------ kernels
def linear_kernel(X, Z) -> np.ndarray:
    return np.asarray(X, float) @ np.asarray(Z, float).T


def poly_kernel(X, Z, degree: int = 3, c: float = 1.0) -> np.ndarray:
    return (np.asarray(X, float) @ np.asarray(Z, float).T + c) ** degree


def rbf_kernel(X, Z, gamma: float = 1.0) -> np.ndarray:
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    d2 = (X**2).sum(1)[:, None] - 2 * X @ Z.T + (Z**2).sum(1)[None, :]
    return np.exp(-gamma * np.clip(d2, 0, None))


def gram_matrix(X, kernel: str = "rbf", **kwargs) -> np.ndarray:
    X = np.asarray(X, float)
    if kernel == "linear":
        return linear_kernel(X, X)
    if kernel == "poly":
        return poly_kernel(X, X, **kwargs)
    if kernel == "rbf":
        return rbf_kernel(X, X, **kwargs)
    raise ValueError(f"unknown kernel {kernel!r}")


def is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Mercer check: all eigenvalues non-negative."""
    K = np.asarray(K, float)
    return bool(np.min(np.linalg.eigvalsh((K + K.T) / 2)) >= -tol)


def kernel_perceptron(X, y, kernel: str = "rbf", gamma: float = 1.0, epochs: int = 30):
    """Dual-form perceptron: alpha_i += 1 on mistakes. Returns alpha."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    K = gram_matrix(X, kernel, gamma=gamma) if kernel == "rbf" else gram_matrix(X, kernel)
    n = len(X)
    alpha = np.zeros(n)
    for _ in range(epochs):
        errors = 0
        for i in range(n):
            f = float((alpha * y) @ K[:, i])
            if (1 if f >= 0 else -1) != y[i]:
                alpha[i] += 1
                errors += 1
        if errors == 0:
            break
    return alpha


def kernel_predict(X, y, alpha, x_query, kernel: str = "rbf", gamma: float = 1.0) -> float:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    xq = np.atleast_2d(np.asarray(x_query, float))
    if kernel == "rbf":
        k = rbf_kernel(X, xq, gamma).ravel()
    elif kernel == "poly":
        k = poly_kernel(X, xq).ravel()
    else:
        k = linear_kernel(X, xq).ravel()
    return float((alpha * y) @ k)


class SVMNumPy:
    """Soft-margin linear SVM with a hinge-loss training loop."""

    def __init__(self, C: float = 1.0, lr: float = 0.01, epochs: int = 500, seed: int = 0):
        self.C, self.lr, self.epochs, self.seed = C, lr, epochs, seed

    def fit(self, X, y):
        self.w, self.b = svm_hinge_sgd(X, y, self.C, self.lr, self.epochs, self.seed)
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.atleast_2d(np.asarray(X, float)) @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def margin(self) -> float:
        return margin_width(self.w)
