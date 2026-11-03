"""Modules 03-05 — linear models: least squares, ridge, lasso, logistic regression."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "design_matrix",
    "lstsq_scratch",
    "polyfit_scratch",
    "polyval_scratch",
    "LinearRegressionScratch",
    "gradient_descent_linear",
    "LinearRegressionNumPy",
    "RidgeRegressionNumPy",
    "soft_threshold",
    "lasso_coordinate_descent",
    "sigmoid",
    "logit",
    "binary_cross_entropy",
    "LogisticRegressionScratch",
    "LogisticRegressionNumPy",
    "softmax",
    "cross_entropy",
]


# --------------------------------------------------------------- pure python
def design_matrix(X: Sequence[Sequence[float]], bias: bool = True) -> list[list[float]]:
    """Prepend a column of ones so the bias is just another weight."""
    rows = [list(map(float, row)) for row in X]
    if bias:
        return [[1.0] + row for row in rows]
    return rows


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular matrix: features are collinear")
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [a - f * c for a, c in zip(M[r], M[col])]
    return [row[-1] for row in M]


def _transpose(M: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(r) for r in zip(*M)]


def _matmul(A, B):
    Bt = _transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def _matvec(A, v):
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def lstsq_scratch(X: Sequence[Sequence[float]], t: Sequence[float]) -> list[float]:
    """Least squares via the normal equations. Returns [b, w1, ..., wm]."""
    A = design_matrix(X)
    At = _transpose(A)
    return _solve(_matmul(At, A), _matvec(At, list(t)))


def polyfit_scratch(xs: Sequence[float], ys: Sequence[float], degree: int) -> list[float]:
    """Polynomial fit; coefficients ascending in degree (c0 + c1 x + ...)."""
    X = [[x**k for k in range(1, degree + 1)] for x in xs]
    return lstsq_scratch(X, ys)


def polyval_scratch(coeffs: Sequence[float], x: float) -> float:
    return sum(c * x**k for k, c in enumerate(coeffs))


class LinearRegressionScratch:
    """Least squares by the normal equations, pure Python."""

    def fit(self, X, t):
        theta = lstsq_scratch(X, t)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X) -> list[float]:
        return [self.intercept_ + sum(w * xj for w, xj in zip(self.coef_, row)) for row in X]

    def score(self, X, t) -> float:
        pred = self.predict(X)
        t = list(t)
        ss_res = sum((a - b) ** 2 for a, b in zip(t, pred))
        mean = sum(t) / len(t)
        ss_tot = sum((a - mean) ** 2 for a in t)
        return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def gradient_descent_linear(
    X: Sequence[Sequence[float]],
    t: Sequence[float],
    lr: float = 0.1,
    epochs: int = 2000,
) -> list[float]:
    """Full-batch gradient descent on the MSE of a linear model."""
    A = design_matrix(X)
    n, m = len(A), len(A[0])
    theta = [0.0] * m
    for _ in range(epochs):
        pred = _matvec(A, theta)
        resid = [p - ti for p, ti in zip(pred, t)]
        grad = [sum(A[i][j] * resid[i] for i in range(n)) / n for j in range(m)]
        theta = [th - lr * g for th, g in zip(theta, grad)]
    return theta


# -------------------------------------------------------------------- numpy
class LinearRegressionNumPy:
    """Least squares via ``lstsq`` (QR/SVD based, numerically stable)."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, t):
        X = np.asarray(X, float)
        t = np.asarray(t, float)
        A = np.hstack([np.ones((len(X), 1)), X]) if self.fit_intercept else X
        theta, *_ = np.linalg.lstsq(A, t, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def score(self, X, t) -> float:
        t = np.asarray(t, float)
        pred = self.predict(X)
        ss_res = float(((t - pred) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


class RidgeRegressionNumPy:
    """Ridge regression: (X'X + lambda I) w = X'y, intercept not penalised."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, t):
        X = np.asarray(X, float)
        t = np.asarray(t, float)
        if self.fit_intercept:
            self.mu_ = X.mean(0)
            self.t_mean_ = t.mean()
            Xc = X - self.mu_
        else:
            self.mu_ = np.zeros(X.shape[1])
            self.t_mean_ = 0.0
            Xc = X
        G = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(G, Xc.T @ (t - self.t_mean_))
        self.intercept_ = float(self.t_mean_ - self.mu_ @ self.coef_)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def soft_threshold(v: float, k: float) -> float:
    """Soft-thresholding operator used by lasso coordinate descent."""
    if v > k:
        return v - k
    if v < -k:
        return v + k
    return 0.0


def lasso_coordinate_descent(
    X: np.ndarray, t: np.ndarray, alpha: float = 0.1, iters: int = 500, tol: float = 1e-8
) -> np.ndarray:
    """Minimise ||Xw - t||^2 / (2n) + alpha ||w||_1 by coordinate descent."""
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    n, m = X.shape
    w = np.zeros(m)
    col_sq = (X**2).sum(0) / n
    for _ in range(iters):
        w_old = w.copy()
        for j in range(m):
            r = t - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ r / n
            w[j] = soft_threshold(rho, alpha) / col_sq[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


# -------------------------------------------------------- logistic regression
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p):
    p = np.asarray(p, float)
    return np.log(p / (1 - p))


def binary_cross_entropy(y_hat, t) -> float:
    y_hat = np.clip(np.asarray(y_hat, float), 1e-12, 1 - 1e-12)
    t = np.asarray(t, float)
    return float(-np.mean(t * np.log(y_hat) + (1 - t) * np.log(1 - y_hat)))


def softmax(Z):
    Z = np.atleast_2d(np.asarray(Z, float))
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, T) -> float:
    P = np.clip(np.asarray(P, float), 1e-12, 1 - 1e-12)
    T = np.asarray(T, float)
    if P.ndim == 1:
        return float(-np.mean(T * np.log(P) + (1 - T) * np.log(1 - P)))
    return float(-np.mean(np.sum(T * np.log(P), axis=1)))


class LogisticRegressionScratch:
    """Binary logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr: float = 0.5, epochs: int = 1000, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, t):
        X = [list(map(float, row)) for row in X]
        t = list(map(float, t))
        n, m = len(X), len(X[0])
        w, b = [0.0] * m, 0.0
        self.losses_: list[float] = []
        for _ in range(self.epochs):
            p = [1.0 / (1.0 + math.exp(-(sum(wj * xj for wj, xj in zip(w, row)) + b))) for row in X]
            err = [pi - ti for pi, ti in zip(p, t)]
            gw = [sum(err[i] * X[i][j] for i in range(n)) / n + self.l2 * w[j] for j in range(m)]
            gb = sum(err) / n
            w = [wj - self.lr * g for wj, g in zip(w, gw)]
            b -= self.lr * gb
            self.losses_.append(
                -sum(ti * math.log(max(pi, 1e-12)) + (1 - ti) * math.log(max(1 - pi, 1e-12))
                     for pi, ti in zip(p, t)) / n
            )
        self.coef_, self.intercept_ = w, b
        return self

    def predict_proba(self, X) -> list[float]:
        return [
            1.0 / (1.0 + math.exp(-(sum(w * xj for w, xj in zip(self.coef_, row)) + self.intercept_)))
            for row in X
        ]

    def predict(self, X, threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def score(self, X, t) -> float:
        preds = self.predict(X)
        return sum(p == ti for p, ti in zip(preds, t)) / len(list(t))


class LogisticRegressionNumPy:
    """Vectorised binary or multi-class logistic regression (softmax output)."""

    def __init__(self, lr: float = 0.5, epochs: int = 2000, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, t):
        X = np.asarray(X, float)
        t = np.asarray(t, float)
        n, m = X.shape
        self.losses_: list[float] = []
        if t.ndim == 1 and set(np.unique(t)) <= {0.0, 1.0}:
            w, b = np.zeros(m), 0.0
            for _ in range(self.epochs):
                p = sigmoid(X @ w + b)
                err = p - t
                w -= self.lr * (X.T @ err / n + self.l2 * w)
                b -= self.lr * err.mean()
                self.losses_.append(binary_cross_entropy(p, t))
            self.coef_, self.intercept_ = w, b
            self.classes_ = np.array([0, 1])
            self.multi_class_ = False
        else:
            T = np.asarray(t, float)
            if T.ndim == 1:
                self.classes_ = np.unique(T)
                T = np.eye(len(self.classes_))[np.searchsorted(self.classes_, T)]
            else:
                self.classes_ = np.arange(T.shape[1])
            W, b = np.zeros((m, T.shape[1])), np.zeros(T.shape[1])
            for _ in range(self.epochs):
                Y = softmax(X @ W + b)
                G = (Y - T) / n
                W -= self.lr * (X.T @ G + self.l2 * W)
                b -= self.lr * G.sum(0)
                self.losses_.append(cross_entropy(Y, T))
            self.coef_, self.intercept_ = W, b
            self.multi_class_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        if not self.multi_class_:
            p = sigmoid(X @ self.coef_ + self.intercept_)
            return np.column_stack([1 - p, p])
        return softmax(X @ self.coef_ + self.intercept_)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def score(self, X, t) -> float:
        return float(np.mean(self.predict(X) == np.asarray(t)))
