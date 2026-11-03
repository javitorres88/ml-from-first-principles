"""Modules 06-07 — neural networks: perceptron, MLP with manual backprop."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .optim import Adam, SGD, SGDMomentum, he_init, xavier_init

__all__ = [
    "step_function",
    "sign",
    "Perceptron",
    "MLPScratch",
    "MLPNumPy",
    "gradient_check",
    "count_parameters",
    "ACTIVATIONS",
]


def step_function(x: float) -> int:
    return 1 if x > 0 else 0


def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


class Perceptron:
    """Rosenblatt's 1957 perceptron with the classic update rule."""

    def __init__(self, n_features: int, lr: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-0.1, 0.1, n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.where(X @ self.w + self.b > 0, 1, -1)

    def fit(self, X, y, epochs: int = 100):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        for _ in range(epochs):
            errors = 0
            for xi, ti in zip(X, y):
                o = 1.0 if xi @ self.w + self.b > 0 else -1.0
                if o != ti:
                    self.w += self.lr * (ti - o) * xi
                    self.b += self.lr * (ti - o)
                    errors += 1
            if errors == 0:
                break
        return self

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


ACTIVATIONS = {
    "relu": (lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float)),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "sigmoid": (
        lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        lambda z: (1 / (1 + np.exp(-np.clip(z, -500, 500))))
        * (1 - 1 / (1 + np.exp(-np.clip(z, -500, 500)))),
    ),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}


class MLPNumPy:
    """A configurable MLP with a manual forward and backward pass.

    Parameters
    ----------
    sizes : list of ints, e.g. ``[4, 16, 16, 3]``
    activation : one of ``relu``, ``tanh``, ``sigmoid``, ``linear``
    out_activation : ``linear`` for regression, ``sigmoid`` for binary,
        ``softmax`` for multi-class
    loss : ``mse`` or ``cross_entropy``
    init : ``he`` (ReLU), ``xavier`` (tanh/sigmoid) or ``small``
    """

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "relu",
        out_activation: str = "linear",
        loss: str = "mse",
        init: str = "he",
        seed: int = 0,
        l2: float = 0.0,
    ):
        self.sizes = list(sizes)
        self.activation = activation
        self.out_activation = out_activation
        self.loss = loss
        self.init = init
        self.l2 = l2
        rng = np.random.default_rng(seed)
        self.params: list[list[np.ndarray]] = []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            if init == "he":
                W, b = he_init(n_in, n_out, rng)
            elif init == "xavier":
                W, b = xavier_init(n_in, n_out, rng)
            else:
                W, b = rng.normal(0, 0.01, (n_in, n_out)), np.zeros(n_out)
            self.params.append([W, b])

    # ------------------------------------------------------------- forward
    def forward(self, X) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        A = np.atleast_2d(np.asarray(X, float))
        cache: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (W, b) in enumerate(self.params):
            Z = A @ W + b
            last = i == len(self.params) - 1
            if last:
                if self.out_activation == "sigmoid":
                    A = ACTIVATIONS["sigmoid"][0](Z)
                elif self.out_activation == "softmax":
                    Zm = Z - Z.max(axis=1, keepdims=True)
                    E = np.exp(Zm)
                    A = E / E.sum(axis=1, keepdims=True)
                else:
                    A = Z
            else:
                A = ACTIVATIONS[self.activation][0](Z)
            cache.append((A, Z))
        return A, cache

    def predict(self, X) -> np.ndarray:
        out, _ = self.forward(X)
        if self.out_activation in ("sigmoid", "softmax"):
            return out.argmax(1) if out.ndim > 1 and out.shape[1] > 1 else (out.ravel() >= 0.5).astype(int)
        return out.ravel()

    def predict_proba(self, X) -> np.ndarray:
        return self.forward(X)[0]

    # ------------------------------------------------------------ backward
    def backward(self, X, T, cache) -> list[list[np.ndarray]]:
        X = np.atleast_2d(np.asarray(X, float))
        T = np.asarray(T, float)
        n = X.shape[0]
        grads: list[list[np.ndarray] | None] = [None] * len(self.params)
        dA = cache[-1][0] - (T if T.ndim > 1 else T.reshape(-1, 1) if cache[-1][0].ndim > 1 else T)
        if self.loss == "mse":
            dA = 2 * dA / n
        else:
            dA = dA / n
        for i in reversed(range(len(self.params))):
            A_prev = X if i == 0 else cache[i - 1][0]
            Z = cache[i][1]
            last = i == len(self.params) - 1
            if last:
                dZ = dA
            else:
                dZ = dA * ACTIVATIONS[self.activation][1](Z)
            W = self.params[i][0]
            dW = A_prev.T @ dZ + self.l2 * W
            db = dZ.sum(0)
            grads[i] = [dW, db]
            dA = dZ @ W.T
        return grads  # type: ignore[return-value]

    def loss_value(self, X, T) -> float:
        out, _ = self.forward(X)
        T = np.asarray(T, float)
        if self.loss == "mse":
            return float(np.mean((out - (T if T.ndim == out.ndim else T.reshape(out.shape))) ** 2))
        out = np.clip(out, 1e-12, 1 - 1e-12)
        if out.ndim == 1 or out.shape[1] == 1:
            o = out.ravel()
            t = T.ravel()
            return float(-np.mean(t * np.log(o) + (1 - t) * np.log(1 - o)))
        return float(-np.mean(np.sum(T * np.log(out), axis=1)))

    def fit(self, X, y, epochs: int = 500, lr: float = 0.05, optim: str = "adam", batch_size: int | None = None):
        opt = {"sgd": SGD, "momentum": SGDMomentum, "adam": Adam}[optim](lr=lr)
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        self.history_: list[float] = []
        for _ in range(epochs):
            if batch_size and batch_size < len(X):
                idx = np.random.default_rng().permutation(len(X))[:batch_size]
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y
            out, cache = self.forward(Xb)
            grads = self.backward(Xb, yb, cache)
            opt.step(self.params, grads)
            self.history_.append(self.loss_value(X, y))
        return self


class MLPScratch:
    """A one-hidden-layer MLP in pure Python, for reading."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / math.sqrt(n_in), (n_hidden, n_in))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1 / math.sqrt(n_hidden), n_hidden)
        self.b2 = 0.0

    def forward(self, x) -> tuple[np.ndarray, float]:
        z = self.W1 @ np.asarray(x, float) + self.b1
        h = np.tanh(z)
        out = float(self.W2 @ h + self.b2)
        return h, out

    def predict(self, X):
        return np.array([self.forward(x)[1] for x in np.atleast_2d(X)])


def gradient_check(model: MLPNumPy, X, T, eps: float = 1e-6, n_checks: int = 8) -> float:
    """Maximum relative error between analytic and numerical gradients."""
    out, cache = model.forward(X)
    grads = model.backward(X, T, cache)
    worst = 0.0
    for i, layer in enumerate(model.params):
        for j in range(len(layer)):
            flat = layer[j].ravel()
            for k in range(min(flat.size, n_checks)):
                old = flat[k]
                flat[k] = old + eps
                lp = model.loss_value(X, T)
                flat[k] = old - eps
                lm = model.loss_value(X, T)
                flat[k] = old
                num = (lp - lm) / (2 * eps)
                ana = grads[i][j].ravel()[k]
                worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


def count_parameters(model: MLPNumPy) -> int:
    return int(sum(W.size + b.size for W, b in model.params))
