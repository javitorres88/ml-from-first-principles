"""Module 07 — optimisers and initialisation schemes."""

from __future__ import annotations

import numpy as np

__all__ = [
    "SGD",
    "SGDMomentum",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "he_init",
    "xavier_init",
    "lecun_init",
    "clip_gradients",
    "clip_grad_norm",
]


def _zeros_like(params):
    return [[np.zeros_like(p) for p in layer] for layer in params]


class SGD:
    """Vanilla gradient descent: theta <- theta - lr * grad."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, params, grads):
        for layer, glayer in zip(params, grads):
            for p, g in zip(layer, glayer):
                if self.weight_decay:
                    g = g + self.weight_decay * p
                p -= self.lr * g


class SGDMomentum:
    """Momentum: v <- rho*v + g ; theta <- theta - lr*v."""

    def __init__(self, lr: float = 0.01, rho: float = 0.9):
        self.lr = lr
        self.rho = rho
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = _zeros_like(params)
        for i, (layer, glayer) in enumerate(zip(params, grads)):
            for j, (p, g) in enumerate(zip(layer, glayer)):
                self.v[i][j] = self.rho * self.v[i][j] + g
                p -= self.lr * self.v[i][j]


class AdaGrad:
    """Accumulate squared gradients; damp steep directions."""

    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = _zeros_like(params)
        for i, (layer, glayer) in enumerate(zip(params, grads)):
            for j, (p, g) in enumerate(zip(layer, glayer)):
                self.s[i][j] += g * g
                p -= self.lr * g / (np.sqrt(self.s[i][j]) + self.eps)


class RMSProp:
    """Exponentially decaying average of squared gradients."""

    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = _zeros_like(params)
        for i, (layer, glayer) in enumerate(zip(params, grads)):
            for j, (p, g) in enumerate(zip(layer, glayer)):
                self.s[i][j] = self.beta * self.s[i][j] + (1 - self.beta) * g * g
                p -= self.lr * g / (np.sqrt(self.s[i][j]) + self.eps)


class Adam:
    """Adaptive moment estimation with bias correction."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = _zeros_like(params)
            self.v = _zeros_like(params)
        self.t += 1
        for i, (layer, glayer) in enumerate(zip(params, grads)):
            for j, (p, g) in enumerate(zip(layer, glayer)):
                self.m[i][j] = self.b1 * self.m[i][j] + (1 - self.b1) * g
                self.v[i][j] = self.b2 * self.v[i][j] + (1 - self.b2) * g * g
                m_hat = self.m[i][j] / (1 - self.b1**self.t)
                v_hat = self.v[i][j] / (1 - self.b2**self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def he_init(n_in: int, n_out: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """He initialisation: std = sqrt(2 / n_in). For ReLU networks."""
    rng = rng or np.random.default_rng(0)
    return rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)), np.zeros(n_out)


def xavier_init(n_in: int, n_out: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Glorot/Xavier: std = sqrt(1 / n_in). For tanh/sigmoid networks."""
    rng = rng or np.random.default_rng(0)
    return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out)), np.zeros(n_out)


def lecun_init(n_in: int, n_out: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """LeCun: std = sqrt(1 / n_in). For SELU networks."""
    rng = rng or np.random.default_rng(0)
    return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out)), np.zeros(n_out)


def clip_gradients(grads, v: float):
    """Rescale gradients so the global norm is at most v, preserving direction."""
    total = float(np.sqrt(sum(float((g**2).sum()) for layer in grads for g in layer)))
    if total > v:
        scale = v / total
        grads = [[g * scale for g in layer] for layer in grads]
    return grads, total


def clip_grad_norm(g, v: float) -> np.ndarray:
    """Clip a single gradient array by its L2 norm."""
    n = float(np.linalg.norm(g))
    return g if n <= v else g * (v / n)
