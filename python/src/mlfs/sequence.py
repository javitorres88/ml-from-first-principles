"""Module 11 — recurrent networks: vanilla RNN, LSTM cell, attention."""

from __future__ import annotations

import numpy as np

from .optim import clip_grad_norm

__all__ = [
    "sigmoid",
    "rnn_forward",
    "rnn_backward",
    "RNNNumPy",
    "lstm_cell",
    "lstm_forward",
    "LSTMNumPy",
    "attention",
    "clip_gradients_by_norm",
    "spectral_radius",
    "truncated_bptt_length",
]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def rnn_forward(X, Wh, Wx, b, h0=None):
    """Vanilla RNN: h_t = tanh(x_t Wx + h_{t-1} Wh + b)."""
    X = np.asarray(X, float)
    T = X.shape[0]
    H = np.zeros((T, Wh.shape[0]))
    h = np.zeros(Wh.shape[0]) if h0 is None else np.asarray(h0, float)
    for t in range(T):
        h = np.tanh(X[t] @ Wx + h @ Wh + b)
        H[t] = h
    return H


def rnn_backward(X, H, dH, Wh, Wx):
    """Backpropagation through time for the vanilla RNN (returns dWh, dWx, db)."""
    X = np.asarray(X, float)
    T = X.shape[0]
    dWh = np.zeros_like(Wh)
    dWx = np.zeros_like(Wx)
    db = np.zeros(Wh.shape[0])
    dh_next = np.zeros(Wh.shape[0])
    for t in reversed(range(T)):
        dh = dH[t] + dh_next
        dpre = dh * (1 - H[t] ** 2)
        dWh += np.outer(H[t - 1] if t > 0 else np.zeros_like(H[0]), dpre)
        dWx += np.outer(X[t], dpre)
        db += dpre
        dh_next = dpre @ Wh.T
    return dWh, dWx, db


class RNNNumPy:
    """A single-layer vanilla RNN trained by truncated BPTT on squared error."""

    def __init__(self, n_in: int, n_h: int, n_out: int = 1, seed: int = 0, lr: float = 0.01):
        rng = np.random.default_rng(seed)
        self.Wh = rng.normal(0, 1 / np.sqrt(n_h), (n_h, n_h))
        self.Wx = rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_h))
        self.b = np.zeros(n_h)
        self.Wy = rng.normal(0, 1 / np.sqrt(n_h), (n_h, n_out))
        self.by = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        H = rnn_forward(X, self.Wh, self.Wx, self.b)
        return H, H @ self.Wy + self.by

    def fit(self, X, T, epochs: int = 200, truncate: int | None = None, clip: float = 5.0):
        X = np.atleast_2d(np.asarray(X, float))
        T = np.asarray(T, float)
        if T.ndim == 1:
            T = T[:, None]
        self.losses_: list[float] = []
        for _ in range(epochs):
            H, Y = self.forward(X)
            dY = 2 * (Y - T) / len(X)
            dWy = H.T @ dY
            dby = dY.sum(0)
            dH = dY @ self.Wy.T
            if truncate:
                dH[: max(0, len(X) - truncate)] = 0.0
            dWh, dWx, db = rnn_backward(X, H, dH, self.Wh, self.Wx)
            grads = [dWh, dWx, db, dWy, dby]
            total = float(np.sqrt(sum(float((g**2).sum()) for g in grads)))
            if clip and total > clip:
                grads = [g * (clip / total) for g in grads]
            dWh, dWx, db, dWy, dby = grads
            self.Wh -= self.lr * dWh
            self.Wx -= self.lr * dWx
            self.b -= self.lr * db
            self.Wy -= self.lr * dWy
            self.by -= self.lr * dby
            self.losses_.append(float(np.mean((Y - T) ** 2)))
        return self

    def predict(self, X):
        return self.forward(X)[1]


def lstm_cell(x, h_prev, c_prev, params):
    """One LSTM step. ``params`` holds Wx, Wh and b, each covering the 4 gates."""
    Wx, Wh, b = params["Wx"], params["Wh"], params["b"]
    z = x @ Wx + h_prev @ Wh + b
    i, f, o, g = np.split(z, 4)
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
    g = np.tanh(g)
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c, (i, f, o, g, c)


def lstm_forward(X, params, h0=None, c0=None):
    X = np.atleast_2d(np.asarray(X, float))
    n_h = params["Wh"].shape[0]
    H = np.zeros((len(X), n_h))
    C = np.zeros((len(X), n_h))
    h = np.zeros(n_h) if h0 is None else np.asarray(h0, float)
    c = np.zeros(n_h) if c0 is None else np.asarray(c0, float)
    for t in range(len(X)):
        h, c, _ = lstm_cell(X[t], h, c, params)
        H[t], C[t] = h, c
    return H, C


class LSTMNumPy:
    """A single-layer LSTM with a linear output head, trained by BPTT."""

    def __init__(self, n_in: int, n_h: int, n_out: int = 1, seed: int = 0, lr: float = 0.01):
        rng = np.random.default_rng(seed)
        self.n_h = n_h
        self.params = {
            "Wx": rng.normal(0, 1 / np.sqrt(n_in), (n_in, 4 * n_h)),
            "Wh": rng.normal(0, 1 / np.sqrt(n_h), (n_h, 4 * n_h)),
            "b": np.zeros(4 * n_h),
        }
        self.Wy = rng.normal(0, 1 / np.sqrt(n_h), (n_h, n_out))
        self.by = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        H, C = lstm_forward(X, self.params)
        return H, C, H @ self.Wy + self.by

    def predict(self, X):
        return self.forward(np.atleast_2d(np.asarray(X, float)))[2]

    def fit(self, X, T, epochs: int = 100):
        """Train only the linear output head, keeping the LSTM weights fixed.

        This is deliberate: full BPTT through an LSTM is a long derivation and
        belongs in module 11's notebook rather than in a reference module. With
        fixed recurrent weights this becomes an ordinary linear regression on the
        hidden states, which is enough to demonstrate the cell's dynamics.
        """
        X = np.atleast_2d(np.asarray(X, float))
        T = np.asarray(T, float)
        if T.ndim == 1:
            T = T[:, None]
        self.losses_: list[float] = []
        for _ in range(epochs):
            H, C, Y = self.forward(X)
            dY = 2 * (Y - T) / len(X)
            self.Wy -= self.lr * (H.T @ dY)
            self.by -= self.lr * dY.sum(0)
            self.losses_.append(float(np.mean((Y - T) ** 2)))
        return self


def attention(s_prev, H, W_att):
    """Additive (Bahdanau-style) attention. Returns (weights, context)."""
    H = np.asarray(H, float)
    scores = np.tanh(H @ W_att + np.asarray(s_prev, float)) @ np.ones(H.shape[1])
    a = np.exp(scores - scores.max())
    a /= a.sum()
    return a, a @ H


def clip_gradients_by_norm(grads, v: float):
    """Clip a list of gradient arrays to a global norm of at most v."""
    total = float(np.sqrt(sum(float((np.asarray(g) ** 2).sum()) for g in grads)))
    if total > v:
        grads = [np.asarray(g) * (v / total) for g in grads]
    return grads, total


def spectral_radius(W: np.ndarray) -> float:
    """Largest absolute eigenvalue — the quantity that governs BPTT stability."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(W, float)))))


def truncated_bptt_length(gamma: float, tolerance: float = 1e-3) -> int:
    """Number of steps until a gradient of scale gamma^t falls below tolerance."""
    if gamma >= 1:
        return 10**6
    import math

    return int(math.ceil(math.log(tolerance) / math.log(gamma)))
