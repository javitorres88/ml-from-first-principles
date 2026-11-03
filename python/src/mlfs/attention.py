"""Module 12 — attention and positional encoding."""

from __future__ import annotations

import numpy as np

__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "causal_mask",
    "padding_mask",
    "multi_head_attention",
    "positional_encoding",
    "layer_norm",
    "MultiHeadAttention",
    "TransformerBlock",
]


def softmax(x, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask: np.ndarray | None = None):
    """softmax(QK^T / sqrt(d_k)) V. Returns (output, weights)."""
    Q = np.asarray(Q, float)
    K = np.asarray(K, float)
    V = np.asarray(V, float)
    dk = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_mask(n: int) -> np.ndarray:
    """Lower-triangular boolean mask: position t may attend to <= t."""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(lengths, max_len: int | None = None) -> np.ndarray:
    """Boolean mask over keys; padded positions are False."""
    lengths = np.asarray(lengths)
    max_len = max_len or int(lengths.max())
    return np.arange(max_len)[None, :] < lengths[:, None]


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads: int, mask=None):
    """Split the feature dimension into heads, attend in each, concatenate."""
    X = np.asarray(X, float)
    n, d = X.shape
    if d % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    heads = []
    for h in range(n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        o, _ = scaled_dot_product_attention(Q[:, sl], K[:, sl], V[:, sl], mask)
        heads.append(o)
    return np.concatenate(heads, axis=-1) @ Wo


def positional_encoding(max_pos: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding, shape (max_pos, d_model)."""
    pos = np.arange(max_pos)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    omega = 1.0 / (10000 ** (i / d_model))
    pe = np.zeros((max_pos, d_model))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe


def layer_norm(x, gamma=None, beta=None, eps: float = 1e-5) -> np.ndarray:
    """Normalise across the feature dimension for each position."""
    x = np.asarray(x, float)
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    gamma = np.ones(x.shape[-1]) if gamma is None else np.asarray(gamma, float)
    beta = np.zeros(x.shape[-1]) if beta is None else np.asarray(beta, float)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


class MultiHeadAttention:
    """Self-contained multi-head attention layer with learned projections."""

    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(d_model)
        self.n_heads = n_heads
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def __call__(self, X, mask=None):
        out = multi_head_attention(X, self.Wq, self.Wk, self.Wv, self.Wo, self.n_heads, mask)
        return out


class TransformerBlock:
    """Pre-norm transformer block: LN -> attention -> residual -> LN -> MLP -> residual."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.W1 = rng.normal(0, scale, (d_model, mlp_ratio * d_model))
        self.c1 = np.zeros(mlp_ratio * d_model)
        self.W2 = rng.normal(0, scale, (mlp_ratio * d_model, d_model))
        self.c2 = np.zeros(d_model)
        self.g1 = np.ones(d_model)
        self.b1 = np.zeros(d_model)
        self.g2 = np.ones(d_model)
        self.b2 = np.zeros(d_model)

    def __call__(self, X, mask=None):
        X = np.asarray(X, float)
        A = self.attn(layer_norm(X, self.g1, self.b1), mask)
        X = X + A
        H = layer_norm(X, self.g2, self.b2)
        X = X + np.maximum(H @ self.W1 + self.c1, 0) @ self.W2 + self.c2
        return X


def count_parameters(block: TransformerBlock) -> int:
    """Parameters of a pre-norm block, ignoring LayerNorm gains and biases."""
    d = block.W2.shape[1]
    return int(4 * d * d + 8 * d * d)
