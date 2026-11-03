"""Module 10 — convolutional layers: cross-correlation, pooling, batch norm."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "conv2d_scratch",
    "max_pool2d_scratch",
    "out_size",
    "conv_params",
    "conv_flops",
    "receptive_field",
    "im2col",
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    "global_avg_pool",
    "batch_norm",
    "BatchNorm",
    "grad_cam",
]


# --------------------------------------------------------------- pure python
def conv2d_scratch(X, V, stride: int = 1, pad: int = 0):
    """2-D cross-correlation on a single-channel image (lists of lists)."""
    H, W = len(X), len(X[0])
    K = len(V)
    if pad:
        X = [[0.0] * (W + 2 * pad) for _ in range(pad)] + [
            [0.0] * pad + list(map(float, row)) + [0.0] * pad for row in X
        ] + [[0.0] * (W + 2 * pad) for _ in range(pad)]
        H, W = H + 2 * pad, W + 2 * pad
    Ho, Wo = (H - K) // stride + 1, (W - K) // stride + 1
    out = [[0.0] * Wo for _ in range(Ho)]
    for i in range(Ho):
        for j in range(Wo):
            acc = 0.0
            for a in range(K):
                for b in range(K):
                    acc += V[a][b] * X[i * stride + a][j * stride + b]
            out[i][j] = acc
    return out


def max_pool2d_scratch(X, k: int = 2, stride: int = 2):
    H, W = len(X), len(X[0])
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    return [
        [max(X[i * stride + a][j * stride + b] for a in range(k) for b in range(k)) for j in range(Wo)]
        for i in range(Ho)
    ]


def out_size(H: int, K: int, P: int = 0, S: int = 1) -> int:
    """H' = (H - K + 2P)/S + 1."""
    return (H - K + 2 * P) // S + 1


def conv_params(C_in: int, C_out: int, K: int) -> int:
    return C_out * (C_in * K * K + 1)


def conv_flops(H: int, W: int, C_in: int, C_out: int, K: int, P: int = 0, S: int = 1) -> int:
    return out_size(H, K, P, S) * out_size(W, K, P, S) * C_out * C_in * K * K


def receptive_field(layers: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Receptive field and total stride for a list of (kernel, stride) pairs."""
    rf, jump = 1, 1
    for K, S in layers:
        rf += (K - 1) * jump
        jump *= S
    return rf, jump


# -------------------------------------------------------------------- numpy
def im2col(X: np.ndarray, K: int, stride: int = 1, pad: int = 0):
    """(C, H, W) -> (H'*W', C*K*K) patch matrix. The trick behind fast conv."""
    C, H, W = X.shape
    if pad:
        X = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
        H, W = H + 2 * pad, W + 2 * pad
    Ho, Wo = (H - K) // stride + 1, (W - K) // stride + 1
    cols = np.empty((Ho * Wo, C * K * K))
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            cols[idx] = X[:, i * stride : i * stride + K, j * stride : j * stride + K].ravel()
            idx += 1
    return cols, Ho, Wo


def conv2d(X: np.ndarray, W: np.ndarray, b: np.ndarray | None = None, stride: int = 1, pad: int = 0):
    """X: (C_in, H, W); W: (C_out, C_in, K, K) -> (C_out, H', W')."""
    X = np.asarray(X, float)
    W = np.asarray(W, float)
    C_out, C_in, K, _ = W.shape
    cols, Ho, Wo = im2col(X, K, stride, pad)
    b = np.zeros(C_out) if b is None else np.asarray(b, float)
    out = cols @ W.reshape(C_out, -1).T + b
    return out.T.reshape(C_out, Ho, Wo)


def max_pool2d(X: np.ndarray, k: int = 2, stride: int = 2) -> np.ndarray:
    C, H, W = X.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.empty((C, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            out[:, i, j] = X[:, i * stride : i * stride + k, j * stride : j * stride + k].max(axis=(1, 2))
    return out


def avg_pool2d(X: np.ndarray, k: int = 2, stride: int = 2) -> np.ndarray:
    C, H, W = X.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.empty((C, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            out[:, i, j] = X[:, i * stride : i * stride + k, j * stride : j * stride + k].mean(axis=(1, 2))
    return out


def global_avg_pool(X: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (C,)."""
    return X.mean(axis=(1, 2))


def batch_norm(X: np.ndarray, gamma=None, beta=None, eps: float = 1e-5) -> np.ndarray:
    """Normalise over all axes except the channel axis (axis 0)."""
    mu = X.mean(axis=tuple(range(1, X.ndim)), keepdims=True)
    var = X.var(axis=tuple(range(1, X.ndim)), keepdims=True)
    shape = (1, -1) + (1,) * (X.ndim - 2)
    gamma = np.ones(X.shape[0]) if gamma is None else gamma
    beta = np.zeros(X.shape[0]) if beta is None else beta
    return gamma.reshape(shape) * (X - mu) / np.sqrt(var + eps) + beta.reshape(shape)


class BatchNorm:
    """Batch normalisation layer with running statistics for inference."""

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        shape = (1, -1) + (1,) * (X.ndim - 2)
        if self.training:
            axes = tuple(range(1, X.ndim))
            mu = X.mean(axis=axes, keepdims=True)
            var = X.var(axis=axes, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.ravel()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.ravel()
        else:
            mu = self.running_mean.reshape(shape)
            var = self.running_var.reshape(shape)
        return self.gamma.reshape(shape) * (X - mu) / np.sqrt(var + self.eps) + self.beta.reshape(shape)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


def grad_cam(feature_maps: np.ndarray, grads: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Grad-CAM heatmap from (C, H, W) feature maps and their gradients."""
    weights = np.asarray(grads).mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * np.asarray(feature_maps)).sum(0), 0)
    cam = cam - cam.min()
    return cam / (cam.max() + eps)
