"""Module 16 — interpretation: Shapley values and permutation importance."""

from __future__ import annotations

import itertools
import random
from math import factorial
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "shapley_exact",
    "shapley_subsets",
    "shapley_montecarlo",
    "check_efficiency",
    "linear_shap",
    "kernel_shap_linear",
    "permutation_importance",
    "saliency",
    "grad_cam",
]


def shapley_exact(value_fn: Callable[[set], float], players: Sequence) -> dict:
    """Average marginal contribution over all permutations (exact)."""
    players = list(players)
    M = len(players)
    phi = {p: 0.0 for p in players}
    for perm in itertools.permutations(players):
        before: set = set()
        for p in perm:
            phi[p] += (value_fn(before | {p}) - value_fn(before)) / factorial(M)
            before.add(p)
    return phi


def shapley_subsets(value_fn: Callable[[set], float], players: Sequence) -> dict:
    """The closed form with the |S|!(M-|S|-1)!/M! weights."""
    players = list(players)
    M = len(players)
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for size in range(M):
            for subset in itertools.combinations(others, size):
                S = set(subset)
                w = factorial(size) * factorial(M - size - 1) / factorial(M)
                total += w * (value_fn(S | {p}) - value_fn(S))
        phi[p] = total
    return phi


def shapley_montecarlo(value_fn: Callable[[set], float], players: Sequence,
                       samples: int = 2000, seed: int = 0) -> dict:
    """Permutation sampling estimate of the Shapley values."""
    rng = random.Random(seed)
    players = list(players)
    phi = {p: 0.0 for p in players}
    for _ in range(samples):
        perm = players[:]
        rng.shuffle(perm)
        before: set = set()
        for p in perm:
            phi[p] += (value_fn(before | {p}) - value_fn(before)) / samples
            before.add(p)
    return phi


def check_efficiency(value_fn: Callable[[set], float], players: Sequence, phi: dict) -> dict:
    """Verify the efficiency axiom: sum(phi) == v(all players)."""
    total = sum(phi.values())
    grand = value_fn(set(players))
    return {"sum_phi": total, "v_all": grand, "efficiency_holds": abs(total - grand) < 1e-9}


def linear_shap(w, x, background) -> tuple[np.ndarray, float]:
    """Exact SHAP values for a linear model: phi_i = w_i (x_i - E[x_i])."""
    w = np.asarray(w, float)
    x = np.asarray(x, float)
    bg = np.atleast_2d(np.asarray(background, float))
    baseline = bg.mean(0)
    return w * (x - baseline), float(w @ baseline)


def kernel_shap_linear(w, b: float, x, background, samples: int = 500, seed: int = 0):
    """Monte-Carlo SHAP for a linear model — should match ``linear_shap``."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    x = np.asarray(x, float)
    bg = np.atleast_2d(np.asarray(background, float))
    M = len(x)
    phi = np.zeros(M)
    for _ in range(samples):
        perm = rng.permutation(M)
        before: list[int] = []
        ref = bg[rng.integers(len(bg))]
        for p in perm:
            base = ref.copy()
            base[before] = x[before]
            with_p = base.copy()
            with_p[p] = x[p]
            phi[p] += ((w @ with_p + b) - (w @ base + b)) / samples
            before.append(p)
    return phi


def permutation_importance(model, X, y, metric: Callable, n_repeats: int = 5, seed: int = 0) -> np.ndarray:
    """Drop in performance when each feature's values are shuffled."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict(X))
    imps = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - metric(y, model.predict(Xp)))
        imps[j] = float(np.mean(drops))
    return imps


def saliency(input_grad: np.ndarray) -> np.ndarray:
    """Per-pixel importance: the magnitude of the input gradient."""
    return np.abs(np.asarray(input_grad, float))


def grad_cam(feature_maps: np.ndarray, grads: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Grad-CAM heatmap from (C, H, W) maps and their gradients."""
    weights = np.asarray(grads, float).mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * np.asarray(feature_maps, float)).sum(0), 0)
    cam = cam - cam.min()
    return cam / (cam.max() + eps)
