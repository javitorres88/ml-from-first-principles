"""Module 05 — model selection: cross-validation, grid search, learning curves."""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, Sequence

import numpy as np

from .foundations import k_fold_indices, stratified_split

__all__ = [
    "k_fold_cv",
    "stratified_k_fold_cv",
    "grid_search",
    "one_standard_error_choice",
    "learning_curve",
    "bias_variance_decomposition",
]


def k_fold_cv(
    fit_predict: Callable, X, y, k: int = 5, seed: int = 0, metric: Callable | None = None
) -> dict[str, float]:
    """Generic k-fold cross-validation.

    ``fit_predict(X_tr, y_tr, X_te)`` must return predictions for ``X_te``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    metric = metric or (lambda t, p: float(np.mean(np.asarray(t) == np.asarray(p))))
    scores = []
    for test_idx in k_fold_indices(len(X), k, seed):
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
        pred = fit_predict(X[train_idx], y[train_idx], X[test_idx])
        scores.append(metric(y[test_idx], pred))
    scores_arr = np.asarray(scores)
    return {"mean": float(scores_arr.mean()), "std": float(scores_arr.std()), "scores": scores}


def stratified_k_fold_cv(
    fit_predict: Callable, X, y, k: int = 5, seed: int = 0, metric: Callable | None = None
) -> dict[str, float]:
    """k-fold CV where each fold preserves the class distribution."""
    X = np.asarray(X)
    y = np.asarray(y)
    metric = metric or (lambda t, p: float(np.mean(np.asarray(t) == np.asarray(p))))
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, v in enumerate(idx):
            folds[i % k].append(int(v))
    scores = []
    for test_idx in folds:
        test_idx = np.asarray(test_idx)
        train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
        pred = fit_predict(X[train_idx], y[train_idx], X[test_idx])
        scores.append(metric(y[test_idx], pred))
    scores_arr = np.asarray(scores)
    return {"mean": float(scores_arr.mean()), "std": float(scores_arr.std()), "scores": scores}


def grid_search(
    build_model: Callable,
    param_grid: dict[str, Sequence],
    X,
    y,
    k: int = 5,
    seed: int = 0,
    metric: Callable | None = None,
) -> dict:
    """Exhaustive search over ``param_grid`` using cross-validation.

    ``build_model(**params)`` must return an object with ``fit(X, y)`` and
    ``predict(X)``.
    """
    keys = list(param_grid)
    results = []
    for values in itertools.product(*(param_grid[k_] for k_ in keys)):
        params = dict(zip(keys, values))
        model = build_model(**params)

        def fit_predict(X_tr, y_tr, X_te, _model=model):
            from copy import deepcopy

            m = deepcopy(_model)
            m.fit(X_tr, y_tr)
            return m.predict(X_te)

        res = k_fold_cv(fit_predict, X, y, k, seed, metric)
        results.append({"params": params, **res})
    best = max(results, key=lambda r: r["mean"])
    return {"best_params": best["params"], "best_score": best["mean"], "results": results}


def one_standard_error_choice(results: list[dict], simplest_key: str = "alpha") -> dict:
    """Return the simplest parameter set within one SE of the best score.

    ``simplest`` means the largest ``simplest_key`` (interpreted as a
    regularisation strength) among the candidates inside the band.
    """
    best = max(results, key=lambda r: r["mean"])
    band = best["mean"] - best["std"]
    inside = [r for r in results if r["mean"] >= band]
    return max(inside, key=lambda r: r["params"].get(simplest_key, 0))


def learning_curve(
    build_model: Callable,
    X,
    y,
    fractions: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    k: int = 5,
    seed: int = 0,
    metric: Callable | None = None,
) -> dict[str, list[float]]:
    """Train/validation score as a function of training-set size."""
    X = np.asarray(X)
    y = np.asarray(y)
    metric = metric or (lambda t, p: float(np.mean(np.asarray(t) == np.asarray(p))))
    rng = np.random.default_rng(seed)
    train_scores, val_scores, sizes = [], [], []
    order = rng.permutation(len(X))
    for frac in fractions:
        n = max(2, int(frac * len(X)))
        idx = order[:n]
        Xs, ys = X[idx], y[idx]
        tr_s, va_s = [], []
        for test_idx in k_fold_indices(len(Xs), k, seed):
            train_idx = np.setdiff1d(np.arange(len(Xs)), test_idx)
            model = build_model()
            model.fit(Xs[train_idx], ys[train_idx])
            tr_s.append(metric(ys[train_idx], model.predict(Xs[train_idx])))
            va_s.append(metric(ys[test_idx], model.predict(Xs[test_idx])))
        train_scores.append(float(np.mean(tr_s)))
        val_scores.append(float(np.mean(va_s)))
        sizes.append(n)
    return {"sizes": sizes, "train": train_scores, "validation": val_scores}


def bias_variance_decomposition(
    fit_predict: Callable,
    truth: Callable,
    sample_fn: Callable,
    n_train: int = 20,
    reps: int = 200,
    grid: np.ndarray | None = None,
    noise: float = 0.3,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    """Monte-Carlo estimate of bias^2, variance and irreducible noise.

    ``sample_fn(rng, n)`` returns ``(X, y)``; ``fit_predict(X, y, grid)``
    returns predictions on ``grid``.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.02, 0.98, 40) if grid is None else np.asarray(grid)
    preds = np.empty((reps, grid.size))
    for r in range(reps):
        X, y = sample_fn(rng, n_train)
        preds[r] = fit_predict(X, y, grid)
    mean_pred = preds.mean(0)
    truth_vals = truth(grid)
    return {
        "bias2": float(((mean_pred - truth_vals) ** 2).mean()),
        "variance": float(preds.var(0).mean()),
        "noise": float(noise**2),
        "mean_prediction": mean_pred,
        "grid": grid,
    }
