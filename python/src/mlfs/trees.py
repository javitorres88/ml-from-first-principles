"""Module 08 — decision trees, bagging, AdaBoost."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

__all__ = [
    "gini",
    "entropy",
    "misclassification",
    "impurity",
    "weighted_impurity",
    "best_split",
    "CARTScratch",
    "grow_tree",
    "predict_tree",
    "RandomForestNumPy",
    "Stump",
    "AdaBoostScratch",
]


def _proportions(labels: Sequence[int]) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=int))
    total = counts.sum()
    return counts / total if total else np.zeros_like(counts, dtype=float)


def gini(labels: Sequence[int]) -> float:
    p = _proportions(labels)
    return float(1.0 - (p**2).sum())


def entropy(labels: Sequence[int]) -> float:
    p = _proportions(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def misclassification(labels: Sequence[int]) -> float:
    p = _proportions(labels)
    return float(1.0 - p.max()) if p.size else 0.0


_CRITERIA = {"gini": gini, "entropy": entropy, "misclassification": misclassification}


def impurity(labels: Sequence[int], criterion: str = "gini") -> float:
    return _CRITERIA[criterion](labels)


def weighted_impurity(left: Sequence[int], right: Sequence[int], criterion: str = "gini") -> float:
    n = len(left) + len(right)
    if n == 0:
        return 0.0
    f = _CRITERIA[criterion]
    return (len(left) * f(left) + len(right) * f(right)) / n


def best_split(X, y, criterion: str = "gini"):
    """Exhaustive search over features and midpoints. Returns (feature, threshold, score)."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    best = (None, None, np.inf)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for a, b in zip(values, values[1:]):
            thr = (a + b) / 2
            mask = X[:, f] <= thr
            if mask.all() or (~mask).all():
                continue
            score = weighted_impurity(y[mask], y[~mask], criterion)
            if score < best[2]:
                best = (f, float(thr), float(score))
    return best


class CARTScratch:
    """CART classifier in pure Python (list-based, for reading)."""

    def __init__(self, max_depth: int = 4, min_samples: int = 2, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = self._grow([list(r) for r in X], list(y), 0)
        return self

    def _grow(self, X, y, depth):
        counts = Counter(y)
        leaf = {"leaf": True, "label": counts.most_common(1)[0][0], "n": len(y)}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples or len(counts) == 1:
            return leaf
        best = None
        for f in range(len(X[0])):
            values = sorted(set(row[f] for row in X))
            for a, b in zip(values, values[1:]):
                thr = (a + b) / 2
                left = [yi for row, yi in zip(X, y) if row[f] <= thr]
                right = [yi for row, yi in zip(X, y) if row[f] > thr]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue
                score = weighted_impurity(left, right, self.criterion)
                if best is None or score < best[0]:
                    best = (score, f, thr, left, right)
        if best is None:
            return leaf
        _, f, thr, left_y, right_y = best
        left_X = [row for row in X if row[f] <= thr]
        right_X = [row for row in X if row[f] > thr]
        return {
            "leaf": False,
            "f": f,
            "thr": thr,
            "left": self._grow(left_X, left_y, depth + 1),
            "right": self._grow(right_X, right_y, depth + 1),
        }

    def predict(self, X):
        return np.array([self._predict_one(node, row) for row in np.atleast_2d(X) for node in [self.tree_]])

    @staticmethod
    def _predict_one(node, row):
        while not node["leaf"]:
            node = node["left"] if row[node["f"]] <= node["thr"] else node["right"]
        return node["label"]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


def grow_tree(X, y, depth: int = 0, max_depth: int = 4, min_samples: int = 2,
              criterion: str = "gini", n_classes: int | None = None) -> dict:
    """Vectorised recursive CART growth."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n_classes = n_classes or int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes)
    leaf = {"leaf": True, "label": int(counts.argmax()), "n": len(y), "counts": counts}
    if depth >= max_depth or len(y) < 2 * min_samples or (counts > 0).sum() == 1:
        return leaf
    f, thr, _ = best_split(X, y, criterion)
    if f is None:
        return leaf
    mask = X[:, f] <= thr
    if mask.sum() < min_samples or (~mask).sum() < min_samples:
        return leaf
    return {
        "leaf": False, "f": f, "thr": thr, "n": len(y),
        "left": grow_tree(X[mask], y[mask], depth + 1, max_depth, min_samples, criterion, n_classes),
        "right": grow_tree(X[~mask], y[~mask], depth + 1, max_depth, min_samples, criterion, n_classes),
    }


def predict_tree(node: dict, X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    out = np.empty(len(X), int)
    for i, row in enumerate(X):
        cur = node
        while not cur["leaf"]:
            cur = cur["left"] if row[cur["f"]] <= cur["thr"] else cur["right"]
        out[i] = cur["label"]
    return out


def leaf_importances(node: dict, n_features: int, out: np.ndarray | None = None) -> np.ndarray:
    """Impurity-decrease feature importance accumulated over a tree."""
    out = np.zeros(n_features) if out is None else out
    if node["leaf"]:
        return out
    counts = node.get("counts")
    if counts is not None:
        parent_imp = 1.0 - ((counts / counts.sum()) ** 2).sum()
        left_n, right_n = node["left"]["n"], node["right"]["n"]
        total = left_n + right_n
        out[node["f"]] += node["n"] * parent_imp
    leaf_importances(node["left"], n_features, out)
    leaf_importances(node["right"], n_features, out)
    return out


class RandomForestNumPy:
    """Bagging over trees with random feature subspaces."""

    def __init__(self, n_trees: int = 25, max_depth: int = 5, n_features: int | None = None, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = self.n_features or max(1, int(np.sqrt(d)))
        self.trees_: list[tuple[dict, np.ndarray]] = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n)
            feats = rng.choice(d, k, replace=False)
            tree = grow_tree(X[idx][:, feats], y[idx], max_depth=self.max_depth,
                             n_classes=int(y.max()) + 1)
            self.trees_.append((tree, feats))
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        classes = np.unique(np.concatenate([np.array([t["leaf"] and 0 or 0]) for t, _ in self.trees_]))
        votes = np.zeros((len(X), 2))
        for tree, feats in self.trees_:
            for i, p in enumerate(predict_tree(tree, X[:, feats])):
                votes[i, p] += 1
        return votes / votes.sum(1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class Stump:
    """A depth-1 tree: one feature, one threshold, two labels."""

    def fit(self, X, y, sample_weight=None):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, int)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        best = None
        for f in range(X.shape[1]):
            values = np.unique(X[:, f])
            for a, b in zip(values, values[1:]):
                thr = (a + b) / 2
                pred = (X[:, f] > thr).astype(int)
                err = float(w[pred != y].sum())
                err = min(err, float(w.sum()) - err)
                if best is None or err < best[0]:
                    best = (err, f, float(thr))
        _, self.f, self.thr = best
        mask = X[:, self.f] <= self.thr
        self.left = int(Counter(y[mask]).most_common(1)[0][0]) if mask.any() else 1
        self.right = int(Counter(y[~mask]).most_common(1)[0][0]) if (~mask).any() else 0
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.where(X[:, self.f] <= self.thr, self.left, self.right)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class AdaBoostScratch:
    """AdaBoost with decision stumps, following the lecture's formulation."""

    def __init__(self, n_estimators: int = 20, lr: float = 1.0):
        self.n_estimators = n_estimators
        self.lr = lr

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, int)
        n = len(y)
        w = np.full(n, 1.0 / n)
        self.models_: list[tuple[Stump, float]] = []
        self.errors_: list[float] = []
        for _ in range(self.n_estimators):
            m = Stump().fit(X, y, w)
            pred = m.predict(X)
            r = float(w[pred != y].sum() / w.sum())
            r = min(max(r, 1e-10), 1 - 1e-10)
            alpha = self.lr * 0.5 * np.log((1 - r) / r)
            w = w * np.exp(alpha * (pred != y) * 2 - alpha)
            w /= w.sum()
            self.models_.append((m, alpha))
            self.errors_.append(r)
            if r < 1e-10:
                break
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        scores = np.zeros(len(X))
        for m, alpha in self.models_:
            scores += alpha * np.where(m.predict(X) == 1, 1.0, -1.0)
        return scores

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
