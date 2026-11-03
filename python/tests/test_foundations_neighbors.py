"""Tests for mlfs.foundations and mlfs.neighbors."""

import numpy as np
import pytest

from mlfs import foundations as F
from mlfs import neighbors as N

sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


# ------------------------------------------------------------------ metrics
def test_confusion_matrix_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    assert np.array_equal(F.confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))


def test_scratch_confusion_matches_numpy_version():
    y_true = [1] * 10 + [0] * 30
    y_pred = [1] * 7 + [0] * 3 + [1] * 4 + [0] * 26
    cm_scratch = F.confusion_matrix_scratch(y_true, y_pred)
    cm_np = F.confusion_matrix(y_true, y_pred)
    assert cm_scratch["tp"] == cm_np[1, 1]
    assert cm_scratch["fp"] == cm_np[0, 1]
    assert cm_scratch["fn"] == cm_np[1, 0]
    assert cm_scratch["tn"] == cm_np[0, 0]


def test_metrics_identities():
    cm = np.array([[90, 0], [10, 0]])  # always predict negative, 10% prevalence
    m = F.metrics(cm)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["sensitivity"] == pytest.approx(0.0)
    assert m["specificity"] == pytest.approx(1.0)
    # accuracy = prevalence * sens + (1 - prevalence) * spec
    assert m["accuracy"] == pytest.approx(m["prevalence"] * m["sensitivity"] + (1 - m["prevalence"]) * m["specificity"])


def test_scratch_metrics_agree_with_numpy():
    cm = np.array([[50, 10], [5, 35]])
    a = F.metrics(cm)
    b = F.metrics_scratch({"tp": 35, "fp": 10, "tn": 50, "fn": 5})
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy"):
        assert a[key] == pytest.approx(b[key])


def test_stratified_split_preserves_prevalence():
    X, y = load_iris(return_X_y=True)
    mask = F.stratified_split(y, frac=0.2, seed=1)
    for cls in np.unique(y):
        full = (y == cls).mean()
        held = (y[mask] == cls).sum() / mask.sum()
        assert abs(full - held) < 0.02


def test_k_fold_covers_every_index_once():
    folds = F.k_fold_indices(23, k=5, seed=3)
    assert sorted(np.concatenate(folds).tolist()) == list(range(23))


def test_regression_metrics():
    y = [1.0, 2.0, 3.0]
    p = [1.0, 2.0, 5.0]
    assert F.mse(y, p) == pytest.approx(4 / 3)
    assert F.rmse(y, p) == pytest.approx(np.sqrt(4 / 3))
    assert F.mae(y, p) == pytest.approx(2 / 3)
    assert F.r2_score(y, p) < 0


# --------------------------------------------------------------------- k-NN
def test_pairwise_euclidean_matches_manual():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    D = N.pairwise_distances(X, X)
    assert D[0, 1] == pytest.approx(5.0)
    assert D[1, 0] == pytest.approx(5.0)
    assert D[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_knn_matches_sklearn(metric):
    X, y = load_iris(return_X_y=True)
    Xs, mu, sd = F.standardize(X)
    sk_metric = "cosine" if metric == "cosine" else metric
    ours = N.KNNNumPy(k=5, metric=metric).fit(Xs, y).predict(Xs)
    theirs = KNeighborsClassifier(n_neighbors=5, metric=sk_metric).fit(Xs, y).predict(Xs)
    # cosine ties can differ by a hair; require >= 98% agreement
    assert np.mean(ours == theirs) > 0.98


def test_knn_scratch_matches_numpy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    ours = N.KNNScratch(k=3).fit(X.tolist(), y.tolist()).predict(X.tolist())
    theirs = N.KNNNumPy(k=3).fit(X, y).predict(X)
    assert np.mean(np.array(ours) == theirs) > 0.95


def test_mahalanobis_reduces_to_euclidean_for_identity_cov():
    cov_inv = np.eye(2)
    assert N.mahalanobis([0, 0], [3, 4], cov_inv) == pytest.approx(5.0)
