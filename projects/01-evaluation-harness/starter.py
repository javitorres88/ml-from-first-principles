"""Starter: evaluation harness. Fill in the TODOs."""
import numpy as np


def split(X, y, groups=None, test_size=0.2, stratify=True, seed=0):
    """Return (train_idx, test_idx). Respect groups when provided."""
    # TODO: implement stratified and/or grouped splitting
    raise NotImplementedError


def confusion_matrix(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]]."""
    # TODO: implement without scikit-learn, then verify against it
    raise NotImplementedError


def report(cm):
    """Return a dict of accuracy, sensitivity, specificity, precision, F1, balanced accuracy."""
    # TODO: implement and translate the numbers into domain language
    raise NotImplementedError


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    tr, te = split(X, y)
    print(len(tr), len(te))
