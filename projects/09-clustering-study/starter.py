"""Starter: k-means, k-means++ and DBSCAN from scratch."""
import numpy as np


def kmeans(X, k, iters=100, seed=0, init="++"):
    """Return (centers, labels, inertia)."""
    raise NotImplementedError


def silhouette(X, labels):
    raise NotImplementedError


def dbscan(X, eps, min_pts):
    """Return (labels, n_clusters); -1 means noise."""
    raise NotImplementedError


if __name__ == "__main__":
    print("compare against sklearn.cluster.KMeans and DBSCAN")
