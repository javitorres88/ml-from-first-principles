"""Starter: exact Shapley values and permutation importance."""
import itertools
from math import factorial


def shapley_exact(value_fn, players):
    """Average marginal contribution over all permutations."""
    raise NotImplementedError


def shapley_montecarlo(value_fn, players, samples=2000, seed=0):
    raise NotImplementedError


def permutation_importance(model, X, y, metric, n_repeats=5):
    raise NotImplementedError


if __name__ == "__main__":
    print("verify efficiency: sum(phi) == v(all players)")
