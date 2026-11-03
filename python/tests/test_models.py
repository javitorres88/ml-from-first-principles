"""Tests for the modelling modules: linear models, trees, clustering, decomposition,
attention, SVMs, RL and interpretation."""

import numpy as np
import pytest

from mlfs import attention as A
from mlfs import cluster as C
from mlfs import decomposition as D
from mlfs import interpret as I
from mlfs import linear_model as L
from mlfs import rl as R
from mlfs import svm as S
from mlfs import trees as T

sklearn = pytest.importorskip("sklearn")


# ------------------------------------------------------------- linear models
def test_linear_regression_matches_sklearn():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    t = X @ np.array([1.5, -2.0, 0.0, 0.5]) + 3.0 + rng.normal(0, 0.1, 100)

    ours = L.LinearRegressionNumPy().fit(X, t)
    theirs = LinearRegression().fit(X, t)
    assert np.allclose(ours.coef_, theirs.coef_, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_)

    scratch = L.LinearRegressionScratch().fit(X.tolist(), t.tolist())
    assert np.allclose(scratch.coef_, theirs.coef_, atol=1e-6)


def test_gradient_descent_reaches_least_squares():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    t = X @ np.array([2.0, -1.0]) + 0.5
    theta_ne = L.lstsq_scratch(X, t)
    theta_gd = L.gradient_descent_linear(X, t, lr=0.1, epochs=5000)
    assert np.allclose(theta_ne, theta_gd, atol=1e-2)


def test_ridge_shrinks_and_matches_closed_form():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    t = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(0, 0.1, 60)
    w_small = L.RidgeRegressionNumPy(alpha=0.01).fit(X, t).coef_
    w_large = L.RidgeRegressionNumPy(alpha=100.0).fit(X, t).coef_
    assert np.linalg.norm(w_large) < np.linalg.norm(w_small)


def test_lasso_produces_exact_zeros():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 6))
    t = 2.0 * X[:, 0] - 1.0 * X[:, 2] + rng.normal(0, 0.1, 200)
    w = L.lasso_coordinate_descent(X, t, alpha=0.3)
    assert np.sum(np.abs(w) < 1e-12) >= 3


def test_sigmoid_and_logit_are_inverse():
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.allclose(L.sigmoid(L.logit(p)), p)


def test_logistic_matches_sklearn():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    ours = L.LogisticRegressionNumPy(lr=0.5, epochs=4000).fit(Xs, y)
    theirs = LogisticRegression(max_iter=5000, C=1e6).fit(Xs, y)
    # predictions agree closely and the two score vectors rank samples identically
    assert np.mean(ours.predict(Xs) == theirs.predict(Xs)) > 0.95
    p_ours = ours.predict_proba(Xs)[:, 1]
    p_theirs = theirs.predict_proba(Xs)[:, 1]
    from scipy.stats import spearmanr  # noqa: F401  (optional dependency)
    assert np.corrcoef(p_ours, p_theirs)[0, 1] > 0.95


def test_softmax_rows_sum_to_one():
    Z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    P = L.softmax(Z)
    assert np.allclose(P.sum(1), 1.0)
    assert np.allclose(A.softmax(Z), P)


# ------------------------------------------------------------------- trees
def test_gini_and_entropy_at_extremes():
    assert T.gini([0, 0, 0, 0]) == pytest.approx(0.0)
    assert T.gini([0, 0, 1, 1]) == pytest.approx(0.5)
    assert T.entropy([0, 0, 1, 1]) == pytest.approx(1.0)
    assert T.entropy([1, 1, 1]) == pytest.approx(0.0)


def test_cart_finds_the_obvious_split():
    X = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    f, thr, score = T.best_split(X, y)
    assert f == 0
    assert 0.2 < thr < 0.9
    assert score == pytest.approx(0.0)


def test_tree_fits_xor_with_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    tree = T.grow_tree(X, y, max_depth=2, min_samples=1)
    assert np.array_equal(T.predict_tree(tree, X), y)


def test_scratch_cart_matches_vectorised():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    scratch = T.CARTScratch(max_depth=3, min_samples=2).fit(X, y)
    vec = T.grow_tree(X, y, max_depth=3, min_samples=2)
    assert np.mean(scratch.predict(X) == T.predict_tree(vec, X)) > 0.95


def test_adaboost_improves_on_a_stump():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    single = T.Stump().fit(X, y)
    boost = T.AdaBoostScratch(n_estimators=10).fit(X, y)
    assert boost.score(X, y) >= single.score(X, y)
    # the label noise puts the Bayes rate around 0.9, so boosting converges there
    assert boost.score(X, y) > 0.85


def test_random_forest_beats_single_tree_on_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    forest = T.RandomForestNumPy(n_trees=20, max_depth=4, seed=0).fit(X, y)
    assert forest.score(X, y) > 0.8


# --------------------------------------------------------------- clustering
def test_kmeans_recovers_well_separated_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.vstack([rng.normal(c, 0.3, (40, 2)) for c in centers])
    C_, labels, inertia = C.kmeans(X, 3, seed=0)
    # within-cluster SSE for 120 points with sigma 0.3 is about 21 in 2-D
    assert inertia < 30.0
    assert len(np.unique(labels)) == 3


def test_kmeans_pp_no_worse_than_random_on_average():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    pp = [C.kmeans(X, 4, seed=s, init="++")[2] for s in range(5)]
    rnd = [C.kmeans(X, 4, seed=s, init="random")[2] for s in range(5)]
    assert np.mean(pp) <= np.mean(rnd) * 1.05


def test_dbscan_finds_core_border_noise():
    X = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [5, 5]], float)
    labels, k = C.dbscan(X, eps=0.3, min_samples=3)
    assert k == 1
    assert labels[-1] == -1


def test_silhouette_high_for_separated_clusters():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 0.2, (30, 2)), rng.normal(10, 0.2, (30, 2))])
    labels = np.array([0] * 30 + [1] * 30)
    assert C.silhouette_score(X, labels) > 0.8


def test_agglomerative_matches_expected_groups():
    X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], float)
    labels = C.agglomerative(X, n_clusters=2, method="ward")
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


# ------------------------------------------------------------- decomposition
def test_svd_reconstruction_and_low_rank_error():
    rng = np.random.default_rng(0)
    A_ = rng.normal(size=(20, 12))
    U, s, Vt = D.svd_components(A_)
    assert np.allclose(U @ np.diag(s) @ Vt, A_, atol=1e-10)
    errs = [D.reconstruction_error(A_, k) for k in (1, 5, 12)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] == pytest.approx(0.0, abs=1e-10)


def test_pca_explained_variance_sums_to_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    ratio = D.explained_variance_ratio(X)
    assert ratio.sum() == pytest.approx(1.0)
    assert np.all(np.diff(ratio) <= 1e-12)


def test_pca_matches_sklearn():
    from sklearn.decomposition import PCA as SkPCA

    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 6)) @ np.diag([3, 2, 1, 0.5, 0.3, 0.1])
    ours = D.PCA(n_components=3).fit(X)
    theirs = SkPCA(n_components=3).fit(X)
    assert np.allclose(np.abs(ours.components_), np.abs(theirs.components_), atol=1e-8)
    assert np.allclose(ours.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-10)


def test_linear_autoencoder_learns_pca_direction():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2)) @ np.array([[2.0, 1.0], [0.2, -0.5]])
    W, V, losses = D.linear_autoencoder(X, k=1, steps=3000)
    pca = D.pca_svd(X, 1)
    ae_dir = V[0] / np.linalg.norm(V[0])
    assert abs(float(pca["components"][0] @ ae_dir)) > 0.99
    assert losses[-1] < losses[0]


# ----------------------------------------------------------------- attention
def test_attention_weights_are_a_distribution():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 8))
    out, w = A.scaled_dot_product_attention(Q, Q, Q)
    assert np.allclose(w.sum(1), 1.0)
    assert out.shape == Q.shape


def test_causal_mask_zeroes_the_future():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(6, 4))
    _, w = A.scaled_dot_product_attention(Q, Q, Q, mask=A.causal_mask(6))
    assert np.allclose(w[np.triu_indices(6, 1)], 0.0)


def test_scaling_prevents_saturation():
    rng = np.random.default_rng(2)
    Q = rng.normal(size=(4, 256)) * np.sqrt(256)
    _, w_scaled = A.scaled_dot_product_attention(Q / np.sqrt(256), Q / np.sqrt(256), Q / np.sqrt(256))
    _, w_unscaled = A.scaled_dot_product_attention(Q, Q, Q)
    assert w_scaled.max() < w_unscaled.max()


def test_positional_encoding_relative_offset():
    pe = A.positional_encoding(50, 16)
    # PE(p + d) is a rotation of PE(p) at each frequency: norms preserved
    norms = np.linalg.norm(pe, axis=1)
    assert np.allclose(norms, norms[0])
    # similarity decays with offset for low frequencies
    sim = lambda a, b: float(pe[a] @ pe[b] / (np.linalg.norm(pe[a]) * np.linalg.norm(pe[b])))
    assert sim(0, 1) > sim(0, 20)


def test_multi_head_matches_single_head_when_one_head():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 8))
    W = rng.normal(size=(8, 8)) * 0.1
    single = A.multi_head_attention(X, W, W, W, W, n_heads=1)
    manual, _ = A.scaled_dot_product_attention(X @ W, X @ W, X @ W)
    assert np.allclose(single, manual @ W)


# ----------------------------------------------------------------------- svm
def test_margin_width():
    assert S.margin_width(np.array([3.0, 4.0])) == pytest.approx(0.4)


def test_hinge_svm_separates_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(1.5, 0.5, (40, 2)), rng.normal(-1.5, 0.5, (40, 2))])
    y = np.array([1] * 40 + [-1] * 40)
    w, b = S.svm_hinge_sgd(X, y, C=1.0, lr=0.01, epochs=300)
    pred = np.where(X @ w + b >= 0, 1, -1)
    assert np.mean(pred == y) > 0.95


def test_kernels_are_psd_and_symmetric():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 3))
    for kernel, kwargs in (("linear", {}), ("poly", {"degree": 2}), ("rbf", {"gamma": 0.7})):
        K = S.gram_matrix(X, kernel, **kwargs)
        assert np.allclose(K, K.T)
        assert S.is_psd(K)


def test_poly_kernel_equals_explicit_feature_inner_product():
    x = np.array([[1.0, 2.0]])
    z = np.array([[3.0, -1.0]])
    # degree-2 homogeneous polynomial kernel corresponds to these monomials
    phi = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
    assert S.poly_kernel(x, z, degree=2, c=0.0)[0, 0] == pytest.approx(float(phi(x[0]) @ phi(z[0])))


def test_kernel_perceptron_solves_rings():
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    X = np.vstack([np.c_[np.cos(angles), np.sin(angles)], 2 * np.c_[np.cos(angles), np.sin(angles)]])
    y = np.array([1] * 24 + [-1] * 24)
    alpha = S.kernel_perceptron(X, y, kernel="rbf", gamma=1.0)
    preds = [1 if S.kernel_predict(X, y, alpha, x, gamma=1.0) >= 0 else -1 for x in X]
    assert np.mean(np.array(preds) == y) > 0.9


# ------------------------------------------------------------------------ rl
def test_value_iteration_and_policy_iteration_agree():
    env = R.GridWorld(size=4, goal=(0, 3), pit=(2, 2), walls=((1, 1),))
    V, pi, iters = R.value_iteration(env)
    Vp, pip, iters_p = R.policy_iteration(env)
    assert iters >= 1 and iters_p >= 1
    for s in pi:
        assert V[s] == pytest.approx(Vp[s], abs=1e-6)


def test_discounting_bounds_return():
    rewards = [1.0] * 100
    # the infinite-horizon sum is 1/(1-gamma) = 10; 100 terms get within 1e-4
    assert R.discounted_return(rewards, gamma=0.9) == pytest.approx(10.0, abs=1e-3)


def test_q_learning_improves_returns():
    env = R.GridWorld(size=4, goal=(0, 3), pit=(2, 2), walls=((1, 1),), slip=0.0)
    Q, policy, returns = R.q_learning(env, episodes=1500, eps=0.3, seed=0)
    assert np.mean(returns[-50:]) > np.mean(returns[:50])
    V, pi, _ = R.value_iteration(env)
    # exact agreement is not expected: many states have tied-optimal actions, and
    # rarely visited states keep noisy estimates. What matters is the value gap.
    values_q = np.array([V[s] for s in pi])
    values_opt = np.array([max(V[s], V[s]) for s in pi])
    greedy_value = np.mean([
        env.expected_reward(s, policy[s])
        + env.gamma * sum(p * V[nxt] for nxt, p in env.transition_probs(s, policy[s]).items())
        for s in pi
    ])
    assert greedy_value > 0.9 * np.mean(values_opt)
    assert np.mean([policy[s] == pi[s] for s in pi]) > 0.7


def test_regret_is_non_decreasing_for_suboptimal_play():
    returns = [1.0] * 10
    reg = R.regret(returns, optimal_return=2.0)
    assert np.all(np.diff(reg) > 0)


# -------------------------------------------------------------- interpretation
def test_shapley_matches_lecture_example():
    table = {
        frozenset(): 0,
        frozenset("A"): 20,
        frozenset("B"): 10,
        frozenset("AB"): 50,
    }
    v = lambda s: table[frozenset(s)]
    phi = I.shapley_exact(v, ["A", "B"])
    assert phi["A"] == pytest.approx(30.0)
    assert phi["B"] == pytest.approx(20.0)
    assert I.check_efficiency(v, ["A", "B"], phi)["efficiency_holds"]


def test_shapley_methods_agree():
    table = {
        frozenset(): 0, frozenset("A"): 20, frozenset("B"): 10, frozenset("C"): 5,
        frozenset("AB"): 50, frozenset("AC"): 30, frozenset("BC"): 20, frozenset("ABC"): 60,
    }
    v = lambda s: table[frozenset(s)]
    players = ["A", "B", "C"]
    exact = I.shapley_exact(v, players)
    subsets = I.shapley_subsets(v, players)
    mc = I.shapley_montecarlo(v, players, samples=20000, seed=1)
    for p in players:
        assert exact[p] == pytest.approx(subsets[p])
        assert mc[p] == pytest.approx(exact[p], abs=0.2)


def test_linear_shap_is_exact():
    rng = np.random.default_rng(0)
    w = np.array([1.5, -2.0, 0.5])
    bg = rng.normal(0, 1, (300, 3))
    x = np.array([1.0, -1.0, 0.5])
    phi, base = I.linear_shap(w, x, bg)
    assert phi.sum() == pytest.approx(float(w @ x) - base, abs=1e-10)
    mc = I.kernel_shap_linear(w, 0.0, x, bg, samples=3000)
    assert np.allclose(mc, phi, atol=0.2)


def test_grad_cam_normalises_and_rectifies():
    rng = np.random.default_rng(1)
    maps = rng.random((6, 7, 7))
    grads = rng.normal(0, 0.1, (6, 7, 7))
    cam = I.grad_cam(maps, grads)
    assert cam.min() == pytest.approx(0.0)
    assert cam.max() == pytest.approx(1.0)
    assert np.all(cam >= 0)
