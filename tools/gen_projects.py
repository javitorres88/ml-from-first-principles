"""Generate the 16 portfolio-project starter folders.

Each folder gets a README (brief + deliverables + rubric summary) and a
runnable starter script with TODOs. Run from the repository root:

    python tools/gen_projects.py
"""

import os
import textwrap

PROJECTS = [
    ("01-evaluation-harness", "An evaluation harness you would trust with a patient", "foundations",
     "Build a reusable evaluation package: stratified/grouped splits, confusion matrix and metric "
     "reporting, cross-validated model comparison, and a leakage audit.",
     ["Repository with README, evaluate.py and report.ipynb",
      "Unit tests proving metric parity with scikit-learn",
      "One-page report with dataset, protocol, CV table, confusion matrix and interpretation",
      "Leakage audit listing every transformer and where it was fitted"]),
    ("02-classifier-bakeoff", "The classifier bake-off", "knn-evaluation",
     "Quantify how much of k-NN's performance comes from the algorithm, the distance metric, and "
     "preprocessing, using a 7x2x3 CV grid with and without standardisation.",
     ["Notebook with the full grid as a heatmap",
      "knn.py with three metrics and weighted/uniform voting",
      "Report answering which factor mattered most",
      "Learning curve against a majority-class baseline"]),
    ("03-regression-clinic", "A regression you can defend in a meeting", "linear-regression",
     "Fit least squares three ways, report conditioning, diagnose residuals, and choose a ridge "
     "penalty by cross-validation.",
     ["Notebook comparing normal equations, lstsq and gradient descent",
      "Coefficient path and residual diagnostics figures",
      "One-page summary for a domain expert"]),
    ("04-risk-score", "A clinical risk score with odds ratios you can defend", "logistic-regression",
     "Build an interpretable binary classifier, report odds ratios with intervals, and check "
     "calibration and threshold choice.",
     ["Odds-ratio table with confidence intervals",
      "Calibration curve",
      "Threshold analysis with the confusion matrix at the deployed threshold",
      "Memo for a clinical audience"]),
    ("05-regularization-study", "A regularisation and validation study", "regularization",
     "Compare ridge and lasso paths, run nested cross-validation, and measure the optimism gap "
     "caused by leakage.",
     ["Coefficient paths for ridge and lasso",
      "Nested vs. non-nested CV comparison",
      "Bias-variance Monte Carlo on your data",
      "Leakage experiment quantified"]),
    ("06-mlp-from-scratch", "Build an MLP and prove it should exist", "neural-networks",
     "Implement a configurable MLP with a manual backward pass, gradient-check it, and study depth "
     "and activation choice.",
     ["mlp.py with gradient checks",
      "Depth/activation sweep figures",
      "Composition-trap demonstration",
      "Report on where the MLP loses to a linear model"]),
    ("07-micrograd", "A mini autograd engine", "backpropagation",
     "Implement reverse-mode automatic differentiation with a Tensor class, then train real "
     "networks with SGD, momentum and Adam.",
     ["micrograd.py with a Tensor class and optimisers",
      "Gradient checks plus parity tests against PyTorch",
      "Optimiser comparison notebook"]),
    ("08-interpretable-tree", "A tree you can hand to a stakeholder", "trees-ensembles",
     "Grow CART from scratch, compare with ensembles, and quantify the accuracy you trade for "
     "interpretability.",
     ["Tree diagram plus written rules",
      "Model comparison table with error bars",
      "Impurity vs. permutation importance analysis",
      "Stakeholder memo"]),
    ("09-clustering-study", "Discovering structure nobody labelled", "clustering",
     "Run k-means, DBSCAN and hierarchical clustering on unlabelled data and defend every choice, "
     "including k.",
     ["Selection curves (inertia, silhouette, gap)",
      "k-means and DBSCAN implementations with tests",
      "Cluster profile table",
      "Interpretation for a domain reader"]),
    ("10-cnn-from-scratch", "A CNN from scratch, then a real one", "cnns",
     "Implement convolution, pooling and batch norm with backward passes, train on MNIST, and "
     "compare with PyTorch.",
     ["cnn.py with gradient-checked backward passes",
      "Learning curves for both implementations",
      "Learned filter visualisation",
      "Explanation of the speed gap"]),
    ("11-char-lm", "A character-level language model, three ways", "sequences",
     "Train RNN, LSTM and GRU models on the same text and measure what gating buys in loss and in "
     "dependency length.",
     ["rnn.py with truncated BPTT",
      "Architecture comparison table",
      "Accuracy vs. dependency-length figure",
      "Generated samples at several checkpoints"]),
    ("12-mini-transformer", "Build and interpret a small transformer", "transformers",
     "Implement attention and a transformer block, train a small GPT, and ablate scaling, position "
     "encoding and masking.",
     ["attention.py with parity tests against PyTorch",
      "Ablation table",
      "Attention heatmaps with interpretation",
      "Reproducibility notes"]),
    ("13-compression-study", "Compress, visualise, reconstruct", "dimensionality-reduction",
     "Compare PCA, kernel PCA, t-SNE and autoencoders on one dataset and quantify what each "
     "preserves.",
     ["Scree and reconstruction figures",
      "PCA and autoencoder implementations",
      "Classifier accuracy before and after compression",
      "Visualisation comparison"]),
    ("14-svm-small-data", "SVMs on small, high-dimensional data", "svm-kernels",
     "Find where SVMs still win: sweep C and gamma, quantify the effect of scaling, and analyse "
     "support-vector sparsity.",
     ["C x gamma CV heatmap",
      "Hinge-loss SVM implementation",
      "Scaling experiment on identical folds",
      "Sparsity analysis tied to KKT conditions"]),
    ("15-rl-gridworld", "Solve, then learn, then generalise", "reinforcement-learning",
     "Take one MDP through value iteration, tabular Q-learning and linear function approximation, "
     "and measure regret.",
     ["MDP specification and exact solution",
      "Learning curves over five seeds",
      "Regret comparison for three exploration schedules",
      "Reflection on reward design"]),
    ("16-interpretability-report", "An interpretability report a stakeholder can act on", "interpretation",
     "Explain a model you built earlier: exact Shapley values, global and local attributions, a "
     "shortcut audit, and an honest caveats section.",
     ["Shapley implementation with a test against shap",
      "Global and local explanation figures",
      "Shortcut audit with quantified impact",
      "Caveats section"]),
]

STARTERS = {
    "foundations": '''"""Starter: evaluation harness. Fill in the TODOs."""
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
''',
    "linear_model": '''"""Starter: regression clinic."""
import numpy as np


def fit_normal_equations(X, t):
    """Solve A^T A w = A^T t with a bias column."""
    raise NotImplementedError


def fit_lstsq(X, t):
    """Use np.linalg.lstsq; compare conditioning with the normal equations."""
    raise NotImplementedError


def fit_gradient_descent(X, t, lr=0.1, epochs=2000):
    raise NotImplementedError


def residual_report(X, t, w, b):
    """Return residuals, and figures/diagnostics for the write-up."""
    raise NotImplementedError


if __name__ == "__main__":
    print("start by loading a regression dataset with at least 8 features")
''',
    "neural_network": '''"""Starter: MLP from scratch with a manual backward pass."""
import numpy as np


class MLP:
    def __init__(self, sizes, activation="relu", seed=0):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def backward(self, X, T, cache):
        raise NotImplementedError

    def fit(self, X, y, epochs=500, lr=0.05):
        raise NotImplementedError


def gradient_check(model, X, T, eps=1e-6):
    """Compare analytic and numerical gradients; return the max relative error."""
    raise NotImplementedError


if __name__ == "__main__":
    print("target: gradient check below 1e-6 for every parameter")
''',
    "backpropagation": '''"""Starter: a mini reverse-mode autodiff engine."""
import numpy as np


class Tensor:
    """A value plus the information needed to backpropagate through it."""

    def __init__(self, data, parents=(), backward_fn=None, requires_grad=False):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    def backward(self):
        """Topological sort, then a reverse sweep with gradient accumulation."""
        raise NotImplementedError


class SGD:
    def __init__(self, params, lr=0.01):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


if __name__ == "__main__":
    print("target: match PyTorch gradients to 1e-6 on random inputs")
''',
    "trees_ensembles": '''"""Starter: CART from scratch plus an ensemble comparison."""
import numpy as np


def gini(labels):
    raise NotImplementedError


def best_split(X, y):
    """Return (feature, threshold, weighted impurity)."""
    raise NotImplementedError


def grow(X, y, max_depth=4, min_samples=2):
    raise NotImplementedError


def predict_tree(node, X):
    raise NotImplementedError


if __name__ == "__main__":
    print("verify your splits against sklearn.tree.DecisionTreeClassifier")
''',
    "clustering": '''"""Starter: k-means, k-means++ and DBSCAN from scratch."""
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
''',
    "dimensionality_reduction": '''"""Starter: PCA by SVD and a linear autoencoder."""
import numpy as np


def pca(X, k=None):
    """Centre, take the SVD, return components and explained variance."""
    raise NotImplementedError


def low_rank(A, k):
    raise NotImplementedError


def linear_autoencoder(X, k=1, lr=0.05, steps=2000):
    raise NotImplementedError


if __name__ == "__main__":
    print("target: the autoencoder's decoder direction aligns with PC1 (|cos| > 0.99)")
''',
    "svm_kernels": '''"""Starter: hinge-loss SVM and kernel methods."""
import numpy as np


def hinge_loss(X, y, w, b, C=1.0):
    raise NotImplementedError


def svm_sgd(X, y, C=1.0, lr=0.01, epochs=500):
    raise NotImplementedError


def rbf_kernel(X, Z, gamma=1.0):
    raise NotImplementedError


def kernel_perceptron(X, y, gamma=1.0, epochs=30):
    raise NotImplementedError


if __name__ == "__main__":
    print("sweep C and gamma with cross-validation and plot the heatmap")
''',
    "reinforcement_learning": '''"""Starter: grid world, value iteration and Q-learning."""
import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class GridWorld:
    def __init__(self, size=5, goal=(0, 4), pit=(2, 2), walls=((1, 1), (3, 3)),
                 living=-0.04, gamma=0.9, slip=0.1):
        raise NotImplementedError

    def step(self, state, action, rng):
        raise NotImplementedError


def value_iteration(env, tol=1e-8):
    raise NotImplementedError


def q_learning(env, episodes=2000, alpha=0.2, eps=0.2, seed=0):
    raise NotImplementedError


if __name__ == "__main__":
    print("compare the Q-learning policy with the exact one from value iteration")
''',
    "interpretation": '''"""Starter: exact Shapley values and permutation importance."""
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
''',
}

DEFAULT_STARTER = '''"""Starter for this project. See README.md for the brief."""
import numpy as np

if __name__ == "__main__":
    print("read the module page for the full brief, then fill this in")
'''


def main(root: str = ".") -> None:
    base = os.path.join(root, "projects")
    os.makedirs(base, exist_ok=True)
    for slug, title, module, brief, deliverables in PROJECTS:
        folder = os.path.join(base, slug)
        os.makedirs(folder, exist_ok=True)
        readme = f"""# {title}

**Module:** [{module}](../../modules/{module}.html) · **Project:** `{slug}`

## Brief

{brief}

## Deliverables

{os.linesep.join(f"- [ ] {d}" for d in deliverables)}

## How to work

1. Read the module page end to end, including the interactive labs.
2. Copy `starter.py` and build up from there — keep the functions small enough to test.
3. Write the tests *before* the report: parity with a reference implementation is the
   strongest evidence your numbers mean anything.
4. Keep a `requirements.txt` and a fixed random seed so the results reproduce.

## Definition of done

- [ ] Every number in the report can be regenerated by running one command.
- [ ] Any from-scratch implementation is tested against a reference implementation.
- [ ] The report states the protocol, the metric, the uncertainty, and the limitations.
- [ ] A reader outside the course could follow the reasoning without the lecture notes.

## Rubric summary

| Criterion | Weight |
| --- | --- |
| Correctness and testing | 30% |
| Experimental design and protocol | 30% |
| Analysis and interpretation | 25% |
| Communication | 15% |

See the module page for the detailed rubric and stretch goals.
"""
        with open(os.path.join(folder, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(readme)
        starter = STARTERS.get(module, DEFAULT_STARTER)
        with open(os.path.join(folder, "starter.py"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(starter)
    print(f"generated {len(PROJECTS)} project folders in {base}")


if __name__ == "__main__":
    main(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
