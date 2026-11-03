"""mlfs — machine learning from first principles.

A small, readable reference implementation library that accompanies the
interactive course at the repository root. Every algorithm is provided in
three flavours where it makes sense:

* ``*Scratch``  — pure Python, no third-party dependencies, written for reading
* ``*NumPy``    — vectorised, still explicit
* a thin wrapper around the scikit-learn equivalent, used in the tests as the
  parity reference

Sub-packages are named after the modules of the course:

======================  ==================================================
module                  contents
======================  ==================================================
``foundations``         metrics, confusion matrices, splits, cross-validation
``neighbors``           k-nearest neighbours, distance metrics
``linear_model``        least squares, ridge, lasso, logistic regression
``model_selection``     k-fold CV, grid search, learning curves
``neural_network``      perceptron, MLP with manual backprop
``optim``               SGD, momentum, RMSProp, Adam, initialisation schemes
``trees``               CART, random forest, AdaBoost
``cluster``             k-means, k-means++, DBSCAN, agglomerative clustering
``conv``                2-D convolution, pooling, batch norm
``sequence``            vanilla RNN, LSTM cell, attention
``attention``           scaled dot-product attention, positional encoding
``decomposition``       SVD, PCA, linear autoencoder
``svm``                 hinge-loss SVM, kernels, kernel perceptron
``rl``                  grid worlds, value iteration, Q-learning
``interpret``           Shapley values, permutation importance
======================  ==================================================
"""

from . import (  # noqa: F401
    attention,
    cluster,
    conv,
    decomposition,
    foundations,
    interpret,
    linear_model,
    model_selection,
    neighbors,
    neural_network,
    optim,
    rl,
    sequence,
    svm,
    trees,
)

__version__ = "0.1.0"

__all__ = [
    "attention",
    "cluster",
    "conv",
    "decomposition",
    "foundations",
    "interpret",
    "linear_model",
    "model_selection",
    "neighbors",
    "neural_network",
    "optim",
    "rl",
    "sequence",
    "svm",
    "trees",
    "__version__",
]
