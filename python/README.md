# `mlfs` — machine learning from first principles

The reference library behind the [ML from First Principles](../README.md) course. Every code block in the
course appears here as a tested module, so the code you read is the code you can run.

```bash
cd python
pip install -e ".[dev]"     # numpy, pytest, scikit-learn, shap, matplotlib
pytest                      # 53 tests, including parity checks against scikit-learn
```

```python
from mlfs import linear_model, neighbors, trees, cluster, attention

X = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]]
t = [3.0, 3.0, 7.0, 7.0]

model = linear_model.LinearRegressionNumPy().fit(X, t)
print(model.coef_, model.intercept_)

knn = neighbors.KNNNumPy(k=1, metric="euclidean").fit(X, [0, 0, 1, 1])
print(knn.predict([[3.5, 3.5]]))
```

## Three implementations per algorithm

| Suffix | What it is | Why it exists |
| --- | --- | --- |
| `*Scratch` | pure Python, no third-party imports | written to be read line by line |
| `*NumPy` | vectorised, still explicit | shows the same maths without the loops |
| *(tests only)* | the scikit-learn equivalent | the parity reference the tests assert against |

The tests check three things: that the scratch and NumPy versions agree to floating-point tolerance, that both agree
with scikit-learn (or a NumPy/SciPy reference where scikit-learn has no equivalent), and that the documented
examples in the course still run.

## Sub-packages

| Module | Contents |
| --- | --- |
| `mlfs.foundations` | metrics, confusion matrices, train/test splits, cross-validation |
| `mlfs.neighbors` | k-nearest neighbours, distance metrics, weighted voting |
| `mlfs.linear_model` | least squares, ridge, lasso, logistic regression |
| `mlfs.model_selection` | k-fold CV, grid search, learning curves |
| `mlfs.neural_network` | perceptron, MLP with manual backpropagation |
| `mlfs.optim` | SGD, momentum, RMSProp, Adam, initialisation schemes |
| `mlfs.trees` | CART, random forest, AdaBoost |
| `mlfs.cluster` | k-means, k-means++, DBSCAN, agglomerative clustering |
| `mlfs.conv` | 2-D convolution, pooling, batch norm |
| `mlfs.sequence` | vanilla RNN, LSTM cell, attention |
| `mlfs.attention` | scaled dot-product attention, positional encoding |
| `mlfs.decomposition` | SVD, PCA, linear autoencoder |
| `mlfs.svm` | hinge-loss SVM, kernels, kernel perceptron |
| `mlfs.rl` | grid worlds, value iteration, Q-learning |
| `mlfs.interpret` | Shapley values, permutation importance |

`torch` is optional (`pip install -e ".[torch]"`) and only used to show the framework equivalent of the
deep-learning modules.

## Layout

```
python/
├── pyproject.toml
├── src/mlfs/          # the library
└── tests/             # test_models.py, test_foundations_neighbors.py
```

## Licence

MIT — see [LICENSE](../LICENSE).
