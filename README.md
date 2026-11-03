# ML from First Principles

An interactive, open-source course that rebuilds machine learning from scratch — intuition, mathematics, live
simulations, and three implementations of every algorithm — built as a companion to **CSC 74020 Machine Learning**
(Prof. Jonathan Gryak, CUNY Graduate Center, Fall 2025).

```
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │ 1 · Intuition │ → │ 2 · Mathematics│ → │ 3 · Labs      │ → │ 4 · Code      │
    │ + real world  │   │ + derivations  │   │ sliders, sims │   │ py / np / sk  │
    └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
                                                    ↓
                                    ┌───────────────────────────────┐
                                    │ 5 · Assessment  6 · Project   │
                                    │ quizzes · cards · practice    │
                                    └───────────────────────────────┘
```

## What is in here

| Path | What it is |
| --- | --- |
| `index.html` | Landing page and curriculum map (open it in a browser, no build step needed) |
| `modules/` | The 16 module pages (generated — see `content/`) |
| `content/modules/` | Authored module bodies, one HTML partial per module |
| `assets/` | CSS, the canvas plotting toolkit, per-module lab scripts, vendored KaTeX |
| `python/` | `mlfs` — the tested Python library that mirrors every code block ([its README](python/README.md)) |
| `projects/` | 16 portfolio-project folders with briefs and starter code |
| `tools/` | Build script, module registry, screenshot helpers, project generator |
| `docs/` | [Site map](docs/site-map.md) (module → lecture → source) and [instructor notes](docs/instructor-notes.md) |

## The 16 modules

| # | Module | Source lecture |
| --- | --- | --- |
| 01 | What Machine Learning Actually Is | Notes 1 |
| 02 | Nearest Neighbours & Honest Evaluation | Notes 2 |
| 03 | Linear Models & Least Squares | Notes 2 |
| 04 | Logistic Regression & Classification Loss | Notes 2 |
| 05 | Regularization, Bias–Variance & Model Selection | Notes 2 & 5 |
| 06 | Artificial Neural Networks | Notes 3 |
| 07 | Backpropagation & Training Deep Networks | Notes 4 |
| 08 | Decision Trees, Bagging & Boosting | Notes 5 |
| 09 | Unsupervised Learning: Clustering | Notes 6 |
| 10 | Convolutional Neural Networks | Notes 7 |
| 11 | Sequences: RNNs, LSTMs & Attention | Notes 8 |
| 12 | Transformers & Large Language Models | Notes 10 |
| 13 | Dimensionality Reduction & Representation Learning | Notes 11 |
| 14 | Support Vector Machines & Kernels | Notes 12 |
| 15 | Reinforcement Learning | Notes 13 |
| 16 | Model Interpretation & What Comes Next | Notes 14 |

Every module contains, in this order: **intuition** (visual explanation + real-world examples), **mathematical
foundations** (derivations, linear algebra, calculus, probability, step-by-step proofs), **interactive
visualizations** (2–3 canvas labs with sliders and simulations), **coding** (pure Python → NumPy →
scikit-learn/PyTorch), **assessment** (quiz with explanations, flashcards, graded practice problems with
solutions) and a **portfolio project** with a rubric.

## Run the site locally

The site is static. Any file server works:

```bash
# from the repository root
python -m http.server 8000
# then open http://localhost:8000
```

Rebuild the module pages after editing `content/` or `tools/modules.mjs`:

```bash
node tools/build.mjs        # writes modules/*.html and index.html
# or, with npm installed
npm run build               # same thing
npm run check               # rebuild + verify sections, lab ids, links, script syntax
npm run serve               # static server on :8000
npm run qa                  # browser QA against :8000 (needs a local Chrome)
```

`npm run check` is the cheap, deterministic gate: it rebuilds, syntax-checks every browser script, and asserts that
every module page has its six sections in order, that every lab id in `tools/modules.mjs` has a container in the page,
and that no local link is broken. `npm run qa` drives a real browser over all 17 pages and checks that KaTeX rendered,
that the quizzes score full marks when answered correctly, that flashcards flip, solutions reveal, progress persists,
and that every lab canvas repaints when its sliders or buttons are used.

CI (`.github/workflows/pages.yml`) runs the build, the stale-page check, `tools/check.mjs` and `pytest`, then deploys
to GitHub Pages on `main`.

## Use the Python library

```bash
cd python
pip install -e ".[dev]"
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

Each sub-package mirrors one module and provides a `*Scratch` (pure Python), a `*NumPy` (vectorised) and — where
useful — a scikit-learn comparison in `python/tests/`.

## Interactive labs

The labs are hand-written canvas visualisations (no charting library, no network calls, no tracking). Examples:

- **Module 01** — confusion matrix and the accuracy paradox; polynomial degree sweep showing overfitting
- **Module 03** — residual squares you can drag, the loss surface with gradient descent, ill-conditioned polynomial fits
- **Module 07** — four optimisers racing on an elongated bowl; initialisation vs. depth
- **Module 09** — k-means step-through, DBSCAN on two moons, the elbow and silhouette curves
- **Module 12** — score/weight matrices with the causal mask and the √d_k scaling; sinusoidal positional encodings
- **Module 15** — value iteration and Q-learning in a grid world
- **Module 16** — exact Shapley values for a three-player game; raw contributions vs. SHAP

## Sources and attribution

The primary source is the CSC 74020 lecture notes by **Prof. Jonathan Gryak** (CUNY Graduate Center, Fall 2025):
Lectures 1–14, including the decks for Transformers/LLMs, Dimensionality Reduction, SVMs, Reinforcement Learning and
Model Interpretation. Notation, terminology, definitions, worked examples and the topic sequence follow those notes.

Supplementary readings referenced where a derivation benefits from a second treatment: Bishop, *Pattern Recognition
and Machine Learning*; Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*; Géron, *Hands-On Machine
Learning*; Raschka, *Machine Learning with PyTorch and Scikit-Learn*; Zhang et al., *Dive into Deep Learning*; Sutton &
Barto, *Reinforcement Learning: An Introduction*.

Interactive labs, code, quizzes, flashcards and projects are original to this repository.

## Contributing

Corrections to the mathematics, clearer derivations, and additional labs are welcome. Please:

1. Edit the partial in `content/modules/`, not the generated page in `modules/`.
2. Run `node tools/build.mjs` and check the page in a browser.
3. If you change code, mirror it in `python/src/mlfs/` and run `pytest`.
4. Keep the six-part structure of every module intact.

## License

MIT — see [LICENSE](LICENSE). The vendored KaTeX distribution keeps its own MIT license in
`assets/vendor/katex/LICENSE`.
