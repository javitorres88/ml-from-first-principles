# Site map

How the published site maps onto the CSC 74020 lecture sequence, and where every piece of source lives.

## Pipeline

```
content/modules/NN-<id>.html   (authored partials — edit these)
        │
        │  node tools/build.mjs   ← tools/modules.mjs holds the registry
        ▼
modules/<id>.html + index.html  (generated — never edit by hand)
        │
        │  loads at runtime
        ▼
assets/css/site.css · assets/js/site.js · assets/js/viz.js · assets/js/mla.js
assets/js/labs/NN-<id>.js       (one lab script per module)
assets/vendor/katex/            (vendored, offline maths typesetting)
```

The build is deterministic: `node tools/build.mjs` rewrites `modules/*.html`, `index.html` and
`assets/favicon.svg` from `tools/modules.mjs` + `content/modules/*.html`. CI fails if the generated pages differ from
what is committed.

## Module ↔ lecture ↔ source

| # | Module page | Lecture | Authored partial | Lab script | Python (`mlfs`) | Project |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | `modules/foundations.html` | Notes 1 | `content/modules/01-foundations.html` | `assets/js/labs/01-foundations.js` | `mlfs.foundations` | `projects/01-evaluation-harness/` |
| 02 | `modules/knn-evaluation.html` | Notes 2 | `content/modules/02-knn-evaluation.html` | `assets/js/labs/02-knn-evaluation.js` | `mlfs.neighbors` | `projects/02-classifier-bakeoff/` |
| 03 | `modules/linear-regression.html` | Notes 2 | `content/modules/03-linear-regression.html` | `assets/js/labs/03-linear-regression.js` | `mlfs.linear_model` | `projects/03-regression-clinic/` |
| 04 | `modules/logistic-regression.html` | Notes 2 | `content/modules/04-logistic-regression.html` | `assets/js/labs/04-logistic-regression.js` | `mlfs.linear_model` | `projects/04-risk-score/` |
| 05 | `modules/regularization.html` | Notes 2 & 5 | `content/modules/05-regularization.html` | `assets/js/labs/05-regularization.js` | `mlfs.linear_model`, `mlfs.model_selection` | `projects/05-regularization-study/` |
| 06 | `modules/neural-networks.html` | Notes 3 | `content/modules/06-neural-networks.html` | `assets/js/labs/06-neural-networks.js` | `mlfs.neural_network` | `projects/06-mlp-from-scratch/` |
| 07 | `modules/backpropagation.html` | Notes 4 | `content/modules/07-backpropagation.html` | `assets/js/labs/07-backpropagation.js` | `mlfs.neural_network`, `mlfs.optim` | `projects/07-micrograd/` |
| 08 | `modules/trees-ensembles.html` | Notes 5 | `content/modules/08-trees-ensembles.html` | `assets/js/labs/08-trees-ensembles.js` | `mlfs.trees` | `projects/08-interpretable-tree/` |
| 09 | `modules/clustering.html` | Notes 6 | `content/modules/09-clustering.html` | `assets/js/labs/09-clustering.js` | `mlfs.cluster` | `projects/09-clustering-study/` |
| 10 | `modules/cnns.html` | Notes 7 | `content/modules/10-cnns.html` | `assets/js/labs/10-cnns.js` | `mlfs.conv` | `projects/10-cnn-from-scratch/` |
| 11 | `modules/sequences.html` | Notes 8 | `content/modules/11-sequences.html` | `assets/js/labs/11-sequences.js` | `mlfs.sequence` | `projects/11-char-lm/` |
| 12 | `modules/transformers.html` | Notes 10 | `content/modules/12-transformers.html` | `assets/js/labs/12-transformers.js` | `mlfs.attention` | `projects/12-mini-transformer/` |
| 13 | `modules/dimensionality-reduction.html` | Notes 11 | `content/modules/13-dimensionality-reduction.html` | `assets/js/labs/13-dimensionality-reduction.js` | `mlfs.decomposition` | `projects/13-compression-study/` |
| 14 | `modules/svm-kernels.html` | Notes 12 | `content/modules/14-svm-kernels.html` | `assets/js/labs/14-svm-kernels.js` | `mlfs.svm` | `projects/14-svm-small-data/` |
| 15 | `modules/reinforcement-learning.html` | Notes 13 | `content/modules/15-reinforcement-learning.html` | `assets/js/labs/15-reinforcement-learning.js` | `mlfs.rl` | `projects/15-rl-gridworld/` |
| 16 | `modules/interpretation.html` | Notes 14 | `content/modules/16-interpretation.html` | `assets/js/labs/16-interpretation.js` | `mlfs.interpret` | `projects/16-interpretability-report/` |

There is no Lecture 9 deck in the course folder; the sequence therefore skips from Notes 8 to Notes 10 between modules
11 and 12, matching the material actually taught.

## Six-part structure of every module

The build assumes — and CI checks — this order, which is also the order of the in-page rail:

| Section id | Heading | What belongs there |
| --- | --- | --- |
| `objectives` | Learning objectives | 4–6 measurable outcomes, injected from `tools/modules.mjs` |
| `intuition` | 1 · Intuition | the idea in plain language, a visual, real-world examples |
| `math` | 2 · Mathematical foundations | derivations, linear algebra, calculus, probability, proofs |
| `labs` | 3 · Interactive visualizations | 2–3 `<div class="lab" data-lab="…">` blocks driven by the module's lab script |
| `code` | 4 · Coding from scratch | three tabbed panes: Pure Python · NumPy · scikit-learn/PyTorch |
| `assessment` | 5 · Assessment | quiz with explanations, flashcards, practice problems + solutions |
| `project` | 6 · Portfolio project | brief, deliverables, rubric, stretch goals |

## Assets

| Path | Role |
| --- | --- |
| `assets/js/viz.js` | the canvas plotting toolkit — axes, scales, scatter, lines, contours, animation loop |
| `assets/js/mla.js` | tiny linear-algebra helpers used by the lab scripts (matrices, softmax, losses) |
| `assets/js/site.js` | theme toggle, nav, code tabs, copy buttons, quiz and flashcard behaviour, progress in `localStorage` |
| `assets/css/site.css` | the whole design system — tokens, layout, light/dark themes, print styles |
| `assets/data/iris.json` | the one dataset the labs load (everything else is generated in the browser) |
| `assets/vendor/katex/` | KaTeX 0.16.11, vendored so the site works offline |

## Repository layout

| Path | What it is |
| --- | --- |
| `index.html` | landing page + curriculum map (generated) |
| `modules/` | the 16 module pages (generated) |
| `content/modules/` | authored module bodies (the source of truth for prose) |
| `assets/` | CSS, canvas toolkit, per-module lab scripts, vendored KaTeX, iris data |
| `python/` | the tested `mlfs` library mirroring the code blocks |
| `projects/` | 16 project folders: brief, rubric, `starter.py` |
| `tools/` | `build.mjs`, `modules.mjs` (registry), `check.mjs` + `qa.mjs` (QA), `gen_projects.py`, screenshot helper |
| `docs/` | this map and the instructor notes |
| `_extracted/`, `_tmp/`, `_shots/` | local scratch (gitignored): lecture text dumps, KaTeX unpack, QA screenshots |
