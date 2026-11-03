import fs from "node:fs";
import path from "node:path";
import { modules, sectionOrder } from "./modules.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");
const CONTENT = path.join(ROOT, "content", "modules");
const OUT = path.join(ROOT, "modules");

const read = (p) => fs.readFileSync(p, "utf8");
const write = (p, s) => {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, s);
};

/* ------------------------------------------------------------------ shared */
function head({ title, description, depth = 1, extraScripts = [] }) {
  const up = "../".repeat(depth);
  return `<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${title}</title>
<meta name="description" content="${description}" />
<meta property="og:title" content="${title}" />
<meta property="og:description" content="${description}" />
<meta property="og:type" content="article" />
<link rel="icon" href="${up}assets/favicon.svg" type="image/svg+xml" />
<link rel="stylesheet" href="${up}assets/vendor/katex/katex.min.css" />
<link rel="stylesheet" href="${up}assets/css/site.css" />
<script defer src="${up}assets/vendor/katex/katex.min.js"></script>
<script defer src="${up}assets/vendor/katex/auto-render.min.js"></script>
<script defer src="${up}assets/js/viz.js"></script>
<script defer src="${up}assets/js/mla.js"></script>
<script defer src="${up}assets/js/site.js"></script>
${extraScripts.map((s) => `<script defer src="${up}${s}"></script>`).join("\n")}
</head>`;
}

function topbar(depth = 1) {
  const up = "../".repeat(depth);
  return `<a class="skip-link" href="#main">Skip to content</a>
<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="${up}index.html">
      <span class="brand-mark">ML</span>
      <span>ML from First Principles<small>Interactive course · CUNY CSC 74020</small></span>
    </a>
    <nav class="topnav" aria-label="Main">
      <a class="opt" href="${up}index.html#curriculum">Curriculum</a>
      <a class="opt" href="${up}index.html#python">Python library</a>
      <a class="opt" href="${up}index.html#projects">Projects</a>
      <a class="opt" href="${up}index.html#sources">Sources</a>
      <button class="icon-btn" data-theme-toggle type="button" aria-label="Toggle theme">☾</button>
      <button class="icon-btn" id="navToggle" type="button" aria-expanded="false" aria-controls="navMenu" aria-label="Menu">☰</button>
    </nav>
  </div>
  <div id="navMenu" hidden style="border-top:1px solid var(--line);background:var(--panel)">
    <div class="shell" style="padding:12px 20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 18px;font-size:.86rem">
      ${modules.map((m) => `<a href="${up}modules/${m.id}.html" style="padding:4px 0;color:var(--ink-2)">${m.num} · ${m.title}</a>`).join("\n      ")}
    </div>
  </div>
</header>`;
}

function footer(depth = 1) {
  const up = "../".repeat(depth);
  return `<footer class="site">
  <div class="shell">
    <div>
      <strong>ML from First Principles</strong><br />
      An interactive, open-source companion to CSC 74020 (Machine Learning), Fall 2025, taught by Prof. Jonathan Gryak at the CUNY Graduate Center.
    </div>
    <div>
      <a href="${up}index.html">Home</a> ·
      <a href="${up}index.html#curriculum">Curriculum</a> ·
      <a href="${up}index.html#python">Python library</a> ·
      <a href="${up}index.html#sources">Sources</a>
    </div>
  </div>
</footer>`;
}

function rail(mod) {
  return `<aside class="rail" aria-label="On this page">
  <h4>${mod.num} · ${mod.kicker}</h4>
  <ol>
    ${sectionOrder.map((s) => `<li><a href="#${s.id}">${s.label}</a></li>`).join("\n    ")}
  </ol>
  <h4 style="margin-top:22px">All modules</h4>
  <ol class="rail-sub">
    ${modules.map((m) => `<li><a href="${m.id}.html"${m.id === mod.id ? ' style="color:var(--accent);font-weight:600"' : ""}>${m.num} ${m.title}</a></li>`).join("\n    ")}
  </ol>
  <div style="margin-top:22px">
    <button class="btn btn-sm" data-mark-complete type="button" style="width:100%;justify-content:center">Mark module complete</button>
  </div>
</aside>`;
}

function pager(mod) {
  const i = modules.findIndex((m) => m.id === mod.id);
  const prev = modules[i - 1];
  const next = modules[i + 1];
  return `<nav class="pager" aria-label="Module navigation">
  ${prev ? `<a class="prev" href="${prev.id}.html"><span>← Previous</span><b>${prev.num} · ${prev.title}</b></a>` : `<a class="prev" href="../index.html"><span>← Back</span><b>Curriculum</b></a>`}
  ${next ? `<a class="next" href="${next.id}.html"><span>Next →</span><b>${next.num} · ${next.title}</b></a>` : `<a class="next" href="../index.html#curriculum"><span>Done →</span><b>Back to curriculum</b></a>`}
</nav>`;
}

/* ------------------------------------------------------------------- pages */
function buildModule(mod) {
  const src = path.join(CONTENT, `${mod.num}-${mod.id}.html`);
  if (!fs.existsSync(src)) {
    console.warn(`! missing content for ${mod.id} (${src})`);
    return false;
  }
  const body = read(src);
  const objectives = mod.objectives.map((o) => `<li>${o}</li>`).join("\n      ");
  const html = `${head({
    title: `${mod.num} · ${mod.title} — ML from First Principles`,
    description: mod.lede.replace(/"/g, "&quot;"),
    extraScripts: [`assets/js/labs/${mod.num}-${mod.id}.js`],
  })}
<body data-module="${mod.id}">
${topbar()}
<main id="main" class="shell with-rail">
${rail(mod)}
<article class="article">
  <header>
    <span class="kicker">Module ${mod.num} · ${mod.kicker}</span>
    <h1>${mod.title}</h1>
    <p class="lede">${mod.lede}</p>
    <div class="header-meta">
      <span>📚 Source: ${mod.lecture}</span>
      <span>⏱ ${mod.time}</span>
      <span>🧪 ${mod.labs.length} interactive labs</span>
      <span>🐍 3 implementations</span>
    </div>
  </header>

  <section class="block" id="objectives">
    <h2><span class="secnum">◎</span> Learning objectives</h2>
    <div class="card">
      <p class="small muted" style="margin-top:0">By the end of this module you should be able to:</p>
      <ul class="tight">
      ${objectives}
      </ul>
    </div>
  </section>

${body}

  ${pager(mod)}
</article>
</main>
${footer()}
</body>
</html>
`;
  write(path.join(OUT, `${mod.id}.html`), html);
  return true;
}

/* ------------------------------------------------------------------- index */
function buildIndex() {
  const cards = modules
    .map(
      (m) => `      <a class="module-card card" href="modules/${m.id}.html" data-progress-for="${m.id}">
        <span class="progress-dot" aria-hidden="true"></span>
        <span class="num">Module ${m.num} · ${m.kicker}</span>
        <h3>${m.title}</h3>
        <p>${m.summary}</p>
        <div class="meta">${m.tags.map((t) => `<span class="chip">${t}</span>`).join("")}<span class="chip chip-accent">${m.time}</span></div>
      </a>`
    )
    .join("\n");

  const html = `${head({
    title: "ML from First Principles — an interactive machine learning course",
    description:
      "Learn machine learning from first principles: intuition, derivations, interactive labs, from-scratch Python, NumPy and scikit-learn, quizzes and portfolio projects.",
    depth: 0,
  })}
<body data-module="home">
${topbar(0)}
<main id="main">
  <section class="hero">
    <div class="shell">
      <span class="eyebrow">Interactive course · 16 modules</span>
      <h1>Machine learning,<br />from first principles.</h1>
      <p class="lede">
        Every module takes one idea and rebuilds it four ways: the <strong>intuition</strong> and where it shows up in the
        world, the <strong>mathematics</strong> derived step by step, an <strong>interactive lab</strong> you can drag,
        and the same algorithm written three times — <strong>pure Python</strong>, <strong>NumPy</strong>, and
        <strong>scikit-learn</strong>. Then you prove you own it with quizzes, flashcards and a portfolio project.
      </p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="modules/foundations.html">Start with Module 01 →</a>
        <a class="btn" href="#curriculum">See the curriculum</a>
        <a class="btn btn-ghost" href="#python">Python library</a>
      </div>
      <div style="margin-top:34px;max-width:420px">
        <div style="display:flex;justify-content:space-between;font-size:.8rem;color:var(--ink-3);margin-bottom:6px">
          <span>Your progress</span><span data-overall-label>0 of 16 modules complete</span>
        </div>
        <div style="height:6px;border-radius:4px;background:var(--line)" >
          <div data-overall-progress="16" style="height:100%;width:0;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent-2));transition:width .3s"></div>
        </div>
      </div>
    </div>
  </section>

  <section class="shell" style="padding-bottom:20px">
    <div class="grid grid-3">
      <div class="card">
        <h3>1 · Intuition first</h3>
        <p class="small muted">A visual explanation and three real-world examples before a single symbol appears. If you cannot say what a model <em>does</em> in one sentence, the maths will not help.</p>
      </div>
      <div class="card">
        <h3>2 · Mathematics, derived</h3>
        <p class="small muted">Linear algebra, calculus and probability appear exactly where they are needed — with numbered, expandable derivations and proofs you can check line by line.</p>
      </div>
      <div class="card">
        <h3>3 · Labs you can drag</h3>
        <p class="small muted">Every module ships canvas-based simulations: sliders for learning rate, k, λ, C, γ, temperature. Break them on purpose. That is the fastest way to build intuition.</p>
      </div>
      <div class="card">
        <h3>4 · Three implementations</h3>
        <p class="small muted">Pure-Python from scratch (no libraries, so nothing hides), then vectorised NumPy, then the scikit-learn equivalent — with a note on when to use which.</p>
      </div>
      <div class="card">
        <h3>5 · Assessment that bites</h3>
        <p class="small muted">Multiple-choice quizzes with explanations for every distractor, a flashcard deck, and graded practice problems with full worked solutions.</p>
      </div>
      <div class="card">
        <h3>6 · A portfolio project</h3>
        <p class="small muted">Each module ends with a scoped project — brief, deliverables, rubric and starter code — so the course leaves you with artefacts, not just notes.</p>
      </div>
    </div>
  </section>

  <section class="shell" id="curriculum" style="padding:56px 20px 10px;scroll-margin-top:80px">
    <h2 style="margin-bottom:8px">Curriculum</h2>
    <p class="lede" style="margin-bottom:26px">Sixteen modules, following the lecture sequence of CSC 74020. Each one is self-contained; the ordering below is the recommended path.</p>
    <div class="grid grid-3">
${cards}
    </div>
  </section>

  <section class="shell" id="python" style="padding:56px 20px 10px;scroll-margin-top:80px">
    <h2 style="margin-bottom:8px">The <code>mlfs</code> Python library</h2>
    <p class="lede" style="margin-bottom:22px">
      Every code block in this course also exists as a tested module in the repository, so you can <code>pip install -e .</code>
      and run the exact code you read. Each algorithm ships with three implementations and a test suite that checks them
      against each other and against scikit-learn.
    </p>
    <div class="grid grid-2">
      <div class="card">
        <h3>From scratch, then vectorised</h3>
        <pre style="margin:0;overflow-x:auto"><code>from mlfs.linear_model import LinearRegressionScratch, LinearRegressionNumPy
from sklearn.linear_model import LinearRegression

m1 = LinearRegressionScratch().fit(X, y)   # loops + the normal equations
m2 = LinearRegressionNumPy().fit(X, y)     # vectorised, pinv-based
m3 = LinearRegression().fit(X, y)          # scikit-learn

assert abs(m1.coef_[0] - m3.coef_[0]) &lt; 1e-8</code></pre>
      </div>
      <div class="card">
        <h3>What is inside</h3>
        <ul class="tight small">
          <li><code>mlfs/</code> — 15 sub-packages covering the course (kNN, trees, SVM, PCA, CNN, RNN, attention, RL…)</li>
          <li><code>tests/</code> — 53 tests, including parity checks against scikit-learn / NumPy references</li>
          <li><code>assets/js/labs/</code> — the 16 lab scripts behind the interactive figures</li>
          <li><code>projects/</code> — starter code and rubrics for the 16 portfolio projects</li>
        </ul>
        <p class="small muted" style="margin-bottom:0">Dependencies: <code>numpy</code>, <code>scikit-learn</code>, <code>pytest</code>. PyTorch is optional and only used in the deep-learning notebooks.</p>
      </div>
    </div>
  </section>

  <section class="shell" id="projects" style="padding:56px 20px 10px;scroll-margin-top:80px">
    <h2 style="margin-bottom:8px">Portfolio projects</h2>
    <p class="lede" style="margin-bottom:22px">Each module ends with a project brief: a problem, a dataset suggestion, deliverables, a rubric and stretch goals. Do four of them and you have a portfolio.</p>
    <div class="grid grid-4">
      <div class="card"><h4>Regression clinic</h4><p class="small muted">Predict and explain — with a residual analysis a statistician would accept.</p></div>
      <div class="card"><h4>Classifier bake-off</h4><p class="small muted">Same data, six algorithms, one honest cross-validated leaderboard.</p></div>
      <div class="card"><h4>From-scratch deep net</h4><p class="small muted">Autograd, MLP, CNN — no frameworks, then match PyTorch.</p></div>
      <div class="card"><h4>Interpretability report</h4><p class="small muted">Explain a model's decisions to a non-technical stakeholder.</p></div>
    </div>
  </section>

  <section class="shell" id="sources" style="padding:56px 20px 80px;scroll-margin-top:80px">
    <h2 style="margin-bottom:8px">Sources &amp; attribution</h2>
    <div class="card">
      <p class="small" style="margin-top:0">
        The primary source for this site is the CSC 74020 <em>Machine Learning</em> lecture notes by
        <strong>Prof. Jonathan Gryak</strong> (Department of Computer Science, CUNY Graduate Center, Fall 2025):
        Lectures 1–14, including the slide decks for Transformers/LLMs, Dimensionality Reduction, SVMs, Reinforcement
        Learning and Model Interpretation. Notation, terminology, definitions, worked examples and the sequence of topics
        follow those notes closely.
      </p>
      <p class="small">
        Supplementary readings from the course reading list are referenced where a derivation benefits from a second
        treatment: Bishop, <em>Pattern Recognition and Machine Learning</em> (2006); Hastie, Tibshirani &amp; Friedman,
        <em>The Elements of Statistical Learning</em>; Géron, <em>Hands-On Machine Learning with Scikit-Learn, Keras &amp;
        TensorFlow</em>; Raschka, <em>Machine Learning with PyTorch and Scikit-Learn</em>; Zhang et al.,
        <em>Dive into Deep Learning</em>; Sutton &amp; Barto, <em>Reinforcement Learning: An Introduction</em>.
      </p>
      <p class="small muted" style="margin-bottom:0">
        Interactive labs, code, quizzes, flashcards and projects are original to this repository. Figures are drawn in
        the browser with the small canvas toolkit in <code>assets/js/viz.js</code> — no charting library, no tracking, no
        network calls. Math is typeset with a locally vendored KaTeX.
      </p>
    </div>
  </section>
</main>
${footer(0)}
</body>
</html>
`;
  write(path.join(ROOT, "index.html"), html);
}

/* ------------------------------------------------------------------- build */
fs.mkdirSync(OUT, { recursive: true });
let built = 0;
for (const mod of modules) if (buildModule(mod)) built++;
buildIndex();

/* favicon */
write(
  path.join(ROOT, "assets", "favicon.svg"),
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset=".7" stop-color="#7c3aed"/><stop offset="1" stop-color="#ec4899"/></linearGradient></defs><rect width="64" height="64" rx="14" fill="url(#g)"/><text x="32" y="42" font-family="ui-sans-serif,system-ui,sans-serif" font-size="26" font-weight="800" fill="#fff" text-anchor="middle">ML</text></svg>
`
);

console.log(`built ${built}/${modules.length} module pages + index.html`);
