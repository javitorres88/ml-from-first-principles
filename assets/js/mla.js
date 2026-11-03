/* ==========================================================================
   mla.js — small numerical helpers shared by the interactive labs.
   Deliberately tiny and readable; not a replacement for NumPy.
   ========================================================================== */
(function () {
  "use strict";
  const A = {};

  A.zeros = (n, m) => Array.from({ length: n }, () => new Array(m).fill(0));
  A.T = (M) => M[0].map((_, j) => M.map((row) => row[j]));
  A.matmul = (X, Y) =>
    X.map((row) => Y[0].map((_, j) => row.reduce((s, v, k) => s + v * Y[k][j], 0)));
  A.matvec = (X, v) => X.map((row) => row.reduce((s, x, i) => s + x * v[i], 0));
  A.dot = (u, v) => u.reduce((s, x, i) => s + x * v[i], 0);
  A.norm = (v) => Math.sqrt(A.dot(v, v));

  /** Solve A x = b for square A (Gaussian elimination, partial pivoting). */
  A.solve = function (Amat, b) {
    const n = Amat.length;
    const M = Amat.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-12) continue;
      [M[c], M[p]] = [M[p], M[c]];
      const piv = M[c][c];
      for (let k = c; k <= n; k++) M[c][k] /= piv;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c];
        if (!f) continue;
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((row) => row[n]);
  };

  /** Least squares via the normal equations (adds a column of ones if bias). */
  A.lstsq = function (X, y, bias = true) {
    const Xa = bias ? X.map((r) => [1].concat(r)) : X.map((r) => r.slice());
    const Xt = A.T(Xa);
    return A.solve(A.matmul(Xt, Xa), A.matvec(Xt, y));
  };

  /** Ridge regression: solve (XᵀX + λI) w = Xᵀy  (bias not penalised). */
  A.ridge = function (X, y, lambda, bias = true) {
    const Xa = bias ? X.map((r) => [1].concat(r)) : X.map((r) => r.slice());
    const Xt = A.T(Xa);
    const G = A.matmul(Xt, Xa);
    for (let i = bias ? 1 : 0; i < G.length; i++) G[i][i] += lambda;
    return A.solve(G, A.matvec(Xt, y));
  };

  /** Polynomial fit y ≈ Σ c_k x^k, returns coefficients ascending in degree. */
  A.polyfit = function (xs, ys, deg) {
    const X = xs.map((x) => Array.from({ length: deg }, (_, k) => Math.pow(x, k + 1)));
    return A.lstsq(X, ys, true);
  };
  A.polyval = (c, x) => c.reduce((s, v, k) => s + v * Math.pow(x, k), 0);

  A.dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  /** k-NN prediction: returns {label, votes:[[label,count],...], neighbours:[i,...]} */
  A.knn = function (X, y, k, q) {
    const d = X.map((p, i) => [A.dist(p, q), i]).sort((a, b) => a[0] - b[0]);
    const nb = d.slice(0, k);
    const votes = new Map();
    nb.forEach(([, i]) => votes.set(y[i], (votes.get(y[i]) || 0) + 1));
    let best = null, bc = -1;
    votes.forEach((c, l) => { if (c > bc) { bc = c; best = l; } });
    return { label: best, votes: Array.from(votes), neighbours: nb.map(([, i]) => i), dists: nb.map(([dd]) => dd) };
  };

  /** k-means with k-means++ init. Returns full history for animation. */
  A.kmeans = function (X, k, opts) {
    opts = opts || {};
    const rng = opts.rng || Math.random;
    const iters = opts.iters || 40;
    const plusplus = opts.init !== "random";
    let centers;
    if (plusplus) {
      centers = [X[Math.floor(rng() * X.length)].slice()];
      while (centers.length < k) {
        const d2 = X.map((p) => Math.min(...centers.map((c) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)));
        const tot = d2.reduce((a, b) => a + b, 0);
        let r = rng() * tot, idx = 0;
        for (let i = 0; i < d2.length; i++) { r -= d2[i]; if (r <= 0) { idx = i; break; } }
        centers.push(X[idx].slice());
      }
    } else {
      const pool = X.slice();
      centers = [];
      for (let i = 0; i < k; i++) centers.push(pool.splice(Math.floor(rng() * pool.length), 1)[0].slice());
    }
    const history = [{ centers: centers.map((c) => c.slice()), labels: null }];
    let labels = new Array(X.length).fill(-1);
    for (let it = 0; it < iters; it++) {
      labels = X.map((p) => {
        let best = 0, bd = Infinity;
        centers.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; best = j; } });
        return best;
      });
      const sums = Array.from({ length: k }, () => [0, 0, 0]);
      X.forEach((p, i) => { const s = sums[labels[i]]; s[0] += p[0]; s[1] += p[1]; s[2]++; });
      const next = centers.map((c, j) => (sums[j][2] ? [sums[j][0] / sums[j][2], sums[j][1] / sums[j][2]] : c.slice()));
      history.push({ centers: next.map((c) => c.slice()), labels: labels.slice() });
      const moved = next.some((c, j) => Math.hypot(c[0] - centers[j][0], c[1] - centers[j][1]) > 1e-9);
      centers = next;
      if (!moved) break;
    }
    const inertia = X.reduce((s, p, i) => s + (p[0] - centers[labels[i]][0]) ** 2 + (p[1] - centers[labels[i]][1]) ** 2, 0);
    return { centers, labels, inertia, history };
  };

  A.silhouette = function (X, labels) {
    const ks = Array.from(new Set(labels));
    if (ks.length < 2) return 0;
    let total = 0;
    X.forEach((p, i) => {
      const byC = new Map();
      X.forEach((q, j) => {
        if (i === j) return;
        const c = labels[j];
        if (!byC.has(c)) byC.set(c, []);
        byC.get(c).push(A.dist(p, q));
      });
      const own = byC.get(labels[i]) || [];
      const a = own.length ? own.reduce((s, v) => s + v, 0) / own.length : 0;
      let b = Infinity;
      byC.forEach((ds, c) => { if (c !== labels[i]) b = Math.min(b, ds.reduce((s, v) => s + v, 0) / ds.length); });
      total += (b - a) / Math.max(a, b);
    });
    return total / X.length;
  };

  /** PCA via the 2×2 covariance eigen-decomposition (for 2-D data). */
  A.pca2 = function (X) {
    const n = X.length;
    const mx = X.reduce((s, p) => s + p[0], 0) / n;
    const my = X.reduce((s, p) => s + p[1], 0) / n;
    let sxx = 0, syy = 0, sxy = 0;
    X.forEach((p) => { const dx = p[0] - mx, dy = p[1] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; });
    sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
    const tr = sxx + syy, det = sxx * syy - sxy * sxy;
    const l1 = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    const l2 = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    let v1 = Math.abs(sxy) > 1e-12 ? [l1 - syy, sxy] : sxx >= syy ? [1, 0] : [0, 1];
    const nv = Math.hypot(v1[0], v1[1]) || 1;
    v1 = [v1[0] / nv, v1[1] / nv];
    const v2 = [-v1[1], v1[0]];
    return { mean: [mx, my], cov: [[sxx, sxy], [sxy, syy]], eigenvalues: [l1, l2], axes: [v1, v2], explained: [l1 / (l1 + l2), l2 / (l1 + l2)] };
  };

  A.sigmoid = (z) => 1 / (1 + Math.exp(-z));
  A.logit = (p) => Math.log(p / (1 - p));
  A.softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

  /** Binary confusion matrix from counts. */
  A.metrics = function (tp, fp, tn, fn) {
    const n = tp + fp + tn + fn;
    const sens = tp + fn ? tp / (tp + fn) : NaN;
    const spec = tn + fp ? tn / (tn + fp) : NaN;
    const prec = tp + fp ? tp / (tp + fp) : NaN;
    const npv = tn + fn ? tn / (tn + fn) : NaN;
    return {
      n, tp, fp, tn, fn,
      accuracy: (tp + tn) / n,
      sensitivity: sens,
      specificity: spec,
      precision: prec,
      npv,
      f1: Number.isFinite(prec) && Number.isFinite(sens) && prec + sens > 0 ? (2 * prec * sens) / (prec + sens) : NaN,
      balanced: (sens + spec) / 2,
      prevalence: (tp + fn) / n,
    };
  };

  /** Fit a logistic regression by gradient descent (2-D input + bias). */
  A.logisticFit = function (X, y, opts) {
    opts = opts || {};
    const lr = opts.lr || 0.5, iters = opts.iters || 400, l2 = opts.l2 || 0;
    let w = [0, 0], b = 0;
    const n = X.length;
    const hist = [];
    for (let t = 0; t < iters; t++) {
      let gw0 = 0, gw1 = 0, gb = 0, loss = 0;
      for (let i = 0; i < n; i++) {
        const z = w[0] * X[i][0] + w[1] * X[i][1] + b;
        const p = A.sigmoid(z);
        const e = p - y[i];
        gw0 += e * X[i][0]; gw1 += e * X[i][1]; gb += e;
        loss += -(y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12));
      }
      w[0] -= lr * (gw0 / n + l2 * w[0]);
      w[1] -= lr * (gw1 / n + l2 * w[1]);
      b -= lr * (gb / n);
      if (t % 10 === 0) hist.push({ w: w.slice(), b, loss: loss / n });
    }
    return { w, b, loss: hist.length ? hist[hist.length - 1].loss : NaN, history: hist };
  };

  window.MLA = A;
})();
