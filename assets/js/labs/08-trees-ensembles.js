/* Module 08 labs — CART partitions, impurity criteria, AdaBoost. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  const COLORS = ["#4f46e5", "#0ea5e9", "#f59e0b"];

  function impurity(counts, criterion) {
    const n = counts.reduce((a, b) => a + b, 0);
    if (!n) return 0;
    const p = counts.map((c) => c / n);
    if (criterion === "gini") return 1 - p.reduce((s, v) => s + v * v, 0);
    if (criterion === "entropy") return -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);
    return 1 - Math.max(...p); // misclassification
  }

  function buildTree(pts, depth, maxDepth, minSamples, criterion, K) {
    const counts = new Array(K).fill(0);
    pts.forEach((p) => counts[p.c]++);
    const n = pts.length;
    const imp = impurity(counts, criterion);
    const label = counts.indexOf(Math.max(...counts));
    if (depth >= maxDepth || n < 2 * minSamples || imp === 0) return { leaf: true, label, n, counts, imp };

    let best = null;
    for (let f = 0; f < 2; f++) {
      const vals = Array.from(new Set(pts.map((p) => p.x[f]))).sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thr = (vals[i] + vals[i + 1]) / 2;
        const left = pts.filter((p) => p.x[f] <= thr);
        const right = pts.filter((p) => p.x[f] > thr);
        if (left.length < minSamples || right.length < minSamples) continue;
        const cl = new Array(K).fill(0), cr = new Array(K).fill(0);
        left.forEach((p) => cl[p.c]++); right.forEach((p) => cr[p.c]++);
        const score = (left.length * impurity(cl, criterion) + right.length * impurity(cr, criterion)) / n;
        if (!best || score < best.score) best = { f, thr, score, left, right };
      }
    }
    if (!best) return { leaf: true, label, n, counts, imp };
    return {
      leaf: false, f: best.f, thr: best.thr, n, counts, imp,
      left: buildTree(best.left, depth + 1, maxDepth, minSamples, criterion, K),
      right: buildTree(best.right, depth + 1, maxDepth, minSamples, criterion, K),
    };
  }

  function predict(node, x) {
    while (!node.leaf) node = x[node.f] <= node.thr ? node.left : node.right;
    return node.label;
  }

  function leaves(node, out) {
    if (node.leaf) { out.push(node); return out; }
    leaves(node.left, out); leaves(node.right, out);
    return out;
  }

  /* ------------------------------------------------------------------ lab 1
     CART: partitions, depth, and overfitting.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-tree"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 250 });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 9, ymin: 0.4, ymax: 1.02, height: 190 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["leaves", "—"], ["train accuracy", "—"], ["hold-out accuracy", "—"], ["root impurity", "—"]]);
    const rng = V.mulberry32(23);

    const pts = [], test = [];
    const labelOf = (a, b) => (a < 0.35 ? 0 : b < 0.55 ? 1 : 2);
    for (let i = 0; i < 150; i++) {
      const a = rng(), b = rng();
      // 12% label noise so that deep trees visibly overfit
      const c = rng() < 0.12 ? Math.floor(rng() * 3) : labelOf(a, b);
      pts.push({ x: [a + V.gauss(rng) * 0.02, b + V.gauss(rng) * 0.02], c });
    }
    for (let i = 0; i < 400; i++) {
      const a = rng(), b = rng();
      const c = rng() < 0.12 ? Math.floor(rng() * 3) : labelOf(a, b);
      test.push({ x: [a + V.gauss(rng) * 0.02, b + V.gauss(rng) * 0.02], c });
    }

    let depth = 3, minSamples = 5, criterion = "gini";
    let tree = null;

    function acc(data) {
      return data.reduce((s, p) => s + (predict(tree, p.x) === p.c ? 1 : 0), 0) / data.length;
    }

    function update() {
      tree = buildTree(pts, 0, depth, minSamples, criterion, 3);
      const lv = leaves(tree, []);
      out.update({ leaves: lv.length, "train accuracy": (acc(pts) * 100).toFixed(1) + "%", "hold-out accuracy": (acc(test) * 100).toFixed(1) + "%", "root impurity": tree.imp.toFixed(3) });

      fig.draw((f, ctx, t) => {
        f.regions((x, y) => predict(tree, [x, y]), {
          nx: 90, ny: 80,
          color: (v) => `${COLORS[v]}22`,
        });
        f.axes({ xstep: 0.25, ystep: 0.25, gridAlpha: 0.25 });
        // split lines
        (function walk(node) {
          if (node.leaf) return;
          if (node.f === 0) f.seg(node.thr, 0, node.thr, 1, { color: t.ink, width: 1.4 });
          else f.seg(0, node.thr, 1, node.thr, { color: t.ink, width: 1.4 });
          walk(node.left); walk(node.right);
        })(tree);
        pts.forEach((p) => f.dot(p.x[0], p.x[1], { color: COLORS[p.c], r: 3.2 }));
        f.text(f.pad.l + 8, f.pad.t + 12, `${criterion} · depth ≤ ${depth} · min samples ${minSamples}`, { color: t.ink, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        const trainAcc = [], testAcc = [];
        for (let d = 1; d <= 9; d++) {
          const tr = buildTree(pts, 0, d, minSamples, criterion, 3);
          const at = (data) => data.reduce((s, p) => s + (predict(tr, p.x) === p.c ? 1 : 0), 0) / data.length;
          trainAcc.push([d, at(pts)]);
          testAcc.push([d, at(test)]);
        }
        f.axes({ xstep: 1, ystep: 0.1, xlabel: "tree depth", ylabel: "accuracy" });
        f.line(trainAcc, { color: t.accent, width: 2.2 });
        f.line(testAcc, { color: "#dc2626", width: 2.2 });
        f.seg(depth, 0.4, depth, 1.02, { color: t.line2, dash: [4, 4], width: 1.2 });
        f.text(f.pad.l + 8, f.pad.t + 12, "— training   — hold-out", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "max depth", min: 1, max: 9, step: 1, value: depth, format: (v) => String(v), onInput: (v) => { depth = v; update(); } });
    V.slider(panel, { label: "min samples per leaf", min: 1, max: 20, step: 1, value: minSamples, format: (v) => String(v), onInput: (v) => { minSamples = v; update(); } });
    V.segmented(panel, {
      label: "splitting criterion",
      options: [{ label: "Gini", value: "gini" }, { label: "entropy", value: "entropy" }, { label: "misclass", value: "misclass" }],
      value: "gini",
      onChange: (v) => { criterion = v; update(); },
    });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Impurity criteria and split quality.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-gini"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 320 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["Gini", "—"], ["entropy", "—"], ["misclassification", "—"], ["best threshold", "—"], ["weighted impurity", "—"]]);

    let p = 0.3;
    let thr = 0.5;

    // a 1-D dataset whose class-1 rate rises with x
    const rng = V.mulberry32(13);
    const data = [];
    for (let i = 0; i < 120; i++) {
      const x = rng();
      data.push({ x, c: rng() < x ? 1 : 0 });
    }

    function weightedImpurity(t) {
      const left = data.filter((d) => d.x <= t), right = data.filter((d) => d.x > t);
      const imp = (arr) => {
        const c1 = arr.filter((d) => d.c === 1).length;
        return impurity([arr.length - c1, c1], "gini");
      };
      return (left.length * imp(left) + right.length * imp(right)) / data.length;
    }

    let bestT = 0.5, bestScore = Infinity;
    for (let i = 1; i < 60; i++) {
      const t = i / 60;
      const s = weightedImpurity(t);
      if (s < bestScore) { bestScore = s; bestT = t; }
    }

    function update() {
      const g = 1 - (p * p + (1 - p) * (1 - p));
      const h = -(p * Math.log2(p || 1e-12) + (1 - p) * Math.log2(1 - p || 1e-12));
      out.update({ Gini: g.toFixed(3), entropy: h.toFixed(3), misclassification: (1 - Math.max(p, 1 - p)).toFixed(3), "best threshold": bestT.toFixed(2), "weighted impurity": weightedImpurity(thr).toFixed(3) });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 0.2, ystep: 0.2, xlabel: "p = proportion of class 1", ylabel: "impurity" });
        f.fn((x) => 1 - (x * x + (1 - x) * (1 - x)), { color: "#4f46e5", width: 2.6 });
        f.fn((x) => -(x * Math.log2(x || 1e-12) + (1 - x) * Math.log2(1 - x || 1e-12)) / 2, { color: "#0ea5e9", width: 2, dash: [5, 4] });
        f.fn((x) => 1 - Math.max(x, 1 - x), { color: "#f59e0b", width: 1.8, dash: [2, 3] });
        f.seg(p, 0, p, 1, { color: t.line2, width: 1 });
        f.dot(p, g, { color: "#4f46e5", r: 5.5, ring: t.panel });
        f.text(f.pad.l + 8, f.pad.t + 12, "— Gini   ┄ entropy/2   ┄ misclassification", { color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 27, `data split at x = ${thr.toFixed(2)} → impurity ${weightedImpurity(thr).toFixed(3)} (best ${bestScore.toFixed(3)} at ${bestT.toFixed(2)})`, { color: t.ink, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "class-1 proportion p", min: 0, max: 1, step: 0.01, value: p, format: (v) => v.toFixed(2), onInput: (v) => { p = v; update(); } });
    V.slider(panel, { label: "candidate split x", min: 0.02, max: 0.98, step: 0.01, value: thr, format: (v) => v.toFixed(2), onInput: (v) => { thr = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     AdaBoost with decision stumps.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-boosting"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: -1.6, ymax: 1.6, height: 230 });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 1, ymin: -1.3, ymax: 1.3, height: 200 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["rounds", "—"], ["weighted error r", "—"], ["α", "—"], ["train accuracy", "—"]]);
    const rng = V.mulberry32(29);

    // 1-D two-class data with overlap
    const data = [];
    for (let i = 0; i < 80; i++) {
      const x = rng();
      const y = x > 0.55 ? 1 : -1;
      data.push({ x, y: rng() < 0.15 ? -y : y });
    }
    let w = data.map(() => 1 / data.length);
    let stumps = [];

    function bestStump() {
      let best = null;
      for (let i = 1; i < 100; i++) {
        const thr = i / 100;
        for (const dir of [1, -1]) {
          let err = 0;
          data.forEach((d, k) => {
            const pred = (d.x > thr ? 1 : -1) * dir;
            if (pred !== d.y) err += w[k];
          });
          if (!best || err < best.err) best = { thr, dir, err };
        }
      }
      return best;
    }

    function addRound() {
      const s = bestStump();
      const r = Math.min(Math.max(s.err, 1e-6), 1 - 1e-6);
      const alpha = 0.5 * Math.log((1 - r) / r);
      stumps.push({ ...s, alpha, r });
      // reweight
      let z = 0;
      data.forEach((d, k) => {
        const pred = (d.x > s.thr ? 1 : -1) * s.dir;
        w[k] = w[k] * Math.exp(pred === d.y ? -alpha : alpha);
        z += w[k];
      });
      w = w.map((v) => v / z);
      update();
    }

    function ensemble(x) {
      return stumps.reduce((s, st) => s + st.alpha * ((x > st.thr ? 1 : -1) * st.dir), 0);
    }

    function update() {
      const acc = data.reduce((s, d) => s + ((ensemble(d.x) >= 0 ? 1 : -1) === d.y ? 1 : 0), 0) / data.length;
      const last = stumps[stumps.length - 1];
      out.update({ rounds: stumps.length, "weighted error r": last ? last.r.toFixed(3) : "—", "α": last ? last.alpha.toFixed(3) : "—", "train accuracy": stumps.length ? (acc * 100).toFixed(1) + "%" : "—" });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 0.2, ystep: 0.5, xlabel: "x", ylabel: "y", gridAlpha: 0.3 });
        data.forEach((d, k) => {
          f.dot(d.x, d.y, { color: d.y === 1 ? "#dc2626" : "#1d4ed8", r: 2 + 5 * Math.sqrt(w[k] * data.length) });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, "dot size ∝ sample weight after the last round", { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 0.2, ystep: 0.5, xlabel: "x", ylabel: "F(x)", gridAlpha: 0.3 });
        if (stumps.length) {
          const pts = [];
          for (let i = 0; i <= 200; i++) { const x = i / 200; pts.push([x, V.clamp(ensemble(x), -1.3, 1.3)]); }
          f.line(pts, { color: t.accent, width: 2.2 });
          stumps.forEach((st) => {
            const pts2 = [];
            for (let i = 0; i <= 100; i++) {
              const x = i / 100;
              pts2.push([x, V.clamp(st.alpha * ((x > st.thr ? 1 : -1) * st.dir), -1.3, 1.3)]);
            }
            f.line(pts2, { color: "#f59e0b", width: 1, alpha: 0.5 });
          });
        }
        f.seg(f.xr[0], 0, f.xr[1], 0, { color: t.line2, width: 1 });
        f.text(f.pad.l + 8, f.pad.t + 12, "— ensemble F(x) = Σ αⱼ hⱼ(x)   thin = individual stumps", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.buttons(panel, [
      { label: "Add boosting round", primary: true, onClick: addRound },
      { label: "Add 5 rounds", onClick: () => { for (let i = 0; i < 5; i++) addRound(); } },
      { label: "Reset", onClick: () => { stumps = []; w = data.map(() => 1 / data.length); update(); } },
    ]);
    update();
  };
})();
