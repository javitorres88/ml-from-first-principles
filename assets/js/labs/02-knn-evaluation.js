/* Module 02 labs — k-NN, distance metrics, and thresholds under prevalence. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  const CLASS_COLORS = ["#4f46e5", "#0ea5e9", "#f59e0b"];

  /* ------------------------------------------------------------------ lab 1
     k-NN with a draggable query point and decision regions.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-knn"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1, xmax: 1, ymin: -1, ymax: 1, height: 360 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["neighbours", "—"], ["vote", "—"], ["prediction", "—"], ["k-th distance", "—"]]);
    const rng = V.mulberry32(11);

    const centers = [[-0.45, 0.35], [0.42, 0.28], [0.0, -0.5]];
    const X = [], y = [];
    centers.forEach((c, ci) => {
      for (let i = 0; i < 26; i++) {
        X.push([c[0] + V.gauss(rng) * 0.19, c[1] + V.gauss(rng) * 0.19]);
        y.push(ci);
      }
    });

    let k = 5, q = [0.05, 0.05], showRegions = true;

    function predict(px, py) {
      const d = X.map((p, i) => [Math.hypot(p[0] - px, p[1] - py), i]).sort((a, b) => a[0] - b[0]);
      const nb = d.slice(0, k);
      const votes = [0, 0, 0];
      nb.forEach(([, i]) => votes[y[i]]++);
      let best = 0;
      votes.forEach((v, c) => { if (v > votes[best]) best = c; });
      return { nb, votes, label: best };
    }

    function update() {
      const r = predict(q[0], q[1]);
      out.update({
        neighbours: r.nb.map(([, i]) => "c" + y[i]).join(" "),
        vote: r.votes.join(" / "),
        prediction: "class " + r.label,
        "k-th distance": r.nb[r.nb.length - 1][0].toFixed(3),
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      if (showRegions) {
        f.regions((x, yy) => predict(x, yy).label, {
          nx: 46, ny: 40,
          color: (v) => ["rgba(79,70,229,0.10)", "rgba(14,165,233,0.10)", "rgba(245,158,11,0.10)"][v] || "#fff",
        });
      }
      f.axes({ xstep: 0.5, ystep: 0.5, gridAlpha: 0.35 });
      X.forEach((p, i) => f.dot(p[0], p[1], { color: CLASS_COLORS[y[i]], r: 3.6 }));
      const r = predict(q[0], q[1]);
      const rk = r.nb[r.nb.length - 1][0];
      f.circle(q[0], q[1], rk, { dataRadius: true, stroke: t.ink3, dash: [4, 4], width: 1.2 });
      r.nb.forEach(([, i]) => {
        f.circle(X[i][0], X[i][1], 8, { stroke: t.ink, width: 1.6 });
        f.seg(q[0], q[1], X[i][0], X[i][1], { color: t.ink3, width: 0.8, alpha: 0.6 });
      });
      f.dot(q[0], q[1], { color: t.ink, r: 5.5, ring: t.panel });
      f.text(f.pad.l + 8, f.pad.t + 12, "drag the black query point", { color: t.ink3, font: "11px ui-sans-serif" });
    });

    fig.onPointer((p, kind) => {
      if (!p.inside) return;
      q = [V.clamp(p.x, -1, 1), V.clamp(p.y, -1, 1)];
      update();
    }, { hover: true });

    V.slider(panel, { label: "k (neighbours)", min: 1, max: 25, step: 2, value: k, format: (v) => String(v), onInput: (v) => { k = v; update(); } });
    V.toggle(panel, { label: "Show decision regions", value: true, onChange: (v) => { showRegions = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     The shape of a distance metric.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-distance"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1.6, xmax: 1.6, ymin: -1.6, ymax: 1.6, height: 360 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["metric", "—"], ["neighbours", "—"], ["prediction", "—"], ["k-th distance", "—"]]);
    const rng = V.mulberry32(5);

    const X = [], y = [];
    for (let i = 0; i < 60; i++) {
      const cls = i % 2;
      X.push([(cls ? 0.55 : -0.5) + V.gauss(rng) * 0.42, V.gauss(rng) * 0.55]);
      y.push(cls);
    }
    const q = [0.15, 0.55];
    let metric = "euclid", k = 5;
    const s = [1.0, 0.45]; // anisotropy for the scaled metric

    function dist(a, b) {
      const dx = a[0] - b[0], dy = a[1] - b[1];
      if (metric === "euclid") return Math.hypot(dx, dy);
      if (metric === "manhattan") return Math.abs(dx) + Math.abs(dy);
      // scaled (Mahalanobis-like): divide each axis by its scale
      return Math.hypot(dx / s[0], dy / s[1]);
    }

    function predict() {
      const d = X.map((p, i) => [dist(p, q), i]).sort((a, b) => a[0] - b[0]);
      const nb = d.slice(0, k);
      const votes = [0, 0];
      nb.forEach(([, i]) => votes[y[i]]++);
      return { nb, votes, label: votes[1] > votes[0] ? 1 : 0 };
    }

    function update() {
      const r = predict();
      out.update({
        metric: metric === "euclid" ? "Euclidean ℓ2" : metric === "manhattan" ? "Manhattan ℓ1" : "scaled (Mahalanobis)",
        neighbours: r.nb.map(([, i]) => y[i]).join(" "),
        prediction: "class " + r.label,
        "k-th distance": r.nb[r.nb.length - 1][0].toFixed(3),
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      f.axes({ xstep: 0.5, ystep: 0.5, gridAlpha: 0.4 });
      // unit ball around the query: {p : dist(p,q) = r_k}
      const r = predict();
      const rk = r.nb[r.nb.length - 1][0];
      const pts = [];
      for (let a = 0; a < 360; a += 3) {
        const th = (a * Math.PI) / 180;
        let px, py;
        if (metric === "euclid") { px = q[0] + rk * Math.cos(th); py = q[1] + rk * Math.sin(th); }
        else if (metric === "manhattan") {
          // |dx| + |dy| = rk
          const c = Math.cos(th), sn = Math.sin(th);
          const denom = Math.abs(c) + Math.abs(sn);
          px = q[0] + (rk * c) / denom; py = q[1] + (rk * sn) / denom;
        } else {
          px = q[0] + rk * s[0] * Math.cos(th); py = q[1] + rk * s[1] * Math.sin(th);
        }
        pts.push([px, py]);
      }
      f.line(pts.concat([pts[0]]), { color: t.accent, width: 1.8, dash: [5, 4] });
      X.forEach((p, i) => f.dot(p[0], p[1], { color: CLASS_COLORS[y[i]], r: 3.6, alpha: 0.9 }));
      r.nb.forEach(([, i]) => f.circle(X[i][0], X[i][1], 8, { stroke: t.ink, width: 1.5 }));
      f.dot(q[0], q[1], { color: t.ink, r: 5.5, ring: t.panel });
      f.text(f.pad.l + 8, f.pad.t + 12, "dashed curve = the k-th neighbour ball", { color: t.ink3, font: "11px ui-sans-serif" });
    });

    V.segmented(panel, {
      label: "Distance metric",
      options: [{ label: "ℓ2", value: "euclid" }, { label: "ℓ1", value: "manhattan" }, { label: "scaled", value: "scaled" }],
      value: "euclid",
      onChange: (v) => { metric = v; update(); },
    });
    V.slider(panel, { label: "k (neighbours)", min: 1, max: 15, step: 2, value: k, format: (v) => String(v), onInput: (v) => { k = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     Threshold, prevalence and the sensitivity/specificity trade-off.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-threshold"] = function (root) {
    const fig = V.makeIn(root, { xmin: -4, xmax: 4, ymin: 0, ymax: 1, height: 320 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [
      ["accuracy", "—"], ["sensitivity", "—"], ["specificity", "—"], ["precision", "—"], ["predicted positive", "—"],
    ]);
    const rng = V.mulberry32(3);

    let dprime = 1.6, thr = 0, prevalence = 0.3;
    const N = 1200;
    let scores = [], labels = [];

    function sample() {
      scores = []; labels = [];
      for (let i = 0; i < N; i++) {
        const pos = rng() < prevalence;
        const v = (pos ? dprime : 0) + V.gauss(rng);
        scores.push(v); labels.push(pos ? 1 : 0);
      }
    }

    function counts() {
      let tp = 0, fp = 0, tn = 0, fn = 0;
      scores.forEach((v, i) => {
        const pred = v >= thr ? 1 : 0;
        if (labels[i]) { pred ? tp++ : fn++; } else { pred ? fp++ : tn++; }
      });
      return M.metrics(tp, fp, tn, fn);
    }

    function update() {
      const m = counts();
      out.update({
        accuracy: (m.accuracy * 100).toFixed(1) + "%",
        sensitivity: (m.sensitivity * 100).toFixed(1) + "%",
        specificity: (m.specificity * 100).toFixed(1) + "%",
        precision: Number.isFinite(m.precision) ? (m.precision * 100).toFixed(1) + "%" : "—",
        "predicted positive": m.tp + m.fp,
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      const bins = 34, w = 8 / bins;
      const hN = new Array(bins).fill(0), hP = new Array(bins).fill(0);
      scores.forEach((v, i) => {
        const b = V.clamp(Math.floor((v + 4) / w), 0, bins - 1);
        (labels[i] ? hP : hN)[b]++;
      });
      const max = Math.max(...hN.map((v, i) => Math.max(v, hP[i]))) || 1;
      const barW = f.iw / bins;
      for (let b = 0; b < bins; b++) {
        const x0 = -4 + b * w;
        const px = f.sx(x0);
        const negH = (hN[b] / max) * f.ih;
        const posH = (hP[b] / max) * f.ih;
        // positives: filled; negatives: outlined. Overlaid, not stacked.
        ctx.fillStyle = "rgba(14,165,233,0.45)";
        ctx.fillRect(px, f.sy(0) - posH, barW - 1, posH);
        ctx.strokeStyle = "rgba(220,38,38,0.85)";
        ctx.lineWidth = 1;
        ctx.strokeRect(px + 0.5, f.sy(0) - negH + 0.5, barW - 1.5, Math.max(negH - 1, 0));
      }
      f.axes({ xstep: 1, xlabel: "classifier score" });
      f.seg(thr, 0, thr, 1, { color: t.ink, width: 2, dash: [5, 4] });
      f.text(f.sx(thr) + 6, f.pad.t + 12, "threshold", { color: t.ink, font: "600 11px ui-sans-serif" });
      // legend
      const lx = f.pad.l + 10, ly = f.pad.t + f.ih - 30;
      ctx.fillStyle = "rgba(14,165,233,0.45)"; ctx.fillRect(lx, ly - 6, 10, 9);
      f.text(lx + 15, ly, "positives", { color: t.ink3, font: "11px ui-sans-serif" });
      ctx.strokeStyle = "rgba(220,38,38,0.85)"; ctx.strokeRect(lx + 0.5, ly - 20.5, 9, 8);
      f.text(lx + 15, ly - 15, "negatives", { color: t.ink3, font: "11px ui-sans-serif" });
    });

    fig.onPointer((p) => {
      if (!p.inside) return;
      thr = V.clamp(p.x, -4, 4);
      update();
    }, { hover: true });

    V.slider(panel, { label: "Class separation d′", min: 0, max: 3.5, step: 0.1, value: dprime, format: (v) => v.toFixed(1), onInput: (v) => { dprime = v; sample(); update(); } });
    V.slider(panel, { label: "Prevalence", min: 0.02, max: 0.6, step: 0.02, value: prevalence, format: (v) => (v * 100).toFixed(0) + "%", onInput: (v) => { prevalence = v; sample(); update(); } });
    V.slider(panel, { label: "Threshold", min: -2, max: 3.5, step: 0.05, value: thr, format: (v) => v.toFixed(2), onInput: (v) => { thr = v; update(); } });
    V.buttons(panel, [{ label: "New sample", onClick: () => { sample(); update(); } }]);

    sample();
    update();
  };
})();
