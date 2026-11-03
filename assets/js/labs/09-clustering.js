/* Module 09 labs — k-means, choosing k, DBSCAN. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  const PAL = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];

  function blobs(seed, specs) {
    const rng = V.mulberry32(seed);
    const pts = [];
    specs.forEach(([cx, cy, n, s]) => {
      for (let i = 0; i < n; i++) pts.push([cx + V.gauss(rng) * s, cy + V.gauss(rng) * s]);
    });
    return pts;
  }

  /* ------------------------------------------------------------------ lab 1
     k-means, step by step.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-kmeans"] = function (root) {
    const fig = V.makeIn(root, { xmin: -3.4, xmax: 3.4, ymin: -3.0, ymax: 3.0, height: 380 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["iteration", "—"], ["inertia (SSE)", "—"], ["centroid moves", "—"], ["silhouette", "—"]]);
    const X = blobs(31, [[-1.6, 1.2, 55, 0.5], [1.5, 1.0, 55, 0.55], [0.0, -1.6, 55, 0.5], [1.9, -1.5, 25, 0.35]]);

    let k = 3, init = "++", seed = 5;
    let state = null, iter = 0, timer = null;

    function runKmeans(kk, initMode, sd) {
      return M.kmeans(X, kk, { rng: V.mulberry32(sd), init: initMode === "++" ? "pp" : "random", iters: 30 });
    }

    function reset() {
      stop();
      const res = runKmeans(k, init, seed);
      state = { history: res.history, idx: 0 };
      iter = 0;
      update();
    }
    function stepOnce() {
      if (!state) return;
      if (state.idx < state.history.length - 1) {
        state.idx++; iter++;
        update();
      } else stop();
    }
    function stop() { clearInterval(timer); timer = null; }

    function current() { return state.history[state.idx]; }

    function update() {
      const cur = current();
      const labels = cur.labels || X.map((p) => {
        let best = 0, bd = Infinity;
        cur.centers.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; best = j; } });
        return best;
      });
      const inertia = X.reduce((s, p, i) => s + (p[0] - cur.centers[labels[i]][0]) ** 2 + (p[1] - cur.centers[labels[i]][1]) ** 2, 0);
      const sil = state.idx > 0 ? M.silhouette(X, labels) : 0;
      const prev = state.idx > 0 ? state.history[state.idx - 1].centers : null;
      const moved = prev ? Math.max(...cur.centers.map((c, j) => Math.hypot(c[0] - prev[j][0], c[1] - prev[j][1]))) : 0;
      out.update({ iteration: iter, "inertia (SSE)": inertia, "centroid moves": moved, silhouette: sil });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.3 });
        X.forEach((p, i) => {
          f.seg(p[0], p[1], cur.centers[labels[i]][0], cur.centers[labels[i]][1], { color: PAL[labels[i] % PAL.length], width: 0.6, alpha: 0.35 });
          f.dot(p[0], p[1], { color: PAL[labels[i] % PAL.length], r: 3.6 });
        });
        cur.centers.forEach((c, j) => {
          f.circle(c[0], c[1], 9, { fill: PAL[j % PAL.length], stroke: "#fff", width: 2 });
          f.text(f.sx(c[0]), f.sy(c[1]), "✕", { align: "center", color: "#fff", font: "700 11px ui-sans-serif" });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, `k = ${k} · init ${init === "++" ? "k-means++" : "random"}`, { color: t.ink, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "clusters k", min: 2, max: 6, step: 1, value: k, format: (v) => String(v), onInput: (v) => { k = v; reset(); } });
    V.segmented(panel, {
      label: "initialisation",
      options: [{ label: "k-means++", value: "++" }, { label: "random", value: "rnd" }],
      value: "++",
      onChange: (v) => { init = v; reset(); },
    });
    V.buttons(panel, [
      { label: "Run", primary: true, onClick: () => { if (timer) return; timer = setInterval(stepOnce, 550); } },
      { label: "Step", onClick: stepOnce },
      { label: "New seed", onClick: () => { seed++; reset(); } },
    ]);
    reset();
  };

  /* ------------------------------------------------------------------ lab 2
     Choosing k: inertia, silhouette, elbow.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-elbow"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 1.5, xmax: 8.5, ymin: 0, ymax: 1, height: 220 });
    const fig2 = new V.Fig(c2, { xmin: -3.4, xmax: 3.4, ymin: -3.0, ymax: 3.0, height: 220 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["k", "—"], ["inertia", "—"], ["silhouette", "—"], ["verdict", "—"]]);
    const X = blobs(77, [[-1.5, 1.1, 60, 0.55], [1.6, 1.0, 60, 0.55], [0.1, -1.7, 60, 0.55]]);

    let k = 3;
    const results = {};
    for (let kk = 2; kk <= 8; kk++) {
      const r = M.kmeans(X, kk, { rng: V.mulberry32(9), init: "pp" });
      results[kk] = { inertia: r.inertia, sil: M.silhouette(X, r.labels), labels: r.labels, centers: r.centers };
    }
    const inertias = Object.values(results).map((r) => r.inertia);
    const maxI = Math.max(...inertias), minI = Math.min(...inertias);
    const sils = Object.values(results).map((r) => r.sil);

    function update() {
      const r = results[k];
      const verdict = r.sil > 0.6 ? "clear structure" : r.sil > 0.4 ? "weak / overlapping" : "k is wrong";
      out.update({ k, inertia: r.inertia, silhouette: r.sil, verdict });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 0.25, xlabel: "k", ylabel: "normalised" });
        const norm = (v) => (v - minI) / (maxI - minI || 1);
        f.line(Object.entries(results).map(([kk, rr]) => [+kk, norm(rr.inertia)]), { color: "#4f46e5", width: 2.4 });
        f.line(Object.entries(results).map(([kk, rr]) => [+kk, rr.sil]), { color: "#0f9d58", width: 2.2, dash: [5, 4] });
        Object.entries(results).forEach(([kk, rr]) => {
          f.dot(+kk, norm(rr.inertia), { color: "#4f46e5", r: +kk === k ? 5.5 : 3 });
          f.dot(+kk, rr.sil, { color: "#0f9d58", r: +kk === k ? 5 : 2.5 });
        });
        f.seg(k, 0, k, 1, { color: t.line2, width: 1, dash: [3, 3] });
        f.text(f.pad.l + 8, f.pad.t + 12, "— inertia (normalised, lower better)   ┄ silhouette (higher better)", { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.3 });
        X.forEach((p, i) => f.dot(p[0], p[1], { color: PAL[r.labels[i] % PAL.length], r: 3.4 }));
        r.centers.forEach((c, j) => f.circle(c[0], c[1], 8, { fill: PAL[j % PAL.length], stroke: "#fff", width: 2 }));
      });
    }

    V.slider(panel, { label: "k", min: 2, max: 8, step: 1, value: k, format: (v) => String(v), onInput: (v) => { k = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     DBSCAN: core, border, noise.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-dbscan"] = function (root) {
    const fig = V.makeIn(root, { xmin: -3.4, xmax: 3.4, ymin: -3.0, ymax: 3.0, height: 400 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["clusters", "—"], ["core points", "—"], ["border", "—"], ["noise", "—"]]);

    const rng = V.mulberry32(101);
    const X = [];
    // two moons
    for (let i = 0; i < 70; i++) {
      const th = Math.PI * (i / 69);
      X.push([-1.2 + 1.1 * Math.cos(th), 0.8 - 1.1 * Math.sin(th)]);
    }
    for (let i = 0; i < 70; i++) {
      const th = Math.PI * (i / 69);
      X.push([1.2 - 1.1 * Math.cos(th), -0.8 + 1.1 * Math.sin(th)]);
    }
    X.forEach((p) => { p[0] += V.gauss(rng) * 0.06; p[1] += V.gauss(rng) * 0.06; });
    // noise
    for (let i = 0; i < 22; i++) X.push([-3 + rng() * 6, -2.8 + rng() * 5.6]);

    let eps = 0.3, minPts = 6;

    function dbscan() {
      const n = X.length;
      const neighbours = X.map((p, i) => X.map((q, j) => (i === j ? -1 : Math.hypot(p[0] - q[0], p[1] - q[1]))).map((d, j) => [d, j]).filter(([d]) => d >= 0 && d <= eps).map(([, j]) => j));
      const core = neighbours.map((nb) => nb.length + 1 >= minPts);
      const labels = new Array(n).fill(-1);
      let c = 0;
      for (let i = 0; i < n; i++) {
        if (!core[i] || labels[i] !== -1) continue;
        const stack = [i];
        labels[i] = c;
        while (stack.length) {
          const cur = stack.pop();
          if (!core[cur]) continue;
          neighbours[cur].forEach((j) => { if (labels[j] === -1) { labels[j] = c; stack.push(j); } });
        }
        c++;
      }
      return { labels, core, neighbours, k: c };
    }

    function update() {
      const r = dbscan();
      const coreCount = r.core.filter(Boolean).length;
      const noise = r.labels.filter((l) => l === -1).length;
      out.update({ clusters: r.k, "core points": coreCount, border: X.length - coreCount - noise, noise });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.25 });
        X.forEach((p, i) => {
          if (r.labels[i] === -1) {
            f.dot(p[0], p[1], { color: "#9ca3af", r: 3 });
          } else if (r.core[i]) {
            f.dot(p[0], p[1], { color: PAL[r.labels[i] % PAL.length], r: 4.6, ring: "#fff" });
          } else {
            f.circle(p[0], p[1], 4, { stroke: PAL[r.labels[i] % PAL.length], width: 1.6 });
          }
        });
        f.text(f.pad.l + 8, f.pad.t + 12, `eps = ${eps.toFixed(2)} · minPts = ${minPts}`, { color: t.ink, font: "11px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 27, "filled = core · hollow = border · grey = noise", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "eps (radius)", min: 0.08, max: 1.4, step: 0.02, value: eps, format: (v) => v.toFixed(2), onInput: (v) => { eps = v; update(); } });
    V.slider(panel, { label: "minPts", min: 2, max: 14, step: 1, value: minPts, format: (v) => String(v), onInput: (v) => { minPts = v; update(); } });
    update();
  };
})();
