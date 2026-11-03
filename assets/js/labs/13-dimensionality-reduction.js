/* Module 13 labs — PCA, SVD low-rank approximation, linear autoencoder. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     PCA: the direction of maximum variance.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-pca"] = function (root) {
    const fig = V.makeIn(root, { xmin: -4, xmax: 4, ymin: -3.4, ymax: 3.4, height: 380 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["angle θ", "—"], ["variance along θ", "—"], ["PC1 variance", "—"], ["explained (PC1)", "—"]]);

    const rng = V.mulberry32(3);
    const X = [];
    for (let i = 0; i < 160; i++) {
      const t = V.gauss(rng), s = V.gauss(rng);
      X.push([1.6 * t + 0.25 * s, 0.9 * t - 0.2 * s]);   // correlated cloud
    }
    const pca = M.pca2(X);
    let theta = 20;

    function varianceAlong(deg) {
      const th = (deg * Math.PI) / 180;
      const u = [Math.cos(th), Math.sin(th)];
      const mean = X.reduce((a, p) => [a[0] + p[0] / X.length, a[1] + p[1] / X.length], [0, 0]);
      let v = 0, proj = [];
      X.forEach((p) => {
        const d = [p[0] - mean[0], p[1] - mean[1]];
        const t = d[0] * u[0] + d[1] * u[1];
        proj.push(t);
        v += t * t;
      });
      return v / (X.length - 1);
    }

    function update() {
      const vTheta = varianceAlong(theta);
      out.update({
        "angle θ": theta.toFixed(0) + "°",
        "variance along θ": vTheta,
        "PC1 variance": pca.eigenvalues[0],
        "explained (PC1)": (pca.explained[0] * 100).toFixed(1) + "%",
      });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.3, xlabel: "x₁", ylabel: "x₂" });
        X.forEach((p) => f.dot(p[0], p[1], { color: "rgba(79,70,229,0.55)", r: 3 }));
        const th = (theta * Math.PI) / 180;
        const L = 3.2;
        f.arrow(0, 0, L * Math.cos(th), L * Math.sin(th), { color: "#dc2626", width: 2.6 });
        f.arrow(0, 0, L * Math.cos(th + Math.PI), L * Math.sin(th + Math.PI), { color: "#dc2626", width: 2.6, head: 0.1 });
        // principal axes
        const [v1, v2] = pca.axes;
        f.arrow(0, 0, 3.0 * v1[0], 3.0 * v1[1], { color: "#0f9d58", width: 2, dash: [5, 4] });
        f.arrow(0, 0, 2.2 * v2[0], 2.2 * v2[1], { color: "#0f9d58", width: 1.4, dash: [3, 3] });
        // projections onto the candidate axis
        X.slice(0, 40).forEach((p) => {
          const tproj = p[0] * Math.cos(th) + p[1] * Math.sin(th);
          const q = [tproj * Math.cos(th), tproj * Math.sin(th)];
          f.seg(p[0], p[1], q[0], q[1], { color: "#dc2626", width: 0.6, alpha: 0.35 });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, "red = your axis (with projections) · green dashed = true principal axes", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "axis angle θ", min: 0, max: 180, step: 1, value: theta, format: (v) => v + "°", onInput: (v) => { theta = v; update(); } });
    V.buttons(panel, [{ label: "Snap to PC1", primary: true, onClick: () => { theta = (Math.atan2(pca.axes[0][1], pca.axes[0][0]) * 180 / Math.PI + 180) % 180; update(); } }]);
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Low-rank approximation: singular values and compression.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-svd"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 240, pad: { l: 10, r: 10, t: 10, b: 10 } });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 40, ymin: 0, ymax: 1, height: 190 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["rank k", "—"], ["stored values", "—"], ["compression", "—"], ["energy kept", "—"]]);

    // synthetic 40x40 image: smooth gradients + shapes
    const N = 40;
    const A = [];
    for (let i = 0; i < N; i++) {
      const row = [];
      for (let j = 0; j < N; j++) {
        let v = 0.5 + 0.4 * Math.sin(i / 4) * Math.cos(j / 6);
        const dx = i - 26, dy = j - 14;
        if (dx * dx + dy * dy < 40) v += 0.45;
        if (Math.abs(i - 10) < 3 && Math.abs(j - 28) < 3) v += 0.5;
        row.push(Math.max(0, Math.min(1, v)));
      }
      A.push(row);
    }

    // power-iteration SVD for the top components
    function topSVD(k) {
      let R = A.map((r) => r.slice());
      const comps = [];
      for (let c = 0; c < k; c++) {
        let u = new Array(N).fill(0).map((_, i) => Math.sin(i + c));
        let v, sigma;
        for (let it = 0; it < 60; it++) {
          v = new Array(N).fill(0);
          for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) v[j] += R[i][j] * u[i];
          const nv = Math.hypot(...v) || 1;
          v = v.map((x) => x / nv);
          u = new Array(N).fill(0);
          for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) u[i] += R[i][j] * v[j];
          sigma = Math.hypot(...u);
          u = u.map((x) => x / (sigma || 1));
        }
        comps.push({ u, v, sigma });
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) R[i][j] -= sigma * u[i] * v[j];
      }
      return comps;
    }

    const all = topSVD(40);
    const total = all.reduce((s, c) => s + c.sigma * c.sigma, 0);
    let k = 4;

    function update() {
      const comps = all.slice(0, k);
      const rec = Array.from({ length: N }, () => new Array(N).fill(0));
      comps.forEach((c) => {
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) rec[i][j] += c.sigma * c.u[i] * c.v[j];
      });
      const energy = comps.reduce((s, c) => s + c.sigma * c.sigma, 0) / (total || 1);
      const stored = k * (N + N + 1);
      out.update({ "rank k": k, "stored values": stored.toLocaleString(), compression: `${(N * N / stored).toFixed(1)}×`, "energy kept": (energy * 100).toFixed(1) + "%" });

      fig.draw((f, ctx, t) => {
        const cell = Math.min(f.iw / N, f.ih / N);
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            const v = V.clamp(rec[i][j], 0, 1);
            ctx.fillStyle = `rgb(${Math.round(255 - 160 * v)},${Math.round(250 - 130 * v)},${Math.round(255 - 40 * v)})`;
            ctx.fillRect(f.pad.l + j * cell, f.pad.t + i * cell, Math.ceil(cell), Math.ceil(cell));
          }
        }
        f.text(f.pad.l + 4, f.pad.t + 12, `rank-${k} reconstruction`, { color: "#111827", font: "600 11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        const maxS = all[0].sigma || 1;
        f.axes({ xstep: 5, ystep: 0.25, xlabel: "singular value index", ylabel: "σᵢ / σ₁" });
        f.line(all.map((c, i) => [i + 1, c.sigma / maxS]), { color: t.accent, width: 2.2 });
        f.seg(k, 0, k, 1, { color: "#dc2626", dash: [4, 3], width: 1.4 });
        f.text(f.pad.l + 8, f.pad.t + 12, "the red line is your truncation rank — the elbow is where to stop", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "truncation rank k", min: 1, max: 40, step: 1, value: k, format: (v) => String(v), onInput: (v) => { k = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     A linear autoencoder learns the PCA subspace.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-autoencoder"] = function (root) {
    const fig = V.makeIn(root, { xmin: -4, xmax: 4, ymin: -3.4, ymax: 3.4, height: 360 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["epoch", "—"], ["reconstruction MSE", "—"], ["angle vs. PC1", "—"], ["‖w‖", "—"]]);

    const rng = V.mulberry32(8);
    const X = [];
    for (let i = 0; i < 120; i++) {
      const t = V.gauss(rng), s = V.gauss(rng) * 0.25;
      X.push([1.7 * t + s, 1.0 * t - 0.3 * s]);
    }
    const pca = M.pca2(X);
    const pc1Angle = (Math.atan2(pca.axes[0][1], pca.axes[0][0]) * 180) / Math.PI;

    // encoder: h = w·x ; decoder: x' = v·h
    let w = [0.9, 0.1], v = [0.9, 0.1], epoch = 0, timer = null;

    function mse() {
      let s = 0;
      X.forEach((x) => {
        const h = w[0] * x[0] + w[1] * x[1];
        s += (x[0] - v[0] * h) ** 2 + (x[1] - v[1] * h) ** 2;
      });
      return s / (X.length * 2);
    }

    function train(steps) {
      const lr = 0.02;
      for (let s = 0; s < steps; s++) {
        let gw = [0, 0], gv = [0, 0];
        X.forEach((x) => {
          const h = w[0] * x[0] + w[1] * x[1];
          const r0 = v[0] * h - x[0], r1 = v[1] * h - x[1];
          const dpre = r0 * v[0] + r1 * v[1];      // dMSE/dh
          gw[0] += dpre * x[0];
          gw[1] += dpre * x[1];
          gv[0] += r0 * h;
          gv[1] += r1 * h;
        });
        w = [w[0] - lr * gw[0] / X.length, w[1] - lr * gw[1] / X.length];
        v = [v[0] - lr * gv[0] / X.length, v[1] - lr * gv[1] / X.length];
        epoch++;
      }
      update();
    }

    function angleOf(vec) {
      return (Math.atan2(vec[1], vec[0]) * 180) / Math.PI;
    }

    function update() {
      const a = angleOf(v);
      const diff = Math.min(Math.abs(a - pc1Angle), 180 - Math.abs(a - pc1Angle));
      out.update({ epoch, "reconstruction MSE": mse(), "angle vs. PC1": diff.toFixed(1) + "°", "‖w‖": Math.hypot(w[0], w[1]) });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.3, xlabel: "x₁", ylabel: "x₂" });
        X.forEach((x) => {
          const h = w[0] * x[0] + w[1] * x[1];
          const q = [v[0] * h, v[1] * h];
          f.seg(x[0], x[1], q[0], q[1], { color: "#dc2626", width: 0.7, alpha: 0.4 });
          f.dot(x[0], x[1], { color: "rgba(79,70,229,0.5)", r: 3 });
          f.dot(q[0], q[1], { color: "#dc2626", r: 2.4 });
        });
        const n = Math.hypot(...v) || 1;
        f.arrow(0, 0, (v[0] / n) * 3.2, (v[1] / n) * 3.2, { color: "#dc2626", width: 2.6 });
        f.arrow(0, 0, 3.0 * pca.axes[0][0], 3.0 * pca.axes[0][1], { color: "#0f9d58", width: 2, dash: [5, 4] });
        f.text(f.pad.l + 8, f.pad.t + 12, "red = learned decoder direction · green dashed = PC1", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    function stop() { clearInterval(timer); timer = null; }

    V.buttons(panel, [
      { label: "Train", primary: true, onClick: () => { if (timer) return; timer = setInterval(() => train(20), 60); } },
      { label: "Train ×200", onClick: () => { stop(); train(200); } },
      { label: "Reset", onClick: () => { stop(); w = [0.9, 0.1]; v = [0.9, 0.1]; epoch = 0; update(); } },
    ]);
    update();
  };
})();
