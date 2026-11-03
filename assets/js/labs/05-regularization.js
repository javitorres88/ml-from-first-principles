/* Module 05 labs — ridge/lasso geometry, bias-variance decomposition, cross-validation. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* coordinate-descent lasso for two coefficients (soft-thresholding) */
  function lasso2(X, y, lambda, iters) {
    let w = [0, 0];
    const n = X.length;
    const colNorm = [0, 1].map((j) => X.reduce((s, r) => s + r[j] * r[j], 0) / n);
    for (let it = 0; it < (iters || 60); it++) {
      for (let j = 0; j < 2; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) {
          const pred = X[i][0] * w[0] + X[i][1] * w[1];
          rho += X[i][j] * (y[i] - pred + X[i][j] * w[j]);
        }
        rho /= n;
        const t = lambda / 2;
        w[j] = rho > t ? (rho - t) / colNorm[j] : rho < -t ? (rho + t) / colNorm[j] : 0;
      }
    }
    return w;
  }

  /* ------------------------------------------------------------------ lab 1
     Ridge vs lasso: constraint geometry and coefficient paths.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-ridge-lasso"] = function (root) {
    const fig = V.makeIn(root, { xmin: -2.2, xmax: 2.6, ymin: -2.2, ymax: 2.6, height: 400 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["penalty", "—"], ["λ", "—"], ["w₁", "—"], ["w₂", "—"], ["‖w‖₁", "—"]]);
    const rng = V.mulberry32(9);

    // two correlated features; true weights (1.6, 0.8)
    const X = [], y = [];
    for (let i = 0; i < 60; i++) {
      const a = V.gauss(rng), b = 0.8 * a + 0.6 * V.gauss(rng);
      X.push([a, b]);
      y.push(1.6 * a + 0.8 * b + V.gauss(rng) * 0.5);
    }
    const n = X.length;
    const sse = (w) => X.reduce((s, r, i) => s + (y[i] - w[0] * r[0] - w[1] * r[1]) ** 2, 0) / n;

    let mode = "ridge", lambda = 0.15;

    function solution(l) {
      if (l <= 1e-9) return M.ridge(X, y, 1e-9, false);
      return mode === "ridge" ? M.ridge(X, y, l, false) : lasso2(X, y, l);
    }

    function update() {
      const w = solution(lambda);
      out.update({
        penalty: mode === "ridge" ? "ℓ2 (circle)" : "ℓ1 (diamond)",
        "λ": lambda.toFixed(3),
        "w₁": w[0].toFixed(3),
        "w₂": w[1].toFixed(3),
        "‖w‖₁": (Math.abs(w[0]) + Math.abs(w[1])).toFixed(3),
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      // heat of the unpenalised loss
      const vals = [];
      let min = Infinity, max = -Infinity;
      for (let j = 0; j < 64; j++) {
        const row = [];
        for (let i = 0; i < 64; i++) {
          const w1 = V.lerp(f.xr[0], f.xr[1], i / 63);
          const w2 = V.lerp(f.yr[0], f.yr[1], j / 63);
          const v = Math.log10(sse([w1, w2]) + 1e-2);
          row.push(v); min = Math.min(min, v); max = Math.max(max, v);
        }
        vals.push(row);
      }
      f.heat(vals, {
        min,
        max: min + 0.4 * (max - min),
        color: (v) => {
          const c = V.clamp((v - min) / (0.4 * (max - min) || 1), 0, 1);
          const stops = [[248, 250, 255], [199, 210, 254], [129, 140, 248], [79, 70, 229], [49, 46, 129]];
          const p = c * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), fr = p - i;
          const a = stops[i], b2 = stops[i + 1];
          return `rgb(${Math.round(V.lerp(a[0], b2[0], fr))},${Math.round(V.lerp(a[1], b2[1], fr))},${Math.round(V.lerp(a[2], b2[2], fr))})`;
        },
      });
      f.axes({ xstep: 1, ystep: 1, xlabel: "w₁", ylabel: "w₂", gridAlpha: 0.3 });

      // constraint region at the optimum: the solution lies on its boundary
      const wStar = solution(1e-9);
      const wNow = solution(lambda);
      const tRad = mode === "ridge"
        ? Math.max(1e-3, Math.hypot(wNow[0], wNow[1]))
        : Math.max(1e-3, Math.abs(wNow[0]) + Math.abs(wNow[1]));
      const pts = [];
      for (let a = 0; a < 361; a++) {
        const th = (a * Math.PI) / 180;
        if (mode === "ridge") pts.push([tRad * Math.cos(th), tRad * Math.sin(th)]);
        else {
          const c = Math.cos(th), s = Math.sin(th);
          const d = Math.abs(c) + Math.abs(s);
          pts.push([(tRad * c) / d, (tRad * s) / d]);
        }
      }
      f.line(pts, { color: t.ink, width: 1.8, dash: [5, 4] });

      // coefficient path as lambda grows
      const path = [];
      for (let i = 0; i <= 60; i++) {
        const l = 1e-3 + (i / 60) * 1.2;
        path.push(solution(l));
      }
      f.line(path, { color: "#f59e0b", width: 2.2, alpha: 0.95 });
      f.dot(wStar[0], wStar[1], { color: "#111827", r: 4.5, ring: "#fff" });

      const w = wNow;
      f.dot(w[0], w[1], { color: "#dc2626", r: 6.5, ring: "#fff" });
      // legend on a light panel so it stays readable over any background
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.fillRect(f.pad.l + 4, f.pad.t + 4, 300, 30);
      f.text(f.pad.l + 8, f.pad.t + 14, "white = low loss, indigo = high loss", { color: "#111827", font: "11px ui-sans-serif" });
      f.text(f.pad.l + 8, f.pad.t + 28, "dashed = constraint · amber = path · black = OLS", { color: "#111827", font: "11px ui-sans-serif" });
    });

    V.segmented(panel, {
      label: "Penalty",
      options: [{ label: "Ridge ℓ2", value: "ridge" }, { label: "Lasso ℓ1", value: "lasso" }],
      value: "ridge",
      onChange: (v) => { mode = v; update(); },
    });
    V.slider(panel, { label: "λ (penalty strength)", min: 0, max: 1.2, step: 0.01, value: lambda, format: (v) => v.toFixed(2), onInput: (v) => { lambda = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Bias-variance decomposition by Monte Carlo.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-bias-variance"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: -2, ymax: 2, height: 250 });
    const fig2 = new V.Fig(c2, { xmin: 0.5, xmax: 10.5, ymin: 0, ymax: 1, height: 210 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["bias²", "—"], ["variance", "—"], ["noise", "—"], ["total", "—"]]);
    const truth = (x) => Math.sin(2 * Math.PI * x) * 0.9;

    let degree = 3, noise = 0.3, nTrain = 20, reps = 60;

    function experiment() {
      const grid = M && null; // placeholder to keep linters quiet
      const xs = Array.from({ length: 40 }, (_, i) => 0.02 + (0.96 * i) / 39);
      const preds = []; // reps x xs
      const rng = V.mulberry32(1234);
      for (let r = 0; r < reps; r++) {
        const X = [], y = [];
        for (let i = 0; i < nTrain; i++) {
          const x = rng();
          X.push(x);
          y.push(truth(x) + V.gauss(rng) * noise);
        }
        const coeffs = M.polyfit(X, y, degree);
        preds.push(xs.map((x) => M.polyval(coeffs, x)));
      }
      const mean = xs.map((_, i) => preds.reduce((s, p) => s + p[i], 0) / reps);
      const bias2 = xs.map((x, i) => (mean[i] - truth(x)) ** 2);
      const variance = xs.map((_, i) => preds.reduce((s, p) => s + (p[i] - mean[i]) ** 2, 0) / reps);
      const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      return { xs, preds, mean, bias2: avg(bias2), variance: avg(variance), noise: noise * noise };
    }

    function curve(deg) {
      const xs = Array.from({ length: 30 }, (_, i) => 0.02 + (0.96 * i) / 29);
      const rng = V.mulberry32(77);
      const preds = [];
      for (let r = 0; r < 40; r++) {
        const X = [], y = [];
        for (let i = 0; i < nTrain; i++) { const x = rng(); X.push(x); y.push(truth(x) + V.gauss(rng) * noise); }
        const c = M.polyfit(X, y, deg);
        preds.push(xs.map((x) => M.polyval(c, x)));
      }
      const mean = xs.map((_, i) => preds.reduce((s, p) => s + p[i], 0) / preds.length);
      const bias2 = xs.reduce((s, x, i) => s + (mean[i] - truth(x)) ** 2, 0) / xs.length;
      const variance = xs.reduce((s, _, i) => s + preds.reduce((ss, p) => ss + (p[i] - mean[i]) ** 2, 0) / preds.length, 0) / xs.length;
      return { bias2, variance };
    }

    let curves = null;

    function update() {
      const e = experiment();
      out.update({ "bias²": e.bias2, variance: e.variance, noise: e.noise, total: e.bias2 + e.variance + e.noise });
      if (!curves) curves = Array.from({ length: 10 }, (_, d) => curve(d + 1));

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 0.2, ystep: 0.5, xlabel: "x", ylabel: "y" });
        e.preds.forEach((p) => f.line(e.xs.map((x, i) => [x, p[i]]), { color: "#4f46e5", width: 0.8, alpha: 0.18 }));
        f.fn(truth, { color: t.ink, width: 2.4 });
        f.line(e.xs.map((x, i) => [x, e.mean[i]]), { color: "#dc2626", width: 2, dash: [5, 4] });
        f.text(f.pad.l + 8, f.pad.t + 12, "— truth   ┄ mean prediction   thin = 60 fits", { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        const maxY = Math.max(...curves.map((c) => c.bias2 + c.variance)) * 1.15 + e.noise;
        f.yr = [0, maxY];
        f.axes({ xstep: 1, ystep: maxY / 4, xlabel: "polynomial degree", ylabel: "error" });
        f.line(curves.map((c, i) => [i + 1, c.bias2]), { color: "#0ea5e9", width: 2.2 });
        f.line(curves.map((c, i) => [i + 1, c.variance]), { color: "#f59e0b", width: 2.2 });
        f.line(curves.map((c, i) => [i + 1, c.bias2 + c.variance + e.noise]), { color: "#dc2626", width: 2.4 });
        f.seg(f.xr[0], e.noise, f.xr[1], e.noise, { color: t.ink3, dash: [4, 4], width: 1.2 });
        f.dot(degree, e.bias2, { color: "#0ea5e9", r: 4.5 });
        f.dot(degree, e.variance, { color: "#f59e0b", r: 4.5 });
        f.dot(degree, e.bias2 + e.variance + e.noise, { color: "#dc2626", r: 5 });
        f.text(f.pad.l + 8, f.pad.t + 12, "— bias²   — variance   — total   ┄ noise", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "model complexity (degree)", min: 1, max: 10, step: 1, value: degree, format: (v) => String(v), onInput: (v) => { degree = v; update(); } });
    V.slider(panel, { label: "noise σ", min: 0.05, max: 0.8, step: 0.05, value: noise, format: (v) => v.toFixed(2), onInput: (v) => { noise = v; curves = null; update(); } });
    V.slider(panel, { label: "training samples N", min: 8, max: 60, step: 2, value: nTrain, format: (v) => String(v), onInput: (v) => { nTrain = v; curves = null; update(); } });
    V.buttons(panel, [{ label: "Resample", onClick: () => { curves = null; update(); } }]);
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     Cross-validation: folds, variance of the estimate, and the 1-SE rule.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-cv"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: -2.5, ymax: 2.5, height: 200, pad: { l: 46, r: 12, t: 16, b: 26 } });
    const fig2 = new V.Fig(c2, { xmin: 0.5, xmax: 9.5, ymin: 0, ymax: 1, height: 240 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["folds", "—"], ["best degree", "—"], ["CV MSE", "—"], ["± sd", "—"]]);
    const rng = V.mulberry32(31);

    const truth = (x) => 0.9 * Math.sin(2 * Math.PI * x);
    const X = [], y = [];
    for (let i = 0; i < 48; i++) { const x = rng(); X.push(x); y.push(truth(x) + V.gauss(rng) * 0.3); }

    let k = 5, shuffleSeed = 1, showSd = true;

    function folds() {
      const idx = X.map((_, i) => i);
      const r = V.mulberry32(shuffleSeed);
      for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      const out2 = Array.from({ length: k }, () => []);
      idx.forEach((v, i) => out2[i % k].push(v));
      return out2;
    }

    function evaluate(fs) {
      const results = [];
      for (let d = 1; d <= 9; d++) {
        const mses = fs.map((test) => {
          const tr = fs.filter((f) => f !== test).flat();
          const coeffs = M.polyfit(tr.map((i) => X[i]), tr.map((i) => y[i]), d);
          const mse = test.reduce((s, i) => s + (y[i] - M.polyval(coeffs, X[i])) ** 2, 0) / test.length;
          return mse;
        });
        const mean = mses.reduce((a, b) => a + b, 0) / mses.length;
        const sd = Math.sqrt(mses.reduce((s, v) => s + (v - mean) ** 2, 0) / mses.length);
        results.push({ d, mean, sd, mses });
      }
      return results;
    }

    function update() {
      const fs = folds();
      const res = evaluate(fs);
      let best = res[0];
      res.forEach((r) => { if (r.mean < best.mean) best = r; });
      out.update({ folds: k, "best degree": best.d, "CV MSE": best.mean, "± sd": best.sd });

      fig.draw((f, ctx, t) => {
        const n = X.length;
        const colors = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#64748b", "#a3e635"];
        const cellW = f.iw / n;
        fs.forEach((fold, fi) => {
          fold.forEach((i) => {
            ctx.fillStyle = colors[fi % colors.length];
            ctx.globalAlpha = 0.85;
            ctx.fillRect(f.pad.l + i * cellW, f.pad.t, Math.max(cellW - 0.5, 1), 16);
          });
        });
        ctx.globalAlpha = 1;
        f.text(f.pad.l, f.pad.t + 30, `fold assignment of ${n} samples (colour = fold)`, { color: t.ink3, font: "11px ui-sans-serif" });
        // fold sizes
        fs.forEach((fold, fi) => {
          const w = (fold.length / n) * f.iw;
          ctx.fillStyle = colors[fi % colors.length];
          ctx.globalAlpha = 0.35;
          ctx.fillRect(f.pad.l + (fi / fs.length) * f.iw, f.pad.t + 42, w - 2, 10);
        });
        ctx.globalAlpha = 1;
        f.text(f.pad.l, f.pad.t + 68, `fold sizes: ${fs.map((f2) => f2.length).join(" / ")}`, { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        const maxY = Math.max(...res.map((r) => r.mean + r.sd)) * 1.1;
        f.yr = [0, maxY];
        f.axes({ xstep: 1, ystep: maxY / 4, xlabel: "polynomial degree", ylabel: "CV MSE" });
        // error bars
        res.forEach((r) => {
          f.seg(r.d, Math.max(0, r.mean - r.sd), r.d, r.mean + r.sd, { color: t.ink3, width: 1.4 });
          f.seg(r.d - 0.15, r.mean - r.sd, r.d + 0.15, r.mean - r.sd, { color: t.ink3, width: 1.4 });
          f.seg(r.d - 0.15, r.mean + r.sd, r.d + 0.15, r.mean + r.sd, { color: t.ink3, width: 1.4 });
        });
        f.line(res.map((r) => [r.d, r.mean]), { color: t.accent, width: 2.2 });
        res.forEach((r) => f.dot(r.d, r.mean, { color: r === best ? "#dc2626" : t.accent, r: r === best ? 6 : 3.5 }));
        // one-standard-error band around the best model
        const band = best.mean + (showSd ? best.sd : 0);
        f.seg(f.xr[0], band, f.xr[1], band, { color: "#dc2626", dash: [5, 4], width: 1.4 });
        f.text(f.pad.l + 8, f.pad.t + 12, "red dot = CV-optimal degree; dashed line = 1-SE band", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "folds k", min: 2, max: 10, step: 1, value: k, format: (v) => String(v), onInput: (v) => { k = v; update(); } });
    V.toggle(panel, { label: "show ± 1 sd band", value: true, onChange: (v) => { showSd = v; update(); } });
    V.buttons(panel, [{ label: "Reshuffle folds", onClick: () => { shuffleSeed++; update(); } }]);
    update();
  };
})();
