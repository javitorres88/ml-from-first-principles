/* Module 03 labs — least squares by hand, the loss surface, polynomial features. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  function makeData(seed, n, w, b, noise) {
    const rng = V.mulberry32(seed);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = -2 + (4 * (i + 0.5)) / n + V.gauss(rng) * 0.05;
      pts.push([x, w * x + b + V.gauss(rng) * noise]);
    }
    return pts;
  }

  /* ------------------------------------------------------------------ lab 1
     Squared error as the area of squares you can see.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-fit-line"] = function (root) {
    const fig = V.makeIn(root, { xmin: -2.3, xmax: 2.3, ymin: -2.5, ymax: 3.5, height: 360 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["slope w", "—"], ["intercept b", "—"], ["SSE", "—"], ["MSE", "—"]]);
    const data = makeData(3, 24, 0.9, 0.6, 0.55);
    const { w: wOpt, b: bOpt } = V.fitLine(data);

    let w = -0.6, b = -0.8;

    function sse(w, b) {
      return data.reduce((s, [x, y]) => s + (y - (w * x + b)) ** 2, 0);
    }

    function update() {
      out.update({ "slope w": w.toFixed(2), "intercept b": b.toFixed(2), SSE: sse(w, b), MSE: sse(w, b) / data.length });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.5 });
      // residual squares
      data.forEach(([x, y]) => {
        const yhat = w * x + b;
        const side = Math.abs(y - yhat);
        const px = f.sx(x), py = f.sy(y), ph = f.sy(yhat);
        ctx.fillStyle = y > yhat ? "rgba(220,38,38,0.13)" : "rgba(15,157,88,0.13)";
        ctx.fillRect(px, Math.min(py, ph), (side / (f.xr[1] - f.xr[0])) * f.iw, Math.abs(ph - py));
        ctx.strokeStyle = y > yhat ? "rgba(220,38,38,0.45)" : "rgba(15,157,88,0.45)";
        ctx.lineWidth = 1;
        ctx.strokeRect(px, Math.min(py, ph), (side / (f.xr[1] - f.xr[0])) * f.iw, Math.abs(ph - py));
      });
      f.fn((x) => wOpt * x + bOpt, { color: t.line2, width: 1.6, dash: [6, 5] });
      f.fn((x) => w * x + b, { color: t.accent, width: 2.6 });
      f.scatter(data, { color: t.ink2, r: 4 });
      f.text(f.pad.l + 8, f.pad.t + 12, "square area = squared residual", { color: t.ink3, font: "11px ui-sans-serif" });
      f.text(f.pad.l + 8, f.pad.t + 27, "dashed = least-squares optimum", { color: t.ink3, font: "11px ui-sans-serif" });
    });

    V.slider(panel, { label: "slope w", min: -1.5, max: 2, step: 0.02, value: w, format: (v) => v.toFixed(2), onInput: (v) => { w = v; update(); } });
    V.slider(panel, { label: "intercept b", min: -2, max: 2.5, step: 0.02, value: b, format: (v) => v.toFixed(2), onInput: (v) => { b = v; update(); } });
    V.buttons(panel, [
      { label: "Jump to optimum", primary: true, onClick: () => { w = wOpt; b = bOpt; update(); } },
      { label: "Random line", onClick: () => { w = -1 + Math.random() * 2.5; b = -1.5 + Math.random() * 3; update(); } },
    ]);
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     The loss surface and gradient descent.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-loss-surface"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1.6, xmax: 2.0, ymin: -2.0, ymax: 2.2, height: 380 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["step", "—"], ["J(w,b)", "—"], ["‖∇J‖", "—"], ["status", "—"]]);
    const data = makeData(3, 24, 0.9, 0.6, 0.55);

    const N = data.length;
    const sxx = data.reduce((s, [x]) => s + x * x, 0) / N;
    const sx = data.reduce((s, [x]) => s + x, 0) / N;
    const sxy = data.reduce((s, [x, y]) => s + x * y, 0) / N;
    const sy = data.reduce((s, [, y]) => s + y, 0) / N;
    // J(w,b) = mean (y - wx - b)^2 ; gradient in closed form
    function J(w, b) {
      return data.reduce((s, [x, y]) => s + (y - w * x - b) ** 2, 0) / N;
    }
    function grad(w, b) {
      let gw = 0, gb = 0;
      data.forEach(([x, y]) => {
        const r = y - (w * x + b);
        gw -= (2 * r * x) / N;
        gb -= (2 * r) / N;
      });
      return [gw, gb];
    }

    let path = [], lr = 0.15, running = false, timer = null, w0 = -1.2, b0 = -1.4;

    function reset() {
      path = [[w0, b0]];
      out.update({ step: 0, "J(w,b)": J(w0, b0), "‖∇J‖": V.Math ? 0 : M.norm(grad(w0, b0)), status: "ready" });
      fig.render();
    }

    function step() {
      const [w, b] = path[path.length - 1];
      const g = grad(w, b);
      const nw = w - lr * g[0], nb = b - lr * g[1];
      if (!Number.isFinite(nw) || Math.abs(nw) > 40) {
        out.update({ status: "diverged — learning rate too large" });
        stop();
        return;
      }
      path.push([nw, nb]);
      const gn = M.norm(g);
      out.update({
        step: path.length - 1,
        "J(w,b)": J(nw, nb),
        "‖∇J‖": gn,
        status: gn < 1e-3 ? "converged" : "descending",
      });
      if (gn < 1e-3 || path.length > 400) stop();
      fig.render();
    }

    function stop() { running = false; clearInterval(timer); }

    fig.draw((f, ctx, t) => {
      const levels = 14;
      const vals = [];
      let min = Infinity, max = -Infinity;
      for (let j = 0; j < 60; j++) {
        const row = [];
        for (let i = 0; i < 60; i++) {
          const w = V.lerp(f.xr[0], f.xr[1], i / 59);
          const b = V.lerp(f.yr[0], f.yr[1], j / 59);
          const v = Math.log10(J(w, b) + 1e-3);
          row.push(v);
          min = Math.min(min, v); max = Math.max(max, v);
        }
        vals.push(row);
      }
      f.heat(vals, {
        min, max,
        color: (v) => {
          const c = V.clamp((v - min) / (max - min || 1), 0, 1);
          const stops = [[239, 246, 255], [191, 219, 254], [147, 197, 253], [96, 165, 250], [59, 130, 246], [37, 99, 235], [30, 58, 138]];
          const p = c * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), fr = p - i;
          const a = stops[i], b2 = stops[i + 1];
          return `rgb(${Math.round(V.lerp(a[0], b2[0], fr))},${Math.round(V.lerp(a[1], b2[1], fr))},${Math.round(V.lerp(a[2], b2[2], fr))})`;
        },
      });
      f.axes({ xstep: 0.5, ystep: 0.5, xlabel: "weight w", ylabel: "bias b", gridAlpha: 0.25 });
      if (path.length > 1) f.line(path, { color: "#f59e0b", width: 2.4 });
      path.forEach((p, i) => f.dot(p[0], p[1], { color: i === path.length - 1 ? "#dc2626" : "#f59e0b", r: i === path.length - 1 ? 5 : 2.6 }));
      const wOpt = (sxy - sx * sy) / (sxx - sx * sx);
      const bOpt = sy - wOpt * sx;
      f.circle(wOpt, bOpt, 7, { stroke: "#fff", width: 2, fill: "rgba(255,255,255,0.25)" });
      f.text(f.pad.l + 8, f.pad.t + 12, "log J(w,b) contours", { color: t.ink, font: "11px ui-sans-serif" });
      f.text(f.pad.l + 8, f.pad.t + 27, "white ring = exact optimum", { color: t.ink, font: "11px ui-sans-serif" });
    });

    V.slider(panel, { label: "learning rate α", min: 0.02, max: 0.6, step: 0.01, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; reset(); } });
    V.buttons(panel, [
      { label: "Run", primary: true, onClick: () => { if (running) return; running = true; timer = setInterval(step, 90); } },
      { label: "Step", onClick: step },
      { label: "Reset", onClick: () => { stop(); reset(); } },
    ]);
    reset();
  };

  /* ------------------------------------------------------------------ lab 3
     Polynomial features: fit, overfit, and extrapolate.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-polynomial"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1.6, xmax: 1.6, ymin: -3, ymax: 3, height: 340 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["degree", "—"], ["parameters", "—"], ["train RMSE", "—"], ["extrapolation RMSE", "—"], ["max |w|", "—"]]);
    const rng = V.mulberry32(21);

    const train = [];
    for (let i = 0; i < 16; i++) {
      const x = -1 + (2 * (i + 0.5)) / 16;
      train.push([x, 0.6 * x * x * x - 0.4 * x + V.gauss(rng) * 0.18]);
    }
    const extra = [];
    for (let i = 0; i < 40; i++) {
      const x = 1.02 + (0.56 * i) / 39;
      extra.push([x, 0.6 * x * x * x - 0.4 * x]);
    }

    let deg = 3, lam = 0;

    function fit() {
      const coeffs = lam > 0
        ? M.ridge(train.map((p) => Array.from({ length: deg }, (_, k) => Math.pow(p[0], k + 1))), train.map((p) => p[1]), lam)
        : M.polyfit(train.map((p) => p[0]), train.map((p) => p[1]), deg);
      const rmse = (pts) => Math.sqrt(pts.reduce((s, [x, y]) => s + (y - M.polyval(coeffs, x)) ** 2, 0) / pts.length);
      out.update({
        degree: deg,
        parameters: deg + 1,
        "train RMSE": rmse(train),
        "extrapolation RMSE": rmse(extra),
        "max |w|": Math.max(...coeffs.map(Math.abs)),
      });
      fig.draw((f, ctx, t) => {
        f.shade(1, f.yr[0], f.xr[1], f.yr[1], { color: "#f59e0b", alpha: 0.09 });
        f.axes({ xstep: 0.5, ystep: 1, xlabel: "x", ylabel: "y" });
        f.fn((x) => 0.6 * x ** 3 - 0.4 * x, { color: t.line2, width: 1.8, dash: [6, 5] });
        f.fn((x) => M.polyval(coeffs, x), { color: t.accent, width: 2.4 });
        f.scatter(train, { color: t.ink2, r: 4 });
        f.text(f.sx(1.08), f.pad.t + 14, "extrapolation", { color: "#b45309", font: "600 11px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 14, "— fit   ┄ truth   ● training data", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "polynomial degree", min: 1, max: 9, step: 1, value: deg, format: (v) => String(v), onInput: (v) => { deg = v; fit(); } });
    V.slider(panel, { label: "ridge λ", min: 0, max: 0.5, step: 0.005, value: lam, format: (v) => v.toFixed(3), onInput: (v) => { lam = v; fit(); } });
    V.buttons(panel, [{ label: "New sample", onClick: () => { /* regenerate noise */ for (const p of train) p[1] = 0.6 * p[0] ** 3 - 0.4 * p[0] + V.gauss(rng) * 0.18; fit(); } }]);
    fit();
  };
})();
