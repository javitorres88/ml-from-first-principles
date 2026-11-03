/* Module 07 labs — backprop by hand, optimiser trajectories, initialisation. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     Backprop on a computation graph, with a numerical gradient check.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-gradcheck"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 250, pad: { l: 10, r: 10, t: 10, b: 10 } });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["f", "—"], ["∂f/∂x", "—"], ["∂f/∂y", "—"], ["∂f/∂z", "—"], ["max |analytic − numeric|", "—"]]);

    let x = -2, y = 5, z = -4;

    // f(x,y,z) = (x + y) * sigmoid(z)   -> exercises +, *, sigmoid
    function f(a, b, c) { return (a + b) * M.sigmoid(c); }

    function gradients() {
      const q = x + y;
      const s = M.sigmoid(z);
      const fwd = q * s;
      const df = 1;
      const dq = df * s;              // d f/d q
      const ds = df * q;              // d f/d s
      const dz = ds * s * (1 - s);    // sigmoid local gradient
      const dx = dq * 1, dy = dq * 1;
      return { q, s, fwd, dx, dy, dz };
    }

    function numeric() {
      const h = 1e-5;
      return {
        dx: (f(x + h, y, z) - f(x - h, y, z)) / (2 * h),
        dy: (f(x, y + h, z) - f(x, y - h, z)) / (2 * h),
        dz: (f(x, y, z + h) - f(x, y, z - h)) / (2 * h),
      };
    }

    function update() {
      const g = gradients(), nm = numeric();
      const err = Math.max(Math.abs(g.dx - nm.dx), Math.abs(g.dy - nm.dy), Math.abs(g.dz - nm.dz));
      out.update({ f: g.fwd.toFixed(5), "∂f/∂x": g.dx.toFixed(5), "∂f/∂y": g.dy.toFixed(5), "∂f/∂z": g.dz.toFixed(5), "max |analytic − numeric|": err.toExponential(2) });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      const g = gradients();
      const nodes = [
        { id: "x", x: 0.08, y: 0.75, label: `x = ${x}`, val: "", grad: g.dx },
        { id: "y", x: 0.08, y: 0.25, label: `y = ${y}`, val: "", grad: g.dy },
        { id: "q", x: 0.38, y: 0.5, label: "q = x + y", val: g.q.toFixed(3), grad: g.dx },
        { id: "z", x: 0.38, y: 0.1, label: `z = ${z}`, val: "", grad: g.dz },
        { id: "s", x: 0.66, y: 0.32, label: "s = σ(z)", val: g.s.toFixed(3), grad: g.dz },
        { id: "f", x: 0.9, y: 0.62, label: "f = q·s", val: g.fwd.toFixed(3), grad: 1 },
      ];
      const at = (n) => [f.sx(n.x), f.sy(n.y)];
      const edges = [["x", "q"], ["y", "q"], ["q", "f"], ["z", "s"], ["s", "f"]];
      ctx.strokeStyle = t.line2; ctx.lineWidth = 1.4;
      edges.forEach(([a, b]) => {
        const A = nodes.find((n) => n.id === a), B = nodes.find((n) => n.id === b);
        const [ax, ay] = at(A), [bx, by] = at(B);
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      });
      nodes.forEach((n) => {
        const [px, py] = at(n);
        ctx.fillStyle = "#fff"; ctx.strokeStyle = t.accent; ctx.lineWidth = 1.6;
        const w = 118, h = 38;
        ctx.beginPath(); ctx.roundRect(px - w / 2, py - h / 2, w, h, 8); ctx.fill(); ctx.stroke();
        f.text(px, py - 5, n.label, { align: "center", color: t.ink, font: "600 11px ui-sans-serif" });
        if (n.val) f.text(px, py + 9, n.val, { align: "center", color: t.ink3, font: "10px ui-sans-serif" });
        f.text(px, py + 30, `∂f/∂${n.id} = ${n.grad.toFixed(3)}`, { align: "center", color: "#dc2626", font: "10px ui-sans-serif" });
      });
      f.text(f.sx(0.02), f.sy(0.98), "forward: left → right · backward: right → left", { color: t.ink3, font: "10px ui-sans-serif" });
    });

    V.slider(panel, { label: "x", min: -5, max: 5, step: 0.5, value: x, format: (v) => String(v), onInput: (v) => { x = v; update(); } });
    V.slider(panel, { label: "y", min: -5, max: 5, step: 0.5, value: y, format: (v) => String(v), onInput: (v) => { y = v; update(); } });
    V.slider(panel, { label: "z", min: -6, max: 6, step: 0.5, value: z, format: (v) => String(v), onInput: (v) => { z = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Optimiser trajectories on an ill-conditioned bowl.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-optimizers"] = function (root) {
    const fig = V.makeIn(root, { xmin: -3.2, xmax: 3.2, ymin: -2.0, ymax: 2.0, height: 400 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["steps", "—"], ["f(x,y)", "—"], ["‖∇f‖", "—"], ["best", "—"]]);

    // ill-conditioned quadratic: f = 0.5*(a x^2 + b y^2)
    const a = 1.0, b = 6.0;
    const f = (p) => 0.5 * (a * p[0] * p[0] + b * p[1] * p[1]);
    const grad = (p) => [a * p[0], b * p[1]];

    let method = "sgd", lr = 0.12, momentum = 0.9, paths = {}, running = false, timer = null;

    function reset() {
      paths = { sgd: [[-2.8, 1.7]], momentum: [[-2.8, 1.7]], rmsprop: [[-2.8, 1.7]], adam: [[-2.8, 1.7]] };
      steps = 0;
      update();
    }
    let steps = 0;

    function optimise(which, p, state) {
      const g = grad(p);
      if (which === "sgd") return { p: [p[0] - lr * g[0], p[1] - lr * g[1]], s: state };
      if (which === "momentum") {
        const v = [momentum * state.v[0] + g[0], momentum * state.v[1] + g[1]];
        return { p: [p[0] - lr * v[0], p[1] - lr * v[1]], s: { v } };
      }
      if (which === "rmsprop") {
        const s = [0.9 * state.s[0] + 0.1 * g[0] ** 2, 0.9 * state.s[1] + 0.1 * g[1] ** 2];
        return { p: [p[0] - lr * g[0] / (Math.sqrt(s[0]) + 1e-8), p[1] - lr * g[1] / (Math.sqrt(s[1]) + 1e-8)], s: { s } };
      }
      // adam
      const m = [0.9 * state.m[0] + 0.1 * g[0], 0.9 * state.m[1] + 0.1 * g[1]];
      const v = [0.999 * state.v[0] + 0.001 * g[0] ** 2, 0.999 * state.v[1] + 0.001 * g[1] ** 2];
      const t = state.t + 1;
      const mh = [m[0] / (1 - 0.9 ** t), m[1] / (1 - 0.9 ** t)];
      const vh = [v[0] / (1 - 0.999 ** t), v[1] / (1 - 0.999 ** t)];
      return { p: [p[0] - lr * mh[0] / (Math.sqrt(vh[0]) + 1e-8), p[1] - lr * mh[1] / (Math.sqrt(vh[1]) + 1e-8)], s: { m, v, t } };
    }

    const states = {
      sgd: {}, momentum: { v: [0, 0] },
      rmsprop: { s: [0, 0] },
      adam: { m: [0, 0], v: [0, 0], t: 0 },
    };

    function step() {
      Object.keys(paths).forEach((k) => {
        const cur = paths[k][paths[k].length - 1];
        const r = optimise(k, cur, states[k]);
        states[k] = r.s;
        paths[k].push(r.p);
      });
      steps++;
      update();
      if (steps > 200) stop();
    }
    function stop() { running = false; clearInterval(timer); }

    function update() {
      const p = paths[method][paths[method].length - 1];
      out.update({ steps, "f(x,y)": f(p), "‖∇f‖": M.norm(grad(p)), best: f(p) < 0.01 ? "converged" : "—" });
      fig.render();
    }

    const COLORS = { sgd: "#dc2626", momentum: "#f59e0b", rmsprop: "#0ea5e9", adam: "#0f9d58" };

    fig.draw((f2, ctx, t) => {
      const vals = [];
      let min = Infinity, max = -Infinity;
      for (let j = 0; j < 60; j++) {
        const row = [];
        for (let i = 0; i < 60; i++) {
          const px = V.lerp(f2.xr[0], f2.xr[1], i / 59);
          const py = V.lerp(f2.yr[0], f2.yr[1], j / 59);
          const v = Math.pow(f([px, py]), 0.25);   // quartic-root scaling keeps the bowl readable
          row.push(v); min = Math.min(min, v); max = Math.max(max, v);
        }
        vals.push(row);
      }
      f2.heat(vals, {
        min, max,
        color: (v) => {
          const c = V.clamp((v - min) / (max - min || 1), 0, 1);
          const stops = [[255, 255, 255], [224, 231, 255], [165, 180, 252], [99, 102, 241], [67, 56, 202], [30, 27, 75]];
          const p = c * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), fr = p - i;
          const a2 = stops[i], b2 = stops[i + 1];
          return `rgb(${Math.round(V.lerp(a2[0], b2[0], fr))},${Math.round(V.lerp(a2[1], b2[1], fr))},${Math.round(V.lerp(a2[2], b2[2], fr))})`;
        },
      });
      f2.axes({ xstep: 1, ystep: 1, xlabel: "θ₁", ylabel: "θ₂", gridAlpha: 0.25 });
      Object.entries(paths).forEach(([k, p]) => {
        const active = k === method;
        f2.line(p, { color: COLORS[k], width: active ? 2.6 : 1.2, alpha: active ? 1 : 0.35 });
        const last = p[p.length - 1];
        f2.dot(last[0], last[1], { color: COLORS[k], r: active ? 5 : 3 });
      });
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fillRect(f2.pad.l + 4, f2.pad.t + 4, 250, 30);
      f2.text(f2.pad.l + 8, f2.pad.t + 12, "— SGD   — momentum   — RMSProp   — Adam", { color: "#111827", font: "11px ui-sans-serif" });
      f2.text(f2.pad.l + 8, f2.pad.t + 27, "condition number κ = 6 (elongated bowl)", { color: "#374151", font: "11px ui-sans-serif" });
    });

    V.segmented(panel, {
      label: "Highlight method",
      options: [{ label: "SGD", value: "sgd" }, { label: "momentum", value: "momentum" }, { label: "RMSProp", value: "rmsprop" }, { label: "Adam", value: "adam" }],
      value: "sgd",
      onChange: (v) => { method = v; update(); },
    });
    V.slider(panel, { label: "learning rate α", min: 0.005, max: 0.25, step: 0.005, value: lr, format: (v) => v.toFixed(3), onInput: (v) => { lr = v; reset(); } });
    V.slider(panel, { label: "momentum ρ", min: 0, max: 0.99, step: 0.01, value: momentum, format: (v) => v.toFixed(2), onInput: (v) => { momentum = v; reset(); } });
    V.buttons(panel, [
      { label: "Run", primary: true, onClick: () => { if (running) return; running = true; timer = setInterval(step, 70); } },
      { label: "Step", onClick: step },
      { label: "Reset", onClick: () => { stop(); reset(); } },
    ]);
    reset();
  };

  /* ------------------------------------------------------------------ lab 3
     Initialisation: does the signal survive the depth?
  ------------------------------------------------------------------- */
  window.MLLabs["lab-init"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0.5, xmax: 10.5, ymin: 0, ymax: 1.2, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["layer 1 std", "—"], ["layer L std", "—"], ["ratio", "—"], ["verdict", "—"]]);

    let depth = 8, width = 128, initStd = 0.01, act = "sigmoid";

    function simulate() {
      const rng = V.mulberry32(19);
      let h = new Array(width).fill(0).map(() => V.gauss(rng));
      const stds = [];
      for (let l = 0; l < depth; l++) {
        const next = new Array(width).fill(0);
        for (let j = 0; j < width; j++) {
          let s = 0;
          for (let i = 0; i < width; i++) s += V.gauss(rng) * initStd * h[i];
          next[j] = act === "relu" ? Math.max(0, s) : act === "tanh" ? Math.tanh(s) : M.sigmoid(s);
        }
        h = next;
        const mu = h.reduce((a, b) => a + b, 0) / width;
        stds.push(Math.sqrt(h.reduce((a, b) => a + (b - mu) ** 2, 0) / width));
      }
      return stds;
    }

    function update() {
      const stds = simulate();
      const ratio = stds[depth - 1] / (stds[0] || 1);
      const verdict = ratio < 0.05 ? "vanishing" : ratio > 5 ? "exploding / saturating" : "healthy";
      out.update({ "layer 1 std": stds[0], "layer L std": stds[depth - 1], ratio, verdict });

      fig.draw((f, ctx, t) => {
        const maxY = Math.max(0.2, Math.min(1.2, Math.max(...stds) * 1.3));
        f.yr = [0, maxY];
        f.axes({ xstep: 1, ystep: maxY / 4, xlabel: "layer", ylabel: "activation std" });
        f.line(stds.map((s, i) => [i + 1, s]), { color: t.accent, width: 2.4 });
        stds.forEach((s, i) => f.dot(i + 1, s, { color: t.accent, r: 3.5 }));
        // reference: what Xavier / He would give
        const ideal = act === "relu" ? Math.sqrt(2 / width) : Math.sqrt(1 / width);
        f.seg(f.xr[0], stds[0], f.xr[1], stds[0], { color: t.ink3, dash: [4, 4], width: 1.2 });
        f.text(f.pad.l + 8, f.pad.t + 12, `init std = ${initStd.toFixed(4)}   ·   He/Xavier would suggest ${ideal.toFixed(4)}`, { color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 27, "dashed = layer-1 scale (we want the curve to stay flat)", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "init std", min: 0.002, max: 0.12, step: 0.002, value: initStd, format: (v) => v.toFixed(3), onInput: (v) => { initStd = v; update(); } });
    V.slider(panel, { label: "depth", min: 2, max: 10, step: 1, value: depth, format: (v) => String(v), onInput: (v) => { depth = v; update(); } });
    V.segmented(panel, {
      label: "activation",
      options: [{ label: "sigmoid", value: "sigmoid" }, { label: "tanh", value: "tanh" }, { label: "ReLU", value: "relu" }],
      value: "sigmoid",
      onChange: (v) => { act = v; update(); },
    });
    V.buttons(panel, [{ label: "Use He/Xavier scale", onClick: () => { initStd = act === "relu" ? Math.sqrt(2 / width) : Math.sqrt(1 / width); update(); } }]);
    update();
  };
})();
