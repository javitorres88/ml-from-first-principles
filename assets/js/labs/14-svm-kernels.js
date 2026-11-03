/* Module 14 labs — margins, soft margins and the kernel trick. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     Margin geometry.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-margin"] = function (root) {
    const fig = V.makeIn(root, { xmin: -2.6, xmax: 2.6, ymin: -2.6, ymax: 2.6, height: 380 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["angle", "—"], ["offset b", "—"], ["margin 2/‖w‖", "—"], ["violations", "—"]]);

    const rng = V.mulberry32(4);
    const X = [], y = [];
    for (let i = 0; i < 34; i++) {
      const cls = i % 2;
      const a = rng() * 2 - 1;
      const r = 1.15 + rng() * 0.9;
      const ang = (cls ? 0.6 : 0.6 + Math.PI) + a * 0.7;
      X.push([r * Math.cos(ang), r * Math.sin(ang)]);
      y.push(cls ? 1 : -1);
    }

    let theta = 200, b = 0;

    function w() { const th = (theta * Math.PI) / 180; return [Math.cos(th), Math.sin(th)]; }

    function update() {
      const [w0, w1] = w();
      const nw = Math.hypot(w0, w1);
      const margin = 2 / nw;
      const viol = X.reduce((s, p, i) => s + (y[i] * (w0 * p[0] + w1 * p[1] + b) < 1 ? 1 : 0), 0);
      out.update({ angle: theta.toFixed(0) + "°", "offset b": b.toFixed(2), "margin 2/‖w‖": margin.toFixed(3), violations: viol });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.3, xlabel: "x₁", ylabel: "x₂" });
        // decision boundary w·x + b = 0 and margins = ±1
        const line = (c) => {
          const pts = [];
          for (let i = 0; i <= 60; i++) {
            const x0 = f.xr[0] + ((f.xr[1] - f.xr[0]) * i) / 60;
            if (Math.abs(w1) < 1e-6) continue;
            pts.push([x0, (c - b - w0 * x0) / w1]);
          }
          return pts;
        };
        if (Math.abs(w1) > 1e-6) {
          f.line(line(0), { color: t.ink, width: 2.2 });
          f.line(line(1), { color: "#0f9d58", width: 1.4, dash: [5, 4] });
          f.line(line(-1), { color: "#0f9d58", width: 1.4, dash: [5, 4] });
        }
        X.forEach((p, i) => {
          const inside = y[i] * (w0 * p[0] + w1 * p[1] + b) < 1;
          f.dot(p[0], p[1], { color: y[i] === 1 ? "#dc2626" : "#1d4ed8", r: inside ? 6 : 4.5, ring: inside ? "#111" : null });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, "solid = decision boundary · dashed = margins · rings = margin violations", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "boundary angle", min: 0, max: 360, step: 1, value: theta, format: (v) => v + "°", onInput: (v) => { theta = v; update(); } });
    V.slider(panel, { label: "offset b", min: -1.5, max: 1.5, step: 0.05, value: b, format: (v) => v.toFixed(2), onInput: (v) => { b = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Kernel perceptron: the kernel trick with an RBF.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-kernel"] = function (root) {
    const fig = V.makeIn(root, { xmin: -2.2, xmax: 2.2, ymin: -2.2, ymax: 2.2, height: 380 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["kernel", "—"], ["γ", "—"], ["support vectors", "—"], ["train accuracy", "—"]]);

    const rng = V.mulberry32(12);
    const X = [], y = [];
    for (let i = 0; i < 90; i++) {
      const a = rng() * 2 * Math.PI;
      const r = (i % 2 === 0 ? 0.75 : 1.5) + V.gauss(rng) * 0.16;
      X.push([r * Math.cos(a), r * Math.sin(a)]);
      y.push(i % 2 === 0 ? 1 : -1);
    }

    let gamma = 1.2, kernel = "rbf";

    function K(a, b) {
      if (kernel === "linear") return a[0] * b[0] + a[1] * b[1];
      const d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
      return Math.exp(-gamma * d2);
    }

    // kernel perceptron (dual form): alpha_i += 1 on mistakes
    const alpha = new Array(X.length).fill(0);
    function train(passes) {
      for (let p = 0; p < passes; p++) {
        let errors = 0;
        for (let i = 0; i < X.length; i++) {
          let f = 0;
          for (let j = 0; j < X.length; j++) if (alpha[j]) f += alpha[j] * y[j] * K(X[j], X[i]);
          if (Math.sign(f || 1) !== y[i]) { alpha[i] += 1; errors++; }
        }
        if (!errors) break;
      }
    }

    function score(x) {
      let f = 0;
      for (let j = 0; j < X.length; j++) if (alpha[j]) f += alpha[j] * y[j] * K(X[j], x);
      return f;
    }

    function reset() {
      alpha.fill(0);
      train(30);
      update();
    }

    function update() {
      const sv = alpha.filter((a) => a > 0).length;
      const acc = X.reduce((s, p, i) => s + ((score(p) >= 0 ? 1 : -1) === y[i] ? 1 : 0), 0) / X.length;
      out.update({ kernel: kernel === "rbf" ? "RBF (Gaussian)" : "linear", "γ": kernel === "rbf" ? gamma.toFixed(2) : "—", "support vectors": sv, "train accuracy": (acc * 100).toFixed(1) + "%" });

      fig.draw((f, ctx, t) => {
        f.regions((a, b) => score([a, b]), {
          nx: 54, ny: 50,
          color: (v) => (v >= 0 ? "rgba(220,38,38,0.10)" : "rgba(29,78,216,0.10)"),
        });
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.25, xlabel: "x₁", ylabel: "x₂" });
        X.forEach((p, i) => f.dot(p[0], p[1], { color: y[i] === 1 ? "#dc2626" : "#1d4ed8", r: alpha[i] ? 6 : 3.6, ring: alpha[i] ? "#fff" : null }));
        f.text(f.pad.l + 8, f.pad.t + 12, "large points with rings = support vectors (αᵢ > 0)", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.segmented(panel, {
      label: "kernel",
      options: [{ label: "RBF", value: "rbf" }, { label: "linear", value: "linear" }],
      value: "rbf",
      onChange: (v) => { kernel = v; reset(); },
    });
    V.slider(panel, { label: "γ (kernel width)", min: 0.1, max: 8, step: 0.1, value: gamma, format: (v) => v.toFixed(1), onInput: (v) => { gamma = v; reset(); } });
    V.buttons(panel, [{ label: "Retrain", primary: true, onClick: reset }]);
    reset();
  };
})();
