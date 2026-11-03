/* Module 06 labs — perceptron learning, activation functions, XOR and hidden layers. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     The perceptron learning rule.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-perceptron"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1.4, xmax: 1.4, ymin: -1.4, ymax: 1.4, height: 360 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["epoch", "—"], ["updates", "—"], ["w", "—"], ["b", "—"], ["errors", "—"]]);
    const rng = V.mulberry32(4);

    const X = [], y = [];
    for (let i = 0; i < 40; i++) {
      const cls = i % 2;
      const cx = cls ? 0.55 : -0.55, cy = cls ? 0.4 : -0.4;
      X.push([cx + V.gauss(rng) * 0.35, cy + V.gauss(rng) * 0.35]);
      y.push(cls ? 1 : -1); // sign convention from the lecture: -1 / +1
    }

    let w = [0.2, -0.1], b = 0, lr = 0.1, epoch = 0, updates = 0, errors = 0, i = 0;

    function reset() {
      w = [V.gauss(rng) * 0.3, V.gauss(rng) * 0.3]; b = 0; epoch = 0; updates = 0; i = 0;
      errors = countErrors(); update();
    }
    function countErrors() {
      return X.reduce((s, x, k) => s + ((w[0] * x[0] + w[1] * x[1] + b >= 0 ? 1 : -1) !== y[k] ? 1 : 0), 0);
    }
    function step() {
      if (i >= X.length) { i = 0; epoch++; }
      const x = X[i], t = y[i];
      const o = w[0] * x[0] + w[1] * x[1] + b >= 0 ? 1 : -1;
      if (o !== t) {
        w[0] += lr * (t - o) * x[0];
        w[1] += lr * (t - o) * x[1];
        b += lr * (t - o);
        updates++;
      }
      i++;
      errors = countErrors();
      update();
    }
    function update() {
      out.update({ epoch, updates, w: `(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`, b: b.toFixed(2), errors });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      f.regions((x, yy) => (w[0] * x + w[1] * yy + b >= 0 ? 1 : 0), {
        nx: 44, ny: 40,
        color: (v) => (v ? "rgba(220,38,38,0.07)" : "rgba(29,78,216,0.07)"),
      });
      f.axes({ xstep: 0.5, ystep: 0.5, gridAlpha: 0.35 });
      X.forEach((p, k) => {
        const pred = w[0] * p[0] + w[1] * p[1] + b >= 0 ? 1 : -1;
        f.dot(p[0], p[1], { color: y[k] === 1 ? "#dc2626" : "#1d4ed8", r: pred !== y[k] ? 6 : 4, ring: pred !== y[k] ? "#111" : null });
      });
      if (Math.abs(w[1]) > 1e-6) f.fn((x) => (-b - w[0] * x) / w[1], { color: t.ink, width: 2.2 });
      // weight vector, drawn from the origin
      const scale = 0.6 / (Math.hypot(w[0], w[1]) || 1);
      f.arrow(0, 0, w[0] * scale, w[1] * scale, { color: "#f59e0b", width: 2.4 });
      f.text(f.pad.l + 8, f.pad.t + 12, "rings = misclassified · amber arrow = weight vector w", { color: t.ink3, font: "11px ui-sans-serif" });
    });

    V.slider(panel, { label: "learning rate η", min: 0.01, max: 0.5, step: 0.01, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    V.buttons(panel, [
      { label: "Train 1 epoch", primary: true, onClick: () => { for (let k = 0; k < X.length; k++) step(); } },
      { label: "Step", onClick: step },
      { label: "Reset", onClick: reset },
    ]);
    reset();
  };

  /* ------------------------------------------------------------------ lab 2
     Activations and the vanishing gradient.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-activations"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: -6, xmax: 6, ymin: -1.6, ymax: 1.6, height: 240 });
    const fig2 = new V.Fig(c2, { xmin: 1, xmax: 30, ymin: 0, ymax: 1, height: 190 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["σ(x)", "—"], ["σ′(x)", "—"], ["Π over L layers", "—"]]);

    const acts = {
      sigmoid: { f: (x) => M.sigmoid(x), d: (x) => { const s = M.sigmoid(x); return s * (1 - s); }, color: "#4f46e5", label: "sigmoid" },
      tanh: { f: (x) => Math.tanh(x), d: (x) => 1 - Math.tanh(x) ** 2, color: "#0ea5e9", label: "tanh" },
      relu: { f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), color: "#0f9d58", label: "ReLU" },
    };
    let which = "sigmoid", x = 1.2;

    function update() {
      const a = acts[which];
      const L = 12;
      const perLayer = a.d(x);
      out.update({ "σ(x)": a.f(x).toFixed(4), "σ′(x)": perLayer.toFixed(4), "Π over L layers": Math.pow(perLayer, L).toExponential(2) });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 2, ystep: 0.5, xlabel: "x", ylabel: "" });
        Object.values(acts).forEach((aa) => {
          const on = aa === a;
          f.fn(aa.f, { color: aa.color, width: on ? 2.6 : 1.3, alpha: on ? 1 : 0.35 });
        });
        f.fn(a.d, { color: a.color, width: 1.6, dash: [4, 3], alpha: 0.9 });
        f.seg(x, -1.6, x, 1.6, { color: t.line2, width: 1 });
        f.dot(x, a.f(x), { color: a.color, r: 5.5, ring: t.panel });
        f.dot(x, a.d(x), { color: a.color, r: 4 });
        f.text(f.pad.l + 8, f.pad.t + 12, "solid = activation, dashed = derivative", { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 5, ystep: 0.25, xlabel: "network depth L", ylabel: "gradient scale" });
        Object.values(acts).forEach((aa) => {
          const on = aa === a;
          const pts = [];
          for (let l = 1; l <= 30; l++) pts.push([l, Math.pow(aa.d(x), l)]);
          f.line(pts, { color: aa.color, width: on ? 2.4 : 1.2, alpha: on ? 1 : 0.3 });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, "|∂L/∂W₁| after L layers = (σ′(x))^L", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    fig.onPointer((p) => { if (!p.inside) return; x = V.clamp(p.x, -6, 6); update(); }, { hover: true });

    V.segmented(panel, {
      label: "Activation",
      options: [{ label: "sigmoid", value: "sigmoid" }, { label: "tanh", value: "tanh" }, { label: "ReLU", value: "relu" }],
      value: "sigmoid",
      onChange: (v) => { which = v; update(); },
    });
    V.slider(panel, { label: "pre-activation x", min: -6, max: 6, step: 0.05, value: x, format: (v) => v.toFixed(2), onInput: (v) => { x = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     XOR: the hidden layer as a change of representation.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-xor"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: -0.3, xmax: 1.3, ymin: -0.3, ymax: 1.3, height: 230 });
    const fig2 = new V.Fig(c2, { xmin: -1.5, xmax: 1.5, ymin: -1.5, ymax: 1.5, height: 230 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["epochs", "—"], ["loss", "—"], ["accuracy", "—"]]);

    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const y = [0, 1, 1, 0];
    const rng = V.mulberry32(8);

    let H = 2, W1, b1, W2, b2, epochs = 0, timer = null;

    function init() {
      W1 = Array.from({ length: H }, () => [V.gauss(rng) * 1.2, V.gauss(rng) * 1.2]);
      b1 = Array.from({ length: H }, () => V.gauss(rng) * 0.5);
      W2 = Array.from({ length: H }, () => V.gauss(rng) * 1.2);
      b2 = 0;
      epochs = 0;
    }

    const tanh = Math.tanh;

    function forward(x) {
      const h = W1.map((row, j) => tanh(row[0] * x[0] + row[1] * x[1] + b1[j]));
      const z = h.reduce((s, hv, j) => s + hv * W2[j], b2);
      return { h, z, p: M.sigmoid(z) };
    }

    function train(steps) {
      for (let s = 0; s < steps; s++) {
        for (let k = 0; k < 4; k++) {
          const { h, z, p } = forward(X[k]);
          const dz = p - y[k];                    // dL/dz for sigmoid + BCE
          // gradients
          const dW2 = h.map((hv) => dz * hv);
          const db2 = dz;
          const dh = W2.map((wj) => dz * wj);
          const dpre = dh.map((d, j) => d * (1 - h[j] * h[j])); // tanh'
          W2 = W2.map((wj, j) => wj - 0.5 * dW2[j]);
          b2 -= 0.5 * db2;
          W1 = W1.map((row, j) => [row[0] - 0.5 * dpre[j] * X[k][0], row[1] - 0.5 * dpre[j] * X[k][1]]);
          b1 = b1.map((bj, j) => bj - 0.5 * dpre[j]);
        }
        epochs++;
      }
      update();
    }

    function update() {
      const preds = X.map((x) => forward(x).p);
      const acc = preds.reduce((s, p, i) => s + ((p >= 0.5 ? 1 : 0) === y[i] ? 1 : 0), 0) / 4;
      const loss = preds.reduce((s, p, i) => s - (y[i] * Math.log(p + 1e-9) + (1 - y[i]) * Math.log(1 - p + 1e-9)), 0) / 4;
      out.update({ epochs, loss, accuracy: (acc * 100).toFixed(0) + "%" });

      fig.draw((f, ctx, t) => {
        f.regions((a, b) => forward([a, b]).p, {
          nx: 56, ny: 50,
          color: (p) => (p > 0.5 ? `rgba(220,38,38,${0.08 + 0.4 * (p - 0.5)})` : `rgba(29,78,216,${0.08 + 0.4 * (0.5 - p)})`),
        });
        f.axes({ xstep: 0.5, ystep: 0.5, gridAlpha: 0.3, xlabel: "x₁", ylabel: "x₂" });
        X.forEach((p, i) => f.dot(p[0], p[1], { color: y[i] ? "#dc2626" : "#1d4ed8", r: 7, ring: "#fff" }));
        f.text(f.pad.l + 8, f.pad.t + 12, "input space: no single line separates XOR", { color: t.ink, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.35, xlabel: "h₁", ylabel: "h₂" });
        X.forEach((p, i) => {
          const { h } = forward(p);
          f.dot(h[0], h[1], { color: y[i] ? "#dc2626" : "#1d4ed8", r: 7, ring: "#fff" });
          f.text(f.sx(h[0]) + 10, f.sy(h[1]) - 10, `(${p[0]},${p[1]})`, { color: t.ink3, font: "10px ui-sans-serif" });
        });
        // the linear decision surface in hidden space: W2·h + b2 = 0
        if (H === 2 && Math.abs(W2[1]) > 1e-6) f.fn((h1) => (-b2 - W2[0] * h1) / W2[1], { color: t.ink, width: 2 });
        f.text(f.pad.l + 8, f.pad.t + 12, "hidden space: the same four points, now linearly separable", { color: t.ink, font: "11px ui-sans-serif" });
      });
    }

    function stop() { clearInterval(timer); timer = null; }

    V.slider(panel, { label: "hidden units", min: 1, max: 6, step: 1, value: H, format: (v) => String(v), onInput: (v) => { stop(); H = v; init(); update(); } });
    V.buttons(panel, [
      { label: "Train", primary: true, onClick: () => { if (timer) return; timer = setInterval(() => train(40), 60); } },
      { label: "Train ×200", onClick: () => { stop(); train(200); } },
      { label: "Reset", onClick: () => { stop(); init(); update(); } },
    ]);
    init();
    update();
  };
})();
