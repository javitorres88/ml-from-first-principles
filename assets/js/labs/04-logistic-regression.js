/* Module 04 labs — the sigmoid, logistic regression fitting, softmax. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     The logistic function and its derivative.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-sigmoid"] = function (root) {
    const fig = V.makeIn(root, { xmin: -6, xmax: 6, ymin: -0.15, ymax: 1.15, height: 320 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["z (logit)", "—"], ["σ(z)", "—"], ["1 − σ(z)", "—"], ["σ′(z)", "—"], ["odds", "—"]]);
    let z = 1.2;

    function update() {
      const s = M.sigmoid(z);
      out.update({
        "z (logit)": z.toFixed(2),
        "σ(z)": s.toFixed(4),
        "1 − σ(z)": (1 - s).toFixed(4),
        "σ′(z)": (s * (1 - s)).toFixed(4),
        odds: (s / (1 - s)).toFixed(3),
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      f.axes({ xstep: 2, ystep: 0.25, xlabel: "z = wᵀx + b", ylabel: "probability" });
      f.line([[-6, 0], [6, 1]], { color: t.line2, dash: [5, 5], width: 1.4 });
      f.fn((x) => M.sigmoid(x), { color: t.accent, width: 2.6 });
      f.fn((x) => M.sigmoid(x) * (1 - M.sigmoid(x)) * 4, { color: "#0ea5e9", width: 1.8, dash: [4, 3] });
      const s = M.sigmoid(z);
      f.seg(z, 0, z, s, { color: t.ink3, dash: [3, 3], width: 1.2 });
      f.seg(-6, s, z, s, { color: t.ink3, dash: [3, 3], width: 1.2 });
      f.dot(z, s, { color: t.accent, r: 6, ring: t.panel });
      f.text(f.pad.l + 8, f.pad.t + 12, "— σ(z)   ┄ linear z→y   ┄ 4σ′(z) (rescaled slope)", { color: t.ink3, font: "11px ui-sans-serif" });
      f.text(f.pad.l + 8, f.pad.t + 27, "drag anywhere to move z", { color: t.ink3, font: "11px ui-sans-serif" });
    });

    fig.onPointer((p) => {
      if (!p.inside) return;
      z = V.clamp(p.x, -6, 6);
      update();
    }, { hover: true });

    V.slider(panel, { label: "z", min: -6, max: 6, step: 0.05, value: z, format: (v) => v.toFixed(2), onInput: (v) => { z = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Fitting a logistic regression by gradient descent.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-logistic-fit"] = function (root) {
    const fig = V.makeIn(root, { xmin: -3, xmax: 3, ymin: -3, ymax: 3, height: 380 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["iteration", "—"], ["cross-entropy", "—"], ["train accuracy", "—"], ["w", "—"], ["b", "—"]]);
    const rng = V.mulberry32(17);

    // two overlapping gaussian classes
    const X = [], y = [];
    for (let i = 0; i < 120; i++) {
      const cls = i % 2;
      const cx = cls ? 0.85 : -0.85, cy = cls ? 0.55 : -0.55;
      X.push([cx + V.gauss(rng) * 0.85, cy + V.gauss(rng) * 0.85]);
      y.push(cls);
    }

    let w = [0, 0], b = 0, lr = 0.9, iters = 0, showProb = true, timer = null;

    function loss() {
      let s = 0;
      for (let i = 0; i < X.length; i++) {
        const p = M.sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b);
        s += -(y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12));
      }
      return s / X.length;
    }
    function accuracy() {
      let c = 0;
      for (let i = 0; i < X.length; i++) {
        const p = M.sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b);
        c += (p >= 0.5 ? 1 : 0) === y[i] ? 1 : 0;
      }
      return c / X.length;
    }
    function stepOnce() {
      const n = X.length;
      let g0 = 0, g1 = 0, gb = 0;
      for (let i = 0; i < n; i++) {
        const p = M.sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b);
        const e = p - y[i];
        g0 += e * X[i][0]; g1 += e * X[i][1]; gb += e;
      }
      w = [w[0] - lr * (g0 / n), w[1] - lr * (g1 / n)];
      b -= lr * (gb / n);
      iters++;
      update();
    }
    function stop() { clearInterval(timer); timer = null; }

    function update() {
      out.update({
        iteration: iters,
        "cross-entropy": loss().toFixed(4),
        "train accuracy": (accuracy() * 100).toFixed(1) + "%",
        w: `(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`,
        b: b.toFixed(2),
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      if (showProb) {
        f.regions((x, yy) => M.sigmoid(w[0] * x + w[1] * yy + b), {
          nx: 60, ny: 52,
          color: (p) => {
            // blue (p=0) -> near white (p=.5) -> red (p=1)
            const c = V.clamp(p, 0, 1);
            const blue = [29, 78, 216], white = [250, 250, 252], red = [220, 38, 38];
            const [a, b2, w2] = c < 0.5 ? [blue, white, c * 2] : [white, red, (c - 0.5) * 2];
            return `rgba(${Math.round(V.lerp(a[0], b2[0], w2))},${Math.round(V.lerp(a[1], b2[1], w2))},${Math.round(V.lerp(a[2], b2[2], w2))},0.42)`;
          },
        });
      }
      f.axes({ xstep: 1, ystep: 1, gridAlpha: 0.4, xlabel: "x₁", ylabel: "x₂" });
      X.forEach((p, i) => f.dot(p[0], p[1], { color: y[i] ? "#dc2626" : "#1d4ed8", r: 4, ring: t.panel }));
      // decision boundary w.x + b = 0
      if (Math.abs(w[1]) > 1e-6) {
        const f0 = (x) => (-b - w[0] * x) / w[1];
        f.fn(f0, { color: t.ink, width: 2.2 });
      } else if (Math.abs(w[0]) > 1e-6) {
        const x0 = -b / w[0];
        f.seg(x0, -3, x0, 3, { color: t.ink, width: 2.2 });
      }
      f.text(f.pad.l + 8, f.pad.t + 12, "shading = P(y=1 | x)", { color: t.ink, font: "11px ui-sans-serif" });
    });

    V.slider(panel, { label: "learning rate", min: 0.05, max: 3, step: 0.05, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    V.toggle(panel, { label: "Show probability shading", value: true, onChange: (v) => { showProb = v; update(); } });
    V.buttons(panel, [
      { label: "Run", primary: true, onClick: () => { if (timer) return; timer = setInterval(() => { stepOnce(); if (iters > 600) stop(); }, 40); } },
      { label: "Step ×10", onClick: () => { for (let i = 0; i < 10; i++) stepOnce(); } },
      { label: "Reset", onClick: () => { stop(); w = [0, 0]; b = 0; iters = 0; update(); } },
    ]);
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     Softmax: from logits to probabilities to loss.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-softmax"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 3, ymin: 0, ymax: 1, height: 190, pad: { l: 40, r: 10, t: 22, b: 34 } });
    const fig2 = new V.Fig(c2, { xmin: -4, xmax: 6, ymin: 0, ymax: 1, height: 170 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["logits z", "—"], ["softmax", "—"], ["cross-entropy", "—"]]);
    const names = ["cat", "dog", "frog"];
    let z = [1.0, 2.0, 0.0], temp = 1.0, target = 0;

    function update() {
      const p = M.softmax(z.map((v) => v / temp));
      out.update({
        "logits z": z.map((v) => v.toFixed(1)).join(", "),
        softmax: p.map((v) => v.toFixed(2)).join(", "),
        "cross-entropy": (-Math.log(p[target] + 1e-12)).toFixed(3),
      });

      fig.draw((f, ctx, t) => {
        const bw = f.iw / 3;
        p.forEach((pv, i) => {
          const x = f.pad.l + i * bw;
          ctx.fillStyle = ["#4f46e5", "#0ea5e9", "#f59e0b"][i];
          ctx.globalAlpha = i === target ? 1 : 0.35;
          ctx.fillRect(x + bw * 0.2, f.sy(pv), bw * 0.6, f.sy(0) - f.sy(pv));
          ctx.globalAlpha = 1;
          f.text(x + bw / 2, f.h - 18, names[i], { align: "center", color: t.ink2, font: "600 12px ui-sans-serif" });
          f.text(x + bw / 2, f.sy(pv) - 10, pv.toFixed(2), { align: "center", color: t.ink, font: "600 12px ui-sans-serif" });
        });
        f.text(f.pad.l, f.pad.t - 8, `true class = ${names[target]} (solid bar)`, { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 2, ystep: 0.25, xlabel: "logit gap z_target − z_other", ylabel: "loss" });
        f.fn((x) => Math.log(1 + Math.exp(-x)), { color: t.accent, width: 2.4 });
        const other = Math.max(...z.filter((_, i) => i !== target));
        const gap = z[target] - other;
        f.dot(gap, Math.log(1 + Math.exp(-gap)), { color: "#dc2626", r: 5 });
        f.text(f.pad.l + 8, f.pad.t + 12, "binary cross-entropy of the target vs. the best rival", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    names.forEach((n, i) => {
      V.slider(panel, {
        label: `logit z · ${n}`, min: -4, max: 6, step: 0.1, value: z[i],
        format: (v) => v.toFixed(1),
        onInput: (v) => { z[i] = v; update(); },
      });
    });
    V.slider(panel, { label: "temperature T", min: 0.25, max: 4, step: 0.05, value: temp, format: (v) => v.toFixed(2), onInput: (v) => { temp = v; update(); } });
    V.select(panel, { label: "true class", options: names.map((n, i) => ({ value: String(i), label: n })), value: "0", onChange: (v) => { target = +v; update(); } });
    update();
  };
})();
