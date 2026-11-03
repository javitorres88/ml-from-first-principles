/* Module 01 labs — pipeline splitting, under/over-fitting, and metrics. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     Train/validation/test splitting: random vs stratified, and class balance.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-pipeline"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1, xmax: 1, ymin: -1, ymax: 1, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [
      ["train", "—"], ["validation", "—"], ["test", "—"], ["positives tr/va/te", "—"], ["worst imbalance", "—"],
    ]);
    const colors = { train: "#4f46e5", val: "#f59e0b", test: "#10b981" };

    const rng = V.mulberry32(7);
    let points = [];
    function generate(prevalence) {
      points = [];
      const n = 90;
      for (let i = 0; i < n; i++) {
        const pos = i < Math.max(1, Math.round(n * prevalence));
        const cx = pos ? 0.34 : -0.34, cy = pos ? 0.22 : -0.16;
        points.push({
          x: cx + V.gauss(rng) * 0.22,
          y: cy + V.gauss(rng) * 0.2,
          y1: pos ? 1 : 0,
          split: null,
        });
      }
    }

    let mode = "stratified", testFrac = 0.25, prevalence = 0.2;

    function assign() {
      const idx = points.map((_, i) => i);
      const shuffle = (arr) => { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };
      points.forEach((p) => (p.split = "train"));
      if (mode === "stratified") {
        [0, 1].forEach((cls) => {
          const ids = shuffle(idx.filter((i) => points[i].y1 === cls));
          const nTest = Math.round(ids.length * testFrac);
          const nVal = Math.round(ids.length * 0.15);
          ids.slice(0, nTest).forEach((i) => (points[i].split = "test"));
          ids.slice(nTest, nTest + nVal).forEach((i) => (points[i].split = "val"));
        });
      } else {
        const ids = shuffle(idx);
        const nTest = Math.round(ids.length * testFrac);
        const nVal = Math.round(ids.length * 0.15);
        ids.slice(0, nTest).forEach((i) => (points[i].split = "test"));
        ids.slice(nTest, nTest + nVal).forEach((i) => (points[i].split = "val"));
      }
      update();
    }

    function stats(split) {
      const s = points.filter((p) => p.split === split);
      const pos = s.filter((p) => p.y1 === 1).length;
      return { n: s.length, pos, frac: s.length ? pos / s.length : 0 };
    }

    function update() {
      const tr = stats("train"), va = stats("val"), te = stats("test");
      const worst = Math.max(Math.abs(tr.frac - prevalence), Math.abs(va.frac - prevalence), Math.abs(te.frac - prevalence));
      out.update({
        train: `${tr.n} · ${(tr.frac * 100).toFixed(0)}% pos`,
        validation: `${va.n} · ${(va.frac * 100).toFixed(0)}% pos`,
        test: `${te.n} · ${(te.frac * 100).toFixed(0)}% pos`,
        "positives tr/va/te": `${tr.pos} / ${va.pos} / ${te.pos}`,
        "worst imbalance": `${(worst * 100).toFixed(1)} pts`,
      });
      fig.render();
    }

    fig.draw((f, ctx, t) => {
      f.axes({ xstep: 0.5, ystep: 0.5, gridAlpha: 0.6 });
      points.forEach((p) => {
        const c = colors[p.split] || t.ink3;
        if (p.y1 === 1) {
          ctx.beginPath();
          ctx.arc(f.sx(p.x), f.sy(p.y), 5, 0, 2 * Math.PI);
          ctx.fillStyle = c; ctx.fill();
        } else {
          ctx.beginPath();
          ctx.arc(f.sx(p.x), f.sy(p.y), 4, 0, 2 * Math.PI);
          ctx.fillStyle = t.panel; ctx.fill();
          ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.stroke();
        }
      });
      f.text(f.pad.l + 8, f.pad.t + 14, "● class 1   ○ class 0", { color: t.ink3, font: "12px ui-sans-serif" });
    });

    const modeSeg = V.segmented(panel, {
      label: "Split strategy",
      options: [{ label: "Stratified", value: "stratified" }, { label: "Random", value: "random" }],
      value: "stratified",
      onChange: (v) => { mode = v; assign(); },
    });
    V.slider(panel, {
      label: "Test fraction", min: 0.1, max: 0.5, step: 0.05, value: 0.25,
      format: (v) => v.toFixed(2), onInput: (v) => { testFrac = v; assign(); },
    });
    V.slider(panel, {
      label: "Class prevalence", min: 0.02, max: 0.5, step: 0.01, value: 0.2,
      format: (v) => (v * 100).toFixed(0) + "% positive",
      onInput: (v) => { prevalence = v; generate(prevalence); assign(); },
    });
    V.buttons(panel, [{ label: "Reshuffle", onClick: assign }]);

    generate(prevalence);
    assign();
  };

  /* ------------------------------------------------------------------ lab 2
     Polynomial degree: underfitting → good fit → overfitting.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-fit"] = function (root) {
    const fig = V.makeIn(root, { xmin: -3.1, xmax: 3.1, ymin: -1.9, ymax: 1.9, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["parameters", "—"], ["train RMSE", "—"], ["test RMSE", "—"], ["verdict", "—"]]);
    const rng = V.mulberry32(42);

    let degree = 3, noise = 0.25;
    let train = [], test = [];

    function truth(x) { return Math.sin(x); }
    function make(noise) {
      const draw = (n, jitter) => {
        const pts = [];
        for (let i = 0; i < n; i++) {
          const x = -3 + (6 * (i + 0.5)) / n + V.gauss(rng) * jitter;
          pts.push([V.clamp(x, -3, 3), truth(x) + V.gauss(rng) * noise]);
        }
        return pts;
      };
      train = draw(18, 0.06);
      test = draw(160, 0.02);
    }

    function rmse(pts, coeffs) {
      const s = pts.reduce((acc, [x, y]) => acc + (y - M.polyval(coeffs, x / 3)) ** 2, 0);
      return Math.sqrt(s / pts.length);
    }

    function fit() {
      const coeffs = M.polyfit(train.map((p) => p[0] / 3), train.map((p) => p[1]), degree);
      const tr = rmse(train, coeffs), te = rmse(test, coeffs);
      const verdict = te > 0.9 ? "overfitting — memorising noise" : te > 0.35 ? "still underfitting" : degree <= 1 ? "underfitting — too rigid" : "good fit";
      out.update({ parameters: degree + 1, "train RMSE": tr, "test RMSE": te, verdict });
      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 0.5 });
        f.fn((x) => truth(x), { color: t.line2, width: 2, dash: [5, 4] });
        f.fn((x) => M.polyval(coeffs, x / 3), { color: t.accent, width: 2.4 });
        f.scatter(train, { color: t.ink2, r: 4 });
        f.text(f.pad.l + 8, f.pad.t + 12, "— model   ┄ truth   ● training data", { color: t.ink3, font: "12px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 28, `degree ${degree} · ${degree + 1} parameters`, { color: t.accent, font: "600 12px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "Polynomial degree", min: 0, max: 12, step: 1, value: degree, format: (v) => String(v), onInput: (v) => { degree = v; fit(); } });
    V.slider(panel, { label: "Noise σ", min: 0, max: 0.6, step: 0.02, value: noise, format: (v) => v.toFixed(2), onInput: (v) => { noise = v; make(noise); fit(); } });
    V.buttons(panel, [{ label: "New sample", onClick: () => { make(noise); fit(); } }]);

    make(noise);
    fit();
  };

  /* ------------------------------------------------------------------ lab 3
     Confusion matrix: prevalence, sensitivity, specificity → metrics.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-metrics"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "12px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 2, ymin: 0, ymax: 2, height: 210, pad: { l: 66, r: 10, t: 30, b: 34 } });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 0.5, ymin: 0, ymax: 1, height: 220 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [
      ["accuracy", "—"], ["sensitivity (recall)", "—"], ["specificity", "—"],
      ["precision (PPV)", "—"], ["F1", "—"], ["balanced acc.", "—"],
    ]);

    let prev = 0.1, sens = 0.9, spec = 0.9, N = 1000;

    function counts(p) {
      const P = Math.round(N * p), Ng = N - P;
      const tp = Math.round(sens * P), fn = P - tp;
      const tn = Math.round(spec * Ng), fp = Ng - tn;
      return { tp, fn, tn, fp };
    }

    function update() {
      const c = counts(prev);
      const m = M.metrics(c.tp, c.fp, c.tn, c.fn);
      out.update({
        accuracy: (m.accuracy * 100).toFixed(1) + "%",
        "sensitivity (recall)": (m.sensitivity * 100).toFixed(1) + "%",
        specificity: (m.specificity * 100).toFixed(1) + "%",
        "precision (PPV)": (m.precision * 100).toFixed(1) + "%",
        F1: m.f1.toFixed(3),
        "balanced acc.": (m.balanced * 100).toFixed(1) + "%",
      });

      fig.draw((f, ctx, t) => {
        const cells = [
          { r: 0, c: 0, label: "TP", v: c.tp, good: true },
          { r: 0, c: 1, label: "FN", v: c.fn, good: false },
          { r: 1, c: 0, label: "FP", v: c.fp, good: false },
          { r: 1, c: 1, label: "TN", v: c.tn, good: true },
        ];
        const w = 1, h = 1;
        const max = Math.max(...cells.map((x) => x.v)) || 1;
        cells.forEach((cell) => {
          const x0 = cell.c * w, y0 = 2 - cell.r * h;
          const alpha = 0.12 + 0.5 * (cell.v / max);
          ctx.fillStyle = cell.good ? `rgba(15,157,88,${alpha})` : `rgba(220,38,38,${alpha})`;
          ctx.fillRect(f.sx(x0), f.sy(y0), f.sx(x0 + w) - f.sx(x0), f.sy(y0 - h) - f.sy(y0));
          f.text(f.sx(x0 + 0.5), f.sy(y0 - 0.32), cell.label, { align: "center", color: t.ink2, font: "600 12px ui-sans-serif" });
          f.text(f.sx(x0 + 0.5), f.sy(y0 - 0.68), String(cell.v), { align: "center", color: t.ink, font: "700 18px ui-sans-serif" });
        });
        ctx.strokeStyle = t.line2; ctx.lineWidth = 1;
        ctx.strokeRect(f.sx(0), f.sy(2), f.sx(2) - f.sx(0), f.sy(0) - f.sy(2));
        f.text(f.pad.l + f.iw / 2, f.h - 8, "predicted", { align: "center", color: t.ink3, font: "11px ui-sans-serif" });
        ctx.save();
        ctx.translate(12, f.pad.t + f.ih / 2);
        ctx.rotate(-Math.PI / 2);
        f.text(0, 0, "actual", { align: "center", color: t.ink3, font: "11px ui-sans-serif" });
        ctx.restore();
        f.text(f.sx(0.5), f.pad.t - 14, "positive", { align: "center", color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.sx(1.5), f.pad.t - 14, "negative", { align: "center", color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.pad.l - 10, f.sy(1.5), "pos", { align: "right", color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.pad.l - 10, f.sy(0.5), "neg", { align: "right", color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 0.1, ystep: 0.25, xlabel: "prevalence", ylabel: "accuracy" });
        const model = [], base = [];
        for (let i = 0; i <= 50; i++) {
          const p = (i / 50) * 0.5;
          const cc = counts(p);
          model.push([p, (cc.tp + cc.tn) / N]);
          base.push([p, (N - Math.round(N * p)) / N]);
        }
        f.line(base, { color: t.ink3, dash: [4, 4], width: 1.6 });
        f.line(model, { color: t.accent, width: 2.4 });
        const cur = counts(prev);
        f.dot(prev, (cur.tp + cur.tn) / N, { color: t.accent, r: 5 });
        f.text(f.pad.l + 10, f.pad.t + f.ih - 26, "— classifier", { color: t.accent, font: "11px ui-sans-serif" });
        f.text(f.pad.l + 10, f.pad.t + f.ih - 12, "┄ always-negative", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "Prevalence", min: 0.01, max: 0.5, step: 0.01, value: prev, format: (v) => (v * 100).toFixed(0) + "%", onInput: (v) => { prev = v; update(); } });
    V.slider(panel, { label: "Sensitivity", min: 0.5, max: 1, step: 0.01, value: sens, format: (v) => (v * 100).toFixed(0) + "%", onInput: (v) => { sens = v; update(); } });
    V.slider(panel, { label: "Specificity", min: 0.5, max: 1, step: 0.01, value: spec, format: (v) => (v * 100).toFixed(0) + "%", onInput: (v) => { spec = v; update(); } });
    V.slider(panel, { label: "Population N", min: 100, max: 5000, step: 100, value: N, format: (v) => String(v), onInput: (v) => { N = v; update(); } });

    update();
  };
})();
