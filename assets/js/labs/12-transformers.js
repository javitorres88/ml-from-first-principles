/* Module 12 labs — scaled dot-product attention and positional encoding. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     Scaled dot-product attention as a heatmap.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-attention-matrix"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 250, pad: { l: 62, r: 12, t: 26, b: 40 } });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 250, pad: { l: 62, r: 12, t: 26, b: 40 } });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["sequence length", "—"], ["scaling", "—"], ["max weight", "—"], ["row sums", "—"]]);

    const tokens = ["The", "cat", "sat", "down", "purr"];
    let dk = 4, useScale = true, causal = false, sharpness = 1;

    // deterministic pseudo-embeddings
    const rng = V.mulberry32(6);
    const D = 6;
    const X = tokens.map(() => Array.from({ length: D }, () => V.gauss(rng)));
    const Wq = Array.from({ length: D }, () => Array.from({ length: D }, () => V.gauss(rng) * 0.5));
    const Wk = Array.from({ length: D }, () => Array.from({ length: D }, () => V.gauss(rng) * 0.5));
    const Q = M.matmul(X, Wq), K = M.matmul(X, Wk);
    const n = tokens.length;

    function scores() {
      const s = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          let dot = 0;
          for (let d = 0; d < Math.min(dk, D); d++) dot += Q[i][d] * K[j][d];
          let v = dot * sharpness;
          if (useScale) v /= Math.sqrt(dk);
          if (causal && j > i) v = -Infinity;
          row.push(v);
        }
        s.push(row);
      }
      return s;
    }

    function weights() {
      return scores().map((row) => {
        const finite = row.filter((v) => Number.isFinite(v));
        const m = Math.max(...finite);
        const e = row.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0));
        const s = e.reduce((a, b) => a + b, 0);
        return e.map((v) => v / s);
      });
    }

    function drawMatrix(f, ctx, t, vals, title, diverging) {
      const cell = Math.min(f.iw / n, f.ih / n);
      const ox = f.pad.l, oy = f.pad.t;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const v = vals[i][j];
          let color;
          if (diverging) {
            const c = V.clamp((v + 2) / 4, 0, 1);
            color = v >= 0 ? `rgba(220,38,38,${0.08 + 0.8 * c})` : `rgba(29,78,216,${0.08 + 0.8 * (1 - c)})`;
          } else {
            color = `rgba(79,70,229,${0.06 + 0.9 * V.clamp(v, 0, 1)})`;
          }
          ctx.fillStyle = color;
          ctx.fillRect(ox + j * cell, oy + i * cell, cell - 2, cell - 2);
          const label = diverging ? v.toFixed(1) : v.toFixed(2);
          if (cell > 34) {
            const light = diverging ? Math.abs(v) < 1.2 : v < 0.45;
            f.text(ox + j * cell + cell / 2 - 1, oy + i * cell + cell / 2, label, { align: "center", color: light ? "#111827" : "#fff", font: "10px ui-monospace, monospace" });
          }
        }
      }
      tokens.forEach((tk, i) => {
        f.text(ox - 8, oy + i * cell + cell / 2, tk, { align: "right", color: t.ink2, font: "11px ui-sans-serif" });
        f.text(ox + i * cell + cell / 2, oy + n * cell + 12, tk, { align: "center", color: t.ink2, font: "11px ui-sans-serif" });
      });
      f.text(ox, oy - 10, title, { color: t.ink3, font: "600 11px ui-sans-serif" });
    }

    function update() {
      const W = weights(), S = scores();
      const rowSums = W.map((r) => r.reduce((a, b) => a + b, 0));
      out.update({
        "sequence length": n,
        scaling: useScale ? `÷√d_k = ÷${Math.sqrt(dk).toFixed(2)}` : "none",
        "max weight": Math.max(...W.flat()).toFixed(3),
        "row sums": rowSums.every((s) => Math.abs(s - 1) < 1e-6) ? "1.000 (all rows)" : "—",
      });

      fig.draw((f, ctx, t) => drawMatrix(f, ctx, t, S.map((r) => r.map((v) => (Number.isFinite(v) ? v : -2))), "scores QKᵀ/√d_k  (red positive, blue negative)", true));
      fig2.draw((f, ctx, t) => drawMatrix(f, ctx, t, W, "attention weights softmax(scores)", false));
    }

    V.slider(panel, { label: "key dimension d_k", min: 1, max: 6, step: 1, value: dk, format: (v) => String(v), onInput: (v) => { dk = v; update(); } });
    V.slider(panel, { label: "score sharpness", min: 0.2, max: 4, step: 0.1, value: sharpness, format: (v) => v.toFixed(1), onInput: (v) => { sharpness = v; update(); } });
    V.toggle(panel, { label: "divide by √d_k", value: true, onChange: (v) => { useScale = v; update(); } });
    V.toggle(panel, { label: "causal mask", value: false, onChange: (v) => { causal = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Sinusoidal positional encoding.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-positional"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 260, pad: { l: 46, r: 12, t: 26, b: 34 } });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 60, ymin: -1.1, ymax: 1.1, height: 200 });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["d_model", "—"], ["dim 0 frequency", "—"], ["p=10 vs p=20 (dim 0)", "—"], ["similarity(p, p+Δ)", "—"]]);

    let dmodel = 16, maxPos = 40, delta = 5;

    function pe(pos, d) {
      const i = Math.floor(d / 2);
      const denom = Math.pow(10000, (2 * i) / dmodel);
      return d % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom);
    }

    function update() {
      const sim = (() => {
        let dot = 0, na = 0, nb = 0;
        for (let d = 0; d < dmodel; d++) {
          const a = pe(10, d), b = pe(10 + delta, d);
          dot += a * b; na += a * a; nb += b * b;
        }
        return dot / (Math.sqrt(na * nb) || 1);
      })();
      out.update({ d_model: dmodel, "dim 0 frequency": (1 / Math.pow(10000, 0)).toFixed(3), "p=10 vs p=20 (dim 0)": `${pe(10, 0).toFixed(3)} vs ${pe(20, 0).toFixed(3)}`, "similarity(p, p+Δ)": sim.toFixed(3) });

      fig.draw((f, ctx, t) => {
        const cell = Math.min(f.iw / dmodel, f.ih / maxPos);
        for (let p = 0; p < maxPos; p++) {
          for (let d = 0; d < dmodel; d++) {
            const v = pe(p, d);
            ctx.fillStyle = v >= 0 ? `rgba(79,70,229,${0.05 + 0.9 * v})` : `rgba(220,38,38,${0.05 + 0.9 * -v})`;
            ctx.fillRect(f.pad.l + d * cell, f.pad.t + p * cell, Math.ceil(cell), Math.ceil(cell));
          }
        }
        f.text(f.pad.l, f.pad.t - 10, "positional encoding matrix (rows = position, cols = dimension)", { color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.pad.l - 8, f.pad.t + 6, "0", { align: "right", color: t.ink3, font: "10px ui-sans-serif" });
        f.text(f.pad.l - 8, f.pad.t + maxPos * cell - 4, String(maxPos - 1), { align: "right", color: t.ink3, font: "10px ui-sans-serif" });
        f.text(f.pad.l + f.iw / 2, f.h - 8, "dimension index i", { align: "center", color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx, t) => {
        f.axes({ xstep: 10, ystep: 0.5, xlabel: "position p", ylabel: "value" });
        [0, 1, 4, 8].filter((d) => d < dmodel).forEach((d, k) => {
          f.fn((p) => pe(p, d), { color: V.palette[k % V.palette.length], width: 1.8, n: 300 });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, "curves: dimension 0, 1, 4, 8 — low dims oscillate fast, high dims slowly", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "d_model", min: 4, max: 32, step: 2, value: dmodel, format: (v) => String(v), onInput: (v) => { dmodel = v; update(); } });
    V.slider(panel, { label: "positions shown", min: 10, max: 60, step: 5, value: maxPos, format: (v) => String(v), onInput: (v) => { maxPos = v; update(); } });
    V.slider(panel, { label: "offset Δ", min: 1, max: 20, step: 1, value: delta, format: (v) => String(v), onInput: (v) => { delta = v; update(); } });
    update();
  };
})();
