/* Module 11 labs — RNN recurrence, gradient flow through time, attention. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     The recurrence: what a hidden state remembers.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-rnn-unroll"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 13, ymin: -1.2, ymax: 1.2, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["mode", "—"], ["h at t = T", "—"], ["mean |h|", "—"], ["memory half-life", "—"]]);

    const seq = [0.6, -0.4, 0.9, 0.2, -0.7, 0.3, 0.8, -0.2, 0.5, -0.6, 0.4, 0.1];
    let wh = 0.9, wx = 0.6, mode = "tanh";

    function run() {
      const hs = [0];
      for (let t = 0; t < seq.length; t++) {
        const z = wh * hs[t] + wx * seq[t];
        hs.push(mode === "tanh" ? Math.tanh(z) : mode === "relu" ? Math.max(0, z) : M.sigmoid(z));
      }
      return hs;
    }

    function update() {
      const hs = run();
      const halfLife = wh <= 0 || wh >= 1 ? (wh >= 1 ? "∞ (no decay)" : "immediate") : (Math.log(0.5) / Math.log(wh)).toFixed(2) + " steps";
      out.update({ mode, "h at t = T": hs[hs.length - 1].toFixed(4), "mean |h|": (hs.reduce((s, v) => s + Math.abs(v), 0) / hs.length).toFixed(4), "memory half-life": halfLife });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 1, ystep: 0.5, xlabel: "time step t", ylabel: "value", gridAlpha: 0.35 });
        f.line(seq.map((v, i) => [i + 1, v]), { color: "#0ea5e9", width: 1.6, dash: [4, 3] });
        f.line(hs.map((v, i) => [i, v]), { color: t.accent, width: 2.6 });
        hs.forEach((v, i) => f.dot(i, v, { color: t.accent, r: 3 }));
        f.text(f.pad.l + 8, f.pad.t + 12, "— hidden state h_t   ┄ input sequence x_t", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.segmented(panel, {
      label: "activation",
      options: [{ label: "tanh", value: "tanh" }, { label: "sigmoid", value: "sigmoid" }, { label: "ReLU", value: "relu" }],
      value: "tanh",
      onChange: (v) => { mode = v; update(); },
    });
    V.slider(panel, { label: "recurrent weight w_h", min: -1.5, max: 1.5, step: 0.05, value: wh, format: (v) => v.toFixed(2), onInput: (v) => { wh = v; update(); } });
    V.slider(panel, { label: "input weight w_x", min: 0, max: 1.5, step: 0.05, value: wx, format: (v) => v.toFixed(2), onInput: (v) => { wx = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Gradient through time: decay, growth, and clipping.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-gradient-flow"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0.5, xmax: 40.5, ymin: -3, ymax: 3, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["spectral radius ρ", "—"], ["‖g‖ after T steps", "—"], ["clipped?", "—"], ["regime", "—"]]);

    let rho = 1.1, T = 25, clip = 5, useClip = false;

    function update() {
      const pts = [];
      let g = 1;
      let clipped = false;
      for (let t = 1; t <= 40; t++) {
        g *= rho;
        if (useClip && Math.abs(g) > clip) { g = Math.sign(g) * clip; clipped = true; }
        pts.push([t, Math.max(-3, Math.min(3, Math.log10(Math.abs(g) + 1e-12)))]);
      }
      const gT = Math.pow(rho, T);
      const regime = Math.abs(rho) < 1 ? "vanishing" : Math.abs(rho) > 1 ? "exploding" : "stable";
      out.update({ "spectral radius ρ": rho.toFixed(2), "‖g‖ after T steps": gT.toExponential(2), "clipped?": clipped ? "yes" : "no", regime });

      fig.draw((f, ctx, t) => {
        f.axes({ xstep: 5, ystep: 1, xlabel: "time steps back", ylabel: "log₁₀ |gradient|", gridAlpha: 0.3 });
        f.line(pts, { color: Math.abs(rho) > 1 ? "#dc2626" : "#0ea5e9", width: 2.4 });
        f.seg(T, -3, T, 3, { color: t.line2, dash: [4, 3], width: 1.2 });
        if (useClip) {
          const lc = Math.log10(clip);
          f.seg(0.5, lc, 40.5, lc, { color: "#0f9d58", dash: [4, 3], width: 1.2 });
          f.text(f.pad.l + 8, f.pad.t + 40, `clip threshold v = ${clip}`, { color: "#0f9d58", font: "11px ui-sans-serif" });
        }
        f.text(f.pad.l + 8, f.pad.t + 12, "‖∂L/∂h₁‖ ≈ ρᵀ · ‖∂L/∂h_T‖", { color: t.ink, font: "11px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 26, "dashed line = the truncation point T", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "spectral radius of W_hh", min: 0.6, max: 1.6, step: 0.02, value: rho, format: (v) => v.toFixed(2), onInput: (v) => { rho = v; update(); } });
    V.slider(panel, { label: "truncation length T", min: 5, max: 40, step: 1, value: T, format: (v) => String(v), onInput: (v) => { T = v; update(); } });
    V.toggle(panel, { label: "gradient clipping", value: false, onChange: (v) => { useClip = v; update(); } });
    V.slider(panel, { label: "clip threshold v", min: 0.5, max: 20, step: 0.5, value: clip, format: (v) => String(v), onInput: (v) => { clip = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     Attention: alignment scores → weights → context vector.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-attention"] = function (root) {
    const stage = root.querySelector(".lab-stage");
    const c1 = document.createElement("canvas");
    stage.prepend(c1);
    const c2 = document.createElement("canvas");
    c2.style.marginTop = "10px";
    stage.append(c2);
    const fig = new V.Fig(c1, { xmin: 0, xmax: 4, ymin: 0, ymax: 1, height: 180, pad: { l: 10, r: 10, t: 24, b: 26 } });
    const fig2 = new V.Fig(c2, { xmin: 0, xmax: 4, ymin: -2, ymax: 2, height: 180, pad: { l: 10, r: 10, t: 24, b: 26 } });

    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["scores e", "—"], ["weights a", "—"], ["context c", "—"], ["Σa", "—"]]);

    const words = ["we", "are", "eating", "bread"];
    const hidden = [[0.8, -0.4], [0.9, -0.3], [-0.2, 0.9], [0.1, 0.6]];
    let scores = [1.2, 1.0, -0.4, 0.2];
    let temperature = 1;

    function weights() { return M.softmax(scores.map((s) => s / temperature)); }

    function update() {
      const a = weights();
      const ctx = [0, 1].map((d) => hidden.reduce((s, h, i) => s + a[i] * h[d], 0));
      out.update({ "scores e": scores.map((s) => s.toFixed(1)).join(", "), "weights a": a.map((v) => v.toFixed(2)).join(", "), "context c": ctx.map((v) => v.toFixed(2)).join(", "), "Σa": a.reduce((x, y) => x + y, 0).toFixed(3) });

      fig.draw((f, ctx2, t) => {
        const bw = f.iw / 4;
        a.forEach((v, i) => {
          const x = f.pad.l + i * bw;
          ctx2.fillStyle = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981"][i];
          ctx2.globalAlpha = 0.25 + 0.75 * v;
          ctx2.fillRect(x + bw * 0.15, f.sy(v), bw * 0.7, f.sy(0) - f.sy(v));
          ctx2.globalAlpha = 1;
          f.text(x + bw / 2, f.h - 14, words[i], { align: "center", color: t.ink2, font: "600 12px ui-sans-serif" });
          f.text(x + bw / 2, f.sy(v) - 8, v.toFixed(2), { align: "center", color: t.ink, font: "11px ui-sans-serif" });
        });
        f.text(f.pad.l, f.pad.t - 10, "attention weights over the input tokens", { color: t.ink3, font: "11px ui-sans-serif" });
      });

      fig2.draw((f, ctx2, t) => {
        const bw = f.iw / 4;
        hidden.forEach((h, i) => {
          const x = f.pad.l + i * bw + bw / 2;
          f.arrow(x, 0, x, h[1], { color: "#cbd5e1", width: 1.6 });
          f.text(x + 6, f.sy(h[1]), `h${i + 1}`, { color: t.ink3, font: "10px ui-sans-serif" });
        });
        // context vector as weighted sum, drawn on the same axis for the second dimension
        f.arrow(f.pad.l + f.iw / 2, 0, f.pad.l + f.iw / 2, ctx[1], { color: "#dc2626", width: 3 });
        f.text(f.pad.l + f.iw / 2 + 8, f.sy(ctx[1]) - 8, `c₂ = ${ctx[1].toFixed(2)}`, { color: "#dc2626", font: "600 11px ui-sans-serif" });
        f.text(f.pad.l, f.pad.t - 10, "second component of each hidden state, and of the context vector", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    words.forEach((w, i) => {
      V.slider(panel, { label: `score e · ${w}`, min: -3, max: 3, step: 0.1, value: scores[i], format: (v) => v.toFixed(1), onInput: (v) => { scores[i] = v; update(); } });
    });
    V.slider(panel, { label: "softmax temperature", min: 0.2, max: 3, step: 0.1, value: temperature, format: (v) => v.toFixed(1), onInput: (v) => { temperature = v; update(); } });
    update();
  };
})();
