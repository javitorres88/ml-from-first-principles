/* Module 10 labs — convolution, shape arithmetic, residual connections. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  const IMG = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
  ];

  const KERNELS = {
    "vertical edge": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    "horizontal edge": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    "blur": [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "sobel x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  };

  /* ------------------------------------------------------------------ lab 1
     Cross-correlation, one window at a time.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-conv"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 20, ymin: 0, ymax: 12, height: 330, pad: { l: 8, r: 8, t: 8, b: 8 } });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["filter", "—"], ["output size", "—"], ["dot product", "—"], ["position", "—"]]);

    let kern = "vertical edge", row = 0, col = 0, stride = 1;

    function apply() {
      const K = KERNELS[kern];
      const H = IMG.length, W = IMG[0].length;
      const oh = Math.floor((H - 3) / stride) + 1, ow = Math.floor((W - 3) / stride) + 1;
      const o = [];
      for (let i = 0; i < oh; i++) {
        const r = [];
        for (let j = 0; j < ow; j++) {
          let s = 0;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += K[a][b] * IMG[i * stride + a][j * stride + b];
          r.push(s);
        }
        o.push(r);
      }
      return { o, oh, ow, K };
    }

    function update() {
      const { o, oh, ow, K } = apply();
      const r = Math.min(row, oh - 1), c = Math.min(col, ow - 1);
      let dot = 0;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) dot += K[a][b] * IMG[r * stride + a][c * stride + b];
      out.update({ filter: kern, "output size": `${oh}×${ow}`, "dot product": dot.toFixed(2), position: `(${r}, ${c})` });

      fig.draw((f, ctx, t) => {
        const cell = Math.min(f.iw / 20, f.ih / 12);
        const ox = f.pad.l + 4, oy = f.pad.t + 18;
        // input image
        for (let i = 0; i < IMG.length; i++) {
          for (let j = 0; j < IMG[0].length; j++) {
            const v = IMG[i][j];
            ctx.fillStyle = `rgba(79,70,229,${0.08 + 0.8 * v})`;
            ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
          }
        }
        // kernel window
        ctx.strokeStyle = "#dc2626"; ctx.lineWidth = 2.5;
        ctx.strokeRect(ox + c * stride * cell, oy + r * stride * cell, 3 * cell, 3 * cell);
        // output map
        const ox2 = ox + IMG[0].length * cell + 26;
        const maxAbs = Math.max(1, ...o.flat().map(Math.abs));
        for (let i = 0; i < oh; i++) {
          for (let j = 0; j < ow; j++) {
            const v = o[i][j] / maxAbs;
            ctx.fillStyle = v >= 0 ? `rgba(15,157,88,${0.08 + 0.75 * v})` : `rgba(220,38,38,${0.08 + 0.75 * -v})`;
            ctx.fillRect(ox2 + j * cell, oy + i * cell, cell - 1, cell - 1);
          }
        }
        ctx.strokeStyle = r === r && c === c ? "#111" : t.line2;
        ctx.lineWidth = 2;
        ctx.strokeRect(ox2 + c * cell, oy + r * cell, cell, cell);

        f.text(ox, f.pad.t + 8, "input", { color: t.ink2, font: "600 11px ui-sans-serif" });
        f.text(ox2, f.pad.t + 8, "output feature map", { color: t.ink2, font: "600 11px ui-sans-serif" });
        // kernel drawn as a 3x3 grid under the input
        const ky = oy + IMG.length * cell + 16;
        f.text(ox, ky - 4, "kernel", { color: t.ink3, font: "600 10px ui-sans-serif" });
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            const v = K[a][b];
            ctx.fillStyle = v >= 0 ? `rgba(15,157,88,${0.1 + 0.6 * Math.min(1, Math.abs(v))})` : `rgba(220,38,38,${0.1 + 0.6 * Math.min(1, Math.abs(v))})`;
            ctx.fillRect(ox + 44 + b * 30, ky + 8 + a * 22, 28, 20);
            f.text(ox + 58 + b * 30, ky + 18 + a * 22, v.toFixed(2), { align: "center", color: "#fff", font: "9px ui-monospace, monospace" });
          }
        }
      });
    }

    V.select(panel, {
      label: "kernel",
      options: Object.keys(KERNELS),
      value: kern,
      onChange: (v) => { kern = v; update(); },
    });
    V.slider(panel, { label: "stride", min: 1, max: 2, step: 1, value: stride, format: (v) => String(v), onInput: (v) => { stride = v; row = 0; col = 0; update(); } });
    V.buttons(panel, [
      { label: "Next window", primary: true, onClick: () => {
        const { oh, ow } = apply();
        col++;
        if (col >= ow) { col = 0; row++; }
        if (row >= oh) { row = 0; }
        update();
      } },
      { label: "Reset", onClick: () => { row = 0; col = 0; update(); } },
    ]);
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Shape arithmetic: H' = (H - K + 2P)/S + 1
  ------------------------------------------------------------------- */
  window.MLLabs["lab-shapes"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 100, ymin: 0, ymax: 100, height: 240, pad: { l: 10, r: 10, t: 10, b: 10 } });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["output H' × W'", "—"], ["parameters", "—"], ["multiply-adds", "—"], ["volume ratio", "—"]]);

    let H = 32, W = 32, Cin = 3, Cout = 10, K = 5, P = 2, S = 1;

    function calc() {
      const Ho = Math.floor((H - K + 2 * P) / S) + 1;
      const Wo = Math.floor((W - K + 2 * P) / S) + 1;
      const params = Cout * (Cin * K * K + 1);
      const flops = Ho * Wo * Cout * Cin * K * K;
      return { Ho, Wo, params, flops };
    }

    function update() {
      const r = calc();
      out.update({ "output H' × W'": `${r.Ho} × ${r.Wo}`, parameters: r.params.toLocaleString(), "multiply-adds": r.flops.toLocaleString(), "volume ratio": (r.Ho * r.Wo * Cout) / (H * W * Cin) });

      fig.draw((f, ctx, t) => {
        const scale = Math.min(f.iw / (H + 2 * P + 4), f.ih / (W + 2 * P + 4));
        const x0 = f.pad.l + 10, y0 = f.pad.t + 24;
        // padding ring
        if (P > 0) {
          ctx.strokeStyle = t.line2; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
          ctx.strokeRect(x0, y0, (W + 2 * P) * scale, (H + 2 * P) * scale);
          ctx.setLineDash([]);
        }
        // input
        ctx.fillStyle = "rgba(79,70,229,0.12)";
        ctx.fillRect(x0 + P * scale, y0 + P * scale, W * scale, H * scale);
        // kernel
        ctx.fillStyle = "rgba(220,38,38,0.35)";
        ctx.fillRect(x0 + P * scale, y0 + P * scale, K * scale, K * scale);
        f.text(x0, f.pad.t + 10, `input ${Cin}×${H}×${W}  →  output ${Cout}×${r.Ho}×${r.Wo}`, { color: t.ink2, font: "600 11px ui-sans-serif" });
        f.text(x0, y0 + (H + 2 * P) * scale + 16, `K=${K}, P=${P}, S=${S}   ·   H' = (H − K + 2P)/S + 1 = (${H} − ${K} + ${2 * P})/${S} + 1 = ${r.Ho}`, { color: t.ink3, font: "11px ui-monospace, monospace" });
      });
    }

    V.slider(panel, { label: "input H", min: 8, max: 64, step: 1, value: H, format: (v) => String(v), onInput: (v) => { H = v; update(); } });
    V.slider(panel, { label: "kernel K", min: 1, max: 11, step: 2, value: K, format: (v) => String(v), onInput: (v) => { K = v; update(); } });
    V.slider(panel, { label: "padding P", min: 0, max: 4, step: 1, value: P, format: (v) => String(v), onInput: (v) => { P = v; update(); } });
    V.slider(panel, { label: "stride S", min: 1, max: 4, step: 1, value: S, format: (v) => String(v), onInput: (v) => { S = v; update(); } });
    V.slider(panel, { label: "input channels C_in", min: 1, max: 8, step: 1, value: Cin, format: (v) => String(v), onInput: (v) => { Cin = v; update(); } });
    V.slider(panel, { label: "filters C_out", min: 1, max: 64, step: 1, value: Cout, format: (v) => String(v), onInput: (v) => { Cout = v; update(); } });
    update();
  };

  /* ------------------------------------------------------------------ lab 3
     Residual connections and the signal through depth.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-residual"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0.5, xmax: 20.5, ymin: 0, ymax: 1.2, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["depth", "—"], ["plain: signal at L", "—"], ["residual: signal at L", "—"], ["ratio", "—"]]);

    let depth = 12, scale = 0.85, shortcut = true;

    function simulate() {
      const rng = V.mulberry32(11);
      const plain = [1], res = [1];
      for (let l = 0; l < depth; l++) {
        const g = 0.5 + rng();                      // random per-layer gain factor
        plain.push(plain[plain.length - 1] * scale * g);
        res.push(res[res.length - 1] * (1 + scale * (g - 0.5)));  // F(x) + x keeps the signal near 1
      }
      return { plain, res };
    }

    function update() {
      const s = simulate();
      out.update({
        depth,
        "plain: signal at L": s.plain[s.plain.length - 1].toExponential(2),
        "residual: signal at L": s.res[s.res.length - 1].toExponential(2),
        ratio: (s.res[s.res.length - 1] / (s.plain[s.plain.length - 1] || 1)).toExponential(2),
      });

      fig.draw((f, ctx, t) => {
        const maxY = Math.max(1.2, Math.min(4, Math.max(...s.plain, ...s.res) * 1.1));
        f.yr = [0, maxY];
        f.axes({ xstep: 2, ystep: maxY / 4, xlabel: "block index", ylabel: "activation scale" });
        f.line(s.plain.map((v, i) => [i + 1, v]), { color: "#dc2626", width: 2.4 });
        f.line(s.res.map((v, i) => [i + 1, v]), { color: "#0f9d58", width: 2.4 });
        f.seg(f.xr[0], 1, f.xr[1], 1, { color: t.line2, dash: [4, 4], width: 1.2 });
        f.text(f.pad.l + 8, f.pad.t + 12, "— plain block (H = F(x))   — residual block (H = F(x) + x)   ┄ input scale", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    V.slider(panel, { label: "number of blocks", min: 2, max: 20, step: 1, value: depth, format: (v) => String(v), onInput: (v) => { depth = v; update(); } });
    V.slider(panel, { label: "per-layer gain", min: 0.5, max: 1.2, step: 0.02, value: scale, format: (v) => v.toFixed(2), onInput: (v) => { scale = v; update(); } });
    update();
  };
})();
