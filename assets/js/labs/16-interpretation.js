/* Module 16 labs — Shapley values and feature attribution. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  /* ------------------------------------------------------------------ lab 1
     Shapley values for a three-player cooperative game.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-shapley"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1, xmax: 3, ymin: -3, ymax: 12, height: 330 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["φ(A)", "—"], ["φ(B)", "—"], ["φ(C)", "—"], ["Σφ", "—"], ["v(ABC)", "—"]]);

    // coalition values the user controls
    const v = { A: 20, B: 10, C: 5, AB: 50, AC: 30, BC: 20, ABC: 60 };
    let current = "AB";

    function permutations(arr) {
      if (arr.length <= 1) return [arr];
      const out2 = [];
      arr.forEach((x, i) => {
        const rest = arr.slice(0, i).concat(arr.slice(i + 1));
        permutations(rest).forEach((p) => out2.push([x, ...p]));
      });
      return out2;
    }

    function coalitionValue(set) {
      if (set.length === 0) return 0;
      const key = set.slice().sort().join("");
      return v[key] ?? 0;
    }

    function shapley() {
      const players = ["A", "B", "C"];
      const perms = permutations(players);
      const phi = { A: 0, B: 0, C: 0 };
      perms.forEach((perm) => {
        const before = [];
        perm.forEach((p) => {
          phi[p] += (coalitionValue([...before, p]) - coalitionValue(before)) / perms.length;
          before.push(p);
        });
      });
      return phi;
    }

    function update() {
      const phi = shapley();
      const sum = phi.A + phi.B + phi.C;
      out.update({ "φ(A)": phi.A.toFixed(2), "φ(B)": phi.B.toFixed(2), "φ(C)": phi.C.toFixed(2), "Σφ": sum.toFixed(2), "v(ABC)": v.ABC });

      fig.draw((f, ctx, t) => {
        const maxAbs = Math.max(1, ...Object.values(phi).map(Math.abs));
        f.yr = [-maxAbs * 0.25, maxAbs * 1.25];
        f.axes({ xstep: 1, ystep: maxAbs / 4, ylabel: "attribution", gridAlpha: 0.3 });
        ["A", "B", "C"].forEach((p, i) => {
          const val = phi[p];
          ctx.fillStyle = val >= 0 ? "rgba(79,70,229,0.75)" : "rgba(220,38,38,0.75)";
          ctx.fillRect(f.sx(i) - 22, f.sy(Math.max(0, val)), 44, Math.abs(f.sy(val) - f.sy(0)));
          f.text(f.sx(i), f.sy(val) - 10, val.toFixed(1), { align: "center", color: t.ink, font: "600 12px ui-sans-serif" });
          f.text(f.sx(i), f.h - 12, p, { align: "center", color: t.ink2, font: "600 13px ui-sans-serif" });
        });
        f.text(f.pad.l + 8, f.pad.t + 12, "efficiency: the bars sum to v(ABC)", { color: t.ink3, font: "11px ui-sans-serif" });
        f.text(f.pad.l + 8, f.pad.t + 27, `last edited coalition: {${current.split("").join(", ")}}`, { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    const specs = [
      ["A alone", "A"], ["B alone", "B"], ["C alone", "C"],
      ["A + B", "AB"], ["A + C", "AC"], ["B + C", "BC"], ["A + B + C", "ABC"],
    ];
    specs.forEach(([label, key]) => {
      V.slider(panel, {
        label: `v(${label})`, min: 0, max: 80, step: 1, value: v[key],
        format: (x) => String(x),
        onInput: (x) => { v[key] = x; current = key; update(); },
      });
    });
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Attribution: linear contributions vs. SHAP values.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-attribution"] = function (root) {
    const fig = V.makeIn(root, { xmin: -1, xmax: 4, ymin: -3, ymax: 3, height: 320 });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["f(x)", "—"], ["baseline E[f]", "—"], ["Σ φ", "—"], ["f(x) − E[f]", "—"]]);

    // a linear model with 3 features; E[x] is the dataset mean
    const w = [1.2, -0.8, 0.5];
    const xbar = [0.0, 0.0, 0.0];
    let x = [1.0, -1.0, 0.5];
    let mode = "shap";

    function f(vec) { return w.reduce((s, wi, i) => s + wi * vec[i], 0); }

    function update() {
      // For a linear model, SHAP values are exactly w_i (x_i - E[x_i])
      const phi = w.map((wi, i) => wi * (x[i] - xbar[i]));
      const raw = w.map((wi, i) => wi * x[i]);
      const contrib = mode === "shap" ? phi : raw;
      const base = f(xbar);
      out.update({
        "f(x)": f(x).toFixed(3),
        "baseline E[f]": base.toFixed(3),
        "Σ φ": contrib.reduce((a, b) => a + b, 0).toFixed(3),
        "f(x) − E[f]": (f(x) - base).toFixed(3),
      });

      fig.draw((f2, ctx, t) => {
        f2.axes({ xstep: 1, ystep: 1, gridAlpha: 0.25 });
        f2.seg(-1, 0, 4, 0, { color: t.line2, width: 1.2 });
        contrib.forEach((val, i) => {
          ctx.fillStyle = val >= 0 ? "rgba(15,157,88,0.8)" : "rgba(220,38,38,0.8)";
          ctx.fillRect(f2.sx(i) - 26, f2.sy(Math.max(0, val)), 52, Math.abs(f2.sy(val) - f2.sy(0)));
          f2.text(f2.sx(i), f2.sy(val) + (val >= 0 ? -10 : 16), val.toFixed(2), { align: "center", color: t.ink, font: "600 12px ui-sans-serif" });
          f2.text(f2.sx(i), f2.h - 12, `x${i + 1} = ${x[i].toFixed(1)}`, { align: "center", color: t.ink2, font: "600 11px ui-sans-serif" });
        });
        f2.text(f2.pad.l + 8, f2.pad.t + 12, mode === "shap" ? "SHAP: wᵢ(xᵢ − E[xᵢ]) — sums to f(x) − E[f]" : "raw contribution: wᵢxᵢ — sums to f(x), not to the difference from baseline", { color: t.ink3, font: "11px ui-sans-serif" });
      });
    }

    [0, 1, 2].forEach((i) => {
      V.slider(panel, { label: `feature x${i + 1}`, min: -2, max: 2, step: 0.1, value: x[i], format: (v) => v.toFixed(1), onInput: (v) => { x[i] = v; update(); } });
    });
    V.segmented(panel, {
      label: "attribution method",
      options: [{ label: "SHAP", value: "shap" }, { label: "raw w·x", value: "raw" }],
      value: "shap",
      onChange: (v) => { mode = v; update(); },
    });
    update();
  };
})();
