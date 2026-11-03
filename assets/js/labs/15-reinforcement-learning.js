/* Module 15 labs — value iteration and Q-learning in a grid world. */
(function () {
  const V = window.Viz, M = window.MLA;
  window.MLLabs = window.MLLabs || {};

  const N = 6;
  const WALLS = new Set(["1,1", "1,2", "4,4", "3,4"]);
  const START = [5, 0];
  const GOAL = [0, 5];
  const PIT = [3, 2];
  const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right

  function rewardOf(r, c) {
    if (r === GOAL[0] && c === GOAL[1]) return 10;
    if (r === PIT[0] && c === PIT[1]) return -10;
    return 0;
  }
  function isWall(r, c) { return r < 0 || c < 0 || r >= N || c >= N || WALLS.has(`${r},${c}`); }

  function step(r, c, a) {
    const [dr, dc] = ACTIONS[a];
    let nr = r + dr, nc = c + dc;
    if (isWall(nr, nc)) { nr = r; nc = c; }
    return [nr, nc];
  }

  function drawGrid(f, ctx, t, valueFn, policyFn, cellText) {
    const cell = Math.min(f.iw / N, f.ih / N);
    const ox = f.pad.l + (f.iw - cell * N) / 2, oy = f.pad.t + (f.ih - cell * N) / 2;
    let vmin = 0, vmax = 0;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      if (isWall(r, c)) continue;
      const v = valueFn(r, c);
      vmin = Math.min(vmin, v); vmax = Math.max(vmax, v);
    }
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const x = ox + c * cell, y = oy + r * cell;
        if (isWall(r, c)) { ctx.fillStyle = "#374151"; ctx.fillRect(x, y, cell - 1, cell - 1); continue; }
        const v = valueFn(r, c);
        const norm = (v - vmin) / (vmax - vmin || 1);
        ctx.fillStyle = v >= 0 ? `rgba(15,157,88,${0.05 + 0.5 * norm})` : `rgba(220,38,38,${0.05 + 0.5 * (1 - norm)})`;
        ctx.fillRect(x, y, cell - 1, cell - 1);
        ctx.strokeStyle = t.line; ctx.strokeRect(x, y, cell - 1, cell - 1);
        if (r === GOAL[0] && c === GOAL[1]) { ctx.fillStyle = "#0f9d58"; ctx.fillRect(x + 4, y + 4, cell - 9, cell - 9); }
        if (r === PIT[0] && c === PIT[1]) { ctx.fillStyle = "#111827"; ctx.fillRect(x + 4, y + 4, cell - 9, cell - 9); }
        if (r === START[0] && c === START[1]) f.text(x + cell / 2, y + cell - 10, "start", { align: "center", color: t.ink3, font: "9px ui-sans-serif" });
        f.text(x + cell / 2, y + cell / 2 + 2, cellText(r, c, v), { align: "center", color: t.ink, font: "11px ui-monospace, monospace" });
        // policy arrow
        if (policyFn) {
          const a = policyFn(r, c);
          if (a != null) {
            const [dr, dc] = ACTIONS[a];
            const cx = x + cell / 2, cy = y + 16;
            f.arrow(cx, cy, cx + dc * cell * 0.22, cy + dr * cell * 0.22, { color: "#4f46e5", width: 1.6, head: 5 });
          }
        }
      }
    }
  }

  /* ------------------------------------------------------------------ lab 1
     Value iteration.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-gridworld"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 400, pad: { l: 8, r: 8, t: 8, b: 8 } });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["iteration", "—"], ["max ΔV", "—"], ["V(start)", "—"], ["steps to goal", "—"]]);

    let gamma = 0.9, living = -0.04, Vt = null, iter = 0, timer = null;

    function init() {
      Vt = Array.from({ length: N }, () => new Array(N).fill(0));
      iter = 0;
    }

    function sweep() {
      const next = Vt.map((row) => row.slice());
      let delta = 0;
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          if (isWall(r, c)) continue;
          if ((r === GOAL[0] && c === GOAL[1]) || (r === PIT[0] && c === PIT[1])) { next[r][c] = rewardOf(r, c); continue; }
          let best = -Infinity;
          ACTIONS.forEach((_, a) => {
            const [nr, nc] = step(r, c, a);
            const val = living + rewardOf(nr, nc) + gamma * Vt[nr][nc];
            if (val > best) best = val;
          });
          delta = Math.max(delta, Math.abs(best - Vt[r][c]));
          next[r][c] = best;
        }
      }
      Vt = next;
      iter++;
      return delta;
    }

    function bestAction(r, c) {
      let best = null, bv = -Infinity;
      ACTIONS.forEach((_, a) => {
        const [nr, nc] = step(r, c, a);
        const val = living + rewardOf(nr, nc) + gamma * Vt[nr][nc];
        if (val > bv) { bv = val; best = a; }
      });
      return best;
    }

    function greedyPath() {
      let r = START[0], c = START[1], steps = 0;
      const seen = new Set();
      while (steps < 40) {
        const key = `${r},${c}`;
        if (seen.has(key)) return "loop";
        seen.add(key);
        if (r === GOAL[0] && c === GOAL[1]) return steps;
        const a = bestAction(r, c);
        [r, c] = step(r, c, a);
        steps++;
      }
      return ">40";
    }

    function update(delta) {
      out.update({ iteration: iter, "max ΔV": delta == null ? "—" : delta.toExponential(2), "V(start)": Vt[START[0]][START[1]].toFixed(2), "steps to goal": iter > 0 ? greedyPath() : "—" });
      fig.draw((f, ctx, t) => drawGrid(f, ctx, t, (r, c) => Vt[r][c], iter > 0 ? bestAction : null, (r, c, v) => v.toFixed(1)));
    }

    function stop() { clearInterval(timer); timer = null; }

    V.slider(panel, { label: "discount γ", min: 0.5, max: 0.99, step: 0.01, value: gamma, format: (v) => v.toFixed(2), onInput: (v) => { gamma = v; init(); update(); } });
    V.slider(panel, { label: "living reward", min: -0.3, max: 0.05, step: 0.01, value: living, format: (v) => v.toFixed(2), onInput: (v) => { living = v; init(); update(); } });
    V.buttons(panel, [
      { label: "Run", primary: true, onClick: () => { if (timer) return; timer = setInterval(() => { const d = sweep(); update(d); if (d < 1e-4) stop(); }, 220); } },
      { label: "Step", onClick: () => update(sweep()) },
      { label: "Reset", onClick: () => { stop(); init(); update(); } },
    ]);
    init();
    update();
  };

  /* ------------------------------------------------------------------ lab 2
     Q-learning with ε-greedy exploration.
  ------------------------------------------------------------------- */
  window.MLLabs["lab-qlearning"] = function (root) {
    const fig = V.makeIn(root, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 400, pad: { l: 8, r: 8, t: 8, b: 8 } });
    const panel = root.querySelector(".lab-panel");
    const out = V.readout(panel, [["episodes", "—"], ["ε", "—"], ["last return", "—"], ["best return", "—"]]);

    let gamma = 0.9, alpha = 0.2, eps = 0.3, living = -0.04;
    let Q = null, episodes = 0, lastReturn = 0, bestReturn = -Infinity, timer = null;
    const rng = V.mulberry32(21);

    function init() {
      Q = Array.from({ length: N }, () => Array.from({ length: N }, () => [0, 0, 0, 0]));
      episodes = 0; lastReturn = 0; bestReturn = -Infinity;
    }

    function runEpisode() {
      let [r, c] = START;
      let total = 0;
      for (let stepCount = 0; stepCount < 60; stepCount++) {
        let a;
        if (rng() < eps) a = Math.floor(rng() * 4);
        else a = Q[r][c].indexOf(Math.max(...Q[r][c]));
        const [nr, nc] = step(r, c, a);
        const rew = living + rewardOf(nr, nc);
        const done = (nr === GOAL[0] && nc === GOAL[1]) || (nr === PIT[0] && nc === PIT[1]);
        const target = rew + (done ? 0 : gamma * Math.max(...Q[nr][nc]));
        Q[r][c][a] += alpha * (target - Q[r][c][a]);
        total += rew;
        r = nr; c = nc;
        if (done) break;
      }
      episodes++;
      lastReturn = total;
      bestReturn = Math.max(bestReturn, total);
      update();
    }

    function bestAction(r, c) {
      const row = Q[r][c];
      const m = Math.max(...row);
      return m === 0 && row.every((v) => v === 0) ? null : row.indexOf(m);
    }

    function update() {
      out.update({ episodes, "ε": eps.toFixed(2), "last return": lastReturn.toFixed(2), "best return": bestReturn === -Infinity ? "—" : bestReturn.toFixed(2) });
      fig.draw((f, ctx, t) => drawGrid(f, ctx, t, (r, c) => Math.max(...Q[r][c]), episodes > 0 ? bestAction : null, (r, c, v) => v.toFixed(1)));
    }

    function stop() { clearInterval(timer); timer = null; }

    V.slider(panel, { label: "learning rate α", min: 0.05, max: 0.9, step: 0.05, value: alpha, format: (v) => v.toFixed(2), onInput: (v) => { alpha = v; } });
    V.slider(panel, { label: "exploration ε", min: 0, max: 1, step: 0.05, value: eps, format: (v) => v.toFixed(2), onInput: (v) => { eps = v; update(); } });
    V.slider(panel, { label: "discount γ", min: 0.5, max: 0.99, step: 0.01, value: gamma, format: (v) => v.toFixed(2), onInput: (v) => { gamma = v; } });
    V.buttons(panel, [
      { label: "Run 200 episodes", primary: true, onClick: () => { if (timer) return; timer = setInterval(() => runEpisode(), 30); setTimeout(stop, 6200); } },
      { label: "1 episode", onClick: runEpisode },
      { label: "Reset", onClick: () => { stop(); init(); update(); } },
    ]);
    init();
    update();
  };
})();
