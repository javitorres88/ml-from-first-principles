/* ==========================================================================
   viz.js — a tiny, dependency-free plotting + controls toolkit for the labs.
   Exposes a single global: window.Viz
   ========================================================================== */
(function () {
  "use strict";

  const palette = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"];

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  const theme = () => ({
    ink: cssVar("--ink", "#12141a"),
    ink2: cssVar("--ink-2", "#3d4351"),
    ink3: cssVar("--ink-3", "#6b7280"),
    line: cssVar("--line", "#e3e5ec"),
    line2: cssVar("--line-2", "#cfd3de"),
    panel: cssVar("--panel", "#ffffff"),
    panel2: cssVar("--panel-2", "#f7f8fb"),
    accent: cssVar("--accent", "#4f46e5"),
    ok: cssVar("--ok", "#0f9d58"),
    bad: cssVar("--bad", "#dc2626"),
    warn: cssVar("--warn", "#b45309"),
  });

  /* ------------------------------------------------------------------ math */
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => lerp(a, b, n === 1 ? 0 : i / (n - 1)));

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* --------------------------------------------------------------- figure */
  class Fig {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.opts = Object.assign({ xmin: 0, xmax: 1, ymin: 0, ymax: 1, height: 340, pad: { l: 46, r: 14, t: 14, b: 36 }, equal: false }, opts || {});
      this.pad = this.opts.pad;
      this.range(this.opts.xmin, this.opts.xmax, this.opts.ymin, this.opts.ymax);
      this._draw = null;
      this._ro = new ResizeObserver(() => this.resize());
      this._ro.observe(canvas.parentElement || canvas);
      this.resize();
    }

    range(xmin, xmax, ymin, ymax) {
      this.xr = [xmin, xmax];
      this.yr = [ymin, ymax];
      return this;
    }

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const parent = this.canvas.parentElement || this.canvas;
      const cssW = Math.max(220, Math.floor(parent.clientWidth - (this.opts.padExtra || 0)));
      const cssH = typeof this.opts.height === "function" ? this.opts.height(cssW) : this.opts.height;
      if (this.w === cssW && this.h === cssH) { if (this._draw) this.render(); return; }
      this.w = cssW; this.h = cssH;
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
      this.canvas.style.height = cssH + "px";
      this.ctx = this.canvas.getContext("2d");
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (this._draw) this.render();
    }

    get iw() { return this.w - this.pad.l - this.pad.r; }
    get ih() { return this.h - this.pad.t - this.pad.b; }

    sx(x) { return this.pad.l + ((x - this.xr[0]) / (this.xr[1] - this.xr[0])) * this.iw; }
    sy(y) { return this.pad.t + this.ih - ((y - this.yr[0]) / (this.yr[1] - this.yr[0])) * this.ih; }
    ix(px) { return this.xr[0] + ((px - this.pad.l) / this.iw) * (this.xr[1] - this.xr[0]); }
    iy(py) { return this.yr[0] + ((this.pad.t + this.ih - py) / this.ih) * (this.yr[1] - this.yr[0]); }

    /** Register a draw callback (called on resize / render). */
    draw(fn) { this._draw = fn; this.render(); return this; }

    render() {
      if (!this._draw) return;
      const t = theme();
      this.ctx.save();
      this.ctx.clearRect(0, 0, this.w, this.h);
      this.ctx.fillStyle = t.panel;
      this.ctx.fillRect(0, 0, this.w, this.h);
      this.t = t;
      this._draw(this, this.ctx, t);
      this.ctx.restore();
    }

    /* ---------------------------------------------------------- primitives */
    axes(o) {
      o = o || {};
      const ctx = this.ctx, t = this.t || theme();
      const xstep = o.xstep, ystep = o.ystep;
      ctx.save();
      ctx.strokeStyle = t.line;
      ctx.fillStyle = t.ink3;
      ctx.lineWidth = 1;
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";

      const xs = xstep ? ticks(this.xr, xstep) : [];
      const ys = ystep ? ticks(this.yr, ystep) : [];
      ctx.beginPath();
      xs.forEach((x) => { const px = Math.round(this.sx(x)) + .5; ctx.moveTo(px, this.pad.t); ctx.lineTo(px, this.pad.t + this.ih); });
      ys.forEach((y) => { const py = Math.round(this.sy(y)) + .5; ctx.moveTo(this.pad.l, py); ctx.lineTo(this.pad.l + this.iw, py); });
      ctx.globalAlpha = o.gridAlpha == null ? .85 : o.gridAlpha;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // zero lines
      if (o.zero !== false) {
        ctx.strokeStyle = t.line2;
        ctx.beginPath();
        if (this.yr[0] < 0 && this.yr[1] > 0) { const py = Math.round(this.sy(0)) + .5; ctx.moveTo(this.pad.l, py); ctx.lineTo(this.pad.l + this.iw, py); }
        if (this.xr[0] < 0 && this.xr[1] > 0) { const px = Math.round(this.sx(0)) + .5; ctx.moveTo(px, this.pad.t); ctx.lineTo(px, this.pad.t + this.ih); }
        ctx.stroke();
      }

      // tick labels
      ctx.fillStyle = t.ink3;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      xs.forEach((x) => ctx.fillText(fmt(x), this.sx(x), this.pad.t + this.ih + 6));
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ys.forEach((y) => ctx.fillText(fmt(y), this.pad.l - 8, this.sy(y)));

      // frame
      ctx.strokeStyle = t.line2;
      ctx.strokeRect(this.pad.l + .5, this.pad.t + .5, this.iw - 1, this.ih - 1);

      if (o.xlabel) {
        ctx.fillStyle = t.ink3; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        ctx.fillText(o.xlabel, this.pad.l + this.iw / 2, this.h - 2);
      }
      if (o.ylabel) {
        ctx.save();
        ctx.translate(11, this.pad.t + this.ih / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = t.ink3; ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(o.ylabel, 0, 0);
        ctx.restore();
      }
      ctx.restore();
      return this;
    }

    line(pts, o) {
      o = o || {};
      const ctx = this.ctx;
      if (!pts.length) return this;
      ctx.save();
      ctx.strokeStyle = o.color || (this.t || theme()).accent;
      ctx.lineWidth = o.width || 2;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      if (o.dash) ctx.setLineDash(o.dash);
      if (o.alpha != null) ctx.globalAlpha = o.alpha;
      ctx.beginPath();
      pts.forEach((p, i) => { const x = this.sx(p[0]), y = this.sy(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      ctx.restore();
      return this;
    }

    fn(f, o) {
      o = o || {};
      const n = o.n || 240;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const x = lerp(this.xr[0], this.xr[1], i / (n - 1));
        const y = f(x);
        if (Number.isFinite(y)) pts.push([x, y]);
      }
      return this.line(pts, o);
    }

    seg(x1, y1, x2, y2, o) {
      o = o || {};
      const ctx = this.ctx;
      ctx.save();
      ctx.strokeStyle = o.color || (this.t || theme()).ink3;
      ctx.lineWidth = o.width || 1.5;
      if (o.dash) ctx.setLineDash(o.dash);
      if (o.alpha != null) ctx.globalAlpha = o.alpha;
      ctx.beginPath(); ctx.moveTo(this.sx(x1), this.sy(y1)); ctx.lineTo(this.sx(x2), this.sy(y2)); ctx.stroke();
      ctx.restore();
      return this;
    }

    arrow(x1, y1, x2, y2, o) {
      o = o || {};
      const ctx = this.ctx;
      this.seg(x1, y1, x2, y2, o);
      const ax = this.sx(x2), ay = this.sy(y2), bx = this.sx(x1), by = this.sy(y1);
      const ang = Math.atan2(ay - by, ax - bx);
      const s = o.head || 7;
      ctx.save();
      ctx.fillStyle = o.color || (this.t || theme()).ink3;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - s * Math.cos(ang - .4), ay - s * Math.sin(ang - .4));
      ctx.lineTo(ax - s * Math.cos(ang + .4), ay - s * Math.sin(ang + .4));
      ctx.closePath(); ctx.fill();
      ctx.restore();
      return this;
    }

    points(xs, ys, o) {
      o = o || {};
      const ctx = this.ctx;
      const r = o.r || 4;
      ctx.save();
      ctx.globalAlpha = o.alpha == null ? 1 : o.alpha;
      for (let i = 0; i < xs.length; i++) {
        ctx.beginPath();
        ctx.arc(this.sx(xs[i]), this.sy(ys[i]), o.rFn ? o.rFn(i) : r, 0, 2 * Math.PI);
        ctx.fillStyle = o.colorFn ? o.colorFn(i) : (o.color || (this.t || theme()).accent);
        ctx.fill();
        if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.strokeWidth || 1.2; ctx.stroke(); }
      }
      ctx.restore();
      return this;
    }

    /** scatter([[x,y],...], {colorFn(i)}) */
    scatter(pts, o) {
      o = o || {};
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      return this.points(xs, ys, o);
    }

    circle(x, y, r, o) {
      o = o || {};
      const ctx = this.ctx;
      const rr = o.dataRadius ? (r / (this.xr[1] - this.xr[0])) * this.iw : r;
      ctx.save();
      ctx.beginPath(); ctx.arc(this.sx(x), this.sy(y), rr, 0, 2 * Math.PI);
      if (o.fill) { ctx.fillStyle = o.fill; ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.fill(); ctx.globalAlpha = 1; }
      if (o.stroke !== false) { ctx.strokeStyle = o.stroke || (this.t || theme()).accent; ctx.lineWidth = o.width || 1.5; if (o.dash) ctx.setLineDash(o.dash); ctx.stroke(); }
      ctx.restore();
      return this;
    }

    dot(x, y, o) {
      o = o || {};
      const ctx = this.ctx;
      ctx.save();
      ctx.beginPath(); ctx.arc(this.sx(x), this.sy(y), o.r || 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = o.color || (this.t || theme()).accent;
      ctx.fill();
      if (o.ring) { ctx.strokeStyle = o.ring; ctx.lineWidth = 1.5; ctx.stroke(); }
      ctx.restore();
      return this;
    }

    text(x, y, s, o) {
      o = o || {};
      const ctx = this.ctx;
      ctx.save();
      ctx.font = o.font || "12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = o.color || (this.t || theme()).ink2;
      ctx.textAlign = o.align || "left";
      ctx.textBaseline = o.baseline || "middle";
      ctx.fillText(s, x, y);
      ctx.restore();
      return this;
    }

    label(x, y, s, o) {
      o = o || {};
      return this.text(this.sx(x), this.sy(y) - (o.dy || 10), s, o);
    }

    /** shade a data-space rectangle */
    shade(x0, y0, x1, y1, o) {
      o = o || {};
      const ctx = this.ctx;
      ctx.save();
      ctx.globalAlpha = o.alpha == null ? .12 : o.alpha;
      ctx.fillStyle = o.color || (this.t || theme()).accent;
      ctx.fillRect(this.sx(x0), this.sy(y1), this.sx(x1) - this.sx(x0), this.sy(y0) - this.sy(y1));
      ctx.restore();
      return this;
    }

    /** heatmap: values[row][col] where row 0 is at the top of the plot area */
    heat(values, o) {
      o = o || {};
      const ctx = this.ctx;
      const rows = values.length, cols = values[0].length;
      const x0 = o.x0 == null ? this.pad.l : this.sx(o.x0);
      const y0 = o.y0 == null ? this.pad.t : this.sy(o.y0);
      const w = o.w || this.iw, h = o.h || this.ih;
      const cw = w / cols, ch = h / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = values[r][c];
          ctx.fillStyle = o.color ? o.color(v) : heatColor(v, o.min, o.max);
          ctx.fillRect(x0 + c * cw, y0 + r * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      return this;
    }

    /** filled decision-region map over the data range, low resolution grid */
    regions(f, o) {
      o = o || {};
      const nx = o.nx || 90, ny = o.ny || 70;
      const vals = [];
      for (let j = 0; j < ny; j++) {
        const row = [];
        for (let i = 0; i < nx; i++) {
          const x = lerp(this.xr[0], this.xr[1], (i + .5) / nx);
          const y = lerp(this.yr[1], this.yr[0], (j + .5) / ny);
          row.push(f(x, y));
        }
        vals.push(row);
      }
      this.heat(vals, o);
      return this;
    }

    /** register pointer interaction in data coordinates */
    onPointer(handler, o) {
      o = o || {};
      const c = this.canvas;
      const pt = (e) => {
        const r = c.getBoundingClientRect();
        const px = e.clientX - r.left, py = e.clientY - r.top;
        return { x: this.ix(px), y: this.iy(py), px, py, inside: px >= this.pad.l && px <= this.pad.l + this.iw && py >= this.pad.t && py <= this.pad.t + this.ih };
      };
      let dragging = false;
      c.style.cursor = o.cursor || "crosshair";
      const down = (e) => { dragging = true; c.setPointerCapture && c.setPointerCapture(e.pointerId); handler(pt(e), "down"); };
      const move = (e) => { const p = pt(e); if (dragging || o.hover) handler(p, "move"); };
      const up = (e) => { dragging = false; handler(pt(e), "up"); };
      c.addEventListener("pointerdown", down);
      c.addEventListener("pointermove", move);
      c.addEventListener("pointerup", up);
      c.addEventListener("pointerleave", (e) => { dragging = false; if (o.onLeave) o.onLeave(); });
      return this;
    }
  }

  function ticks(range, step) {
    const out = [];
    const start = Math.ceil(range[0] / step) * step;
    for (let v = start; v <= range[1] + 1e-9; v += step) out.push(Math.abs(v) < 1e-9 ? 0 : v);
    return out;
  }
  function fmt(v) {
    if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(0);
    return String(Math.round(v * 1000) / 1000);
  }
  function heatColor(v, min, max) {
    // perceptual-ish blue -> white -> red for signed, or light->dark for unsigned
    const t = min != null && max != null ? (v - min) / (max - min || 1) : v;
    const c = clamp(t, 0, 1);
    const stops = [[239, 246, 255], [191, 219, 254], [147, 197, 253], [96, 165, 250], [59, 130, 246], [37, 99, 235], [29, 78, 216]];
    const p = c * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${Math.round(lerp(a[0], b[0], f))},${Math.round(lerp(a[1], b[1], f))},${Math.round(lerp(a[2], b[2], f))})`;
  }

  /* ------------------------------------------------------------- controls */
  function slider(container, spec) {
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    const id = "s" + Math.random().toString(36).slice(2, 8);
    const label = document.createElement("label");
    label.setAttribute("for", id);
    const nameSpan = document.createElement("span");
    nameSpan.textContent = spec.label;
    const out = document.createElement("output");
    label.append(nameSpan, out);
    const input = document.createElement("input");
    input.type = "range";
    input.id = id;
    input.min = spec.min; input.max = spec.max; input.step = spec.step == null ? "any" : spec.step;
    input.value = spec.value;
    const show = (v) => (out.textContent = spec.format ? spec.format(+v) : String(+v));
    show(spec.value);
    input.addEventListener("input", () => { show(input.value); spec.onInput && spec.onInput(+input.value); });
    wrap.append(label, input);
    container.append(wrap);
    const api = {
      el: wrap,
      get value() { return +input.value; },
      set(v, silent) { input.value = v; show(v); if (!silent) spec.onInput && spec.onInput(+v); return api; },
    };
    if (spec.value != null) api.value0 = spec.value;
    return api;
  }

  function select(container, spec) {
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    const label = document.createElement("label");
    label.textContent = spec.label;
    const sel = document.createElement("select");
    (spec.options || []).forEach((o) => {
      const opt = document.createElement("option");
      if (typeof o === "string") { opt.value = o; opt.textContent = o; }
      else { opt.value = o.value; opt.textContent = o.label; }
      sel.append(opt);
    });
    if (spec.value != null) sel.value = spec.value;
    sel.addEventListener("change", () => spec.onChange && spec.onChange(sel.value));
    wrap.append(label, sel);
    container.append(wrap);
    return { el: wrap, get value() { return sel.value; }, set(v) { sel.value = v; return this; }, sel };
  }

  function toggle(container, spec) {
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    const label = document.createElement("label");
    label.style.cursor = "pointer";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = !!spec.value;
    cb.style.width = "auto"; cb.style.marginRight = "8px";
    label.append(cb, document.createTextNode(spec.label));
    cb.addEventListener("change", () => spec.onChange && spec.onChange(cb.checked));
    wrap.append(label);
    container.append(wrap);
    return { el: wrap, get value() { return cb.checked; }, set(v) { cb.checked = v; return this; } };
  }

  function segmented(container, spec) {
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    const label = document.createElement("label");
    label.textContent = spec.label;
    label.style.marginBottom = "6px";
    const seg = document.createElement("div");
    seg.className = "seg";
    const buttons = [];
    spec.options.forEach((o) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = o.label || o;
      b.setAttribute("aria-pressed", String((o.value || o) === spec.value));
      b.addEventListener("click", () => {
        buttons.forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        spec.onChange && spec.onChange(o.value || o);
      });
      buttons.push(b); seg.append(b);
    });
    wrap.append(label, seg);
    container.append(wrap);
    return { el: wrap, buttons };
  }

  function buttons(container, specs) {
    const row = document.createElement("div");
    row.className = "btn-row";
    specs.forEach((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn btn-sm" + (s.primary ? " btn-primary" : "");
      b.textContent = s.label;
      b.addEventListener("click", () => s.onClick && s.onClick());
      row.append(b);
    });
    container.append(row);
    return { el: row };
  }

  function readout(container, entries) {
    const box = document.createElement("div");
    box.className = "readout";
    const cells = {};
    entries.forEach(([k, v]) => {
      const d = document.createElement("div");
      const a = document.createElement("span"); a.textContent = k;
      const b = document.createElement("b"); b.textContent = v == null ? "—" : String(v);
      d.append(a, b); box.append(d); cells[k] = b;
    });
    container.append(box);
    return {
      el: box,
      set(k, v) { if (cells[k]) cells[k].textContent = typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v); },
      update(obj) { Object.entries(obj).forEach(([k, v]) => this.set(k, v)); },
    };
  }

  /** standard lab scaffold: returns {fig, panel, note, controls} */
  function lab(root) {
    const stage = root.querySelector(".lab-stage");
    const panel = root.querySelector(".lab-panel");
    const note = root.querySelector(".lab-note");
    const canvas = document.createElement("canvas");
    stage.prepend(canvas);
    return { canvas, stage, panel, note };
  }

  /* --------------------------------------------------------------- export */
  window.Viz = {
    Fig, palette, theme, cssVar,
    clamp, lerp, linspace, mulberry32, gauss, heatColor, ticks,
    slider, select, toggle, segmented, buttons, readout, lab,
    make(canvas, opts) { return new Fig(canvas, opts); },
    makeIn(root, opts) {
      const stage = root.querySelector(".lab-stage");
      const canvas = document.createElement("canvas");
      stage.prepend(canvas);
      return new Fig(canvas, opts);
    },
    /** polyfill helper: linear regression fit */
    fitLine(pts) {
      const n = pts.length;
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      pts.forEach(([x, y]) => { sx += x; sy += y; sxx += x * x; sxy += x * y; });
      const denom = n * sxx - sx * sx;
      if (Math.abs(denom) < 1e-12) return { w: 0, b: sy / n };
      const w = (n * sxy - sx * sy) / denom;
      return { w, b: (sy - w * sx) / n };
    },
  };
})();
