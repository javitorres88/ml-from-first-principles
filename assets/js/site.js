/* ==========================================================================
   site.js — shared behaviour: theme, nav, TOC, math, code tabs, quizzes,
   flashcards, progress tracking, lab bootstrapping.
   ========================================================================== */
window.MLLabs = window.MLLabs || {};

(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const STORE = "mlfp.v1";

  /* ------------------------------------------------------------- storage */
  const store = {
    read() {
      try { return JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { return {}; }
    },
    write(o) {
      try { localStorage.setItem(STORE, JSON.stringify(o)); } catch (e) { /* ignore */ }
    },
    get(k, dflt) { const o = this.read(); return k in o ? o[k] : dflt; },
    set(k, v) { const o = this.read(); o[k] = v; this.write(o); },
    patch(k, obj) { const o = this.read(); o[k] = Object.assign({}, o[k] || {}, obj); this.write(o); },
  };

  /* --------------------------------------------------------------- theme */
  function initTheme() {
    const saved = store.get("theme", null);
    const sys = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const theme = saved || sys;
    document.documentElement.setAttribute("data-theme", theme);
    $$("[data-theme-toggle]").forEach((b) => {
      const sync = () => {
        b.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "☀" : "☾";
        b.setAttribute("aria-label", document.documentElement.getAttribute("data-theme") === "dark" ? "Switch to light theme" : "Switch to dark theme");
      };
      sync();
      b.addEventListener("click", () => {
        const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        store.set("theme", next);
        sync();
        document.dispatchEvent(new CustomEvent("mlfp:theme"));
      });
    });
  }

  /* ---------------------------------------------------------------- math */
  function renderMath() {
    if (typeof window.renderMathInElement !== "function") return;
    const targets = $$(".article, .math-scope");
    targets.forEach((el) => {
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\[", right: "\\]", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
          ],
          throwOnError: false,
          strict: false,
          trust: true,
        });
      } catch (e) { console.warn("math render failed", e); }
    });
  }

  /* --------------------------------------------------------- code blocks */
  const KEYWORDS = new Set(("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self match case").split(" "));
  const BUILTINS = new Set(("abs all any bool dict enumerate filter float format frozenset getattr hasattr int isinstance iter len list map max min next object open ord pow print property range repr reversed round set setattr slice sorted str sum super tuple type zip input np torch plt pd").split(" "));

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlightPython(src) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b|([+\-*/%=<>!&|^~:]+)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += escapeHtml(src.slice(last, m.index));
      last = re.lastIndex;
      if (m[1]) out += `<span class="tok-c">${escapeHtml(m[1])}</span>`;
      else if (m[2]) out += `<span class="tok-s">${escapeHtml(m[2])}</span>`;
      else if (m[3]) out += `<span class="tok-n">${escapeHtml(m[3])}</span>`;
      else if (m[4]) {
        const w = m[4];
        const cls = KEYWORDS.has(w) ? "tok-k" : BUILTINS.has(w) ? "tok-b" : (/^\s*\(/.test(src.slice(last)) ? "tok-f" : null);
        out += cls ? `<span class="${cls}">${escapeHtml(w)}</span>` : escapeHtml(w);
      } else out += `<span class="tok-o">${escapeHtml(m[5])}</span>`;
    }
    out += escapeHtml(src.slice(last));
    return out;
  }

  function initCodeBlocks() {
    $$(".codeblock").forEach((block) => {
      const panes = $$(".code-pane", block);
      const tabs = $$(".code-tabs button[data-pane]", block);
      panes.forEach((p) => {
        const code = p.querySelector("code");
        if (code && !code.dataset.done) {
          code.innerHTML = highlightPython(code.textContent);
          code.dataset.done = "1";
        }
      });
      tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          tabs.forEach((t) => t.setAttribute("aria-selected", "false"));
          tab.setAttribute("aria-selected", "true");
          panes.forEach((p) => p.classList.toggle("active", p.dataset.pane === tab.dataset.pane));
        });
      });
      const copy = block.querySelector(".copy");
      if (copy) {
        copy.addEventListener("click", async () => {
          const active = block.querySelector(".code-pane.active code") || block.querySelector(".code-pane code");
          if (!active) return;
          try {
            await navigator.clipboard.writeText(active.textContent);
            copy.textContent = "Copied ✓";
          } catch (e) { copy.textContent = "Press ⌘/Ctrl+C"; }
          setTimeout(() => (copy.textContent = "Copy"), 1600);
        });
      }
    });
  }

  /* --------------------------------------------------------------- quiz */
  function initQuiz(quiz) {
    const items = $$(".quiz-item", quiz);
    const state = { answered: 0, correct: 0 };
    const scoreEl = quiz.querySelector("[data-score]");
    const id = quiz.dataset.quiz;

    items.forEach((item, idx) => {
      const answer = item.dataset.answer;
      const opts = $$(".opt", item);
      const explain = item.querySelector(".explain");
      opts.forEach((opt) => {
        opt.addEventListener("click", () => {
          if (item.dataset.done) return;
          item.dataset.done = "1";
          const pick = opt.dataset.key;
          const right = pick === answer;
          opts.forEach((o) => {
            o.disabled = true;
            if (o.dataset.key === answer) o.classList.add("correct");
            else if (o === opt) o.classList.add("wrong");
          });
          if (explain) explain.classList.add("show");
          state.answered++;
          if (right) state.correct++;
          updateScore();
          if (id) store.patch("quiz", { [id]: { answered: state.answered, correct: state.correct, total: items.length } });
        });
      });
    });

    function updateScore() {
      if (!scoreEl) return;
      scoreEl.innerHTML = `<span>Score</span><b>${state.correct} / ${items.length}</b><span class="muted">${state.answered === items.length ? (state.correct === items.length ? "Perfect — you can explain this to someone else." : "Review the explanations above, then retry the ones you missed.") : `${items.length - state.answered} remaining`}</span>`;
    }
    updateScore();

    const reset = quiz.querySelector("[data-reset]");
    if (reset) {
      reset.addEventListener("click", () => {
        items.forEach((item) => {
          delete item.dataset.done;
          $$(".opt", item).forEach((o) => { o.disabled = false; o.classList.remove("correct", "wrong"); });
          const ex = item.querySelector(".explain");
          if (ex) ex.classList.remove("show");
        });
        state.answered = 0; state.correct = 0;
        updateScore();
      });
    }
  }

  /* --------------------------------------------------------- flashcards */
  function initDeck(deck) {
    const cards = $$(".flashcard", deck);
    if (!cards.length) return;
    let i = 0;
    const order = cards.map((_, k) => k);
    const prog = deck.querySelector("[data-deck-progress]");
    const bar = deck.querySelector("[data-deck-bar]");

    cards.forEach((c) => {
      c.setAttribute("tabindex", "0");
      c.setAttribute("role", "button");
      c.setAttribute("aria-label", "Flashcard — activate to flip");
      const flip = () => c.classList.toggle("flipped");
      c.addEventListener("click", flip);
      c.addEventListener("keydown", (e) => {
        if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); }
        if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
        if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
      });
    });

    function show() {
      cards.forEach((c, k) => { c.style.display = order[i] === k ? "" : "none"; c.classList.remove("flipped"); });
      if (prog) prog.textContent = `${i + 1} / ${cards.length}`;
      if (bar) bar.style.width = `${((i + 1) / cards.length) * 100}%`;
    }
    function go(d) { i = (i + d + cards.length) % cards.length; show(); }

    deck.querySelector("[data-prev]")?.addEventListener("click", () => go(-1));
    deck.querySelector("[data-next]")?.addEventListener("click", () => go(1));
    deck.querySelector("[data-shuffle]")?.addEventListener("click", () => {
      for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(Math.random() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
      i = 0; show();
    });
    show();
  }

  /* ----------------------------------------------------------- exercises */
  function initSolutions() {
    $$("[data-reveal]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = document.getElementById(btn.dataset.reveal);
        if (!target) return;
        const open = target.hasAttribute("hidden");
        if (open) target.removeAttribute("hidden"); else target.setAttribute("hidden", "");
        btn.textContent = open ? "Hide solution" : "Show solution";
      });
    });
  }

  /* ------------------------------------------------------------ progress */
  function initProgress() {
    const btn = $("[data-mark-complete]");
    const key = document.body.dataset.module;
    const cardSel = `[data-progress-for="${key}"]`;
    const paint = () => {
      const done = store.get("done", {});
      if (btn) {
        const isDone = !!done[key];
        btn.textContent = isDone ? "✓ Completed" : "Mark module complete";
        btn.classList.toggle("btn-primary", !isDone);
      }
      $$(cardSel).forEach((c) => c.classList.toggle("done", !!done[key]));
    };
    if (btn && key) {
      btn.addEventListener("click", () => {
        const done = store.get("done", {});
        done[key] = !done[key];
        store.set("done", done);
        paint();
      });
    }
    // landing page progress bar
    const bar = $("[data-overall-progress]");
    if (bar) {
      const total = +bar.dataset.overallProgress;
      const done = store.get("done", {});
      const n = Object.values(done).filter(Boolean).length;
      bar.style.width = `${Math.min(100, (n / total) * 100)}%`;
      const label = $("[data-overall-label]");
      if (label) label.textContent = `${n} of ${total} modules complete`;
    }
    paint();
  }

  /* ---------------------------------------------------------------- TOC */
  function initTOC() {
    const links = $$(".rail a[href^='#']");
    if (!links.length) return;
    const map = new Map();
    links.forEach((a) => {
      const el = document.getElementById(a.getAttribute("href").slice(1));
      if (el) map.set(el, a);
    });
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          const a = map.get(e.target);
          if (a) a.classList.add("active");
        }
      });
    }, { rootMargin: "-84px 0px -70% 0px", threshold: 0 });
    map.forEach((_, el) => obs.observe(el));
  }

  /* ------------------------------------------------------------- nav bar */
  function initNav() {
    const toggle = $("#navToggle");
    const menu = $("#navMenu");
    if (toggle && menu) {
      toggle.addEventListener("click", () => {
        const open = menu.hasAttribute("hidden");
        if (open) menu.removeAttribute("hidden"); else menu.setAttribute("hidden", "");
        toggle.setAttribute("aria-expanded", String(open));
      });
    }
  }

  /* ---------------------------------------------------------- lab boot */
  function initLabs() {
    $$("[data-lab]").forEach((root) => {
      const id = root.dataset.lab;
      const fn = window.MLLabs[id];
      if (typeof fn === "function") {
        try { fn(root); } catch (e) { console.error("lab failed:", id, e); }
      } else {
        console.warn("no lab registered for", id);
      }
    });
  }

  /* --------------------------------------------------------------- boot */
  let booted = false;
  function boot() {
    if (booted) return;
    booted = true;
    initTheme();
    initNav();
    renderMath();
    initCodeBlocks();
    $$(".quiz").forEach(initQuiz);
    $$(".deck").forEach(initDeck);
    initSolutions();
    initProgress();
    initTOC();
    initLabs();
  }

  // Deferred scripts run while readyState is "interactive" and DOMContentLoaded
  // fires only after the last of them, so lab scripts are always registered by then.
  if (document.readyState === "complete") boot();
  else {
    document.addEventListener("DOMContentLoaded", boot);
    window.addEventListener("load", boot);
  }

  window.MLFP = { store, highlightPython };
})();
