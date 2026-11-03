/* Browser QA for the whole site: console errors, KaTeX rendering, and whether every interactive widget works.
 *
 * Requires a local Chrome and a running static server:
 *
 *   python -m http.server 8000        # or: npm run serve
 *   node tools/qa.mjs                 # or: npm run qa
 *   node tools/qa.mjs http://127.0.0.1:8000 backprop   # one page
 *
 * Exits non-zero if any page has a problem, so it can gate a release.
 */
import puppeteer from "puppeteer-core";

const CHROME =
  process.env.CHROME_PATH ||
  (process.platform === "win32"
    ? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    : process.platform === "darwin"
      ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
      : "/usr/bin/google-chrome");

const BASE = (process.argv[2] || "http://127.0.0.1:8000").replace(/\/$/, "");
const filter = process.argv[3] || "";

const listHtml = await (await fetch(`${BASE}/index.html`)).text();
const pages = [...new Set([...listHtml.matchAll(/href="modules\/([^"]+\.html)"/g)].map((m) => `modules/${m[1]}`))];
const targets = ["index.html", ...pages].filter((p) => p.includes(filter));

const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--disable-gpu"] });
const ctx = await browser.createBrowserContext();
const rows = [];

for (const target of targets) {
  const page = await ctx.newPage();
  const errs = [];
  page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errs.push(`console: ${m.text()}`);
  });
  page.on("requestfailed", (r) => errs.push(`request failed: ${r.url()}`));
  await page.setViewport({ width: 1280, height: 900 });
  await page.goto(`${BASE}/${target}`, { waitUntil: "networkidle0", timeout: 60000 });
  await new Promise((r) => setTimeout(r, 600));

  const row = { page: target, errs, issues: [] };

  /* rendering */
  const render = await page.evaluate(() => ({
    katexErrors: document.querySelectorAll(".katex-error").length,
    unrendered: (document.body.innerText.match(/(^|\s)\$[^$\n]{2,60}\$/g) || []).length,
    labs: document.querySelectorAll("[data-lab]").length,
    canvases: document.querySelectorAll("[data-lab] canvas").length,
  }));
  if (render.katexErrors) row.issues.push(`${render.katexErrors} KaTeX error node(s)`);
  if (render.unrendered) row.issues.push(`${render.unrendered} unrendered $…$ span(s)`);
  if (render.labs && render.canvases < render.labs) row.issues.push(`${render.labs} labs but only ${render.canvases} canvas(es)`);

  /* code tabs */
  const tabIssues = await page.evaluate(async () => {
    const issues = [];
    for (const block of document.querySelectorAll(".codeblock")) {
      for (const tab of block.querySelectorAll(".code-tabs button[data-pane]")) {
        tab.click();
        await new Promise((r) => setTimeout(r, 30));
        const pane = block.querySelector(`.code-pane[data-pane="${tab.dataset.pane}"]`);
        if (!pane || !pane.classList.contains("active")) issues.push(`pane ${tab.dataset.pane} did not activate`);
      }
    }
    return issues;
  });
  row.issues.push(...tabIssues);

  /* quiz: answer every question correctly, expect full marks and every explanation */
  row.quiz = await page.evaluate(async () => {
    const quiz = document.querySelector("[data-quiz]");
    if (!quiz) return null;
    const items = [...quiz.querySelectorAll(".quiz-item")];
    for (const item of items) {
      const right = item.querySelector(`.opt[data-key="${item.dataset.answer}"]`);
      (right || item.querySelector(".opt")).click();
    }
    await new Promise((r) => setTimeout(r, 120));
    const score = quiz.querySelector("[data-score]")?.textContent.replace(/\s+/g, " ").trim();
    const explains = quiz.querySelectorAll(".explain.show").length;
    quiz.querySelector("[data-reset]")?.click();
    await new Promise((r) => setTimeout(r, 120));
    const reset = quiz.querySelector("[data-score]")?.textContent.replace(/\s+/g, " ").trim();
    return { questions: items.length, score, explains, reset };
  });
  if (row.quiz) {
    if (row.quiz.explains !== row.quiz.questions)
      row.issues.push(`quiz: ${row.quiz.explains}/${row.quiz.questions} explanations shown`);
    if (!/^Score ?\d+ \/ \d+/.test(row.quiz.score || "")) row.issues.push(`quiz score missing: ${row.quiz.score}`);
    else if (!row.quiz.score.includes(`${row.quiz.questions} / ${row.quiz.questions}`))
      row.issues.push(`quiz: perfect run did not score full marks (${row.quiz.score})`);
  }

  /* flashcards */
  row.deck = await page.evaluate(async () => {
    const cards = [...document.querySelectorAll(".flashcard")];
    if (!cards.length) return null;
    const deck = cards[0].closest("section, div");
    cards[0].click();
    await new Promise((r) => setTimeout(r, 60));
    const flipped = cards[0].classList.contains("flipped");
    deck?.querySelector("[data-next]")?.click();
    await new Promise((r) => setTimeout(r, 60));
    return { cards: cards.length, flipped, progress: deck?.querySelector("[data-deck-progress]")?.textContent || null };
  });
  if (row.deck && !row.deck.flipped) row.issues.push("flashcard did not flip on click");

  /* worked solutions */
  row.solutions = await page.evaluate(async () => {
    const btns = [...document.querySelectorAll("[data-reveal]")];
    let opened = 0;
    for (const b of btns) {
      const t = document.getElementById(b.dataset.reveal);
      b.click();
      await new Promise((r) => setTimeout(r, 25));
      if (t && !t.hasAttribute("hidden")) opened++;
      b.click();
    }
    return { buttons: btns.length, opened };
  });
  if (row.solutions.opened !== row.solutions.buttons)
    row.issues.push(`solutions: ${row.solutions.opened}/${row.solutions.buttons} opened`);

  /* progress persistence */
  const mark = await page.$("[data-mark-complete]");
  if (mark) {
    await mark.click();
    await new Promise((r) => setTimeout(r, 150));
    row.stored = await page.evaluate(() => (localStorage.getItem("mlfp.v1") || "").slice(0, 120));
    if (!row.stored) row.issues.push("progress not written to localStorage");
  }

  /* labs: canvases must respond to sliders and buttons */
  row.labs = await page.evaluate(async () => {
    const out = [];
    for (const lab of document.querySelectorAll("[data-lab]")) {
      const canvases = [...lab.querySelectorAll("canvas")];
      if (!canvases.length) {
        out.push({ lab: lab.dataset.lab, error: "no canvas" });
        continue;
      }
      const before = canvases.map((c) => c.toDataURL());
      const sliders = [...lab.querySelectorAll('input[type="range"]')];
      const buttons = [...lab.querySelectorAll("button")].filter((b) => !/reset|shuffle|show|hide|copy/i.test(b.textContent || ""));
      const seen = new Set();
      let repainted = 0;
      const tick = async () => {
        await new Promise((r) => setTimeout(r, 260));
        const now = canvases.map((c) => c.toDataURL());
        if (now.some((u, i) => u !== before[i])) repainted++;
      };
      for (const s of sliders) {
        const lo = Number(s.min || 0);
        const hi = Number(s.max || 1);
        for (const v of [lo, (lo + hi) / 2, hi]) {
          s.value = String(v);
          s.dispatchEvent(new Event("input", { bubbles: true }));
          s.dispatchEvent(new Event("change", { bubbles: true }));
          await tick();
        }
      }
      for (const b of buttons.slice(0, 3)) {
        const label = (b.textContent || "").trim();
        if (seen.has(label)) continue;
        seen.add(label);
        b.click();
        await tick();
        await tick();
      }
      out.push({ lab: lab.dataset.lab, canvases: canvases.length, sliders: sliders.length, buttons: buttons.length, repainted });
    }
    return out;
  });
  for (const l of row.labs) {
    if (l.error) row.issues.push(`${l.lab}: ${l.error}`);
    else if (!l.repainted)
      row.issues.push(`${l.lab}: canvas never changed under interaction (${l.sliders} sliders, ${l.buttons} buttons)`);
  }

  row.issues.push(...errs);
  rows.push(row);
  await page.close();
}
await browser.close();

let bad = 0;
for (const r of rows) {
  const bits = [];
  if (r.quiz) bits.push(`quiz ${r.quiz.questions}q`);
  if (r.deck) bits.push(`${r.deck.cards} cards`);
  if (r.labs?.length) bits.push(`${r.labs.length} labs`);
  if (r.issues.length) {
    bad++;
    console.log(`✗ ${r.page}  ${bits.join(" · ")}`);
    for (const i of r.issues) console.log(`    ${i}`);
  } else {
    console.log(`✓ ${r.page}  ${bits.join(" · ")}`);
  }
}
console.log(bad ? `\n✗ ${bad}/${rows.length} pages with problems` : `\n✓ all ${rows.length} pages clean`);
process.exit(bad ? 1 : 0);
