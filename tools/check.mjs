/* Repository self-check: build, then verify the generated site is internally consistent.
 *
 *   node tools/check.mjs
 *
 * Exits non-zero (and prints a summary) if anything is wrong. Used by `npm run check` and CI.
 */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { modules, sectionOrder } from "./modules.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");
const problems = [];
const fail = (msg) => problems.push(msg);
const rel = (p) => path.relative(ROOT, p).replaceAll("\\", "/");
const exists = (p) => fs.existsSync(path.join(ROOT, p));

/* ---------------------------------------------------------------- 1. build */
try {
  const out = execFileSync(process.execPath, [path.join(ROOT, "tools", "build.mjs")], {
    cwd: ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  process.stdout.write(out);
  if (/missing content/i.test(out)) fail("build reported missing module content");
} catch (err) {
  fail(`build failed: ${err.stderr || err.message}`);
}

/* ------------------------------------------------- 2. scripts parse (ESM ok) */
const scriptFiles = [
  ...fs.readdirSync(path.join(ROOT, "assets", "js")).filter((f) => f.endsWith(".js")).map((f) => `assets/js/${f}`),
  ...fs
    .readdirSync(path.join(ROOT, "assets", "js", "labs"))
    .filter((f) => f.endsWith(".js"))
    .map((f) => `assets/js/labs/${f}`),
  ...fs.readdirSync(path.join(ROOT, "tools")).filter((f) => f.endsWith(".mjs")).map((f) => `tools/${f}`),
];
for (const file of scriptFiles) {
  try {
    execFileSync(process.execPath, ["--check", path.join(ROOT, file)], { stdio: "pipe" });
  } catch (err) {
    fail(`syntax error in ${file}\n${String(err.stderr || "").trim()}`);
  }
}

/* ------------------------------------------- 3. registry ↔ source ↔ assets */
for (const mod of modules) {
  const partial = `content/modules/${mod.num}-${mod.id}.html`;
  if (!exists(partial)) fail(`missing partial ${partial}`);
  const lab = `assets/js/labs/${mod.num}-${mod.id}.js`;
  if (!exists(lab)) fail(`missing lab script ${lab}`);
  const page = `modules/${mod.id}.html`;
  if (!exists(page)) fail(`missing generated page ${page}`);
}
if (fs.readdirSync(path.join(ROOT, "projects"), { withFileTypes: true }).filter((d) => d.isDirectory()).length !== modules.length) {
  fail(`projects/ should hold one folder per module (${modules.length})`);
}

/* ------------------------------------------ 4. sections present and in order */
const pageOf = (mod) => fs.readFileSync(path.join(ROOT, "modules", `${mod.id}.html`), "utf8");
for (const mod of modules) {
  if (!exists(`modules/${mod.id}.html`)) continue;
  const html = pageOf(mod);
  let cursor = -1;
  for (const section of sectionOrder) {
    const at = html.indexOf(`id="${section.id}"`);
    if (at === -1) {
      fail(`${mod.id}: missing section #${section.id}`);
      continue;
    }
    if (at < cursor) fail(`${mod.id}: section #${section.id} is out of order`);
    cursor = at;
  }
  // every lab declared in the registry must have a matching container in the page
  for (const lab of mod.labs) {
    if (!html.includes(`data-lab="${lab}"`)) fail(`${mod.id}: lab container data-lab="${lab}" not found`);
  }
  // every math block must be balanced enough for KaTeX auto-render to find it
  const dollars = (html.match(/(?<!\\)\$/g) || []).length;
  if (dollars % 2 !== 0) fail(`${mod.id}: odd number of $ delimiters (${dollars}) — KaTeX will leave one unrendered`);
}

/* ---------------------------------------------- 5. local links and assets */
const htmlFiles = ["index.html", ...modules.map((m) => `modules/${m.id}.html`)];
for (const file of htmlFiles) {
  if (!exists(file)) {
    fail(`missing page ${file}`);
    continue;
  }
  const html = fs.readFileSync(path.join(ROOT, file), "utf8");
  const refs = new Set();
  for (const m of html.matchAll(/(?:href|src)="([^"]+)"/g)) refs.add(m[1]);
  for (const ref of refs) {
    if (/^(https?:|mailto:|data:|#)/.test(ref)) continue;
    const [clean] = ref.split("#");
    if (!clean) continue;
    const target = path.resolve(path.dirname(path.join(ROOT, file)), clean);
    if (!fs.existsSync(target)) fail(`${file}: broken link → ${ref}`);
  }
}

/* --------------------------------------------------- 6. projects + python */
for (const dir of fs.readdirSync(path.join(ROOT, "projects"), { withFileTypes: true })) {
  if (!dir.isDirectory()) continue;
  for (const need of ["README.md", "starter.py"]) {
    if (!exists(`projects/${dir.name}/${need}`)) fail(`projects/${dir.name}: missing ${need}`);
  }
  const brief = fs.readFileSync(path.join(ROOT, "projects", dir.name, "README.md"), "utf8");
  for (const m of brief.matchAll(/\]\((\.\.\/[^)]+)\)/g)) {
    const target = path.resolve(path.join(ROOT, "projects", dir.name), m[1]);
    if (!fs.existsSync(target)) fail(`projects/${dir.name}/README.md: broken link → ${m[1]}`);
  }
}
for (const file of ["python/README.md", "python/pyproject.toml", "docs/site-map.md", "docs/instructor-notes.md"]) {
  if (!exists(file)) fail(`missing ${file}`);
}

/* --------------------------------------------------------------- summary */
if (problems.length) {
  console.error(`\n✗ ${problems.length} problem(s):`);
  for (const p of problems) console.error(`  • ${p}`);
  process.exit(1);
}
console.log(`\n✓ checked ${modules.length} modules, ${scriptFiles.length} scripts, ${htmlFiles.length} pages — all good`);
