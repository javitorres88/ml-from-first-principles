/* Element screenshots: node tools/shotel.mjs <url> <selector> <out.png> [width] */
import puppeteer from "puppeteer-core";
const [, , url, selector, out, widthArg] = process.argv;
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--hide-scrollbars", "--disable-gpu", "--force-device-scale-factor=1"],
});
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
await page.setViewport({ width: Number(widthArg || 1280), height: 1000 });
await page.goto(url, { waitUntil: "networkidle0", timeout: 60000 });
await new Promise((r) => setTimeout(r, 800));
const els = await page.$$(selector);
if (!els.length) console.log("NO MATCH for", selector);
let i = 0;
for (const el of els) {
  await el.scrollIntoView();
  await new Promise((r) => setTimeout(r, 350));
  const name = els.length > 1 ? out.replace(/\.png$/, `-${++i}.png`) : out;
  await el.screenshot({ path: name });
  console.log("wrote", name);
}
if (errors.length) console.log("CONSOLE ERRORS:\n" + errors.slice(0, 20).join("\n"));
await browser.close();
