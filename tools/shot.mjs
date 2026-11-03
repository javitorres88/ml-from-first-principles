/* Screenshot helper: node tools/shot.mjs <url> <outPrefix> [segments] [width] */
import puppeteer from "puppeteer-core";

const [, , url, out, segsArg, widthArg] = process.argv;
const segments = Number(segsArg || 3);
const width = Number(widthArg || 1280);
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--hide-scrollbars", "--disable-gpu", "--force-device-scale-factor=1"],
});
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.type() + ": " + m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
await page.setViewport({ width, height: 1000 });
await page.goto(url, { waitUntil: "networkidle0", timeout: 60000 });
await new Promise((r) => setTimeout(r, 900));

const height = await page.evaluate(() => document.documentElement.scrollHeight);
const segH = Math.ceil(height / segments);
for (let i = 0; i < segments; i++) {
  await page.evaluate((y) => window.scrollTo(0, y), i * segH);
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: `${out}-${i + 1}.png` });
}
console.log(`page height ${height}px, ${segments} screenshots -> ${out}-*.png`);
if (errors.length) console.log("CONSOLE:\n" + errors.slice(0, 25).join("\n"));
await browser.close();
