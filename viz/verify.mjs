// Drive the viewer in headless Chrome: play each replay, collect console errors, run the arrival
// check inside the page, and capture screenshots. Run from viz/ with the server already up:
//
//   python3 -m http.server 8099 &
//   node verify.mjs http://localhost:8099
//
// Needs puppeteer-core and a Chrome binary; it asserts nothing about the simulation, it only reports
// what the page says about itself.
import { mkdirSync } from "node:fs";
import puppeteer from "puppeteer-core";

const BASE = process.argv[2] || "http://localhost:8099";
const CHROME = process.env.CHROME || "/usr/bin/google-chrome";
const OUT = "shots";
mkdirSync(OUT, { recursive: true });

const MIDDAY = 120 + 240;        // day 1, 12:00
const NIGHT = 230 + 240;         // day 1, 23:00

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--disable-gpu", "--window-size=1600,1000",
         "--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
  defaultViewport: { width: 1600, height: 1000 },
});

const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("requestfailed", (r) => errors.push("requestfailed: " + r.url()));

await page.goto(BASE + "/index.html", { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__ready === true && window.__hasReplay && window.__hasReplay(),
                           { timeout: 60000 });
// the error panel is the viewer's own report of a failed asset
const panel = await page.evaluate(() => {
  const el = document.getElementById("err");
  return el && el.style.display === "block" ? el.textContent : "";
});
if (panel) errors.push("error panel: " + panel.trim());

const ids = await page.evaluate(() =>
  [...document.querySelectorAll("#replaySel option")].map((o) => o.value));

const report = [];
for (const id of ids) {
  const before = errors.length;
  await page.evaluate(async (v) => {
    document.getElementById("replaySel").value = v;
    await window.__load(v);                      // awaited, so the check never races the fetch
  }, id);
  await page.waitForFunction(() => window.__hasReplay(), { timeout: 30000 });
  await new Promise((r) => setTimeout(r, 400));

  // run the page's own arrival check over every journey of the replay
  const check = await page.evaluate(() => window.arrivalCheck());

  // play to the end at 20x, then confirm the final tick was reached
  const played = await page.evaluate(async () => {
    document.getElementById("speed").value = "20";
    document.getElementById("speed").dispatchEvent(new Event("input"));
    document.getElementById("scrub").value = "0";
    document.getElementById("scrub").dispatchEvent(new Event("input"));
    document.getElementById("playBtn").click();
    const maxT = window.__maxT ? window.__maxT() : 1439;
    const t0 = Date.now();
    while (Date.now() - t0 < 120000) {
      await new Promise((r) => setTimeout(r, 250));
      if (window.__tick && window.__tick() >= maxT) break;
    }
    return { reached: window.__tick ? window.__tick() : null, maxT };
  });

  for (const [name, t] of [["midday", MIDDAY], ["night", NIGHT]]) {
    await page.evaluate((tt) => {
      const s = document.getElementById("scrub");
      s.value = String(tt); s.dispatchEvent(new Event("input"));
    }, t);
    await new Promise((r) => setTimeout(r, 700));
    await page.screenshot({ path: `${OUT}/${id}_${name}.png` });
  }

  report.push({ id, check, played, newErrors: errors.slice(before) });
  console.log(`${id}: journeys ${check.journeys}, mismatches ${check.mismatches}, ` +
              `wrongLength ${check.wrongLength}, censored ${check.censoredAtEnd}, ` +
              `played to ${played.reached}/${played.maxT}, new errors ${errors.length - before}`);
}

console.log("\ntotal console errors: " + errors.length);
for (const e of errors.slice(0, 15)) console.log("  " + e);
await browser.close();
console.log("\n" + JSON.stringify(report.map(r => ({ id: r.id, ...r.check, examples: undefined })), null, 1));
