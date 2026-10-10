/*
 * Render the social preview card of the documentation home page.
 *
 * 1. Opens the home page served by `mkdocs serve`, pauses the animation on its
 *    last step (backtesting and comparison) and takes a screenshot of it.
 * 2. Saves that screenshot as images/skforecast-backtesting-comparison.png,
 *    the image shown in the GitHub README.
 * 3. Renders tools/docs/home_page/social_card.html with that screenshot and saves
 *    it as docs/img/social-card-home.png (1200 x 630 px), the image that
 *    docs/overrides/home.html declares in its Open Graph tags.
 *
 * Needs Node.js 22 or newer (built-in WebSocket) and Google Chrome.
 *
 * Usage (from the repository root, with `mkdocs serve` running):
 *     node tools/docs/home_page/make_social_card.mjs [http://127.0.0.1:8000/]
 *
 * Set CHROME_PATH if Chrome is not in the default macOS location.
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HOME_URL = process.argv[2] || "http://127.0.0.1:8000/";
const CHROME =
  process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9339;
const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..", "..");
const OUTPUT = join(ROOT, "docs", "img", "social-card-home.png");
const README_IMAGE = join(ROOT, "images", "skforecast-backtesting-comparison.png");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), "sk-card-"));
const chrome = spawn(CHROME, [
  "--headless=new",
  "--disable-gpu",
  "--hide-scrollbars",
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  "about:blank",
], { stdio: "ignore" });

async function openPage() {
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" });
      return await res.json();
    } catch {
      await sleep(250);
    }
  }
  throw new Error("Chrome did not start");
}

async function connect() {
  const target = await openPage();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  });
  const send = (method, params = {}) =>
    new Promise((r) => {
      const i = ++id;
      pending.set(i, r);
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  const evaluate = async (expression) =>
    (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }))
      .result?.result?.value;
  return { ws, send, evaluate };
}

try {
  // 1. Screenshot of the animation, paused on its last step
  const home = await connect();
  await home.send("Page.enable");
  await home.send("Emulation.setDeviceMetricsOverride", {
    width: 1260, height: 1000, deviceScaleFactor: 2, mobile: false,
  });
  await home.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-color-scheme", value: "light" }],
  });
  await home.send("Page.navigate", { url: HOME_URL });
  await sleep(4000);
  const rect = await home.evaluate(`(() => {
    document.querySelectorAll(".md-consent, .md-banner").forEach((e) => e.remove());
    const play = document.getElementById("play");
    if (play.getAttribute("aria-label") === "Pause animation") play.click();
    document.querySelector('.step[data-step="2"]').click();
    const stage = document.getElementById("stage");
    // Square corners, so the screenshot has no page background in them
    stage.style.borderRadius = "0";
    window.scrollTo(0, 0);
    // Page coordinates, as expected by Page.captureScreenshot
    const top = stage.querySelector(".stage-top").getBoundingClientRect();
    const body = stage.querySelector(".stage-body").getBoundingClientRect();
    return {
      x: top.left + window.scrollX,
      y: top.top + window.scrollY,
      width: top.width,
      height: body.bottom - top.top,
    };
  })()`);
  if (!rect) throw new Error(`No animation found at ${HOME_URL}`);
  await sleep(800);
  const stageShot = await home.send("Page.captureScreenshot", {
    format: "png",
    clip: { ...rect, scale: 1 },
    captureBeyondViewport: true,
  });
  const stageFile = join(profile, "stage.png");
  writeFileSync(stageFile, Buffer.from(stageShot.result.data, "base64"));
  writeFileSync(README_IMAGE, Buffer.from(stageShot.result.data, "base64"));
  console.log(`Written ${README_IMAGE}`);
  home.ws.close();

  // 2. Card
  const card = await connect();
  await card.send("Page.enable");
  await card.send("Emulation.setDeviceMetricsOverride", {
    width: 1200, height: 630, deviceScaleFactor: 1, mobile: false,
  });
  const cardUrl = pathToFileURL(join(HERE, "social_card.html"));
  await card.send("Page.navigate", { url: cardUrl.href });
  await sleep(2500);
  // The card reads nothing from its URL: the screenshot is set from here
  const stageLoaded = await card.evaluate(`new Promise((done) => {
    const img = document.getElementById("stage");
    img.onload = () => done(true);
    img.onerror = () => done(false);
    img.src = ${JSON.stringify(pathToFileURL(stageFile).href)};
  })`);
  if (!stageLoaded) throw new Error(`Could not load ${stageFile} in the card`);
  await card.evaluate("document.fonts.ready.then(() => true)");
  const shot = await card.send("Page.captureScreenshot", {
    format: "png",
    clip: { x: 0, y: 0, width: 1200, height: 630, scale: 1 },
  });
  writeFileSync(OUTPUT, Buffer.from(shot.result.data, "base64"));
  card.ws.close();
  console.log(`Written ${OUTPUT}`);
} finally {
  chrome.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
