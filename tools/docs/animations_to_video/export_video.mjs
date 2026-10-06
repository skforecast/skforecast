/*
 * Export the documentation animations (docs/animations/*.html) to MP4 videos,
 * for example to insert them in a presentation.
 *
 * Every frame is rendered at its exact time with the export mode of the
 * animation engine (?export=1 exposes window.skfRender(t)), captured with the
 * local Google Chrome (puppeteer-core) and encoded to H.264 with ffmpeg
 * (ffmpeg-static). Nothing runs in real time, so the result does not depend on
 * the speed of the machine.
 *
 * Usage (from tools/docs/animations_to_video, after `npm install`):
 *
 *   node export_video.mjs                          # all animations, light and dark
 *   node export_video.mjs global-forecasting       # one animation
 *   node export_video.mjs --theme light --fps 30 --scale 2 --out videos
 *   node export_video.mjs --src ../../../../other-repo/docs/animations name
 *
 * Set CHROME_PATH if Chrome is not installed in its default location.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import ffmpegPath from 'ffmpeg-static';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/* ------------------------------------------------------------- arguments */
const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv.splice(i, 2)[1] : fallback;
};
const theme = opt('theme', 'both');
const fps = Number(opt('fps', 30));
const scale = Number(opt('scale', 2));
const outDir = path.resolve(HERE, opt('out', 'videos'));
const ANIM_DIR = path.resolve(HERE, opt('src', '../../../docs/animations'));
if (!fs.existsSync(ANIM_DIR)) {
  console.error(`Animations folder not found: ${ANIM_DIR}`);
  process.exit(1);
}
const themes = theme === 'both' ? ['light', 'dark'] : [theme];
const names = argv.length ? argv : fs.readdirSync(ANIM_DIR)
  .filter(f => f.endsWith('.html'))
  .map(f => f.replace(/\.html$/, ''));

/* ---------------------------------------------------------------- chrome */
const CHROME = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
].find(p => p && fs.existsSync(p));
if (!CHROME) {
  console.error('Google Chrome not found. Set CHROME_PATH to its executable.');
  process.exit(1);
}

/* ------------------------------------------------ static server for the pages */
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
               '.woff': 'font/woff', '.woff2': 'font/woff2', '.png': 'image/png', '.svg': 'image/svg+xml'};
const server = http.createServer((req, res) => {
  const file = path.join(ANIM_DIR, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(ANIM_DIR) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, {'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream'});
  fs.createReadStream(file).pipe(res);
});
server.listen(0);
await once(server, 'listening');
const base = `http://localhost:${server.address().port}`;

/* ---------------------------------------------------------------- export */
fs.mkdirSync(outDir, {recursive: true});
const browser = await puppeteer.launch({executablePath: CHROME, headless: true});

try {
  for (const name of names) {
    for (const th of themes) {
      const page = await browser.newPage();
      await page.setViewport({width: 1280, height: 720, deviceScaleFactor: scale});
      await page.goto(`${base}/${name}.html?export=1&theme=${th}`, {waitUntil: 'networkidle0'});
      await page.waitForFunction('window.skfReady === true');
      const {duration, height} = await page.evaluate(() => ({
        duration: window.skfDuration,
        height: document.getElementById('skf-stage').viewBox.baseVal.height
      }));
      await page.setViewport({width: 1280, height, deviceScaleFactor: scale});

      const out = path.join(outDir, `${name}-${th}.mp4`);
      const ffmpeg = spawn(ffmpegPath, [
        '-y', '-loglevel', 'error',
        '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', out
      ], {stdio: ['pipe', 'inherit', 'inherit']});

      const frames = Math.round(duration*fps);
      for (let i = 0; i < frames; i++) {
        // render the frame and wait until it is painted
        await page.evaluate(t => {
          window.skfRender(t);
          return new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        }, i/fps);
        const png = await page.screenshot({type: 'png', clip: {x: 0, y: 0, width: 1280, height}});
        if (!ffmpeg.stdin.write(png)) await once(ffmpeg.stdin, 'drain');
        if (i % fps === 0) process.stdout.write(`\r${name} (${th}): ${Math.round(100*i/frames)}%`);
      }
      ffmpeg.stdin.end();
      const [code] = await once(ffmpeg, 'close');
      if (code !== 0) throw new Error(`ffmpeg failed for ${out}`);
      console.log(`\r${name} (${th}): ${path.relative(process.cwd(), out)}`);
      await page.close();
    }
  }
} finally {
  await browser.close();
  server.close();
}
