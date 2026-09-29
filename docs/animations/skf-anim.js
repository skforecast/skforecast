/*!
 * Copyright (c) 2021-present Joaquín Amat Rodrigo and Javier Escobar Ortiz.
 * Part of the skforecast documentation, licensed under CC BY-NC-SA 4.0
 * (https://creativecommons.org/licenses/by-nc-sa/4.0/). It is not covered by the
 * BSD-3-Clause license of the skforecast software.
 * The skforecast trademark and logo are registered with the EUIPO (application
 * number 019109684) and may not be used without permission.
 */
/* Shared engine for the skforecast documentation animations.

   An animation is a pure function of time: `render(t, out)` pushes SVG strings
   for the frame at `t` seconds. The engine builds the stage and the controls,
   resolves the theme (light/dark, synced with the MkDocs Material palette of
   the parent page), runs the playback loop and exposes an export mode.

   Usage (inside an animation page):

     SkfAnim.run({
       title: 'Global forecasting models',
       desc: 'Accessible description of the whole animation.',
       total: 36,        // duration in seconds (loops)
       height: 640,      // stage height (width is always 1280), default 720
       still: 34,        // frame shown when the user prefers reduced motion
       fadeOut: [35.2, 36],
       render(t, o) { SkfAnim.header(t, o, {title, captions, steps}); ... }
     });

   URL parameters: ?theme=light|dark forces the theme, ?t=<seconds> shows a
   still frame, ?export=1 hides the controls and exposes window.skfRender(t),
   window.skfDuration and window.skfReady for frame-by-frame capture.

   Embedding (see docs/stylesheets/extra.css): the iframe height is computed
   from its width with --skf-ratio (height/width of the stage, default 0.5). */
(function(){
"use strict";

/* ----------------------------------------------------------------- math */
const cl   = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = p => p < .5 ? 4*p*p*p : 1 - Math.pow(-2*p + 2, 3)/2;
const lin  = (t, a, b) => cl((t - a)/(b - a));
const P    = (t, a, b) => ease(lin(t, a, b));
const lerp = (a, b, p) => a + (b - a)*p;
const r2   = v => Math.round(v*100)/100;
const V    = n => `var(--${n})`;
const esc  = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
/* pulse: 0 -> 1 -> 0 centered at c with half width w */
const pulse = (t, c, w) => cl(1 - Math.abs(t - c)/w);

/* ------------------------------------------------------ text measurement */
const SANS = "'Work Sans','Roboto','Helvetica Neue',Arial,sans-serif";
const MONO = "'JetBrains Mono','Roboto Mono',Menlo,Consolas,monospace";
const mctx = document.createElement('canvas').getContext('2d');
let mcache = {};
function mw(s, size, weight = 400, cls = 's'){
  const key = cls + weight + size + s;
  if (mcache[key] !== undefined) return mcache[key];
  mctx.font = `${cls === 'i' ? 'italic ' : ''}${weight} ${size}px ${cls === 'm' ? MONO : SANS}`;
  return (mcache[key] = mctx.measureText(s).width);
}

/* ------------------------------------------------------------ primitives */
const hidden = op => op <= 0.004;
function text(x, y, s, o = {}){
  const op = o.op === undefined ? 1 : o.op;
  if (hidden(op)) return '';
  return `<text x="${r2(x)}" y="${r2(y)}" class="${o.cls || 's'}" font-size="${o.size || 14}" font-weight="${o.weight || 400}"` +
    ` text-anchor="${o.anchor || 'start'}" style="fill:${o.fill || V('fg')}" opacity="${r2(op)}">${esc(s)}</text>`;
}
function rect(x, y, w, h, o = {}){
  const op = o.op === undefined ? 1 : o.op;
  if (hidden(op) || w <= 0 || h <= 0) return '';
  return `<rect x="${r2(x)}" y="${r2(y)}" width="${r2(w)}" height="${r2(h)}" rx="${o.rx || 0}"` +
    ` style="fill:${o.fill || 'none'};${o.fo !== undefined ? 'fill-opacity:' + o.fo + ';' : ''}stroke:${o.stroke || 'none'};stroke-width:${o.sw || 1};${o.so !== undefined ? 'stroke-opacity:' + o.so + ';' : ''}${o.dash ? 'stroke-dasharray:' + o.dash + ';' : ''}" opacity="${r2(op)}"/>`;
}
function line(x1, y1, x2, y2, o = {}){
  const op = o.op === undefined ? 1 : o.op;
  if (hidden(op)) return '';
  return `<line x1="${r2(x1)}" y1="${r2(y1)}" x2="${r2(x2)}" y2="${r2(y2)}" style="stroke:${o.stroke || V('grid')};stroke-width:${o.sw || 1};${o.dash ? 'stroke-dasharray:' + o.dash + ';' : ''}stroke-linecap:round" opacity="${r2(op)}"/>`;
}
function circle(x, y, r, o = {}){
  const op = o.op === undefined ? 1 : o.op;
  if (hidden(op) || r <= 0) return '';
  return `<circle cx="${r2(x)}" cy="${r2(y)}" r="${r2(r)}" style="fill:${o.fill || 'none'};stroke:${o.stroke || 'none'};stroke-width:${o.sw || 1};${o.dash ? 'stroke-dasharray:' + o.dash + ';' : ''}" opacity="${r2(op)}"/>`;
}
function path(d, o = {}){
  const op = o.op === undefined ? 1 : o.op;
  if (hidden(op) || !d) return '';
  return `<path d="${d}" style="fill:${o.fill || 'none'};stroke:${o.stroke || 'none'};stroke-width:${o.sw || 1};stroke-linejoin:round;stroke-linecap:round;${o.dash ? 'stroke-dasharray:' + o.dash + ';' : ''}" opacity="${r2(op)}"/>`;
}
/* Polyline through pts [[x, y], ...] drawn up to progress p in [0, 1]
   (p is measured in segments, so points are reached at even intervals). */
function polyline(pts, p = 1){
  if (pts.length < 2 || p <= 0) return '';
  const q = p*(pts.length - 1);
  let d = `M${r2(pts[0][0])} ${r2(pts[0][1])}`;
  for (let k = 1; k < pts.length; k++){
    if (k <= q){ d += ` L${r2(pts[k][0])} ${r2(pts[k][1])}`; continue; }
    const f = q - (k - 1);
    d += ` L${r2(lerp(pts[k-1][0], pts[k][0], f))} ${r2(lerp(pts[k-1][1], pts[k][1], f))}`;
    break;
  }
  return d;
}
/* Curved arrow with horizontal tangents, drawn up to progress p. */
function arrow(o, x0, y0, x1, y1, p, color, op = 1, bend = 38){
  if (p <= 0 || op <= 0) return;
  const s = x1 < x0 ? -1 : 1;
  const d = `M${r2(x0)} ${r2(y0)} C${r2(x0 + s*bend)} ${r2(y0)} ${r2(x1 - s*bend)} ${r2(y1)} ${r2(x1)} ${r2(y1)}`;
  o.push(`<path d="${d}" pathLength="1" style="fill:none;stroke:${color};stroke-width:2;stroke-dasharray:${r2(p)} 1;stroke-linecap:round" opacity="${r2(op)}"/>`);
  if (p > .96) o.push(`<path d="M${r2(x1)} ${r2(y1)} l${-s*9} -5 v10 z" style="fill:${color}" opacity="${r2(op*lin(p, .96, 1))}"/>`);
}
/* Value chip (used for values flying between the chart and the table). */
function chip(o, x, y, s, c, op = 1){
  const w = mw(s, 12.5, 700, 'm') + 14;
  o.push(rect(x - w/2, y - 11, w, 22, {rx:6, fill:V(c), op}));
  o.push(text(x, y + 4.3, s, {size:12.5, cls:'m', weight:700, anchor:'middle', fill: c === 'target' ? '#2a1600' : V('chip'), op}));
}

/* ---------------------------------------------------------------- header */
/* Title, the caption active at t and a right-aligned stepper.
   captions: [[start, end, text], ...]   steps: [[label, start, end], ...] */
function header(t, o, {title, captions = [], steps = []}){
  const a = P(t, 0, .6);
  o.push(text(40, 58, title, {size:26, weight:700, fill:V('title'), op:a}));
  for (const [s, e, msg] of captions){
    if (t < s || t >= e) continue;
    const ca = Math.min(s === 0 ? lin(t, .3, .8) : lin(t, s, s + .35), 1 - lin(t, e - .25, e));
    o.push(text(40, 92, msg, {size:16, fill:V('muted'), op:ca}));
  }
  const items = steps.map(([lab, s, e], n) => {
    const state = t >= e ? 'done' : t >= s ? 'active' : 'todo';
    return {lab, n, state, w: 20 + 8 + mw(lab, 13, 700)};
  });
  const gap = 22;
  let x = 1240 - items.reduce((s, it) => s + it.w, 0) - gap*(items.length - 1);
  for (const it of items){
    const cx = x + 10, cy = 52, on = it.state === 'active', done = it.state === 'done';
    o.push(circle(cx, cy, 10, {fill: on ? V('target') : 'none', stroke: on ? V('target') : done ? V('muted') : V('faint'), sw:1.5, op:a}));
    o.push(text(cx, cy + 4.3, String(it.n + 1), {size:12, weight:700, anchor:'middle', fill: on ? '#001633' : done ? V('muted') : V('faint'), op:a}));
    o.push(text(x + 28, cy + 4.6, it.lab, {size:13, weight: on ? 700 : 400, fill: on ? V('fg') : done ? V('muted') : V('faint'), op:a}));
    x += it.w + gap;
  }
}

/* ----------------------------------------------------------------- brand */
/* skforecast logo and URL in the bottom-right corner, always visible (also in
   exported videos), so the animation keeps its attribution when shared.
   Pass `brand: false` in the configuration to hide it. */
function brand(o, H){
  const url = 'skforecast.org', w = mw(url, 13, 700), s = 26;
  const x = 1240 - w, y = H - 16, lx = x - 8 - s, ly = y - 18;
  o.push(`<g opacity="0.7">` +
    `<image href="img/logo-light.png" class="skf-logo-light" x="${lx}" y="${ly}" width="${s}" height="${s}"/>` +
    `<image href="img/logo-dark.png" class="skf-logo-dark" x="${lx}" y="${ly}" width="${s}" height="${s}"/>` +
    text(x, y, url, {size:13, weight:700, fill:V('muted')}) + `</g>`);
}

/* --------------------------------------------------------------- theming */
const params = new URLSearchParams(location.search);
function resolveTheme(){
  const q = params.get('theme');
  if (q === 'dark' || q === 'light') return q;
  const scheme = el => el && el.getAttribute && el.getAttribute('data-md-color-scheme');
  let s = scheme(document.body);
  try { if (!s && window.parent !== window) s = scheme(window.parent.document.body); } catch(e) {}
  if (s) return s === 'slate' ? 'dark' : 'light';
  return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function applyTheme(){ document.documentElement.setAttribute('data-theme', resolveTheme()); }
function watchTheme(){
  applyTheme();
  if (params.get('theme')) return;
  try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme); } catch(e) {}
  const watch = el => { try { new MutationObserver(applyTheme).observe(el, {attributes:true, attributeFilter:['data-md-color-scheme']}); } catch(e) {} };
  watch(document.body);
  try { if (window.parent !== window) watch(window.parent.document.body); } catch(e) {}
}

/* -------------------------------------------------------------- playback */
const ICON_EXPAND = '<path d="M2 2h5v2H4v3H2zm7 0h5v5h-2V4H9zM2 9h2v3h3v2H2zm10 0h2v5H9v-2h3z"/>';
const ICON_SHRINK = '<path d="M5 2h2v5H2V5h3zm4 0h2v3h3v2H9zM2 9h5v5H5v-3H2zm7 0h5v2h-3v3H9z"/>';
const ICON_RESTART = '<path d="M8 2.5a5.5 5.5 0 1 1-5.2 7.3l1.4-.5A4 4 0 1 0 5.1 5.2L7 7H2V2l1.9 1.9A5.5 5.5 0 0 1 8 2.5z"/>';
function run(cfg){
  const H = cfg.height || 720;
  const host = document.querySelector('.skf-anim');
  host.innerHTML =
    `<svg id="skf-stage" viewBox="0 0 1280 ${H}" role="img" aria-labelledby="skf-title skf-desc">` +
      `<title id="skf-title">${esc(cfg.title)}</title><desc id="skf-desc">${esc(cfg.desc)}</desc><g id="skf-root"></g></svg>` +
    `<div class="skf-controls">` +
      `<button id="skf-play" aria-label="Pause"><svg viewBox="0 0 16 16" id="skf-play-icon"></svg></button>` +
      `<button id="skf-restart" aria-label="Restart"><svg viewBox="0 0 16 16">${ICON_RESTART}</svg></button>` +
      `<input id="skf-scrub" type="range" min="0" max="1000" value="0" step="1" aria-label="Animation position">` +
      `<span class="time" id="skf-time">0.0 s</span>` +
      `<button id="skf-full" aria-label="Full screen"><svg viewBox="0 0 16 16" id="skf-full-icon">${ICON_EXPAND}</svg></button></div>`;
  watchTheme();

  const root = document.getElementById('skf-root');
  const fade = cfg.fadeOut || [cfg.total, cfg.total];
  function render(t){
    const o = [`<rect width="1280" height="${H}" style="fill:var(--bg)"/>`, `<g opacity="${r2(1 - P(t, ...fade))}">`];
    cfg.render(t, o);
    o.push('</g>');
    if (cfg.brand !== false) brand(o, H);
    root.innerHTML = o.join('');
  }

  if (params.get('export') === '1'){
    document.documentElement.classList.add('export');
    window.skfRender = render;
    window.skfDuration = cfg.total;
    document.fonts.ready.then(() => { mcache = {}; render(0); window.skfReady = true; });
    return;
  }

  const btn = document.getElementById('skf-play'), icon = document.getElementById('skf-play-icon');
  const scrub = document.getElementById('skf-scrub'), timeEl = document.getElementById('skf-time');
  const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const tParam = parseFloat(params.get('t'));
  let t = !isNaN(tParam) ? tParam : reduce ? (cfg.still ?? 0) : 0;
  let playing = isNaN(tParam) && !reduce, last = null, visible = true;
  function setIcon(){
    icon.innerHTML = playing ? '<path d="M4 3h3v10H4zM9 3h3v10H9z"/>' : '<path d="M4 2.5v11l9-5.5z"/>';
    btn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }
  function show(){ render(t); scrub.value = Math.round(t/cfg.total*1000); timeEl.textContent = t.toFixed(1) + ' s'; }
  function tick(now){
    if (playing && visible && last !== null) t = (t + Math.min(.1, (now - last)/1000)) % cfg.total;
    last = now;
    show();
    requestAnimationFrame(tick);
  }
  btn.addEventListener('click', () => { playing = !playing; setIcon(); });
  document.getElementById('skf-restart').addEventListener('click', () => { t = 0; playing = true; setIcon(); });
  scrub.addEventListener('input', () => { t = scrub.value/1000*cfg.total; playing = false; setIcon(); });
  // Full screen (the iframe needs the allowfullscreen attribute).
  const full = document.getElementById('skf-full');
  if (!document.fullscreenEnabled) full.style.display = 'none';
  full.addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else host.requestFullscreen().catch(() => {});
  });
  document.addEventListener('fullscreenchange', () => {
    const on = !!document.fullscreenElement;
    document.getElementById('skf-full-icon').innerHTML = on ? ICON_SHRINK : ICON_EXPAND;
    full.setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
  });
  if ('IntersectionObserver' in window){
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(document.getElementById('skf-stage'));
  }
  setIcon();
  document.fonts.ready.then(() => { mcache = {}; });
  requestAnimationFrame(tick);
}

window.SkfAnim = {cl, ease, lin, P, lerp, r2, V, esc, pulse, mw,
  text, rect, line, circle, path, polyline, arrow, chip, header, run};
})();
