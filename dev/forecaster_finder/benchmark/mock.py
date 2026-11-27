"""
Mock of the forecaster finder page, built from results/review/finder.json
(written by build.py).

Nothing is selected at the start. Each answer fills the answer box (models,
skforecast classes, links to the user guides), and the script appears in the
code window when the answers are complete. It writes ../mock.html, a page
without dependencies that opens in a browser.

Usage:
    python -I mock.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
data = json.loads((HERE / "results" / "review" / "finder.json").read_text())
page = {
    "rows": data["rows"],
    "notes": data["notes"],
    "intro": data["intro"],
    "evidence": data["evidence"],
    "partial": data["partial"],
    "script_steps": data["script_steps"],
    "min_history": data["min_history"],
    # The prompt is rebuilt in the page: its text, then the script
    "combos": {c["key"]: {"why": c["why"], "snippet": c["snippet"], "sample": c["sample"],
                          "prompt": c["prompt"].split("```python")[0],
                          "skforecast_ai": c["skforecast_ai"], "agent": c["agent"]}
               for c in data["combos"]},
    "snippets": {k: v["code"] for k, v in data["snippets"].items()},
}
payload = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")

HTML = r"""<title>Maqueta del selector skforecast</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Open+Sans:wght@400;600&family=Ubuntu+Mono:wght@400;700&display=swap">
<style>
/* Layout: one column (max 68rem). The selector is a two-column tool on desktop:
   questions on the left; on the right the answer box (the one loud element,
   navy like the hero of the skforecast home page) and the code window below it.
   Palette and type: the tokens of the skforecast home page (navy and orange). */
:root{
  --bg:#f4f6f9; --surface:#ffffff; --ink:#1b2433; --ink-2:#4d586b; --line:#dfe5ee;
  --heading:#0b1f3a; --accent:#b35a06; --accent-fill:#f79939; --accent-ink:#1f1405; --soft:#eaeff5;
  --good:#1f7a45;
  --stage:#0b1a30; --stage-panel:#12243f; --stage-line:#27406a; --stage-ink:#e6edf6; --stage-muted:#9aabc3;
  --code-bg:#0d1726; --code-line:#1c2940; --code-ink:#dbe3ee; --code-dim:#8796ad;
  --tk-key:#f7a656; --tk-cls:#7cc4f0; --tk-str:#a6d99a; --tk-num:#d4b0ee; --tk-com:#7d8ba1;
  --display:"Poppins","Segoe UI",system-ui,sans-serif;
  --body:"Open Sans","Segoe UI",system-ui,sans-serif;
  --mono:"Ubuntu Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0c1524; --surface:#131f33; --ink:#e3e7ee; --ink-2:#a7b1c1; --line:#263651;
  --heading:#eef3fa; --accent:#f7a656; --accent-fill:#f79939; --accent-ink:#1f1405; --soft:#1a2740;
  --good:#5fd08f;
  --stage:#081120; --stage-panel:#101d33; --stage-line:#27406a; --stage-ink:#e6edf6; --stage-muted:#9aabc3;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0c1524; --surface:#131f33; --ink:#e3e7ee; --ink-2:#a7b1c1; --line:#263651;
  --heading:#eef3fa; --accent:#f7a656; --accent-fill:#f79939; --accent-ink:#1f1405; --soft:#1a2740;
  --good:#5fd08f;
  --stage:#081120; --stage-panel:#101d33; --stage-line:#27406a; --stage-ink:#e6edf6; --stage-muted:#9aabc3;
  color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.55;margin:0}
.wrap{max-width:68rem;margin:0 auto;padding-block:2rem 4rem;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:1.5rem}
h1,h2,h3{font-family:var(--display);color:var(--heading);line-height:1.25;text-wrap:balance;margin:0}
h1{font-size:clamp(1.6rem,3.5vw,2.1rem)}
h2{font-size:1.25rem}
p{margin:0;max-width:70ch}
.eyebrow{font-size:.75rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2)}
.lede{color:var(--ink-2)}
.caption{font-size:.82rem;color:var(--ink-2)}
code{font-family:var(--mono);background:var(--soft);padding:.05em .3em;border-radius:4px}
button{font:inherit;color:inherit;cursor:pointer}
button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--accent-fill);outline-offset:2px}
a{color:var(--accent)}
.intro{display:flex;flex-direction:column;gap:.5rem}
.presets{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}
.presets span{font-size:.85rem;color:var(--ink-2);font-weight:600}
.preset{background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:.3rem .8rem;font-size:.88rem}
.preset:hover{border-color:var(--accent)}
/* The mock of the page block */
.finder{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1.25rem;
  display:grid;grid-template-columns:1fr;gap:1.5rem}
@media (min-width:980px){.finder{grid-template-columns:18.5rem minmax(0,1fr)}}
.finder-head{grid-column:1/-1;display:flex;flex-direction:column;gap:.25rem}
.rows{display:flex;flex-direction:column;gap:1.1rem;min-width:0}
.row{border:0;margin:0;padding:0;min-width:0;display:flex;flex-direction:column;gap:.4rem}
.row legend{font-family:var(--display);font-weight:600;color:var(--heading);padding:0;font-size:.98rem}
.row legend .n{display:inline-block;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;border-radius:50%;
  background:var(--soft);color:var(--ink-2);font-size:.75rem;margin-right:.4rem}
.row.done legend .n{background:var(--accent-fill);color:var(--accent-ink)}
.row .hint{font-size:.82rem;color:var(--ink-2)}
.opts{display:flex;flex-wrap:wrap;gap:.4rem}
.opt{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.4rem .7rem;text-align:left;font-size:.92rem}
.opt:hover{border-color:var(--accent)}
.opt[aria-checked="true"]{background:var(--accent-fill);border-color:var(--accent-fill);color:var(--accent-ink);font-weight:600}
.opt small{display:block;font-size:.76rem;font-weight:400;opacity:.85}
.reset{align-self:flex-start;background:none;border:0;padding:0;color:var(--accent);font-size:.85rem;text-decoration:underline}
.result{min-width:0;display:flex;flex-direction:column;gap:1rem}
/* Answer box */
.answer{background:var(--stage);color:var(--stage-ink);border-radius:12px;padding:1.1rem 1.2rem;display:flex;flex-direction:column;gap:.8rem}
.answer .eyebrow{color:var(--stage-muted);display:flex;justify-content:space-between;gap:1rem}
.answer .headline{font-family:var(--display);font-weight:600;font-size:1.3rem;line-height:1.3;text-wrap:balance;color:#fff}
.answer .headline.pending{color:var(--stage-muted);font-weight:500}
.blocks{display:grid;grid-template-columns:repeat(auto-fit,minmax(11.5rem,1fr));gap:.6rem}
.block{background:var(--stage-panel);border:1px solid var(--stage-line);border-radius:10px;padding:.65rem .75rem;min-width:0;
  display:flex;flex-direction:column;gap:.15rem}
.block.empty{border-style:dashed;background:transparent;color:var(--stage-muted);min-height:5.2rem;justify-content:center;align-items:center;font-size:.85rem}
.block .role{font-size:.68rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--accent-fill)}
.block .fc{font-family:var(--mono);font-weight:700;font-size:1.05rem;line-height:1.25;color:#fff;overflow-wrap:break-word}
.block .fc small{display:block;font-family:var(--body);font-weight:400;font-size:.74rem;color:var(--stage-muted)}
.block .or{font-size:.74rem;color:var(--stage-muted)}
.block .model{font-family:var(--display);font-weight:600;font-size:.9rem;color:var(--tk-cls)}
.block .what{font-size:.8rem;color:var(--stage-muted)}
.block .tag{align-self:flex-start;font-size:.7rem;border:1px solid var(--stage-line);border-radius:999px;padding:0 .45rem;color:var(--stage-ink)}
.block .guides{display:flex;flex-direction:column;gap:.1rem;margin-top:.2rem}
.block a{font-size:.8rem;color:var(--accent-fill)}
.block.flash{animation:flash 1.1s ease-out}
@keyframes flash{0%{border-color:var(--accent-fill);box-shadow:0 0 0 3px rgba(247,153,57,.45)}100%{border-color:var(--stage-line);box-shadow:0 0 0 0 rgba(247,153,57,0)}}
.answer .why{font-size:.92rem;color:var(--stage-ink);display:flex;flex-direction:column;gap:.3rem}
.answer .why a{color:var(--accent-fill)}
/* Script */
.script{display:flex;flex-direction:column;gap:.6rem;min-width:0}
.script h3{font-size:1rem}
.steps{margin:0;padding-left:1.2rem;font-size:.9rem;display:flex;flex-direction:column;gap:.15rem}
.window{background:var(--code-bg);border:1px solid var(--code-line);border-radius:10px;overflow:hidden;min-width:0}
.window-bar{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--code-line)}
.window-bar i{width:10px;height:10px;border-radius:50%;background:#2a3a55;display:block}
.window-bar span{margin-inline:auto;font:13px var(--mono);color:var(--code-dim)}
.window-bar button{font:600 12px var(--display);color:#c3cedd;background:#16243a;border:1px solid #2a3a55;border-radius:6px;padding:3px 10px}
.window-bar button:hover{border-color:#f79939;color:#fff}
.window-bar button[hidden]{display:none}
.window pre{margin:0;padding:14px 16px;font-family:var(--mono);font-size:14px;line-height:1.55;color:var(--code-ink);overflow:auto;max-height:26rem;white-space:pre;tab-size:4}
.window pre code{background:none;padding:0;border-radius:0;color:inherit;font-size:inherit}
.window .fold{display:block;width:100%;text-align:left;background:#101c30;border:0;border-bottom:1px solid var(--code-line);
  color:var(--code-dim);font:13px var(--mono);padding:6px 16px}
.window .fold:hover{color:#fff}
.window .waiting{color:var(--code-dim);min-height:6.5rem}
.window .ln{display:inline-block;min-width:100%}
.window .ln.reveal{opacity:0;animation:reveal .25s ease-out forwards}
.window .ln.changed{animation:changed 1.6s ease-out}
@keyframes reveal{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
@keyframes changed{0%{background:rgba(247,153,57,.35)}100%{background:transparent}}
.mini{background:var(--code-bg);border:1px solid var(--code-line);border-radius:8px;padding:10px 12px;margin:0;font-family:var(--mono);font-size:13px;line-height:1.5;color:var(--code-ink);overflow:auto;white-space:pre}
.mini code{background:none;padding:0;color:inherit;font-size:inherit}
.say{background:var(--soft);border-radius:8px;padding:.5rem .7rem;font-size:.9rem;color:var(--ink)}
.opt2{display:flex;flex-direction:column;gap:.35rem}
.opt2 b{color:var(--ink)}
.tk-k{color:var(--tk-key)}.tk-c{color:var(--tk-cls)}.tk-s{color:var(--tk-str)}.tk-n{color:var(--tk-num)}.tk-m{color:var(--tk-com)}
.actions{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}
.btn{font-size:.9rem;font-weight:600;border-radius:8px;padding:.45rem .85rem;border:1px solid var(--line);background:var(--surface);text-decoration:none;color:var(--ink)}
.btn.primary{background:var(--accent-fill);border-color:var(--accent-fill);color:var(--accent-ink)}
.status{font-size:.82rem;color:var(--good);min-height:1.2em}
details.more{font-size:.88rem;color:var(--ink-2);border-top:1px solid var(--line);padding-top:.5rem}
details.more summary{cursor:pointer;font-weight:600;color:var(--ink)}
details.more .body{display:flex;flex-direction:column;gap:.5rem;padding-top:.5rem}
details.more ul{margin:0;padding-left:1.1rem;display:flex;flex-direction:column;gap:.2rem}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.25rem;display:flex;flex-direction:column;gap:.7rem}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;vertical-align:top;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{font-family:var(--display);font-weight:600;color:var(--heading);font-size:.82rem}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div class="wrap">
  <header class="intro">
    <div class="eyebrow">skforecast · maqueta del selector · 27 nov 2026</div>
    <h1>Qué vería el usuario en cada caso</h1>
    <p class="lede">Maqueta del bloque "Build your first comparison". Al llegar no hay nada elegido: cada respuesta rellena el recuadro de la derecha (modelos, clases de skforecast y enlaces a sus guías) y el script aparece cuando las respuestas están completas. El bloque está en inglés, como iría en la documentación.</p>
    <div class="presets" id="presets"><span>Rellenar con un caso de ejemplo:</span></div>
  </header>

  <section class="finder" aria-label="Forecaster finder mock">
    <div class="finder-head">
      <div class="eyebrow">Not sure where to start?</div>
      <h2>Build your first comparison</h2>
      <p class="lede">Answer four questions about your data. You get the models to try first, what to compare them with and a script that runs as it is.</p>
      <p class="caption" id="intro"></p>
    </div>
    <form class="rows" id="rows" onsubmit="return false"></form>
    <div class="result" aria-live="polite">
      <div class="answer" id="answer"></div>
      <div class="script" id="script"></div>
    </div>
  </section>

  <section class="panel" id="how">
    <div class="eyebrow">Sección de la página, debajo del bloque</div>
    <h2>How we chose these rules</h2>
    <p class="lede" id="how-intro"></p>
    <div class="tbl"><table id="evidence"><thead><tr><th>Your data</th><th>Datasets</th><th>Chronos-2 had the lower error</th><th>LightGBM had the lower error</th><th>LightGBM did worse than the baseline</th></tr></thead><tbody></tbody></table></div>
    <p class="caption" id="how-stats"></p>
    <p class="caption">One configuration per model, not tuned. Chronos-2 is the small model, on CPU. LightGBM uses the settings of the script above. A tuned LightGBM with good features can do better than these figures: that is why it is the model to work on.</p>
  </section>

  <section class="panel">
    <h2>Qué hace cada respuesta</h2>
    <div class="tbl"><table>
      <thead><tr><th>Fila</th><th>En el recuadro de respuesta</th><th>En el script</th></tr></thead>
      <tbody>
        <tr><td>What do you want to forecast?</td><td>Aparecen el titular y los bloques, cada uno encabezado por su forecaster y con el enlace a su guía</td><td>Clase, métrica y baseline</td></tr>
        <tr><td>How many time series?</td><td>El forecaster de LightGBM: de los dos posibles (<code>ForecasterRecursive</code> o <code>ForecasterRecursiveMultiSeries</code>) queda el que corresponde</td><td>Clase multiserie, series escaladas, un baseline por serie</td></tr>
        <tr><td>Your data has one value per...</td><td>Qué repite el baseline; en mensual y trimestral de una serie, aparece el bloque de ARIMA</td><td>Estación, horizonte, lags, calendario</td></tr>
        <tr><td>Will you know anything else in advance?</td><td>La etiqueta "uses your known variables" en los modelos que las usan</td><td>Añade <code>exog</code></td></tr>
      </tbody>
    </table></div>
  </section>

  <section class="panel">
    <h2>Historia mínima con la que se ejecuta el script</h2>
    <p class="lede">Comprobada para cada tipo de script. Con menos datos, <code>TimeSeriesFold</code> da un error que lo dice.</p>
    <div class="tbl"><table id="history"><thead><tr><th>Un valor por</th><th>Valores</th><th>Equivale a</th></tr></thead><tbody></tbody></table></div>
  </section>
</div>

<script type="application/json" id="finder-data">__PAYLOAD__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById("finder-data").textContent);
  const DOCS = "https://skforecast.org/latest/";
  const state = {};
  let previous = {};          // Text of each block in the last render, to flash what changed
  let showImports = false;
  let lastLines = null;       // Lines of the last complete script, to flash the ones that change
  const PRESETS = [
    ["Ventas diarias de una tienda", {target:"number", series:"one", freq:"daily", extra:"no"}],
    ["Demanda mensual, una serie", {target:"number", series:"one", freq:"monthly", extra:"no"}],
    ["Demanda horaria con la temperatura prevista", {target:"number", series:"one", freq:"hourly", extra:"yes"}],
    ["50 productos con promociones", {target:"number", series:"several", freq:"daily", extra:"yes"}],
    ["Nivel de demanda (bajo, medio, alto)", {target:"category", freq:"weekly", extra:"no"}],
  ];
  const SPAN = {hourly:"2 weeks", daily:"3 months", weekly:"2 years", monthly:"3 years", quarterly:"6 years"};
  const SPAN_ES = {hourly:"2 semanas", daily:"3 meses", weekly:"2 años", monthly:"3 años", quarterly:"6 años"};
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const KEYWORDS = new Set(["import","from","for","in","if","else","print","True","False","None","lambda"]);

  function highlight(code){
    return code.split("\n").map(line => {
      let out = "", i = 0;
      while (i < line.length){
        const rest = line.slice(i);
        let m;
        if (rest[0] === "#"){ out += '<span class="tk-m">' + esc(rest) + "</span>"; break; }
        if ((m = rest.match(/^("[^"]*"|'[^']*')/))){ out += '<span class="tk-s">' + esc(m[0]) + "</span>"; i += m[0].length; continue; }
        if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/))){
          const w = m[0];
          out += KEYWORDS.has(w) ? '<span class="tk-k">' + w + "</span>" :
                 /^[A-Z][a-z]/.test(w) ? '<span class="tk-c">' + w + "</span>" : esc(w);
          i += w.length; continue;
        }
        if ((m = rest.match(/^\d+(\.\d+)?/))){ out += '<span class="tk-n">' + m[0] + "</span>"; i += m[0].length; continue; }
        out += esc(rest[0]); i += 1;
      }
      return out;
    }).join("\n");
  }

  const visible = row => !row.show_if || Object.entries(row.show_if).every(([k, v]) => state[k] === v);
  const key = () => ["target", "series", "freq", "extra"].map(k =>
    (k === "series" && state.target !== "number") ? "_" : (state[k] || "_")).join("-");
  const rowsShown = () => D.rows.filter(visible);

  function fromHash(){
    const p = (location.hash || "").replace("#", "").split("-");
    if (p.length !== 4) return;
    const s = {target: {n:"number", c:"category"}[p[0]], series: p[1], freq: p[2], extra: p[3]};
    if (!s.target) return;
    if (s.target === "category") delete s.series;
    const ok = D.rows.every(r => s[r.id] === undefined ? r.id === "series" : r.options.some(o => o[0] === s[r.id]));
    if (ok){ Object.keys(state).forEach(k => delete state[k]); Object.assign(state, s); }
  }

  function renderRows(){
    const form = document.getElementById("rows");
    let n = 0;
    form.innerHTML = rowsShown().map(r => {
      n += 1;
      return '<fieldset class="row' + (state[r.id] ? " done" : "") + '" id="row-' + r.id + '" role="radiogroup" aria-labelledby="lg-' + r.id + '">' +
      '<legend id="lg-' + r.id + '"><span class="n">' + n + "</span>" + esc(r.text) + "</legend>" +
      (r.hint ? '<div class="hint">' + esc(r.hint) + "</div>" : "") +
      '<div class="opts">' + r.options.map(o =>
        '<button type="button" class="opt" role="radio" id="opt-' + r.id + "-" + o[0] + '" data-row="' + r.id + '" data-value="' + o[0] +
        '" aria-checked="' + (state[r.id] === o[0]) + '">' + esc(o[1]) + (o[2] ? "<small>" + esc(o[2]) + "</small>" : "") + "</button>").join("") +
      "</div></fieldset>";
    }).join("") + (Object.keys(state).length ? '<button type="button" class="reset" id="reset">Start over</button>' : "");
  }

  function blockHtml(b, changed){
    const pending = b.forecasters.length > 1;
    const fc = b.forecasters.map((f, n) =>
      (n ? '<span class="or">or</span>' : "") + '<span class="fc">' + esc(f.cls).replace(/([a-z])([A-Z])/g, "$1<wbr>$2") + (f.when ? "<small>" + esc(f.when) + "</small>" : "") + "</span>").join("");
    const guides = b.forecasters.map(f => '<a href="' + DOCS + f.guide + '" target="_blank" rel="noopener">' +
      (pending ? esc(f.cls) + " guide →" : "User guide →") + "</a>").join("");
    return '<div class="block' + (changed ? " flash" : "") + '"><span class="role">' + esc(b.role) + "</span>" + fc +
      '<span class="model">' + esc(b.model) + '</span><span class="what">' + esc(b.text) + "</span>" +
      b.tags.map(t => '<span class="tag">' + esc(t) + "</span>").join("") +
      (guides ? '<span class="guides">' + guides + "</span>" : "") + "</div>";
  }

  function renderAnswer(){
    const P = D.partial[key()];
    const el = document.getElementById("answer");
    const total = rowsShown().length, done = total - P.missing.length;
    const head = '<div class="eyebrow"><span>Your starting point</span><span>' + done + " of " + total + " answered</span></div>";
    if (!P.headline){
      el.innerHTML = head + '<p class="headline pending">Answer the first question to see which forecasters to try.</p>' +
        '<div class="blocks"><div class="block empty">Forecaster</div><div class="block empty">Forecaster</div><div class="block empty">Baseline</div></div>';
      previous = {};
      return;
    }
    const now = {};
    const blocks = P.blocks.map(b => {
      const text = JSON.stringify(b);
      now[b.role] = text;
      const changed = previous[b.role] !== undefined ? previous[b.role] !== text : Object.keys(previous).length > 0;
      return blockHtml(b, changed);
    }).join("");
    previous = now;
    const c = D.combos[key()];
    const why = c ? '<div class="why">' + c.why.map(w => "<p>" + esc(w) + "</p>").join("") +
      '<p><a href="#how" id="how-link">How we chose these rules</a></p></div>' : "";
    el.innerHTML = head + '<p class="headline">' + esc(P.headline) + "</p>" + '<div class="blocks">' + blocks + "</div>" + why;
  }

  function codeHtml(code){
    // One element per line: the first time the script appears its lines come in
    // one after another; after a change of answer, only the new lines flash.
    const lines = code.split("\n");
    const html = highlight(code).split("\n");
    const first = lastLines === null;
    const out = html.map((h, n) => {
      const cls = first ? "ln reveal" : (lastLines.has(lines[n]) || !lines[n].trim() ? "ln" : "ln changed");
      const delay = first ? ' style="animation-delay:' + Math.min(n * 14, 700) + 'ms"' : "";
      return '<span class="' + cls + '"' + delay + ">" + (h || " ") + "</span>";
    }).join("\n");
    return out;
  }

  function renderScript(){
    const P = D.partial[key()];
    const c = D.combos[key()];
    const el = document.getElementById("script");
    if (!c){
      // Outline of the script: what the answers so far already decide
      const left = P.missing.length;
      const outline = P.blocks.map((b, n) => "# " + (n + 1) + ". " + b.model.replace(/^with (a )?/, "") +
        (b.forecasters.length ? " (" + b.forecasters.map(f => f.cls).join(" or ") + ")" : "")).join("\n");
      const wait = "# Your script appears here when you answer the " + (left === 1 ? "last question" : left + " questions left") + ".";
      el.innerHTML = '<div class="window"><div class="window-bar"><i></i><i></i><i></i><span>first_comparison.py</span></div>' +
        '<pre class="waiting"><code><span class="tk-m">' + esc(wait + (outline ? "\n#\n# So far it will compare:\n" + outline : "")) + "</span></code></pre></div>";
      lastLines = null;
      return;
    }
    const code = D.snippets[c.snippet];
    const lines = code.split("\n");
    const cut = lines.findIndex(l => l.startsWith("# Example settings"));
    const imports = lines.slice(0, cut).join("\n").trimEnd(), body = lines.slice(cut).join("\n");
    const nImports = imports.split("\n").length;
    const kind = state.target === "category" ? "category" : "number";
    const shown = showImports ? imports + "\n\n" + body : body;
    const ai = c.skforecast_ai ?
      '<details class="more"><summary>Let skforecast-ai compare more models for you</summary><div class="body">' +
      "<p>skforecast-ai is a separate package that chooses the settings of each model from your data and tries more of them. Add these lines after the script: they use the same data and the same test periods. Its ranking can differ from the one above.</p>" +
      '<pre class="mini"><code>' + highlight(c.skforecast_ai.trimEnd()) + "</code></pre>" +
      '<div class="actions"><button type="button" class="btn" id="copy-ai">Copy lines</button>' +
      '<a class="btn" href="https://ai.skforecast.org/" target="_blank" rel="noopener">skforecast-ai docs</a></div></div></details>' : "";
    const mcp = c.agent ?
      '<div class="opt2"><b>If it supports MCP</b> (Claude Code, Cursor, VS Code, Codex, Claude Desktop): connect the ' +
      '<a href="https://ai.skforecast.org/stable/user-guides/mcp-server.html" target="_blank" rel="noopener">skforecast-ai MCP server</a>' +
      ' and ask it in plain language. It runs the comparison with tools and returns the script that ran; your data is not sent to the model.' +
      '<div class="say" id="agent-text">' + esc(c.agent) + '</div><div><button type="button" class="btn" id="copy-agent">Copy message</button></div></div>' : "";
    el.innerHTML =
      "<h3>What the script does</h3>" +
      '<ol class="steps">' + D.script_steps[kind].map(s => "<li>" + esc(s) + "</li>").join("") + "</ol>" +
      '<div class="window"><div class="window-bar"><i></i><i></i><i></i><span>first_comparison.py</span>' +
      '<button type="button" id="copy-code">Copy</button></div>' +
      '<button type="button" class="fold" id="fold" aria-expanded="' + showImports + '">' + (showImports ? "▾" : "▸") + " imports (" + nImports + " lines)</button>" +
      "<pre><code>" + codeHtml(shown) + "</code></pre></div>" +
      '<p class="caption">Runs as it is: tested on sample data every time the docs are built. Needs at least ' +
      D.min_history[state.freq] + " values" + (SPAN[state.freq] ? " (" + SPAN[state.freq] + ")" : "") + ".</p>" +
      '<div class="actions"><button type="button" class="btn primary" id="copy-code-2">Copy script</button>' +
      '<button type="button" class="btn" id="colab">Run it in Google Colab</button>' +
      '<span class="status" id="status" role="status"></span></div>' + ai +
      '<details class="more"><summary>Using an AI assistant?</summary><div class="body">' + mcp +
      '<div class="opt2"><b>' + (c.agent ? "If it does not" : "Any assistant") + '</b>: copy the script with instructions to adapt it to your data. The assistant is told to ask for a sample of your data first and to keep the forecasters as they are until the script runs.' +
      '<div><button type="button" class="btn" id="copy-prompt">Copy script as a prompt</button></div></div></div></details>' +
      '<details class="more"><summary>Good to know</summary><div class="body"><ul>' +
      D.notes.map(n => "<li>" + esc(n).replace(/`([^`]+)`/g, "<code>$1</code>") + "</li>").join("") + "</ul></div></details>";
    lastLines = new Set(shown.split("\n"));
    el.dataset.code = code;
    el.dataset.colab = c.sample + "\n" + code;
    el.dataset.prompt = c.prompt + "```python\n" + code + "```\n";
    el.dataset.ai = c.skforecast_ai || "";
    el.dataset.agent = c.agent || "";
  }

  function copy(text, message){
    const status = document.getElementById("status");
    const done = ok => { if (status) status.textContent = ok ? message : "Could not copy: select the text and copy it"; };
    try { navigator.clipboard.writeText(text).then(() => done(true), () => done(false)); }
    catch (e) { done(false); }
  }

  function update(){
    renderRows(); renderAnswer(); renderScript();
    const complete = !!D.combos[key()];
    try { history.replaceState(null, "", complete ? "#" + [state.target === "number" ? "n" : "c", state.series || "one", state.freq, state.extra].join("-") : location.pathname + location.search); }
    catch (e) { /* preview without history */ }
  }

  document.getElementById("rows").addEventListener("click", e => {
    if (e.target.id === "reset"){ Object.keys(state).forEach(k => delete state[k]); showImports = false; update(); return; }
    const b = e.target.closest(".opt"); if (!b) return;
    state[b.dataset.row] = b.dataset.value;
    if (state.target === "category") delete state.series;
    update();
    const again = document.getElementById(b.id); if (again) again.focus();
  });
  document.getElementById("rows").addEventListener("keydown", e => {
    const b = e.target.closest(".opt"); if (!b) return;
    const dir = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[e.key]; if (!dir) return;
    e.preventDefault();
    const opts = Array.from(b.parentElement.querySelectorAll(".opt"));
    opts[(opts.indexOf(b) + dir + opts.length) % opts.length].click();
  });
  document.getElementById("script").addEventListener("click", e => {
    const el = document.getElementById("script");
    if (e.target.id === "copy-code" || e.target.id === "copy-code-2") copy(el.dataset.code, "Script copied");
    if (e.target.id === "copy-prompt") copy(el.dataset.prompt, "Prompt copied");
    if (e.target.id === "copy-ai") copy(el.dataset.ai, "Lines copied");
    if (e.target.id === "copy-agent") copy(el.dataset.agent, "Message copied");
    if (e.target.id === "colab") copy(el.dataset.colab, "Script copied with sample data. In the published page this button also opens the notebook: paste it in the second cell.");
    if (e.target.id === "fold"){
      showImports = !showImports;
      lastLines = new Set(D.snippets[D.combos[key()].snippet].split("\n"));
      renderScript(); const f = document.getElementById("fold"); if (f) f.focus();
    }
  });
  document.getElementById("answer").addEventListener("click", e => {
    if (e.target.id === "how-link"){ e.preventDefault(); document.getElementById("how").scrollIntoView({behavior: "smooth"}); }
  });
  const presets = document.getElementById("presets");
  PRESETS.forEach(([label, answers], i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "preset"; b.id = "preset-" + i; b.textContent = label;
    b.addEventListener("click", () => { Object.keys(state).forEach(k => delete state[k]); Object.assign(state, answers); update(); });
    presets.appendChild(b);
  });
  document.getElementById("intro").textContent = D.intro;
  document.getElementById("how-intro").textContent = "Each model was run with the same backtesting on " + D.evidence.datasets +
    " public datasets, with every model using all the data before each test period. Lower error means a lower MASE. One vote per dataset.";
  document.querySelector("#evidence tbody").innerHTML = D.evidence.table.map(r =>
    "<tr><td>" + esc(r.data) + '</td><td class="num">' + r.datasets + '</td><td class="num">' + r.chronos_ahead + " of " + r.datasets +
    '</td><td class="num">' + r.lightgbm_ahead + " of " + r.datasets + '</td><td class="num">' + r.lightgbm_below_baseline + " of " + r.datasets + "</td></tr>").join("");
  document.getElementById("how-stats").textContent = "Monthly and quarterly single series (" + D.evidence.stats.datasets +
    " datasets): ARIMA had a lower error than LightGBM in " + D.evidence.stats.arima_ahead_of_lightgbm + " and than Chronos-2 in " +
    D.evidence.stats.arima_ahead_of_chronos + ".";
  document.querySelector("#history tbody").innerHTML = D.rows.find(r => r.id === "freq").options
    .filter(o => o[0] !== "other")
    .map(o => "<tr><td>" + esc(o[1]) + '</td><td class="num">' + D.min_history[o[0]] + "</td><td>" + SPAN_ES[o[0]] + "</td></tr>").join("");
  window.addEventListener("hashchange", () => { fromHash(); update(); });
  fromHash(); update();
})();
</script>
"""

out = HERE.parent / "mock.html"
out.write_text(HTML.replace("__PAYLOAD__", payload))
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(page['combos'])} combinations, "
      f"{len(page['partial'])} states)")
