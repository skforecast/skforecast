"""
Mock of the forecaster finder page, built from results/review/finder.json
(written by build.py): the four rows of options and the result that the
user would see for every combination, with the snippets that build.py runs.

It writes ../mock.html, a page without dependencies that opens in a browser.

Usage:
    python -I mock.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
data = json.loads((HERE / "results" / "review" / "finder.json").read_text())
page = {
    "rows": data["rows"],
    "default": data["default"],
    "notes": data["notes"],
    "intro": data["intro"],
    "evidence": data["evidence"],
    "min_history": data["min_history"],
    "combos": data["combos"],
    "snippets": {k: v["code"] for k, v in data["snippets"].items()},
}
payload = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")

HTML = r"""<title>Maqueta del selector skforecast</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Open+Sans:wght@400;600&family=Ubuntu+Mono:wght@400;700&display=swap">
<style>
/* Layout: one column (max 68rem); the selector is a two-column tool on desktop
   (rows of options on the left, result on the right) and stacks on phones.
   Palette and type: the tokens of the skforecast home page (navy and orange). */
:root{
  --bg:#f4f6f9; --surface:#ffffff; --ink:#1b2433; --ink-2:#4d586b; --line:#dfe5ee;
  --heading:#0b1f3a; --accent:#b35a06; --accent-fill:#f79939; --accent-ink:#1f1405; --soft:#eaeff5;
  --good:#1f7a45; --good-soft:#e4f4ea;
  --code-bg:#0d1726; --code-line:#1c2940; --code-ink:#dbe3ee; --code-dim:#8796ad;
  --tk-key:#f7a656; --tk-cls:#7cc4f0; --tk-str:#a6d99a; --tk-num:#d4b0ee; --tk-com:#7d8ba1;
  --display:"Poppins","Segoe UI",system-ui,sans-serif;
  --body:"Open Sans","Segoe UI",system-ui,sans-serif;
  --mono:"Ubuntu Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0c1524; --surface:#131f33; --ink:#e3e7ee; --ink-2:#a7b1c1; --line:#263651;
  --heading:#eef3fa; --accent:#f7a656; --accent-fill:#f79939; --accent-ink:#1f1405; --soft:#1a2740;
  --good:#5fd08f; --good-soft:#12301f; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0c1524; --surface:#131f33; --ink:#e3e7ee; --ink-2:#a7b1c1; --line:#263651;
  --heading:#eef3fa; --accent:#f7a656; --accent-fill:#f79939; --accent-ink:#1f1405; --soft:#1a2740;
  --good:#5fd08f; --good-soft:#12301f; color-scheme:dark}
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
code{font-family:var(--mono);background:var(--soft);padding:.05em .3em;border-radius:4px}
button{font:inherit;color:inherit;cursor:pointer}
button:focus-visible,a:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
a{color:var(--accent)}
.intro{display:flex;flex-direction:column;gap:.5rem}
.presets{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}
.presets span{font-size:.85rem;color:var(--ink-2);font-weight:600}
.preset{background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:.3rem .8rem;font-size:.88rem}
.preset:hover{border-color:var(--accent)}
/* The mock of the page block */
.finder{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1.25rem;
  display:grid;grid-template-columns:1fr;gap:1.5rem}
@media (min-width:980px){.finder{grid-template-columns:19rem minmax(0,1fr)}}
.finder-head{grid-column:1/-1;display:flex;flex-direction:column;gap:.2rem}
.rows{display:flex;flex-direction:column;gap:1.1rem;min-width:0}
.row{border:0;margin:0;padding:0;min-width:0;display:flex;flex-direction:column;gap:.4rem}
.row[hidden]{display:none}
.row legend{font-family:var(--display);font-weight:600;color:var(--heading);padding:0;font-size:.98rem}
.row .hint{font-size:.82rem;color:var(--ink-2)}
.opts{display:flex;flex-wrap:wrap;gap:.4rem}
.opt{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.4rem .7rem;text-align:left;font-size:.92rem}
.opt:hover{border-color:var(--accent)}
.opt[aria-checked="true"]{background:var(--accent-fill);border-color:var(--accent-fill);color:var(--accent-ink);font-weight:600}
.opt small{display:block;font-size:.76rem;font-weight:400;opacity:.85}
.result{min-width:0;display:flex;flex-direction:column;gap:1rem}
.headline{font-family:var(--display);font-weight:600;font-size:1.35rem;color:var(--heading);text-wrap:balance}
.reason{display:flex;flex-direction:column;gap:.35rem;font-size:.95rem}
.models{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;font-size:.85rem;color:var(--ink-2)}
.models b{font-family:var(--display);font-weight:600;color:var(--heading);border:1px solid var(--line);border-radius:999px;padding:.1rem .6rem}
details.more{font-size:.86rem;color:var(--ink-2)}
details.more summary{cursor:pointer;font-weight:600;color:var(--ink)}
details.more ul{margin:.4rem 0 0;padding-left:1.1rem;display:flex;flex-direction:column;gap:.2rem}
.caption{font-size:.8rem;color:var(--ink-2)}
.window{background:var(--code-bg);border:1px solid var(--code-line);border-radius:10px;overflow:hidden;min-width:0}
.window-bar{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--code-line)}
.window-bar i{width:10px;height:10px;border-radius:50%;background:#2a3a55;display:block}
.window-bar span{margin-inline:auto;font:13px var(--mono);color:var(--code-dim)}
.window-bar button{font:600 12px var(--display);color:#c3cedd;background:#16243a;border:1px solid #2a3a55;border-radius:6px;padding:3px 10px}
.window-bar button:hover{border-color:#f79939;color:#fff}
.window pre{margin:0;padding:14px 16px;font-family:var(--mono);font-size:14px;line-height:1.55;color:var(--code-ink);overflow:auto;max-height:30rem;white-space:pre;tab-size:4}
.window pre code{background:none;padding:0;border-radius:0;color:inherit;font-size:inherit}
.tk-k{color:var(--tk-key)}.tk-c{color:var(--tk-cls)}.tk-s{color:var(--tk-str)}.tk-n{color:var(--tk-num)}.tk-m{color:var(--tk-com)}
.tested{font-size:.82rem;color:var(--ink-2)}
.actions{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}
.btn{font-size:.9rem;font-weight:600;border-radius:8px;padding:.45rem .85rem;border:1px solid var(--line);background:var(--surface);text-decoration:none;color:var(--ink)}
.btn.primary{background:var(--accent-fill);border-color:var(--accent-fill);color:var(--accent-ink)}
.status{font-size:.82rem;color:var(--good);min-height:1.2em}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.25rem;display:flex;flex-direction:column;gap:.7rem}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;vertical-align:top;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{font-family:var(--display);font-weight:600;color:var(--heading);font-size:.82rem}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <header class="intro">
    <div class="eyebrow">skforecast · maqueta del selector · 27 nov 2026</div>
    <h1>Qué vería el usuario en cada caso</h1>
    <p class="lede">Maqueta del bloque "Build your first comparison" con los datos que genera el prototipo: 36 combinaciones de respuestas, cada una con su titular, una frase que lo explica y el código que se ejecuta sin avisos en cada compilación. Las cifras del estudio no van en el resultado: viven en la sección "How we chose these rules". El bloque está en inglés, como iría en la documentación. Cambia cualquier opción y el resultado se actualiza.</p>
    <div class="presets" id="presets"><span>Casos de ejemplo:</span></div>
  </header>

  <section class="finder" aria-label="Forecaster finder mock">
    <div class="finder-head">
      <div class="eyebrow">Not sure where to start?</div>
      <h2>Build your first comparison</h2>
      <p class="lede">Pick what describes your data. You get the models to try, what to compare them with and a script that runs as it is.</p>
      <p class="caption" id="intro"></p>
    </div>
    <form class="rows" id="rows" onsubmit="return false"></form>
    <div class="result" id="result" aria-live="polite"></div>
  </section>

  <section class="panel">
    <h2>Qué cambia con cada fila</h2>
    <div class="tbl"><table>
      <thead><tr><th>Fila</th><th>Cambia la recomendación</th><th>Cambia el código</th></tr></thead>
      <tbody>
        <tr><td>What do you want to forecast?</td><td>Sí: un número lleva a Chronos-2 y LightGBM; una categoría, a un clasificador y una regla estacional</td><td>Clase, métrica y baseline</td></tr>
        <tr><td>How many time series?</td><td>No: solo la frase que acompaña al titular</td><td>Clase multiserie, series escaladas, un baseline por serie</td></tr>
        <tr><td>Your data has one value per...</td><td>Solo en mensual y trimestral de una serie: añade ARIMA como referencia</td><td>Estación, horizonte, lags, calendario</td></tr>
        <tr><td>Will you know anything else in advance?</td><td>No: solo la frase que acompaña al titular</td><td>Añade <code>exog</code> a los dos candidatos</td></tr>
      </tbody>
    </table></div>
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
    <h2>Historia mínima con la que se ejecuta el código</h2>
    <p class="lede">Comprobada para cada tipo de fragmento. Con menos datos, <code>TimeSeriesFold</code> da un error que lo dice.</p>
    <div class="tbl"><table id="history"><thead><tr><th>Frecuencia</th><th>Valores</th><th>Equivale a</th></tr></thead><tbody></tbody></table></div>
  </section>
</div>

<script type="application/json" id="finder-data">__PAYLOAD__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById("finder-data").textContent);
  const state = Object.assign({}, D.default);
  const PRESETS = [
    ["Ventas diarias de una tienda", {target:"number", series:"one", freq:"daily", extra:"no"}],
    ["Demanda mensual, una serie", {target:"number", series:"one", freq:"monthly", extra:"no"}],
    ["Demanda eléctrica horaria con la temperatura prevista", {target:"number", series:"one", freq:"hourly", extra:"yes"}],
    ["50 productos con promociones planificadas", {target:"number", series:"several", freq:"daily", extra:"yes"}],
    ["Nivel de demanda (bajo, medio, alto) semanal", {target:"category", series:"one", freq:"weekly", extra:"no"}],
    ["Datos cada 15 minutos", {target:"number", series:"one", freq:"other", extra:"no"}],
  ];
  const SPAN = {hourly:"2 semanas", daily:"3 meses", weekly:"2 años", monthly:"3 años", quarterly:"6 años", other:"según la frecuencia"};
  const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
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

  function visible(row){
    return !row.show_if || Object.entries(row.show_if).every(([k, v]) => state[k] === v);
  }
  function combo(){
    return D.combos.find(c => Object.entries(c.answers).every(([k, v]) => state[k] === v));
  }
  function hash(){
    return [state.target === "number" ? "n" : "c", state.series, state.freq, state.extra].join("-");
  }
  function fromHash(){
    const p = (location.hash || "").replace("#", "").split("-");
    if (p.length !== 4) return;
    const s = {target: p[0] === "c" ? "category" : "number", series: p[1], freq: p[2], extra: p[3]};
    const ok = D.rows.every(r => r.options.some(o => o[0] === s[r.id]));
    if (ok) Object.assign(state, s);
  }

  function renderRows(){
    const form = document.getElementById("rows");
    form.innerHTML = D.rows.map(r =>
      '<fieldset class="row" id="row-' + r.id + '"' + (visible(r) ? "" : " hidden") + ' role="radiogroup" aria-labelledby="lg-' + r.id + '">' +
      '<legend id="lg-' + r.id + '">' + esc(r.text) + "</legend>" +
      (r.hint ? '<div class="hint">' + esc(r.hint) + "</div>" : "") +
      '<div class="opts">' + r.options.map(o =>
        '<button type="button" class="opt" role="radio" id="opt-' + r.id + "-" + o[0] + '" data-row="' + r.id + '" data-value="' + o[0] +
        '" aria-checked="' + (state[r.id] === o[0]) + '">' + esc(o[1]) + (o[2] ? "<small>" + esc(o[2]) + "</small>" : "") + "</button>").join("") +
      "</div></fieldset>").join("");
  }

  function renderResult(){
    const c = combo();
    const el = document.getElementById("result");
    if (!c){ el.textContent = "No result for this combination."; return; }
    const code = D.snippets[c.snippet];
    el.innerHTML =
      '<div><div class="eyebrow">Your starting point</div><p class="headline">' + esc(c.headline) + "</p></div>" +
      '<div class="reason">' + c.why.map(w => "<p>" + esc(w) + "</p>").join("") + "</div>" +
      '<div class="models"><span>The script compares:</span>' + c.lineup.map(k => "<b>" + esc(k[1]) + "</b>").join("") + "</div>" +
      '<div class="window"><div class="window-bar"><i></i><i></i><i></i><span>first_comparison.py</span>' +
      '<button type="button" id="copy-code">Copy</button></div><pre><code>' + highlight(code) + "</code></pre></div>" +
      '<div class="tested">Runs as it is: tested on sample data every time the docs are built. Needs at least ' +
      D.min_history[state.freq] + " values" + (state.freq === "other" ? "" : " (" + ({hourly:"2 weeks", daily:"3 months", weekly:"2 years", monthly:"3 years", quarterly:"6 years"}[state.freq]) + ")") + ".</div>" +
      '<div class="actions"><button type="button" class="btn primary" id="copy-prompt">Copy prompt for your AI assistant</button>' +
      '<a class="btn" href="https://ai.skforecast.org/" target="_blank" rel="noopener">Compare models on your own data with skforecast-ai</a>' +
      '<span class="status" id="status" role="status"></span></div>' +
      '<details class="more"><summary>Good to know</summary><ul>' + D.notes.map(n => "<li>" + esc(n).replace(/`([^`]+)`/g, "<code>$1</code>") + "</li>").join("") + "</ul></details>";
    el.dataset.code = code;
    el.dataset.prompt = c.prompt;
  }

  function copy(text, label){
    const status = document.getElementById("status");
    const done = ok => { if (status) status.textContent = ok ? label + " copied" : "Select the text and copy it"; };
    try { navigator.clipboard.writeText(text).then(() => done(true), () => done(false)); }
    catch (e) { done(false); }
  }

  function update(){
    renderRows(); renderResult();
    try { history.replaceState(null, "", "#" + hash()); } catch (e) { /* preview without history */ }
  }

  document.getElementById("rows").addEventListener("click", e => {
    const b = e.target.closest(".opt"); if (!b) return;
    state[b.dataset.row] = b.dataset.value;
    update();
    const again = document.getElementById(b.id); if (again) again.focus();
  });
  document.getElementById("rows").addEventListener("keydown", e => {
    const b = e.target.closest(".opt"); if (!b) return;
    const dir = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1}[e.key]; if (!dir) return;
    e.preventDefault();
    const opts = Array.from(b.parentElement.querySelectorAll(".opt"));
    const next = opts[(opts.indexOf(b) + dir + opts.length) % opts.length];
    next.click();
  });
  document.getElementById("result").addEventListener("click", e => {
    const el = document.getElementById("result");
    if (e.target.id === "copy-code") copy(el.dataset.code, "Code");
    if (e.target.id === "copy-prompt") copy(el.dataset.prompt, "Prompt");
  });
  const presets = document.getElementById("presets");
  PRESETS.forEach(([label, answers], i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "preset"; b.id = "preset-" + i; b.textContent = label;
    b.addEventListener("click", () => { Object.assign(state, answers); update(); });
    presets.appendChild(b);
  });
  document.querySelector("#history tbody").innerHTML = D.rows.find(r => r.id === "freq").options
    .filter(o => o[0] !== "other")
    .map(o => "<tr><td>" + esc(o[1]) + '</td><td class="num">' + D.min_history[o[0]] + "</td><td>" + SPAN[o[0]] + "</td></tr>").join("");
  document.getElementById("intro").textContent = D.intro;
  document.getElementById("how-intro").textContent = "Each model was run with the same backtesting on " + D.evidence.datasets +
    " public datasets, with every model using all the data before each test period. Lower error means a lower MASE. One vote per dataset.";
  document.querySelector("#evidence tbody").innerHTML = D.evidence.table.map(r =>
    "<tr><td>" + esc(r.data) + '</td><td class="num">' + r.datasets + '</td><td class="num">' + r.chronos_ahead + " of " + r.datasets +
    '</td><td class="num">' + r.lightgbm_ahead + " of " + r.datasets + '</td><td class="num">' + r.lightgbm_below_baseline + " of " + r.datasets + "</td></tr>").join("");
  document.getElementById("how-stats").textContent = "Monthly and quarterly single series (" + D.evidence.stats.datasets +
    " datasets): ARIMA had a lower error than LightGBM in " + D.evidence.stats.arima_ahead_of_lightgbm + " and than Chronos-2 in " +
    D.evidence.stats.arima_ahead_of_chronos + ".";
  window.addEventListener("hashchange", () => { fromHash(); renderRows(); renderResult(); });
  fromHash(); update();
})();
</script>
"""

out = HERE.parent / "mock.html"
out.write_text(HTML.replace("__PAYLOAD__", payload))
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(page['combos'])} combinations)")
