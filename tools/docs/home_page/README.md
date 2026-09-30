# Documentation home page

The home page of the documentation (https://skforecast.org/latest/) is not a
Markdown page. It is a custom Material for MkDocs template with its own styles,
a small script, and data files. This folder holds the scripts that generate the
data, and this README explains how the pieces fit together and how to update
them.


## File map

```
docs/
├── README.md                          # Front matter only: `template: home.html`
├── overrides/
│   ├── home.html                      # The page: every section and all its text
│   └── partials/
│       ├── home-data.json             # GENERATED: data of the two animations
│       └── team.html                  # Team cards, shared with the More pages (snippets)
├── stylesheets/home.css               # Styles, all scoped to .sk-home
├── javascripts/home.js                # Animations (listed in mkdocs.yml extra_javascript)
└── img/
    ├── STA-Logo-Default-Color-RGB.png # Sovereign Tech Agency logo
    ├── logo-gcos.svg                  # GC.OS logo (official file from gcos.ai)
    └── social-card-home.png           # GENERATED: preview image when the page is shared
tools/docs/home_page/
├── README.md                          # This file
├── generate_home_data.py              # Writes docs/overrides/partials/home-data.json
├── select_model_configs.py            # Chooses the LightGBM settings and ARIMA exogenous variables
├── make_social_card.mjs               # Writes docs/img/social-card-home.png and the README image
└── social_card.html                   # Layout of the social preview card
```


## How the page is built

- `docs/README.md` is the home page in `nav`. Its front matter selects the
  template (`template: home.html`). Its body is ignored, so the home page does
  not appear in the site search.
- `docs/overrides/home.html` extends `main.html` (so the header, tabs, search,
  version selector, announcement bar and footer are the standard ones) and
  overrides three blocks:
  - `extrahead`: loads Poppins (headings) from Google Fonts and
    `stylesheets/home.css`. Both are loaded only on the home page. With
    `navigation.instant`, Material adds and removes them as the reader enters or
    leaves the page.
  - `site_nav`: empty, so there are no sidebars.
  - `container`: the whole page, inside `<div class="sk-home" id="sk-home">`.
    The content of `partials/home-data.json` goes, HTML-escaped, in the
    `data-home` attribute of that div. It is not in a
    `<script type="application/json">` because instant navigation re-runs the
    scripts of the page without their `type` attribute, which would execute the
    JSON as JavaScript. The team cards come from `partials/team.html`, the
    same file the More pages include with `pymdownx.snippets`.
- Internal links use the `url` filter (`{{ 'user_guides/backtesting.html' | url }}`),
  so every version deployed with mike links to its own pages.
- `home.css` removes the 61rem width limit of `.md-main__inner` and defines
  color tokens. Dark values apply when Material's palette toggle sets
  `data-md-color-scheme="slate"` on `<body>`. The animation stage is dark in
  both schemes.
- `home.js` does nothing on pages without `#sk-home`. It subscribes to
  Material's `document$`, which emits on every page change with instant
  navigation. Before starting a new instance, it destroys the previous one
  (animation frame, observers, `visibilitychange` listener and typing loop).
  It honors `prefers-reduced-motion`: the stage shows its final frame and the
  code window does not type.


## The animations

Both animations show real skforecast outputs, never drawings. The data is
committed in `home-data.json`, so building the docs needs neither LightGBM nor
Chronos.

**Hero animation** (Load your data, Forecast with any model, Backtest and
compare):

| Item | Value |
|:-----|:------|
| Dataset | `vic_electricity`, half-hourly, aggregated to daily totals in GWh |
| Exogenous variables | Daily maximum and mean temperature, public holidays |
| Test period | The last 56 days of the dataset (2014-11-06 to 2014-12-31) |
| Backtesting | 2 folds of 28 days, `refit=False` |
| LightGBM | `ForecasterRecursive`, settings chosen on the 56 days before the test period |
| Chronos-2 | `autogluon/chronos-2-small`, zero-shot, temperature and holidays as covariates |
| ARIMA | `Arima(order=None, seasonal_order=None, m=7)`, exogenous variables chosen on the validation period |
| Intervals | 80% (`interval=[0.1, 0.9]`). LightGBM: bootstrapping with out-of-sample residuals |

**Global models wall**: 36 series picked at random (seed 123) from the 304
quarterly series of `australia_tourism`, forecast 8 quarters ahead by a single
`ForecasterRecursiveMultiSeries` with LightGBM trained once on all of them.

Design decisions, so that the comparison stays honest:

- The test period is fixed as the last 56 days, never chosen to favor a model.
- The three models receive the same exogenous variables.
- The LightGBM configuration and the subset of exogenous variables of ARIMA are
  chosen on a validation period before the test period
  (`select_model_configs.py`), never on the test period. On the current data,
  the validation picks all three variables for ARIMA, the same set the other
  two models use.
- ARIMA with exogenous variables needs the fix of `skforecast.stats.Arima`
  released after 0.25.0. Earlier releases return wrong coefficients, and wrong
  predictions, when `exog` is used together with the intercept or with more
  than one exogenous variable.
- The code shown in the animation is simplified. The note under the stage in
  `home.html` says so and describes the setup. Keep that note in sync with any
  change here.


## Common tasks

### Change a text, a link or a section

Edit `docs/overrides/home.html` and check it with `mkdocs serve`. Link internal
pages with the `url` filter and their `.html` path (the site uses
`use_directory_urls: false`).

### Regenerate the animation data

Needed when the datasets, the models or the API shown on the page change. The
script needs `lightgbm` and `chronos-forecasting`. The first run downloads
Chronos-2 from HuggingFace.

Run the scripts from the repository root with `PYTHONPATH=.`, so they use the
skforecast of the working tree. Run as `python tools/docs/home_page/<script>.py`,
Python would otherwise import the installed release (unless skforecast is
installed in editable mode), which may give different results. The scripts stop
with an explanation when that happens.

```bash
PYTHONPATH=. python tools/docs/home_page/generate_home_data.py              # both animations
PYTHONPATH=. python tools/docs/home_page/generate_home_data.py --only hero  # or --only global
```

The script prints the MAE, the 80% interval coverage and the mean interval
width of each model. The scoreboard of the animation reads the MAE from the
JSON, but the texts that describe the models (chips, note under the stage, code
snippets in `home.js`) are written by hand: review them if the results change.

### Choose the model settings again

```bash
PYTHONPATH=. python tools/docs/home_page/select_model_configs.py            # validation period
PYTHONPATH=. python tools/docs/home_page/select_model_configs.py --windows  # also other test windows
```

Copy the selected LightGBM configuration into `lightgbm_forecaster()` and the
selected exogenous variables into `EXOG_COLUMNS` in `generate_home_data.py`,
then regenerate the data. If ARIMA ends up with a different subset than the
other models, give it its own list and update the chip and the note of the
stage in `home.html`. `--windows` repeats the
comparison on other periods, to check that the ranking is not specific to the
period shown.

### Update the social preview card

The Open Graph and Twitter tags in the `extrahead` block of `home.html` make
LinkedIn, Slack or X show a preview with `docs/img/social-card-home.png` when
the home page is shared. The image is the headline plus a screenshot of the
animation on its last step. The same script saves that screenshot alone as
`images/skforecast-backtesting-comparison.png`, the image of the GitHub README
(referenced with an absolute URL, so PyPI can show it too). Regenerate both
when the animation or the headline changes. It needs Node.js 22 or newer and Google Chrome (set `CHROME_PATH` if
Chrome is not in the default macOS location):

```bash
mkdocs serve                                   # in another terminal
node tools/docs/home_page/make_social_card.mjs      # default URL: http://127.0.0.1:8000/
```

The tags use absolute URLs to `https://skforecast.org/latest/`, as social
networks require, so a new image is only visible once it is deployed.

### Update the number of publications

The home page (trust line and citation section), the About page and the GitHub
README say "70+ scientific publications" and link to a Google Scholar search
instead of listing them. The number comes from OpenAlex: scholarly works
(articles, preprints, theses, book chapters) whose full text mentions
skforecast. To update it:

```bash
curl -s "https://api.openalex.org/works?search=skforecast&filter=type:article|preprint|dissertation|book-chapter&per-page=1" \
  | python -c "import json, sys; print(json.load(sys.stdin)['meta']['count'])"
```

Round it down to the nearest ten and update the three places.

### Change the dataset of the hero animation

Update `load_daily_electricity()` and the constants in `generate_home_data.py`,
the axis ranges and labels in `docs/javascripts/home.js` (`ymin`, `ymax`,
`tmin`, `tmax`, the temperature strip), the code snippets in `CODE` (same file),
the dataset card and the note under the stage in `home.html`.

### Logos

- Sovereign Tech Agency, NumFOCUS and GC.OS logos are the official files, used
  unmodified. GC.OS publishes no brand guidelines.
- The NumFOCUS logo (`docs/img/logo-numfocus-affiliated.png`) is the "Affiliated
  Project" version provided by NumFOCUS, whose [trademark guidelines](https://numfocus.org/trademark-guidelines)
  require prior authorization. That authorization was granted in September 2026.


## Verification

```bash
mkdocs serve
```

Check the home page:
- in light and dark mode, using the palette toggle in the header;
- after navigating to another page and back (instant navigation): the
  animation restarts and the browser console shows no errors;
- at phone width (about 400 px);
- with reduced motion enabled in the operating system.
