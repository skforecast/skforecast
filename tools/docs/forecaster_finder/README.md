# Forecaster finder

The page [Which forecaster should I use?](https://skforecast.org/latest/quick-start/choose-a-forecaster.html)
(`docs/quick-start/choose-a-forecaster.md`) asks four questions about the data
and answers with the forecasters to compare first, the baseline to compare
them with and a script that runs as it is. This folder holds its single source
and the scripts that generate and check everything it publishes.

Scope: the finder decides which forecasters to try and against which baseline.
It does not decide their settings: the values of the scripts are examples per
frequency so that the script runs.


## File map

```
tools/docs/forecaster_finder/
├── README.md                # This file
├── finder.py                # SOURCE: questions, rules, texts and code templates
├── build.py                 # Runs every script and writes the generated files
├── check_snippets.py        # Runs each kind of script with its minimum history
├── check_clean_install.sh   # First real run: new environment, install line, no Hugging Face
├── check_mcp.py             # Drives the skforecast-ai MCP server as an agent would
└── first_comparison.ipynb   # GENERATED: notebook opened by "Run it in Google Colab"
docs/
├── quick-start/choose-a-forecaster.md   # The page; three GENERATED blocks between markers
├── javascripts/finder-data.json         # GENERATED: every state and every script
├── javascripts/finder.js                # The widget (listed in mkdocs.yml extra_javascript)
├── stylesheets/finder.css               # Styles, loaded only on the page
└── overrides/finder.html                # Template of the page: loads the styles
skills/choosing-a-forecaster/SKILL.md    # One GENERATED block between markers
benchmarks/forecaster_finder/            # The study the rules rest on (see its README)
```


## How it works

- `finder.py` has no dependencies. It defines the rows of options (`ROWS`), the
  example settings per frequency (`FREQ`), the rules (`decide`), the answer box
  for any set of answers, complete or not (`blocks`), the sentences shown with
  each result (`REASON`) and the code templates (`render`).
- Every script includes its data block (`data_block`): sample data, so that it
  runs as it is, with comments that say how to load a CSV file instead and a
  check that stops with a clear message when the data has no dates, no
  frequency or no values. What the page shows, what "Copy script" copies and
  what is tested are the same text.
- `build.py` renders the script of each of the 36 combinations of answers and
  runs it in an empty namespace, as pasted into a new notebook, followed by
  the lines of "Forecast with the model you keep" (`forecast_snippet`). It
  fails if a script raises an error or any warning, if a line is
  longer than 88 characters, if a linked user guide does not exist or if the
  results of the study stop supporting a sentence (`evidence()`). Only then it
  writes the generated files.
- The page holds no rules. `finder.js` reads `finder-data.json`, which has the
  answer box of the 105 possible states and the 36 scripts, and renders the
  state of the answers. The answers are kept in the hash of the URL
  (`#answers=number-one-daily-no`), so a result can be shared, survives going to
  a user guide and back, and the home page can link to the page with the first
  question answered.
- Without JavaScript the page is still complete: the evidence, the rules table
  and an example script are generated into the Markdown between
  `<!-- FORECASTER-FINDER:...:START -->` and `<!-- FORECASTER-FINDER:...:END -->`.
  The skill `choosing-a-forecaster` gets the same rules, for AI assistants.
- With `navigation.instant`, `document$` emits on every page change:
  `finder.js` destroys the previous instance and starts a new one, as
  `home.js` does.
- Events are sent to Google Analytics only when the reader accepted it
  (Material defines `gtag` after the consent), with closed identifiers:
  `finder_change` (row, option, complete), `finder_copy` (kind), `finder_open`
  (section), `finder_cta` (target), `finder_feedback` (value) and `finder_reset`.


## Change something

1. Edit `finder.py` (a text, a rule, a setting, a template). Never edit the
   generated files or blocks.
2. Run `python tools/docs/forecaster_finder/build.py`. It needs the packages of
   the scripts: `pip install lightgbm chronos-forecasting` and the `stats`
   extra of skforecast. `--no-run` writes without running, to iterate on texts.
3. If a setting or a template changed, run
   `python tools/docs/forecaster_finder/check_snippets.py`.
4. Run `python tools/ai/generate_ai_context_files.py`: the skill is part of
   `llms-full.txt`.
5. If the LightGBM block of the scripts changed, measure it again
   (`benchmarks/forecaster_finder/study.py --snippet` and `rules.py`): the page
   says that its LightGBM is the one of the study.

The script order is deliberate: baseline, LightGBM, ARIMA and Chronos-2 last,
each result printed as soon as it is computed. Chronos-2 downloads its weights
from Hugging Face on the first run; if that fails, the reader already has the
other results.

The error of the scripts is the mean absolute error, in the units of the
reader's data: the script compares models on the same data, so it needs no
scaled metric. The study compares across datasets and uses the MASE.

`dev/forecaster_finder/user_review/` holds two reviews of the page as a user
would use it (people with their own CSV files). Read them before changing the
questions, the data block or the texts: most of what is there comes from what
failed in those reviews.

The page is a quick test that attracts readers and points them to the right
forecasters. Connecting the script to their own data is up to them: keep the
data block short, and do not grow the script to cover every kind of file.


## Checks

| Check | What it covers | Where it runs |
|:--|:--|:--|
| `build.py --check` | Generated files up to date, sentences supported by the study, links, line length. Runs nothing, needs only pandas | CI, job "Generated files are up to date" |
| `build.py` | Every script runs on sample data without errors or warnings | CI, job "Every script runs as it is" (also weekly) |
| `check_snippets.py` | Each kind of script runs with the minimum history the page states | Same job |
| `build.py --ai` | The lines of skforecast-ai shown under the script | By hand, in an environment with `skforecast-ai[foundation]` |
| `check_mcp.py` | The skforecast-ai MCP server answers the path the page describes | By hand, same environment plus the `mcp` extra |
| `check_clean_install.sh` | Install line of the script in a new environment, first and second run, and the script without access to Hugging Face | By hand (it installs about 6 GB) |

The workflow is `.github/workflows/forecaster-finder.yml`. It installs the
packages as the scripts tell the reader to, without pinned versions. Its weekly
run checks the default branch, that is, the scripts of the published
documentation.

The link of "Run it in Google Colab" points to `first_comparison.ipynb` on the
`main` branch (`COLAB` in `finder.py`), so it works once this folder is there.
