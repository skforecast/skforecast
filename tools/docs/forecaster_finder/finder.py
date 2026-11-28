"""
Forecaster finder: rows of options, decision rules and code templates.

Single source of the page "Which forecaster should I use?"
(docs/quick-start/choose-a-forecaster.md) and of the rules block of the skill
`choosing-a-forecaster`. build.py renders every distinct script, runs it on
sample data with the warnings as errors and writes the data of the page;
check_snippets.py finds the shortest history each script runs with.

The finder decides which forecasters to try and against which baseline. It
does not decide their settings: the values per frequency are examples so that
the script runs, and the machine learning block is measured as it is written
here by `benchmarks/forecaster_finder/study.py --snippet`.
"""

from __future__ import annotations

ROWS = [
    {
        "id": "target",
        "text": "What do you want to forecast?",
        "options": [
            ("number", "A number", "Sales, energy demand, website visits"),
            ("category", "A category", "Low, medium or high demand; failure or no failure"),
        ],
    },
    {
        "id": "series",
        "text": "How many time series?",
        "hint": "A time series is one column of values recorded over time, such as the "
                "daily sales of one store.",
        "options": [
            ("one", "One", "The total sales of a company"),
            ("several", "Several", "The sales of each store or each product"),
        ],
        "show_if": {"target": "number"},
    },
    {
        "id": "freq",
        "text": "Your data has one value per...",
        "hint": "skforecast calls this the frequency of the series.",
        "options": [
            ("hourly", "Hour", ""),
            ("daily", "Day", ""),
            ("weekly", "Week", ""),
            ("monthly", "Month", ""),
            ("quarterly", "Quarter", ""),
            ("other", "Something else", "15 minutes, a year..."),
        ],
    },
    {
        "id": "extra",
        "text": "Will you know anything else in advance?",
        "hint": "Things that affect the values and that you know for future dates. "
                "skforecast calls them exogenous variables.",
        "options": [
            ("yes", "Yes", "Holidays, planned promotions, prices, weather forecasts"),
            ("no", "No", "Choose this too if you only know them up to today, like sensor readings"),
        ],
    },
]

# Example settings per frequency, so that the snippet runs as it is. They are
# not tuned: season (offset of the baseline), horizon and number of test
# periods, lags, rolling mean and calendar features. `min_history` is the
# shortest history the scripts of the frequency must run with (checked by
# check_snippets.py), and `history_text` the same in words.
FREQ = {
    "hourly": dict(alias="h", unit="hours", season=24, steps=24, folds=3,
                   lags="[1, 2, 3, 24, 48, 168]", window=24,
                   calendar='["hour", "day_of_week"]',
                   season_text="same hour the day before", min_history=24 * 14, history_text="2 weeks"),
    "daily": dict(alias="D", unit="days", season=7, steps=14, folds=3, lags="14", window=7,
                  calendar='["day_of_week", "month"]',
                  season_text="same day the week before", min_history=90, history_text="3 months"),
    "weekly": dict(alias="W", unit="weeks", season=52, steps=8, folds=3,
                   lags="[1, 2, 3, 4, 52]", window=4, calendar='["month"]',
                   season_text="same week the year before", min_history=104, history_text="2 years"),
    "monthly": dict(alias="MS", unit="months", season=12, steps=6, folds=2, lags="12",
                    window=None, calendar='["month"]',
                    season_text="same month the year before", min_history=36, history_text="3 years"),
    "quarterly": dict(alias="QS", unit="quarters", season=4, steps=4, folds=2, lags="4",
                      window=None, calendar='["quarter"]',
                      season_text="same quarter the year before", min_history=24, history_text="6 years"),
}
# "Something else": the daily settings, marked as values to edit
FREQ["other"] = dict(FREQ["daily"], season_text="value one season before", edit=True,
                     history_text=None)

ALIAS_NOTE = {
    "hourly": '"h" is one value per hour.',
    "daily": '"D" is one value per day.',
    "weekly": '"W" is one value per week, dated on Sundays ("W-MON" if yours are dated on\n'
              "# Mondays).",
    "monthly": '"MS" is one value per month, dated on its first day ("ME" if yours are dated\n'
               "# on the last day).",
    "quarterly": '"QS" is one value per quarter, dated on its first day ("QE" if yours are\n'
                 "# dated on the last day).",
    "other": 'Write the frequency of your data instead of "D": "15min", "YS" (one value per\n'
             "# year)...",
}

ML = "ml"
FM = "foundation"

HEADLINE = "Compare two forecasters: LightGBM and Chronos-2"
HEADLINE_STATS = ", and keep ARIMA as a reference"
HEADLINE_CATEGORY = "Start with a machine learning classifier and compare it with a simple rule"

INTRO = (
    "We tested each kind of model on {datasets} public datasets, always with the same "
    "backtesting, to see where each one works best. Your answers pick a starting point; "
    "the comparison on your own data decides."
)

# One plain sentence per result. The figures behind each one are on the section
# "How we chose these rules"; build.py fails if the results of the study stop
# supporting a sentence (CHECKS in build.py).
REASON = {
    "one-none": "In our tests on single series like yours, Chronos-2 was usually the most "
                "accurate without any setup, and LightGBM often needed work on its features "
                "to beat the baseline.",
    "one-known": "In our tests Chronos-2 was ahead more often, also with variables known in "
                 "advance, but this is where LightGBM came closest and sometimes won.",
    "several": "In our tests with several series Chronos-2 was ahead more often, and "
               "LightGBM won in some datasets.",
    "stats": "ARIMA is the classic reference for monthly and quarterly data: in our tests it "
             "often did better than LightGBM out of the box.",
    "category": "A classifier is the model for categories. In our test it did not clearly "
                "beat the simple rule, so keep the rule as your baseline.",
}

NOTES = {
    "number": [
        "Need to explain the forecast? LightGBM gives feature importances "
        "(`forecaster.get_feature_importances()`); foundation models do not.",
        "The first run downloads the weights of Chronos-2 (about 110 MB) from Hugging Face and "
        "takes a little longer. If your network blocks Hugging Face, that last block of the "
        "script fails: the results of the other models are already printed.",
        "Forecasting many series often? Chronos-2 needs PyTorch and, at scale, a GPU.",
        "Chronos-2 is Apache-2.0. Other foundation models in skforecast have licenses that "
        "restrict commercial use.",
    ],
    "category": [
        "The classifier learns from the past categories. If the categories come from a "
        "number (a level, a reading), you can also forecast the number and turn the forecast "
        "into a category: choose `A number` in the first question.",
        "Need to explain the forecast? The classifier gives feature importances "
        "(`forecaster.get_feature_importances()`).",
        "The script needs only skforecast and LightGBM: no PyTorch and no downloads.",
    ],
}


# User guide of each skforecast class shown in the answer box (paths of the docs)
GUIDES = {
    "ForecasterFoundation": "user_guides/foundation-forecasting-models.html",
    "ForecasterRecursive": "user_guides/autoregressive-forecaster.html",
    "ForecasterRecursiveMultiSeries": "user_guides/independent-multi-time-series-forecasting.html",
    "ForecasterStats": "user_guides/forecasting-sarimax-arima.html",
    "ForecasterEquivalentDate": "user_guides/forecasting-baseline.html",
    "ForecasterRecursiveClassifier": "user_guides/autoregressive-classification-forecasting.html",
}
EXOG_GUIDE = "user_guides/exogenous-variables.html"
COMPARE_GUIDE = "user_guides/comparing-forecasting-models.html"
# Where to go after the script: (text of the link, page)
NEXT = {
    "number": ("compare more model families the same way", COMPARE_GUIDE),
    "category": ("more about forecasting categories",
                 "user_guides/autoregressive-classification-forecasting.html"),
}

# Notebook with an install cell and a cell to paste the script into, written by
# build.py next to this file. The link works once the file is on that branch.
COLAB = ("https://colab.research.google.com/github/skforecast/skforecast/blob/main/"
         "tools/docs/forecaster_finder/first_comparison.ipynb")

# Bridge between the answer and the script: the answer says which forecasters
# to compare, and the script is that comparison
SCRIPT_TITLE = "Run the comparison on your data"
SCRIPT_INTRO = {
    "number": "This script does it. It keeps your last periods apart as a test, forecasts "
              "them with each model using only the data before, and prints the error of "
              "each one: keep what beats the baseline.",
    "category": "This script does it. It keeps your last periods apart as a test, forecasts "
                "them with the classifier using only the data before, and prints how often "
                "the classifier and the simple rule are right: keep the classifier only if "
                "it beats the rule.",
}


def blocks(a: dict) -> dict:
    """
    Answer box for the answers given so far (`a` may be incomplete): the
    headline, the blocks and the rows still to answer. Each answer fills or
    adds something, so the box is built as the user answers.

    skforecast provides forecasters, so each block is headed by its
    forecaster class (with the link to its user guide), then the model it
    wraps. While the number of series is not known, the block of LightGBM
    shows its two possible forecasters. The baseline, which is the same in
    every case, is a line under the blocks.
    """
    target = a.get("target")
    missing = [r["id"] for r in ROWS if r["id"] not in a
               and all(a.get(k) == v for k, v in r.get("show_if", {}).items())]
    if target is None:
        return {"headline": None, "blocks": [], "baseline": None, "missing": missing,
                "complete": False, "outline": []}
    f = FREQ.get(a.get("freq"))
    season = f["season_text"] if f else None
    exog = a.get("extra") == "yes"
    out = []

    def block(role, forecasters, model, text, tags=()):
        out.append({
            "role": role, "model": model, "text": text, "tags": list(tags),
            "forecasters": [{"cls": cls, "guide": GUIDES[cls], "when": when}
                            for cls, when in forecasters],
        })

    if target == "category":
        headline = HEADLINE_CATEGORY
        block("You train it", [("ForecasterRecursiveClassifier", "")],
              "with a LightGBM classifier", "Machine learning classifier trained on your data.",
              ["uses your known variables"] if exog else [])
        baseline = {"cls": None, "guide": None,
                    "text": "Always compared with a simple rule: the category of the "
                            + (season or "same period one season before") + "."}
    else:
        headline = HEADLINE
        series = a.get("series")
        ml = {"one": [("ForecasterRecursive", "")],
              "several": [("ForecasterRecursiveMultiSeries", "")]}.get(
                  series, [("ForecasterRecursive", "one series"),
                           ("ForecasterRecursiveMultiSeries", "several series")])
        tags = ["uses your known variables"] if exog else []
        # Two candidates with different roles, in no order of preference: the
        # sentence under them says which one is usually ahead (REASON)
        block("You train it", ml, "with LightGBM",
              ("One machine learning model for all your series" if series == "several" else
               "Machine learning model trained on your data")
              + ": add your own variables, explain it, tune it.", tags)
        block("No training", [("ForecasterFoundation", "")], "with Chronos-2",
              "Foundation model, ready to use. Usually the most accurate before any tuning.",
              tags)
        if series == "one" and a.get("freq") in ("monthly", "quarterly"):
            headline += HEADLINE_STATS
            block("Reference", [("ForecasterStats", "")], "with ARIMA", "Classic statistical model.")
        # The baseline is the same in every case: a line under the blocks, not a block
        baseline = {"cls": "ForecasterEquivalentDate", "guide": GUIDES["ForecasterEquivalentDate"],
                    "text": "Always compared with a seasonal baseline"
                            + (", one per series" if series == "several" else "") + ": the "
                            + (season or "value one season before") + "."}
    return {"headline": headline, "blocks": out, "baseline": baseline, "missing": missing,
            "complete": not missing, "outline": outline(a, out)}


def outline(a: dict, shown: list) -> list:
    """
    Steps of the script that the answers so far already decide, in the order
    of the script: shown as comments in the code window while it is incomplete.
    """
    if a["target"] == "category":
        return ["LightGBM classifier (ForecasterRecursiveClassifier)", "Simple rule"]
    by_model = {b["model"]: " or ".join(f["cls"] for f in b["forecasters"]) for b in shown}
    steps = ["Seasonal baseline (ForecasterEquivalentDate)",
             f"LightGBM ({by_model['with LightGBM']})"]
    if "with ARIMA" in by_model:
        steps.append("ARIMA (ForecasterStats)")
    return steps + ["Chronos-2 (ForecasterFoundation)"]


def ai_snippet(a: dict) -> str | None:
    """
    The same comparison with skforecast-ai, which picks the settings of each
    model from the data. It continues the finder script: same data and `cv`.
    Checked by build.py when skforecast-ai is installed. `None` for a
    category, which skforecast-ai does not forecast.
    """
    if a["target"] == "category":
        return None
    several = a["series"] == "several"
    exog = a["extra"] == "yes"
    lines = ["# pip install \"skforecast-ai[foundation]\"", "from skforecast_ai import ForecastingAssistant", ""]
    if several:
        lines.append("data = pd.DataFrame(series)" if not exog else
                     "data = pd.concat([pd.DataFrame(series), exog], axis=1)")
        target = "list(series)"
    elif exog:
        lines.append("data = pd.concat([y, exog], axis=1)  # the other columns are used as exog")
        target = "y.name"
    args = ("data=y, cv=cv" if not (several or exog) else f"data=data, cv=cv, target={target}")
    if a["freq"] == "hourly" and not several:
        # One of its candidates fits a model per step: more than 50 fits with 24 steps
        args += ", confirm_cost=True"
    lines += [
        f"result = ForecastingAssistant().compare({args})  # same data and cv as above",
        "print(result.results.to_string())  # ranking of the models it tried",
    ]
    return "\n".join(lines) + "\n"


def agent_message(a: dict) -> str | None:
    """Sentence to send to an AI agent connected to the skforecast-ai MCP server."""
    if a["target"] == "category":
        return None
    f = FREQ[a["freq"]]
    horizon = (f"the next {f['steps']} {f['unit']}" if not f.get("edit")
               else f"the next {f['steps']} values")
    what = ("each series of data/my_data.csv" if a["series"] == "several"
            else "the column sales of data/my_data.csv")
    extra = (" The columns promotion and price are known in advance: use them as exogenous "
             "variables." if a["extra"] == "yes" else "")
    baseline = "" if a["series"] == "several" else ", against a seasonal naive baseline"
    return (f"Compare forecasting models for {horizon} of {what}{baseline}, and show me "
            f"the script of the best one.{extra}")


def all_partial_answers():
    """Every set of answers, complete or not, that the rows can produce."""
    combos = []
    freqs = [None] + list(FREQ)
    extras = [None, "yes", "no"]
    for freq in freqs:
        for extra in extras:
            base = {k: v for k, v in (("freq", freq), ("extra", extra)) if v is not None}
            combos.append(dict(base))
            combos.append({"target": "category", **base})
            for series in (None, "one", "several"):
                combos.append({"target": "number", **({"series": series} if series else {}), **base})
    return combos


DATA = "# Data " + "-" * 73


def data_block(a: dict, n: int | None = None) -> str:
    """
    Block of the script that defines the data: `y` or `series`, and `exog`.

    It creates sample data, so that the script runs as it is in an empty
    notebook (build.py runs every script like this), and its comments say how
    to load a CSV file instead, what the frequency means and which shape the
    data must have. It ends with a check that stops with a clear message when
    the data has no dates, no frequency or no values: the usual results of
    loading a file with the wrong arguments.
    """
    f = FREQ[a["freq"]]
    n = n or SAMPLE_LENGTH[a["freq"]]
    alias = f["alias"]
    category = a["target"] == "category"
    several = a.get("series") == "several"
    exog = a["extra"] == "yes"

    lines = [
        DATA,
        "# Sample data so that the script runs as it is. To use your own data, replace",
    ]
    if several:
        lines += [
            "# this block. For a CSV file with one row per date and store (columns date,",
            "# store and sales):",
            '#     data = pd.read_csv("my_data.csv", parse_dates=["date"])',
            f'#     wide = data.pivot(index="date", columns="store", values="sales").asfreq("{alias}")',
            "#     series = {name: wide[name] for name in wide.columns}",
        ]
        if exog:
            lines += [
                "# exog: a DataFrame with the dates as index and the variables you know in",
                "# advance, or a dict with one DataFrame per series if they differ between them.",
            ]
    else:
        column = "level" if category else "sales"
        what = "categories" if category else "values"
        lines += [
            f"# this block. For a CSV file with a column of dates and a column of {what}:",
            '#     data = pd.read_csv("my_data.csv", index_col="date", parse_dates=True)',
            f'#     y = data["{column}"].asfreq("{alias}")',
        ]
        if exog:
            lines.append(f'#     exog = data[["promotion", "price"]].asfreq("{alias}")'
                         "  # known in advance")
    lines += [
        f"# {ALIAS_NOTE[a['freq']]} If the values turn into NaN after `asfreq`, the",
        "# dates of the file do not match that frequency. If the dates are written as",
        "# day/month/year, add `dayfirst=True` to `read_csv`.",
        f'index = pd.date_range("2018-01-01", periods={n}, freq="{alias}")',
        "rng = np.random.default_rng(123)",
        "t = np.arange(len(index))",
        f"pattern = 0.02 * t + 10 * np.sin(2 * np.pi * t / {f['season']})",
    ]
    if exog:
        lines += [
            "exog = pd.DataFrame(",
            '    {"promotion": rng.integers(0, 2, len(index)).astype(float)}, index=index',
            ")",
            'pattern = pattern + 3 * exog["promotion"].to_numpy()',
        ]
    if category:
        lines += [
            "values = pd.Series(100 + pattern + rng.normal(0, 1, len(index)), index=index)",
            "y = pd.cut(",
            "    values, bins=[-np.inf, values.quantile(0.2), values.quantile(0.8), np.inf],",
            '    labels=["low", "medium", "high"],',
            ").astype(str)",
        ]
    elif several:
        lines += [
            "series = {",
            '    f"item_{i}": pd.Series(',
            "        50 + 10 * i + pattern + rng.normal(0, 1, len(index)), index=index",
            "    )",
            "    for i in range(5)",
            "}",
        ]
    else:
        lines += [
            "y = pd.Series(",
            '    100 + pattern + rng.normal(0, 1, len(index)), index=index, name="sales"',
            ")",
        ]
    # Stop with a clear message if the data is not what the models need
    if several:
        lines += [
            "for name, s in series.items():",
            "    dated = isinstance(s.index, pd.DatetimeIndex) and s.index.freq is not None",
            "    if not dated or s.isna().all():",
            '        raise ValueError(f"The series {name} needs dates with a frequency, and values.")',
        ]
    else:
        lines += [
            "dated = isinstance(y.index, pd.DatetimeIndex) and y.index.freq is not None",
            "if not dated or y.isna().all():",
            '    raise ValueError("`y` needs dates with a frequency, and values: see above.")',
        ]
    return "\n".join(lines)


SAMPLE_LENGTH = {"hourly": 24 * 60, "daily": 730, "weekly": 52 * 5, "monthly": 180,
                 "quarterly": 80, "other": 730}


def decide(a: dict) -> dict:
    """
    Leaf for the answers `a`: models of the script in its order, headline and
    the key of the sentence shown with it (REASON; build.py checks each
    sentence against the results of the study).

    RULES (evidence in benchmarks/forecaster_finder/):
    - A category: classifier and the seasonal rule, by capability.
    - A number: two candidates with different roles and a seasonal baseline.
      LightGBM is the model the user trains and shapes; Chronos-2 needs no
      training and had the lower error in most datasets of the three types of
      data (one series with and without variables known in advance, several
      series), so the answers change the sentence and the code, not the
      candidates.
    - One monthly or quarterly series: ARIMA as a statistical reference.
    - Order of the script: baseline, LightGBM, ARIMA and Chronos-2 last, with
      each result printed as soon as it is computed. Chronos-2 downloads its
      weights on the first run: if that fails, the rest is already printed.
    """
    if a["target"] == "category":
        return {"leaf": "category", "order": ["classifier"], "stats": False,
                "headline": HEADLINE_CATEGORY}
    several = a["series"] == "several"
    stats = (not several) and a["freq"] in ("monthly", "quarterly")
    leaf = "several" if several else "one-known" if a["extra"] == "yes" else "one-none"
    return {"leaf": leaf, "order": [ML] + (["stats"] if stats else []) + [FM], "stats": stats,
            "headline": HEADLINE + (HEADLINE_STATS if stats else "")}


def reasons(leaf: dict) -> list:
    """Sentences shown under the headline."""
    return [REASON[leaf["leaf"]]] + ([REASON["stats"]] if leaf["stats"] else [])


def prompt(a: dict, leaf: dict, code: str) -> str:
    """Prompt for an AI assistant: the answers, the tested script and what to do with it."""
    data = ("a category (classification)" if a["target"] == "category" else
            "several time series" if a["series"] == "several" else "one time series")
    freq = "an uncommon frequency" if a["freq"] == "other" else f"{a['freq']} data"
    extra = ("I also have variables whose future values I know." if a["extra"] == "yes"
             else "I have no variables known in advance.")
    return (
        f"I want to forecast {data} with skforecast ({freq}). {extra}\n\n"
        f"The script below comes from the skforecast documentation and is tested: it compares "
        f"the models I should try first with the same backtesting. Help me adapt it to my "
        f"data.\n"
        f"- First ask me for a sample of my data and how far ahead I need to forecast.\n"
        f"- Change only the data block and the settings marked as examples (steps, season, "
        f"lags, calendar features). Keep the models and the comparison as they are until the "
        f"script runs.\n"
        f"- When it runs, explain the errors it prints and show me how to forecast with the "
        f"model that I keep.\n"
        f"- If you need the skforecast API, use https://skforecast.org/latest/llms-full.txt.\n\n"
        f"```python\n{code}```\n"
    )


# ==============================================================================
# Code templates
# ==============================================================================
def _imports(names: dict) -> str:
    lines = []
    for module in sorted(names):
        line = f"from {module} import {', '.join(sorted(names[module]))}"
        if len(line) > 88:
            items = "".join(f"    {n},\n" for n in sorted(names[module]))
            line = f"from {module} import (\n{items})"
        lines.append(line)
    return "\n".join(lines)


def _edit(f: dict, text: str = "edit for your data") -> str:
    return f"  # <- {text}" if f.get("edit") else ""


def _steps_season(f: dict, steps: int, repeats: str, note: str = "") -> list:
    """Lines of the horizon and the season, with their comments aligned."""
    unit = f["unit"] if not f.get("edit") else "steps"
    width = max(len(f"steps = {steps}"), len(f"season = {f['season']}")) + 2
    return [
        f"{f'steps = {steps}':<{width}}# {unit} ahead of each forecast{note}{_edit(f, 'edit')}",
        f"{'season = ' + str(f['season']):<{width}}# {repeats} {f['season_text']}"
        f"{_edit(f, 'edit')}",
    ]


def _cv(f: dict, length: str) -> list:
    """Lines of the test periods, the same for every model."""
    return [
        f"# Test: the last {f['folds']} periods. Every model uses all the data before each one.",
        "cv = TimeSeriesFold(",
        f"    steps=steps, initial_train_size={length} - {f['folds']} * steps,",
        "    refit=True, fixed_train_size=False,",
        ")",
    ]


INSTALL = "# pip install {skforecast} {packages}"
# PyTorch for CPU only: pip installs the CUDA packages by default on Linux
INSTALL_CPU = (
    "# Linux without a GPU? Install PyTorch for CPU first (about 4 GB less):\n"
    "#     pip install torch --index-url https://download.pytorch.org/whl/cpu"
)
SETTINGS = "# Example settings so that the script runs: they are not tuned."
METRIC = (
    "# Error of each model: the mean absolute error, in the units of your data{mean}.\n"
    "# Lower is better: keep a model only if it beats the baseline.\n"
    'metric = "mean_absolute_error"'
)
METRIC_SEVERAL = (
    "# Error of each model: the mean absolute error, in the units of your data,\n"
    "# averaged over the series. Lower is better: keep a model only if it beats the\n"
    "# baseline.\n"
    'metric = "mean_absolute_error"'
)
OTHER_FREQUENCY = (
    "# Edit them for your data. With one value every 15 minutes and a daily pattern,\n"
    "# season = 96 and steps = 96 forecast one day; with yearly data, season = 1."
)


def render(a: dict, leaf: dict, n: int | None = None) -> str:
    """
    Python script with the models of the leaf, ready to copy. It includes its
    data (data_block), so it runs as it is. `n` is the length of the sample
    data (default: a comfortable history; check_snippets.py uses the minimum).
    """
    f = FREQ[a["freq"]]
    if leaf["leaf"] == "category":
        return render_classification(a, f, n)

    several = a["series"] == "several"
    use_exog = a["extra"] == "yes"
    imp = {"skforecast.model_selection": {"TimeSeriesFold", "backtesting_forecaster"},
           "skforecast.recursive": {"ForecasterEquivalentDate"}}
    data = "series" if several else "y"
    exog_arg = ", exog=exog" if use_exog else ""

    def show(label):
        # Error of a backtest as a number; with several series, the mean of the series
        if several:
            return ('error = error.set_index("levels").loc["average", metric]  # mean of the series\n'
                    f'print(f"{label + ":":<10} {{error:.3f}}")')
        return f'print(f"{label + ":":<10} {{error[metric].item():.3f}}")'

    def call(function, args):
        line = f"error, _ = {function}({args})"
        return line if len(line) <= 88 else f"error, _ = {function}(\n    {args}\n)"

    head = [SETTINGS] + ([OTHER_FREQUENCY] if f.get("edit") else [])
    length = "len(next(iter(series.values())))" if several else "len(y)"
    head += _steps_season(f, f["steps"], "the baseline repeats the")
    head += _cv(f, length)
    head.append(METRIC_SEVERAL if several else METRIC.format(mean=""))
    blocks = [data_block(a, n), "\n".join(head)]

    # 1. Baseline
    if several:
        blocks.append(
            f"# 1. Baseline: {f['season_text']}, one per series\n"
            f"error = pd.concat([\n"
            f"    backtesting_forecaster(\n"
            f"        ForecasterEquivalentDate(offset=season), y=y, cv=cv, metric=metric,\n"
            f"        show_progress=False,\n"
            f"    )[0]\n"
            f"    for y in series.values()\n"
            f"])\n"
            f'print(f"{"Baseline:":<10} {{error[metric].mean():.3f}}")'
        )
    else:
        blocks.append(
            f"# 1. Baseline: {f['season_text']}\n"
            f"baseline = ForecasterEquivalentDate(offset=season)\n"
            + call("backtesting_forecaster", "baseline, y=y, cv=cv, metric=metric") + "\n"
            + show("Baseline")
        )

    for k, cand in enumerate(leaf["order"], start=2):
        if cand == ML:
            imp.setdefault("lightgbm", set()).add("LGBMRegressor")
            imp["skforecast.preprocessing"] = {"CalendarFeatures"}
            cls = "ForecasterRecursiveMultiSeries" if several else "ForecasterRecursive"
            imp["skforecast.recursive"].add(cls)
            bt = "backtesting_forecaster_multiseries" if several else "backtesting_forecaster"
            imp["skforecast.model_selection"].add(bt)
            win = ""
            if f["window"]:
                imp["skforecast.preprocessing"].add("RollingFeatures")
                win = f'    window_features=RollingFeatures(stats="mean", window_sizes={f["window"]}),\n'
            scaler = ""
            if several:
                imp.setdefault("sklearn.preprocessing", set()).add("StandardScaler")
                scaler = "    transformer_series=StandardScaler(),  # series on different scales\n"
            parts = ["lags"] + (["a rolling mean"] if f["window"] else []) + ["calendar features"]
            parts += ["exog"] if use_exog else []
            what = ", ".join(parts[:-1]) + " and " + parts[-1]
            blocks.append(
                f"# {k}. Machine learning: LightGBM with {what}\n"
                f"forecaster = {cls}(\n"
                f"    estimator=LGBMRegressor(min_child_samples=5, random_state=123, verbose=-1),\n"
                f"    lags={f['lags']},{_edit(f)}\n{win}"
                f"    calendar_features=CalendarFeatures(features={f['calendar']}),\n{scaler}"
                f")\n"
                + call(bt, f"forecaster, {data}={data}{exog_arg}, cv=cv, metric=metric") + "\n"
                + show("LightGBM")
            )
        elif cand == "stats":
            imp["skforecast.recursive"].add("ForecasterStats")
            imp["skforecast.stats"] = {"Arima"}
            imp["skforecast.model_selection"].add("backtesting_stats")
            blocks.append(
                f"# {k}. Statistical reference: ARIMA with automatic order selection\n"
                f"arima = ForecasterStats(estimator=Arima(order=None, seasonal_order=None, m=season))\n"
                f"error, _ = backtesting_stats(\n"
                f"    arima, y=y, cv=cv, metric=metric,\n"
                f"    suppress_warnings=True,  # convergence notices of the orders it tries\n"
                f")\n"
                + show("ARIMA")
            )
        else:
            imp["skforecast.foundation"] = {"FoundationModel", "ForecasterFoundation"}
            imp["skforecast.model_selection"].add("backtesting_foundation")
            what = " with the same exog" if use_exog else ""
            blocks.append(
                f"# {k}. Foundation model: Chronos-2{what}, no training\n"
                f"# (the first run downloads its weights)\n"
                f'chronos = ForecasterFoundation(estimator=FoundationModel("autogluon/chronos-2-small"))\n'
                f"error, _ = backtesting_foundation(\n"
                f"    chronos, series={data}{exog_arg}, cv=cv, metric=metric,\n"
                f"    suppress_warnings=True,  # Chronos-2 is not trained: `refit` has no effect on it\n"
                f")\n"
                + show("Chronos-2")
            )

    # ARIMA needs statsmodels, an optional dependency of skforecast (extra `stats`)
    install = INSTALL.format(skforecast='"skforecast[stats]"' if leaf["stats"] else "skforecast",
                             packages="lightgbm chronos-forecasting")
    return (install + "\n" + INSTALL_CPU + "\nimport numpy as np\nimport pandas as pd\n"
            + _imports(imp) + "\n\n" + "\n\n".join(blocks) + "\n")


def render_classification(a, f, n=None):
    # No more than one season ahead, so that the seasonal rule only uses known values
    steps = min(f["steps"], f["season"])
    exog = a["extra"] == "yes"
    exog_arg = ", exog=exog" if exog else ""
    lines = [
        INSTALL.format(skforecast="skforecast", packages="lightgbm"),
        "import numpy as np",
        "import pandas as pd",
        "from lightgbm import LGBMClassifier",
        "from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster",
        "from skforecast.recursive import ForecasterRecursiveClassifier",
        "from sklearn.metrics import balanced_accuracy_score",
        "",
        data_block(a, n),
        "",
        SETTINGS,
    ]
    lines += [OTHER_FREQUENCY] if f.get("edit") else []
    lines += _steps_season(f, steps, "the rule repeats the category of the",
                           " (no more than one season)")
    lines += _cv(f, "len(y)")
    lines += [
        "# How often each one is right (balanced accuracy, from 0 to 1): higher is better.",
        "# Keep the classifier only if it beats the rule.",
        "",
        "# 1. Machine learning classifier",
        "forecaster = ForecasterRecursiveClassifier(",
        "    estimator=LGBMClassifier(min_child_samples=5, random_state=123, verbose=-1),",
        f"    lags={f['lags']},{_edit(f)}",
        ")",
        "score, pred = backtesting_forecaster(",
        f'    forecaster, y=y{exog_arg}, cv=cv, metric="balanced_accuracy_score"',
        ")",
        'print(f"Classifier: {score[\'balanced_accuracy_score\'].item():.3f}")',
        "",
        f"# 2. Baseline: the category of the {f['season_text']}",
        "rule = y.shift(season).loc[pred.index]",
        'print(f"Rule:       {balanced_accuracy_score(y.loc[pred.index], rule):.3f}")',
    ]
    return "\n".join(lines) + "\n"


def forecast_snippet(a: dict, leaf: dict) -> str:
    """
    Lines that continue the script: they train each model on all the data and
    forecast the next periods. With variables known in advance, the forecast
    needs their values for the dates to forecast: the lines create sample
    ones. build.py runs these lines after every script.
    """
    category = a["target"] == "category"
    several = a.get("series") == "several"
    exog = a["extra"] == "yes"
    fit_exog, pred_exog = (", exog=exog", ", exog=exog_future") if exog else ("", "")
    lines = ["# Train on all your data and forecast the next periods with the model you keep"]
    if exog:
        index = "next(iter(series.values())).index" if several else "y.index"
        lines += [
            "# exog_future: your known variables for the dates to forecast (sample values here)",
            f"index = {index}",
            "dates = pd.date_range(index[-1], periods=steps + 1, freq=index.freq)[1:]",
            'exog_future = pd.DataFrame({"promotion": 0.0}, index=dates)',
        ]
    data = "series=series" if several else "y=y"
    name = "the classifier" if category else "LightGBM"
    lines += [
        f"forecaster.fit({data}{fit_exog})  # {name}",
        f"print(forecaster.predict(steps=steps{pred_exog}))",
    ]
    if not category:
        if leaf["stats"]:
            lines += ["arima.fit(y=y, suppress_warnings=True)  # ARIMA",
                      "print(arima.predict(steps=steps))"]
        lines += [
            f"chronos.fit(series={'series' if several else 'y'}{fit_exog})  # Chronos-2",
            f"print(chronos.predict(steps=steps{pred_exog}))",
        ]
    return "\n".join(lines) + "\n"


def needs(a: dict) -> str:
    """Sentence under the script: the history it needs."""
    f = FREQ[a["freq"]]
    if f.get("edit"):
        return (f"Needs at least {f['min_history']} values with these example settings: "
                "edit them for your data.")
    return f"Needs at least {f['min_history']} values ({f['history_text']})."


def all_answers():
    """Every combination of answers that the rows can produce."""
    combos = []
    for freq in FREQ:
        for extra in ("yes", "no"):
            combos.append({"target": "category", "freq": freq, "extra": extra})
            for series in ("one", "several"):
                combos.append({"target": "number", "series": series, "freq": freq,
                               "extra": extra})
    return combos
