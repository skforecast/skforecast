"""
Forecaster finder: rows of options, decision rules and code templates.

Prototype of the single source for the page and the skill (PLAN.md, section
6). Version 2: four rows on one screen, rules by type of data and snippets
that refit every model in each fold, raise no warnings and run with a short
history. build.py renders every distinct snippet, runs it on synthetic data
with the warnings as errors and writes results/review/finder.json;
check_snippets.py finds the shortest history each snippet runs with.

The ML block of the snippets is measured as it is written here by
`review.py --snippet`.
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
            ("no", "No", "Also if you only know them up to today, like sensor readings"),
        ],
    },
]
DEFAULT = {"target": "number", "series": "one", "freq": "daily", "extra": "no"}

# Example settings per frequency, so that the snippet runs as it is. They are
# not tuned: season (offset of the baseline), horizon and number of test
# periods, lags, rolling mean and calendar features. `min_history` is the
# shortest history the snippets of the frequency must run with (checked by
# check_snippets.py).
FREQ = {
    "hourly": dict(alias="h", unit="hours", season=24, steps=24, folds=3,
                   lags="[1, 2, 3, 24, 48, 168]", window=24,
                   calendar='["hour", "day_of_week"]',
                   season_text="same hour the day before", min_history=24 * 14),
    "daily": dict(alias="D", unit="days", season=7, steps=14, folds=3, lags="14", window=7,
                  calendar='["day_of_week", "month"]',
                  season_text="same day the week before", min_history=90),
    "weekly": dict(alias="W", unit="weeks", season=52, steps=8, folds=3,
                   lags="[1, 2, 3, 4, 52]", window=4, calendar='["month"]',
                   season_text="same week the year before", min_history=104),
    "monthly": dict(alias="MS", unit="months", season=12, steps=6, folds=2, lags="12",
                    window=None, calendar='["month"]',
                    season_text="same month the year before", min_history=36),
    "quarterly": dict(alias="QS", unit="quarters", season=4, steps=4, folds=2, lags="4",
                      window=None, calendar='["quarter"]',
                      season_text="same quarter the year before", min_history=24),
}
# "Something else": the daily settings, marked as values to edit
FREQ["other"] = dict(FREQ["daily"], season_text="same step one season before", edit=True)

ML = "ml"
FM = "foundation"

LINEUP = {
    FM: ("Chronos-2", "ForecasterFoundation", "Foundation model, no training"),
    ML: ("LightGBM", "{ml_class}", "Machine learning model you can improve and explain"),
    "stats": ("ARIMA", "ForecasterStats", "Statistical reference, automatic order selection"),
    "baseline": ("Seasonal baseline", "ForecasterEquivalentDate", "{season_text}"),
    "classifier": ("LightGBM classifier", "ForecasterRecursiveClassifier",
                   "Machine learning classifier"),
    "rule": ("Seasonal rule", "y.shift(season)", "The category of the {season_text}"),
}

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

NOTES = [
    "Need to explain the forecast? LightGBM gives feature importances "
    "(`forecaster.get_feature_importances()`); foundation models do not.",
    "Forecasting many series often? Chronos-2 needs PyTorch and, at scale, a GPU.",
    "Chronos-2 is Apache-2.0. Other foundation models in skforecast have licenses that "
    "restrict commercial use.",
]


def decide(a: dict) -> dict:
    """
    Leaf for the answers `a`: candidates in order, statistical reference,
    headline and the key of the sentence shown with it (REASON; build.py
    checks each sentence against the results of the study).

    RULES (PLAN.md 6.2; evidence in REVISION.md, section 7):
    - A category: classifier and the seasonal rule, by capability.
    - A number: Chronos-2 first, then LightGBM, and the seasonal baseline.
      Chronos-2 had the lower error in most datasets of the three types of
      data (one series with and without variables known in advance, several
      series), so the answers change the sentence and the code, not the order.
    - One monthly or quarterly series: ARIMA as a statistical reference.
    """
    if a["target"] == "category":
        return {
            "leaf": "category", "order": ["classifier"], "stats": False,
            "headline": "Start with a machine learning classifier and compare it with "
                        "a simple rule",
        }
    several = a["series"] == "several"
    stats = (not several) and a["freq"] in ("monthly", "quarterly")
    leaf = "several" if several else "one-known" if a["extra"] == "yes" else "one-none"
    headline = "Start with Chronos-2, then try to beat it with LightGBM"
    if stats:
        headline += ", and keep ARIMA as a reference"
    return {"leaf": leaf, "order": [FM, ML], "stats": stats, "headline": headline}


def reasons(leaf: dict) -> list:
    """Sentences shown under the headline."""
    return [REASON[leaf["leaf"]]] + ([REASON["stats"]] if leaf["stats"] else [])


def lineup(a: dict, leaf: dict) -> list:
    """Cards of the result: (role, name, skforecast class, description)."""
    f = FREQ[a["freq"]]
    fmt = dict(season_text=f["season_text"],
               ml_class="ForecasterRecursiveMultiSeries" if a.get("series") == "several"
               else "ForecasterRecursive")
    cards = []
    if leaf["leaf"] == "category":
        keys = [("Candidate", "classifier"), ("Baseline", "rule")]
    else:
        keys = [("First", leaf["order"][0]), ("Challenger", leaf["order"][1])]
        if leaf["stats"]:
            keys.append(("Reference", "stats"))
        keys.append(("Baseline", "baseline"))
    for role, key in keys:
        name, cls, text = LINEUP[key]
        cards.append((role, name, cls.format(**fmt), text.format(**fmt)))
    return cards


def prompt(a: dict, leaf: dict) -> str:
    """Prompt for an AI assistant with the answers and the plan."""
    data = ("a category (classification)" if a["target"] == "category" else
            "several time series" if a["series"] == "several" else "one time series")
    freq = "an uncommon frequency" if a["freq"] == "other" else f"{a['freq']} data"
    extra = ("I also have variables whose future values I know." if a["extra"] == "yes"
             else "I have no variables known in advance.")
    names = ", then ".join(c[1] for c in lineup(a, leaf))
    return (
        f"I want to forecast {data} with skforecast ({freq}). {extra}\n"
        f"Plan: compare these models with the same backtesting folds, in this order: "
        f"{names}. Keep a model only if it beats the baseline.\n"
        f"Before writing any code, ask me for a sample of my data and how far ahead I "
        f"need to forecast. Use the skforecast API reference at "
        f"https://skforecast.org/latest/llms-full.txt and do not tune anything until "
        f"the comparison runs."
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


HEADER = (
    "# Example settings so that the script runs: they are not tuned.\n"
    "# pip install skforecast {packages}"
)


def render(a: dict, leaf: dict) -> str:
    """Python snippet with the candidates of the leaf, ready to copy."""
    f = FREQ[a["freq"]]
    if leaf["leaf"] == "category":
        return render_classification(a, f)

    several = a["series"] == "several"
    use_exog = a["extra"] == "yes"
    imp = {"skforecast.model_selection": {"TimeSeriesFold", "backtesting_forecaster"},
           "skforecast.recursive": {"ForecasterEquivalentDate"}}
    blocks = []
    data = "series" if several else "y"
    exog_arg = ", exog=exog" if use_exog else ""
    metric = '"mean_absolute_scaled_error"'
    freq_note = ("a DatetimeIndex with a frequency" if f.get("edit")
                 else f'a DatetimeIndex, freq="{f["alias"]}"')

    head = [HEADER.format(packages="lightgbm chronos-forecasting"), ""]
    if several:
        head.append(f"# series: dict with one pandas Series per series, each with {freq_note},")
        head.append("#         e.g. series = {name: data[name] for name in data.columns}")
        length = "len(next(iter(series.values())))"
    else:
        head.append(f"# y: pandas Series with {freq_note},")
        head.append(f'#    e.g. y = data["sales"].asfreq("{f["alias"]}")')
        length = "len(y)"
    if use_exog:
        head.append("# exog: DataFrame with the same index and the values you know in advance")
    head.append(f"steps = {f['steps']}    # {f['unit'] if not f.get('edit') else 'steps'} "
                f"ahead of each forecast{_edit(f, 'edit')}")
    head.append(f"season = {f['season']}{' ' * (4 - len(str(f['season'])))}"
                f"# the baseline repeats the {f['season_text']}{_edit(f, 'edit')}")
    head.append(
        f"cv = TimeSeriesFold(\n"
        f"    steps=steps, initial_train_size={length} - {f['folds']} * steps,  "
        f"# last {f['folds']} periods\n"
        f"    refit=True, fixed_train_size=False,  # every model uses all the data before each period\n"
        f")"
    )
    blocks.append("\n".join(head))

    results = []
    n = 1
    for cand in leaf["order"]:
        if cand == ML:
            imp.setdefault("lightgbm", set()).add("LGBMRegressor")
            imp["skforecast.preprocessing"] = imp.get("skforecast.preprocessing", set()) | {"CalendarFeatures"}
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
            code = (
                f"# {n}. Machine learning: LightGBM with {what}\n"
                f"forecaster = {cls}(\n"
                f"    estimator=LGBMRegressor(min_child_samples=5, random_state=123, verbose=-1),\n"
                f"    lags={f['lags']},{_edit(f)}\n{win}"
                f"    calendar_features=CalendarFeatures(features={f['calendar']}),\n{scaler}"
                f")\n"
                f"metric_ml, _ = {bt}(\n"
                f"    forecaster, {data}={data}{exog_arg}, cv=cv, metric={metric}\n"
                f")"
            )
            blocks.append(code)
            results.append(("LightGBM", "metric_ml"))
        else:
            imp["skforecast.foundation"] = {"FoundationModel", "ForecasterFoundation"}
            imp["skforecast.model_selection"].add("backtesting_foundation")
            what = " with the same exog" if use_exog else ""
            code = (
                f"# {n}. Foundation model: Chronos-2{what}, no training "
                f"(the first run downloads its weights)\n"
                f'chronos = ForecasterFoundation(estimator=FoundationModel("autogluon/chronos-2-small"))\n'
                f"metric_fm, _ = backtesting_foundation(\n"
                f"    chronos, series={data}{exog_arg}, cv=cv, metric={metric},\n"
                f"    suppress_warnings=True,  # `refit` does not apply to it: it always sees the data before\n"
                f")"
            )
            blocks.append(code)
            results.append(("Chronos-2", "metric_fm"))
        n += 1

    if leaf["stats"]:
        imp["skforecast.recursive"].add("ForecasterStats")
        imp["skforecast.stats"] = {"Arima"}
        imp["skforecast.model_selection"].add("backtesting_stats")
        blocks.append(
            f"# {n}. Statistical reference: ARIMA with automatic order selection\n"
            f"arima = ForecasterStats(estimator=Arima(order=None, seasonal_order=None, m=season))\n"
            f"metric_arima, _ = backtesting_stats(\n"
            f"    arima, y=y, cv=cv, metric={metric},\n"
            f"    suppress_warnings=True,  # convergence notices of the orders it tries\n"
            f")"
        )
        results.append(("ARIMA", "metric_arima"))
        n += 1

    if several:
        blocks.append(
            f"# {n}. Baseline: {f['season_text']}, one per series\n"
            f"metric_base = pd.concat([\n"
            f"    backtesting_forecaster(\n"
            f"        ForecasterEquivalentDate(offset=season), y=y, cv=cv, metric={metric},\n"
            f"        show_progress=False,\n"
            f"    )[0]\n"
            f"    for y in series.values()\n"
            f"]).mean().to_frame().T"
        )
    else:
        blocks.append(
            f"# {n}. Baseline: {f['season_text']}\n"
            f"baseline = ForecasterEquivalentDate(offset=season)\n"
            f"metric_base, _ = backtesting_forecaster(baseline, y=y, cv=cv, metric={metric})"
        )
    results.append(("Baseline", "metric_base"))

    get = (lambda v: f'{v}.query("levels == \'average\'")') if several else (lambda v: v)
    rows = ",\n    ".join(
        f'"{label}": {get(var) if var in ("metric_ml", "metric_fm") else var}'
        f'["mean_absolute_scaled_error"].item()'
        for label, var in results
    )
    blocks.append("# Lower is better: keep a model only if it beats the baseline\n"
                  f"print(pd.Series({{\n    {rows}\n}}).sort_values())")
    header = "import pandas as pd\n" + _imports(imp)
    return header + "\n\n" + "\n\n".join(blocks) + "\n"


def render_classification(a, f):
    # No more than one season ahead, so that the seasonal rule only uses known values
    steps = min(f["steps"], f["season"])
    folds = f["folds"]
    exog = a["extra"] == "yes"
    exog_arg = ", exog=exog" if exog else ""
    freq_note = ("a DatetimeIndex with a frequency" if f.get("edit")
                 else f'a DatetimeIndex, freq="{f["alias"]}"')
    lines = [
        "from lightgbm import LGBMClassifier",
        "from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster",
        "from skforecast.recursive import ForecasterRecursiveClassifier",
        "from sklearn.metrics import balanced_accuracy_score",
        "",
        HEADER.format(packages="lightgbm"),
        "",
        f'# y: pandas Series of categories ("low", "medium", "high") with {freq_note}',
    ]
    if exog:
        lines.append("# exog: DataFrame with the same index and the values you know in advance")
    lines += [
        f"steps = {steps}{' ' * (5 - len(str(steps)))}# {f['unit'] if not f.get('edit') else 'steps'} "
        f"ahead of each forecast (no more than one season){_edit(f, 'edit')}",
        f"season = {f['season']}{' ' * (4 - len(str(f['season'])))}# the rule repeats the category of the "
        f"{f['season_text']}{_edit(f, 'edit')}",
        "cv = TimeSeriesFold(",
        f"    steps=steps, initial_train_size=len(y) - {folds} * steps,  # last {folds} periods",
        "    refit=True, fixed_train_size=False,",
        ")",
        "",
        "# 1. Machine learning classifier",
        "forecaster = ForecasterRecursiveClassifier(",
        "    estimator=LGBMClassifier(min_child_samples=5, random_state=123, verbose=-1),",
        f"    lags={f['lags']},{_edit(f)}",
        ")",
        "metric, pred = backtesting_forecaster(",
        f'    forecaster, y=y{exog_arg}, cv=cv, metric="balanced_accuracy_score"',
        ")",
        "",
        f"# 2. Baseline: the category of the {f['season_text']}",
        "rule = y.shift(season).loc[pred.index]",
        "",
        "# Higher is better: keep the classifier only if it beats the rule",
        'print("Classifier:", metric["balanced_accuracy_score"].item())',
        'print("Rule:      ", balanced_accuracy_score(y.loc[pred.index], rule))',
    ]
    return "\n".join(lines) + "\n"


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
