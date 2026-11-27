"""
Forecaster finder: questions, decision rules and code templates.

Single source for the interactive page and the skill. The rules encode the
results of benchmark.py (see the EVIDENCE notes); build.py renders every
distinct snippet, runs it on synthetic data and writes finder.json.
"""

from __future__ import annotations

QUESTIONS = [
    {
        "id": "target",
        "text": "What do you want to forecast?",
        "options": [
            ("number", "A number", "Sales, energy demand, website visits, temperature"),
            ("category", "A category", "Low, medium or high demand; failure or no failure"),
        ],
    },
    {
        "id": "series",
        "text": "How many time series do you want to forecast?",
        "hint": "A time series is one column of values recorded over time, such as the daily sales of one store.",
        "options": [
            ("one", "One", "The total sales of a company"),
            ("several", "Several", "The sales of each store or each product"),
        ],
        "show_if": {"target": "number"},
    },
    {
        "id": "freq",
        "text": "How often are the values recorded?",
        "options": [
            ("hourly", "Every hour", "Or more often: every 15 or 30 minutes"),
            ("daily", "Every day", ""),
            ("weekly", "Every week", ""),
            ("monthly", "Every month", ""),
            ("quarterly", "Every quarter", "Or every year"),
        ],
    },
    {
        "id": "extra",
        "text": "Do you know other things that affect the values?",
        "hint": "In skforecast these are called exogenous variables.",
        "options": [
            ("future", "Yes, and I know their future values",
             "Holidays, planned promotions, prices, weather forecasts"),
            ("past", "Yes, but only up to today",
             "Sensor readings, measured weather, sales of other products"),
            ("none", "No, only the dates", ""),
        ],
    },
    {
        "id": "priority",
        "text": "What matters most to you?",
        "options": [
            ("accuracy", "The most accurate forecast", ""),
            ("cost", "Speed and low cost", "Many series, frequent forecasts, no GPU"),
            ("explain", "Understanding what drives the forecast", "Which variables matter and how much"),
        ],
        "show_if": {"target": "number"},
    },
]

# Settings per frequency: forecast horizon, lags, window and calendar features
# of the snippets (the same as benchmark.py), seasonal offset of the baseline
# and the history threshold of the question.
FREQ = {
    "hourly": dict(alias="h", unit="hours", steps=24, folds=8, lags="[1, 2, 3, 24, 48, 168]",
                   window=24, calendar='["hour", "day_of_week"]', offset=24,
                   offset_text="same hour the day before", ets_m=24),
    "daily": dict(alias="D", unit="days", steps=14, folds=8, lags="14", window=7,
                  calendar='["day_of_week", "month"]', offset=7,
                  offset_text="same day the week before", ets_m=7),
    "weekly": dict(alias="W", unit="weeks", steps=8, folds=6, lags="[1, 2, 3, 4, 52]", window=4,
                   calendar='["month"]', offset=52, offset_text="same week the year before",
                   ets_m=52),
    "monthly": dict(alias="MS", unit="months", steps=12, folds=3, lags="12", window=None,
                    calendar='["month"]', offset=12, offset_text="same month the year before",
                    ets_m=12),
    "quarterly": dict(alias="QS", unit="quarters", steps=4, folds=3, lags="4", window=None,
                      calendar='["quarter"]', offset=4,
                      offset_text="same quarter the year before", ets_m=4),
}

ML = "ml"
FM = "foundation"


def decide(a: dict) -> dict:
    """
    Return the leaf for the answers `a`. Both candidates are always shown,
    with the baseline; the answers decide their order and the code.

    EVIDENCE (benchmark.py, 16 cases on 13 datasets, see evidence.py):
    - Chronos-2 had a lower MASE than LightGBM in 26 of 33 single-series runs
      and in 7 of 8 multi-series runs, and beat the seasonal naive in 40 of 41.
    - LightGBM was 5 to 10 times faster and wins with strong known-in-advance
      variables and a year or more of history (vic_electricity).
    - The amount of history did not change the order of the candidates, so
      there is no question about it.
    - Weekly, monthly and quarterly single series: AutoETS or AutoARIMA beat
      Chronos-2 in 8 of 15 runs, so they are added as a benchmark.
    - Variables known only up to today did not improve any model (3 cases).
    """
    if a["target"] == "category":
        return {"leaf": "classification", "order": ["classifier"], "stats": False,
                "past_only": False}
    several = a["series"] == "several"
    low_freq = a["freq"] in ("weekly", "monthly", "quarterly")
    order = [FM, ML] if a["priority"] == "accuracy" else [ML, FM]
    return {
        "leaf": f"{'several' if several else 'one'}-{a['priority']}",
        "order": order,
        "stats": (not several) and low_freq,
        "past_only": a["extra"] == "past",
    }


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


def render(a: dict, leaf: dict) -> str:
    """Python snippet with the candidates of the leaf, ready to copy."""
    f = FREQ[a["freq"]]
    if leaf["leaf"] == "classification":
        return render_classification(a, f)

    several = a["series"] == "several"
    use_exog = a["extra"] == "future"
    imp = {"skforecast.model_selection": {"TimeSeriesFold"}, "skforecast.recursive": {"ForecasterEquivalentDate"},
           "pandas": set()}
    blocks = []
    data = ("series", "series") if several else ("y", "y")
    exog_arg = ", exog=exog" if use_exog else ""
    cv_note = f"last {f['folds']} x {f['steps']} {f['unit']}"
    head = []
    if several:
        head.append(f'# series: DataFrame, one column per series, DatetimeIndex with freq="{f["alias"]}"')
    else:
        head.append(f'# y: pandas Series with a DatetimeIndex, e.g. y = data["sales"].asfreq("{f["alias"]}")')
    if use_exog:
        head.append("# exog: DataFrame with the same index, values known in advance")
    if a["extra"] == "past":
        head.append("# Variables known only up to today did not improve any model in our tests,")
        head.append("# so the candidates start without them.")
    head.append(f"cv = TimeSeriesFold(steps={f['steps']}, initial_train_size=len({data[0]}) - {f['folds']} * {f['steps']})  # {cv_note}")
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
            parts = ["lags"] + (["rolling mean"] if f["window"] else []) + ["calendar features"]
            parts += ["exog"] if use_exog else []
            what = ", ".join(parts[:-1]) + " and " + parts[-1]
            code = (
                f"# {n}. Machine learning: LightGBM with {what}\n"
                f"forecaster = {cls}(\n"
                f"    estimator=LGBMRegressor(random_state=123, verbose=-1),\n"
                f"    lags={f['lags']},\n{win}"
                f"    calendar_features=CalendarFeatures(features={f['calendar']}),\n"
                f")\n"
                f"mase_ml, _ = {bt}(\n"
                f"    forecaster, {data[0]}={data[1]}{exog_arg}, cv=cv, metric=\"mean_absolute_scaled_error\"\n"
                f")"
            )
            if a.get("priority") == "explain":
                fit = f"forecaster.fit(series=series{exog_arg})" if several else f"forecaster.fit(y=y{exog_arg})"
                code += f"\n{fit}\nforecaster.get_feature_importances()  # Which lags and variables matter"
            blocks.append(code)
            results.append(("LightGBM", "mase_ml"))
        else:
            imp["skforecast.foundation"] = {"FoundationModel", "ForecasterFoundation"}
            imp["skforecast.model_selection"].add("backtesting_foundation")
            what = " with the same exog" if use_exog else ""
            code = (
                f"# {n}. Foundation model: Chronos-2{what}, no training needed\n"
                f'chronos = ForecasterFoundation(estimator=FoundationModel("autogluon/chronos-2-small"))\n'
                f"mase_fm, _ = backtesting_foundation(\n"
                f"    chronos, series={data[1]}{exog_arg}, cv=cv, metric=\"mean_absolute_scaled_error\"\n"
                f")"
            )
            blocks.append(code)
            results.append(("Chronos-2", "mase_fm"))
        n += 1

    if leaf["stats"]:
        imp["skforecast.recursive"].add("ForecasterStats")
        imp["skforecast.stats"] = {"Ets"}
        imp["skforecast.model_selection"].add("backtesting_stats")
        blocks.append(
            f"# {n}. Statistical benchmark: AutoETS\n"
            f"ets = ForecasterStats(estimator=Ets(m={f['ets_m']}))\n"
            f'mase_ets, _ = backtesting_stats(ets, y=y, cv=cv, metric="mean_absolute_scaled_error")'
        )
        results.append(("AutoETS", "mase_ets"))
        n += 1

    imp["skforecast.model_selection"].add("backtesting_forecaster")
    if several:
        blocks.append(
            f"# {n}. Baseline: {f['offset_text']}, one per series\n"
            f"mase_base = pd.concat([\n"
            f"    backtesting_forecaster(\n"
            f"        ForecasterEquivalentDate(offset={f['offset']}), y=series[name], cv=cv,\n"
            f"        metric=\"mean_absolute_scaled_error\", show_progress=False\n"
            f"    )[0]\n"
            f"    for name in series.columns\n"
            f"]).mean().to_frame().T"
        )
    else:
        blocks.append(
            f"# {n}. Baseline: {f['offset_text']}\n"
            f"baseline = ForecasterEquivalentDate(offset={f['offset']})\n"
            f'mase_base, _ = backtesting_forecaster(baseline, y=y, cv=cv, metric="mean_absolute_scaled_error")'
        )
    results.append(("Baseline", "mase_base"))

    get = (lambda v: f'{v}.query("levels == \'average\'")') if several else (lambda v: v)
    rows = ",\n    ".join(
        f'"{label}": {get(var) if var in ("mase_ml", "mase_fm") else var}["mean_absolute_scaled_error"].item()'
        for label, var in results
    )
    blocks.append(f"# Lower is better: keep a model only if it beats the baseline\nprint(pd.Series({{\n    {rows}\n}}).sort_values())")
    header = "import pandas as pd\n" + _imports({k: v for k, v in imp.items() if k != "pandas"})
    return header + "\n\n" + "\n\n".join(blocks) + "\n"


def render_classification(a, f):
    steps = f["offset"]  # One season ahead, so the seasonal baseline only uses known values
    folds = 2 if steps >= 52 else f["folds"]
    exog = a["extra"] == "future"
    exog_arg = ", exog=exog" if exog else ""
    head = 'y: pandas Series of categories, e.g. "low", "medium", "high"'
    lines = [
        "from lightgbm import LGBMClassifier",
        "from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster",
        "from skforecast.recursive import ForecasterRecursiveClassifier",
        "from sklearn.metrics import balanced_accuracy_score",
        "",
        f'# {head}, with freq="{f["alias"]}"',
    ]
    if exog:
        lines.append("# exog: DataFrame with the same index, values known in advance")
    lines += [
        f"cv = TimeSeriesFold(steps={steps}, initial_train_size=len(y) - {folds} * {steps})",
        "",
        "# 1. Machine learning classifier",
        "forecaster = ForecasterRecursiveClassifier(",
        "    estimator=LGBMClassifier(random_state=123, verbose=-1),",
        f"    lags={f['lags']},",
        ")",
        "metric, pred = backtesting_forecaster(",
        f'    forecaster, y=y{exog_arg}, cv=cv, metric=["accuracy_score", "balanced_accuracy_score"]',
        ")",
        "",
        f"# 2. Baseline: the category of the {f['offset_text']}",
        f"baseline = y.shift({f['offset']}).loc[pred.index]",
        "print(metric)",
        'print("Baseline:", balanced_accuracy_score(y.loc[pred.index], baseline))',
    ]
    return "\n".join(lines) + "\n"


def all_answers():
    """Every combination of answers that the questions can produce."""
    combos = []
    for freq in FREQ:
        for extra in ("future", "past", "none"):
            combos.append({"target": "category", "freq": freq, "extra": extra})
            for series in ("one", "several"):
                for priority in ("accuracy", "cost", "explain"):
                    combos.append({"target": "number", "series": series, "freq": freq,
                                   "extra": extra, "priority": priority})
    return combos
