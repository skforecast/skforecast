# Study behind the forecaster finder

The page [Which forecaster should I use?](https://skforecast.org/latest/quick-start/choose-a-forecaster.html)
and the skill `choosing-a-forecaster` tell the reader which forecasters to
compare first. This folder holds the study those rules rest on: its scripts
and its results. The page shows figures computed from `results/rules_*.csv`,
and its build fails if a sentence is no longer supported by them
(`tools/docs/forecaster_finder/build.py`).

It is not part of the performance benchmarks of `benchmarks/run_benchmarks.py`.


## What it measures

- 26 public datasets of `skforecast.datasets`, 35 cases: hourly, daily, weekly,
  monthly and quarterly data, one series and several, with and without
  variables known in advance.
- The same backtesting folds for every model. Every model uses all the data
  before each fold: LightGBM is refitted with an expanding window
  (`refit=True, fixed_train_size=False`), AutoARIMA and AutoETS are refitted by
  `backtesting_stats`, and Chronos-2 receives the history up to the fold.
- Error: MASE, divided by the MASE of the seasonal naive of the case. Below 1,
  the model beats the baseline.
- One configuration per model, chosen before looking at the test period.
  "LightGBM tuned" is the exception: 20 trials of `bayesian_search_forecaster`
  on a validation period taken from the training data.
- One vote per dataset in each type of data (the mean of its runs), so that a
  dataset with many variants does not count several times.

| Model in the results | What it is |
|:--|:--|
| `LightGBM snippet v2 [refit]` | The LightGBM that the scripts of the page write (`tools/docs/forecaster_finder/finder.py`) |
| `LightGBM [refit]`, `LightGBM+ [refit]` | LightGBM with the lags of the case, and a stronger one (more lags, two rolling windows, mild regularization) |
| `LightGBM tuned [refit]` | 20 trials of search on validation |
| `Chronos-2` | `autogluon/chronos-2-small`, on CPU |
| `AutoETS`, `AutoARIMA` | `Ets` and `Arima` with automatic selection, in the series with a season of 24 or less |
| `Seasonal naive` | `ForecasterEquivalentDate` |


## Results

`python rules.py` prints them and writes the files the page reads.

| Type of data | Datasets | Chronos-2 had the lower error | LightGBM did worse than the baseline | LightGBM tuned had a lower error than Chronos-2 |
|:--|--:|--:|--:|--:|
| One series, nothing known in advance | 16 | 14 | 7 | 4 |
| One series, variables known in advance | 7 | 5 | 1 | 1 |
| Several series | 10 | 8 | 2 | 2 |

- Chronos-2 was not worse than the baseline in any dataset.
- The search improved LightGBM in 18 of the 33 dataset and type of data pairs
  and made it worse by more than 5% in 11.
- In the 8 monthly and quarterly datasets with one series, AutoARIMA had a
  lower error than the LightGBM of the page in 6 and than Chronos-2 in 3.

Limits: public datasets (Chronos-2 may have seen some of them; the control
with `autogluon/chronos-2-synth`, trained only on synthetic data, is in the
main results), one LightGBM configuration or a light automatic search, the
small Chronos-2 on CPU, variables known in advance taken from observed values,
and a single dataset for classification. It does not measure a LightGBM built
with knowledge of the problem.


## Files

| File | What it is |
|:--|:--|
| `cases.py` | The 15 original cases and the helpers shared by the study (loading, folds, metric) |
| `cases_d4.py` | 20 more cases, to reach six or more datasets per type of data |
| `study.py` | The study: runs the models on the cases and writes one CSV per case |
| `rules.py` | Evidence per rule: reads the CSV files and writes `rules_runs.csv`, `rules_cells.csv`, `rules_stats.csv` and `rules_tuned.csv` |
| `recount.py` | Effect of the protocol on the 15 original cases (frozen LightGBM against refitted, by run and by dataset) |
| `results/` | One CSV per case (`d4_` for the added cases), controls (`extra_*`), per-series and per-fold errors (`detail/`) and the versions (`environment*.json`) |


## Reproduce

In an environment with the versions of `requirements.txt` and the repository
installed (`pip install -e .`). Times on 4 cores without GPU.

```bash
cd benchmarks/forecaster_finder
python -I study.py              # the 15 original cases (35 min)
python -I study.py --extra      # other sites and subsets, classification (15 min)
python -I study.py --horizon    # a horizon four times shorter (7 min)
python -I study.py --d4         # the 20 added cases (25 min)
python -I study.py --snippet    # the LightGBM of the page, as it is written (3 min)
python -I study.py --arima      # AutoARIMA, also with exog (2 min)
python -I study.py --tune       # LightGBM with 20 trials of search (50 min)
python -I rules.py              # figures of the page
python ../../tools/docs/forecaster_finder/build.py   # checks the sentences and rewrites the page
```

Run the study again in every minor version and when the version of
`chronos-forecasting` or LightGBM changes. If a figure changes, the page
changes with it; if a rule stops holding, rewrite its sentence in
`tools/docs/forecaster_finder/finder.py`.
