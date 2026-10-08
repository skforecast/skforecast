# Changelog

All significant changes to this project are documented in this release file.

| Legend                                                     |                                       |
|:-----------------------------------------------------------|:--------------------------------------|
| <span class="badge text-bg-feature">Feature</span>         | New feature                           |
| <span class="badge text-bg-enhancement">Enhancement</span> | Improvement in existing functionality |
| <span class="badge text-bg-api-change">API Change</span>   | Changes in the API                    |
| <span class="badge text-bg-fix">Fix</span>                 | Bug fix                               |
| <span class="badge text-bg-docs">Docs</span>               | Documentation improvement             |



## 0.26.0 <small>In development</small> { id="0.26.0" }

The main changes in this release are:

+ <span class="badge text-bg-enhancement">Enhancement</span> Faster import of the forecaster modules. `numba` is no longer imported when `skforecast.recursive`, `skforecast.direct`, `skforecast.preprocessing` or `skforecast.model_selection` are imported. It is loaded on the first use of <code>[RollingFeatures]</code> or <code>[RollingFeaturesClassification]</code>.

+ <span class="badge text-bg-enhancement">Enhancement</span> Faster `fit` of <code>[ForecasterRecursiveMultiSeries]</code> with many series, with less memory and the same predictions: with 500 series, 10 to 22% faster with LightGBM and more than 10 times faster with `series_weights`; with 300 series and `encoding='onehot'`, about 3 times faster.

+ <span class="badge text-bg-enhancement">Enhancement</span> Faster predictions: `predict` of <code>[ForecasterRecursiveMultiSeries]</code> is 4.9x faster with 5000 series (2.3x with 500), and the forecasters with an `ExtraTreesRegressor` predict 15x faster.

+ <span class="badge text-bg-enhancement">Enhancement</span> Faster <code>[Arima]</code>: seasonal models fit 1.4x to 2.2x faster (an ARIMA(1,1,1)(1,1,1)[12] on 500 observations, from 0.51 to 0.27 seconds; the automatic order selection with `m=12`, from about 3.2 to 1.4-1.7 seconds). The speedups do not change the results.

+ <span class="badge text-bg-feature">Feature</span> New functions <code>[get_model_info]</code> and <code>[list_adapters]</code> in `skforecast.foundation` to query, without installing the backend or loading the weights, the capabilities and requirements of the foundation models: adapter, default `context_length`, exogenous variable support, supported quantiles, backend package, license restriction and Hugging Face gating.

+ <span class="badge text-bg-feature">Feature</span> The skforecast workflow skills can now be installed in your own coding agent: as a Claude Code plugin (`/plugin marketplace add skforecast/skforecast`) or, for Cursor, GitHub Copilot, Codex, Gemini CLI and other agents, with `npx skills add skforecast/skforecast`. See [Install skforecast context in your agent](../quick-start/ai-assisted-forecasting.md#install-skforecast-context-in-your-agent).

+ <span class="badge text-bg-feature">Feature</span> Skforecast documentation is available in [Context7](https://context7.com/skforecast/skforecast), so any MCP compatible agent can query it with the library `/skforecast/skforecast`.

+ <span class="badge text-bg-api-change">API Change</span> The minimum supported versions of pandas and scikit-learn are now 2.2 and 1.6 (previously 2.1 and 1.4), because skforecast did not work correctly with the older ones.

+ <span class="badge text-bg-fix">Fix</span> <code>[Ets]</code> now estimates its smoothing parameters. A compilation flag disabled the checks of missing components, so every model without damping kept the starting values (alpha=0.1, beta=0.01, gamma=0.01). The estimates now agree with `statsmodels` and R's `forecast::ets`, and the prediction intervals of models without an analytical variance are reproducible and about 17 times faster.

+ <span class="badge text-bg-fix">Fix</span> Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> with `encoding='onehot'` where the predictions were wrong when the series were not in alphabetical order, for example when they were named `'s1'` to `'s10'`.

+ <span class="badge text-bg-docs">Docs</span> The examples and tutorials pages are now a filterable card grid: every tutorial shows an icon, a one line summary and topic tags, and can be narrowed down with a search box and level/topic filters. [Examples](../examples/examples_english.md)

+ <span class="badge text-bg-docs">Docs</span> New home page of the documentation: what skforecast does in one screen, with an animation of real forecasts from LightGBM, Chronos-2 and ARIMA, and sections on model families, global models, production features and AI assistants. [Home](../README.md)

+ <span class="badge text-bg-docs">Docs</span> New section "Install skforecast context in your agent" in the [AI-assisted forecasting](../quick-start/ai-assisted-forecasting.md) guide.


**Added**

+ New functions <code>[get_model_info]</code> and <code>[list_adapters]</code> in `skforecast.foundation`. They return <code>[FoundationModelInfo]</code>, a frozen dataclass with the capabilities and requirements of a foundation model (`adapter`, `model_id_prefixes`, `default_model_id`, `default_context_length`, `backend_package`, `allow_exog`, `supports_past_only_covariates`, `supports_categorical_covariates`, `supports_heterogeneous_covariates`, `supports_nan_in_series`, `supported_quantiles`, `requires_hf_auth`, `requires_provider_auth`, `weights_repo_id`, `weights_in_hf_cache`, `license`, `license_url` and `commercial_use_restricted`). `weights_repo_id` is the Hugging Face repository the weights are actually downloaded from, which is not always `model_id` (e.g. `jingang/TabICL` for `soda-inria/tabicl`), and `weights_in_hf_cache` tells whether they are stored in the Hugging Face Hub cache. `license` is informed for every supported model as an SPDX identifier (e.g. `Apache-2.0`), or as the license name of the Hugging Face model card when it is not a standard license, and `commercial_use_restricted` tells whether it restricts commercial use. `license_url` links to the license file of the provider, or to the model card of `weights_repo_id` for standard licenses. `list_adapters(as_frame=True)` returns the same information as a pandas DataFrame with one row per adapter. Everything is read from the adapter classes and from the same license registry used by `LicenseWarning`, so tools built on skforecast do not need to keep their own copy.

+ New class attributes in every foundation model adapter: `supports_categorical_covariates`, `requires_hf_auth`, `requires_provider_auth`, `weights_repo_id`, `weights_in_hf_cache`, `backend_package`, `default_model_id` and `SUPPORTED_QUANTILES` (`None` when any quantile level in `(0, 1)` is accepted). The installation hints of the `ImportError` raised when a backend is missing are built from `backend_package`.

+ New argument `include_drift` in <code>[Arima]</code> to include a linear drift term when the order is specified manually (`d + D <= 1`), equivalent to `include.drift` in R's `forecast::Arima`. The `best_params_` attribute found by the automatic model selection now also includes `fit_intercept` and `include_drift`, so passing them to `set_params` fits exactly the selected model.

+ Claude Code plugin marketplace (`.claude-plugin/marketplace.json`) that publishes the `skills/` folder as the `skforecast` plugin.

+ `context7.json` to configure how [Context7](https://context7.com/skforecast/skforecast) indexes the documentation (scratch, asset and unrelated folders are excluded) and to give coding agents a short list of rules that prevent the most common mistakes when generating skforecast code.

+ New section "Install skforecast context in your agent" in the [AI-assisted forecasting](../quick-start/ai-assisted-forecasting.md) guide.


**Changed**

+ `numba` is imported and the rolling statistics of <code>[RollingFeatures]</code> and <code>[RollingFeaturesClassification]</code> are JIT compiled on their first use instead of when `skforecast.preprocessing` is imported. This removes around 0.3 seconds from the import of every forecaster module (about 65% of the time spent by skforecast itself once numpy, pandas and scikit-learn are loaded). Behavior is unchanged.

+ `fit` of <code>[ForecasterRecursiveMultiSeries]</code> is faster and uses less memory with many series. The rows of each series in the training matrix are located once, instead of once per series, to split the in-sample residuals and the sample weights, and the training matrix is built in a single pre-allocated block, without the copies that merged its columns. With 500 series of 2,000 observations and a LightGBM of 25 trees, `fit` is 10 to 22% faster (with or without exogenous variables or `calendar_features`), about 3 times faster with `encoding='onehot'` (300 series) and more than 10 times faster with `series_weights` (from about 33 to 2.4 seconds). With `Ridge` it is almost 2 times faster; with heavier estimators the gain is proportionally smaller, and with `encoding=None` without exogenous variables or calendar features there is no change. The peak memory of `create_train_X_y` is halved with float exogenous variables and drops by 60% with `calendar_features`, and that of `fit` drops by 40% with `encoding='onehot'`. Results are unchanged, except for the matrices returned by `create_train_X_y` (next entry).

+ `predict` and the other prediction methods of <code>[ForecasterRecursiveMultiSeries]</code> are faster with many series when they use the last window stored in the forecaster. The series to predict were looked up in a list (quadratic in the number of series) and the last window was built aligning the index of every series. With `LinearRegression`, 24 lags and 24 steps, `predict` takes 1.8 ms instead of 2.6 ms with 50 series, 7.7 ms instead of 17.6 ms with 500 series and 71 ms instead of 349 ms with 5000 series.

+ The prediction methods of all the forecasters check their inputs faster: the missing values of `last_window` and `exog` are checked on their numpy values. With <code>[ForecasterRecursive]</code>, `LinearRegression`, 24 lags and 5 exogenous variables, the check takes 70 µs instead of 116 µs, and `predict(24)` 333 µs instead of 384 µs (174 µs instead of 200 µs without exogenous variables).

+ <code>[ForecasterRecursive]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> with an `ExtraTreesRegressor` or an `ExtraTreeRegressor` as estimator use the same fast prediction path as `RandomForestRegressor` and `DecisionTreeRegressor`, which predicts with each tree directly. The predictions are the same, also with missing values. With an `ExtraTreesRegressor(n_estimators=100)` and 24 lags, `predict(100)` of <code>[ForecasterRecursive]</code> takes 33 ms instead of 508 ms.

+ The backtesting and hyperparameter search functions copy a <code>[ForecasterRnn]</code> twice as fast (72 ms instead of 156 ms with a model of 126,000 parameters), because its Keras model is copied once instead of twice.

+ <code>[multivariate_time_series_corr]</code> only computes the correlations with `time_series` instead of the whole correlation matrix of the lags, with the same results. With 5 series of 2000 values and 24 lags, it takes 19 ms instead of 22 ms with `method='pearson'`, 98 ms instead of 327 ms with `'spearman'` and 87 ms instead of 769 ms with `'kendall'`.

+ In the matrices returned by `create_train_X_y` of <code>[ForecasterRecursiveMultiSeries]</code>, the one-hot columns of the series (`encoding='onehot'`) and the calendar features that <code>[CalendarFeatures]</code> returns as integers are now `float`, as the one-hot columns already were in `create_predict_X` and the calendar features in the other forecasters, and, when `series` is a dict, the index of `X_train` always has the name of the index of the series (with a wide or long DataFrame it has no name, as before). With `series_weights` or `weight_func`, `create_sample_weights` now raises a `ValueError` in three cases: when the rows of a series are not contiguous in `X_train`, when the forecaster has not created the encoding of the series yet, or, with `encoding='onehot'`, when the one-hot column of any series is missing.

+ `fit` of <code>[ForecasterRecursiveMultiSeries]</code> and the searches with <code>[OneStepAheadFold]</code> (<code>[grid_search_forecaster_multiseries]</code>, <code>[random_search_forecaster_multiseries]</code> and <code>[bayesian_search_forecaster_multiseries]</code>) now raise a `ValueError` when the estimator modifies the training matrix in place, for example `LinearRegression(copy_X=False)` or a pipeline with `StandardScaler(copy=False)`. The training matrix is no longer copied before it reaches the estimator, and it is used again after training: `fit` calculates the in-sample residuals with it, and the searches fit the next candidate with it. The grid and random searches, as with any other error, skip the candidate with a `RuntimeWarning` that includes the message, and the Bayesian search stops. Before, depending on the encoding and the exogenous variables, these estimators worked, gave wrong in-sample residuals or search metrics without any warning, or failed with an unrelated error. Use the default copy behavior of the estimator (`copy_X=True`, `copy=True`). The check compares a sample of up to 100 rows of the matrix before and after training, so a modification limited to other rows is not detected, and it ignores the cells that were NaN before training, so a step that fills them in place, such as `SimpleImputer(copy=False)`, is allowed. In `fit` it does not apply when the residuals are not calculated, as in backtesting without prediction intervals.

+ The examples and tutorials pages (English, Spanish and Chinese) are rendered as Material card grids with a search box and level/topic filter chips. Each tutorial now carries a one line summary and tags, and a language switcher links the three pages. The page URLs are unchanged. The three pages are generated at build time from a single source of truth, `tools/docs/hooks/examples.yml`, so the languages can no longer drift apart; add or edit a tutorial there rather than in the Markdown pages.

+ The `__repr__`, `_repr_html_` and `summary` of <code>[ForecasterStats]</code>, and the `params` column of `get_estimators_info`, only show the estimator parameters that differ from their default values. The full set of parameters is still available in the `estimator_params_` attribute.

+ In <code>[Arima]</code>, the in-sample fitted values and residuals (`fitted_values_`, `in_sample_residuals_`, `get_fitted_values` and `get_residuals`) of the first `d + D * m` observations are now NaN. These observations are dominated by the diffuse initialization of the Kalman filter, are excluded from the likelihood and have no meaningful one-step-ahead prediction (the first fitted value was 0). `get_score` and `summary` ignore them.

+ Faster <code>[Arima]</code>. The Kalman filter that computes the likelihood follows the structure of the state space model without branching on every element, and skips the exact zeros of the seasonal AR and differencing polynomials. Seasonal models fit 1.4x to 2.2x faster: ARIMA(1,1,1)(1,1,1)[12] from 0.51 to 0.27 seconds with 500 observations and from 6.9 to 4.2 seconds with 5000, ARIMA(1,0,1)(1,0,1)[24] from 1.44 to 0.67 seconds, and the automatic order selection with `m=12` (`order=None`) from about 3.2 to 1.4-1.7 seconds. Models with an intercept or exogenous variables fit up to 1.5x faster on long series (ARIMA(1,0,1) with an intercept and 5000 observations, from 38 to 28 milliseconds), and `predict_interval` is about 1.2x faster. Non-seasonal models without intercept are unchanged. These optimizations give results identical, bit for bit, to those of the previous implementation (only the models fitted by maximum likelihood on series with missing values change, because of the fix of the Kalman filter described below).

+ The home page of the documentation is now a custom Material template, `docs/overrides/home.html`, instead of the rendered `docs/README.md`. Its animations show real skforecast outputs, stored in `docs/overrides/partials/home-data.json` and regenerated with the scripts in `tools/docs/home_page/` (see its README). The forecasters table stays in [Introduction to forecasting](../introduction-forecasting/introduction-forecasting.md), and the citation is now at the end of the home page. The list of publications citing skforecast is replaced by a link to a Google Scholar search (70+ publications). The team cards of the home page and of the About, Consulting and Sponsorship pages are a single component (`docs/overrides/partials/team.html`, included in Markdown with `pymdownx.snippets`) that also supports the dark scheme.

+ The GitHub README follows the new home page: the model families and production features replace the old "Why use skforecast?" list, an image shows the backtesting of LightGBM, Chronos-2 and ARIMA, and the quick example includes the same workflow with a foundation model. Sections already covered by the documentation (table of contents, what is new, examples) are removed, the badges are reduced to four rows (package, meta, testing and community), the installation comes before the example, every feature links to its user guide, the AI tools are grouped in one section with their badges, the contributors are shown in a picture generated by contrib.rocks, and the sponsorship section shows the Open Collective, Buy Me a Coffee and GitHub Sponsors buttons. The banner has a dark mode version (`images/banner-landing-page-skforecast-dark.png`) with the original colors of the logo.

+ Repository metadata: new `SECURITY.md` (supported versions and private reporting of vulnerabilities, linked from the issue chooser) and pull request template; the Code of Conduct is updated to Contributor Covenant 2.1; the copyright line of `LICENSE` names the copyright holders and no longer needs a yearly update; the PyPI summary is shorter, and the project links add a Funding link and use the canonical URLs.

+ `CITATION.cff` is updated: current description and keywords, repository URL, concept DOI and the ORCID of both authors, and a valid SPDX license identifier (`BSD-3-Clause`, the previous value did not validate against the CFF schema). Zenodo takes the metadata of each release from this file, and it no longer has a `version` or `date-released` to update by hand, as in scikit-learn or pandas: Zenodo takes both from each GitHub release. The citations in the documentation and the README no longer contain a version that had to be edited by hand: they recommend citing the version used, from Zenodo, and give the DOI that always resolves to the latest release.

+ The documentation site serves its fonts and other external assets (images, scripts) from its own domain, using the Material for MkDocs `privacy` plugin, so visitors no longer connect to Google Fonts or other third parties before accepting cookies. Math is now rendered with KaTeX instead of MathJax: it is about four times lighter, renders faster and is served from the site (`docs/vendor/katex`, updated with `tools/docs/vendor_katex.py`). Badges with live values (downloads, versions) are still loaded from their original sources.

+ The tqdm progress bars saved in the documentation notebooks are now static HTML bars with their final values (with a plain text fallback), instead of Jupyter widgets that needed require.js and a script from unpkg.com. `tools/docs/execute_notebooks/execute_notebooks.py` applies the conversion (`tools/docs/execute_notebooks/static_widgets.py`) after executing each notebook. The documentation search now also finds class names by any of their words (for example, "Recursive" finds `ForecasterRecursive`) and keeps version numbers such as 0.25.0 whole.

+ The smoothing parameters and initial states of <code>[Ets]</code> are now optimized with L-BFGS-B, from four starting points, with a finite-difference gradient computed in compiled code, and the best solution is refined with a compiled Nelder-Mead. The likelihood of ETS models is multimodal and Nelder-Mead alone (2000 iterations) did not converge for seasonal models. Estimates and predictions of `Ets` differ from previous versions; the log-likelihoods now match those of `statsmodels` `ETSModel` (e.g. AirPassengers: 1130.00 vs 1129.97 for AAA, 1045.00 vs 1044.98 for MAM, as -2 log-likelihood; small differences come from the lower bound 1e-4 of the smoothing parameters, as in R). Since the previous version did not estimate the smoothing parameters, simple models take longer to fit (a few milliseconds), additive seasonal models take about the same time as before (AAA with `m=12` and 144 observations: 40 ms), multiplicative seasonal models take somewhat longer (MAM: 65 ms instead of 40 ms) and the automatic selection with `m=12` takes about 0.9 seconds instead of 0.7.

+ <code>[select_features]</code> and <code>[select_features_multiseries]</code> now sample the records without replacement and keep them in their original order. Previously they were sampled with replacement (around 22% of the sampled rows were duplicates with `subsample=0.5`), so selectors with internal cross-validation, such as `RFECV` or `SequentialFeatureSelector`, could see the same record in train and validation. In <code>[select_features]</code>, because of the time order, a `TimeSeriesSplit` can now be used as the `cv` of the selector (in <code>[select_features_multiseries]</code> the series are stacked one after another, so it does not give a temporal validation). The selected features for a given `random_state` may differ from previous versions.

+ The foundation model adapters share a private base class, `_AdapterBase` (`skforecast/foundation/_adapter_base.py`), that declares the contract <code>[FoundationModel]</code> relies on: the capability class attributes and the `fit`, `predict`, `get_params` and `set_params` methods. An adapter that does not declare one of the capability attributes in its own class body now raises a `TypeError` when the class is defined, instead of failing later in `get_model_info` or at predict time. Behavior is unchanged.

+ The <code>[LicenseWarning]</code> raised when TabPFN-TS is loaded now names the license of the weights that the backend actually downloads, `TabPFN-3.5 License v1.0`, and links to `Prior-Labs/tabpfn_3_5`. It previously pointed to the license of `Prior-Labs/tabpfn_3`, an older version of the weights. `tabpfn-time-series>=1.3` is now the documented minimum version of the backend, since earlier versions download other weights; it is the version the `ImportError` hint asks to install.

+ The `theforecastingcompany/t0*` checkpoints are no longer gated on the Hugging Face Hub, so the documentation of `T0Adapter` no longer asks to accept the license and authenticate before using them. The `OSError` raised when the checkpoint configuration cannot be downloaded now points first to the model ID and the network connection.

+ The minimum supported versions of pandas and scikit-learn are now 2.2 and 1.6 (previously 2.1 and 1.4), because skforecast did not work correctly with the older ones:
    + pandas 2.1: `fit` failed with a target series of a nullable dtype (`Int64` or `Float64`), raising `TypeError: ufunc 'isnan' not supported for the input types`, and <code>[reshape_series_wide_to_long]</code> and <code>[calculate_distance_from_holiday]</code> (when the holiday column has missing values) raised errors, since they use features added in pandas 2.2.
    + scikit-learn 1.4 and 1.5: `ExtraTreesRegressor`, `ExtraTreeRegressor` and their classifiers do not accept missing values, but skforecast treated them as if they did. With <code>[ForecasterRecursiveMultiSeries]</code>, <code>[backtesting_forecaster_multiseries]</code> raised `ValueError: Input X contains NaN` when the last window of a series had missing values, instead of skipping the predictions of that series. With scikit-learn 1.4, fitting a `LinearRegression` on the training matrices returned by `create_train_X_y` also raised `ValueError: cannot set WRITEABLE flag to True of this array`.

+ The minimum supported version of statsmodels (`stats` and `plotting` extras) is now 0.13.2 (previously 0.13). statsmodels 0.13.0 cannot be installed in the Python versions supported by skforecast (it has no wheel for them and the build from source fails), and 0.13.1 cannot be imported with pandas 2, the minimum required by skforecast (`ImportError: cannot import name 'Int64Index' from 'pandas'`).

+ The minimum supported version of keras (`deeplearning` extra) is now 3.3 (previously 3.0). With keras 3.0 to 3.2 and the PyTorch backend, a Keras model cannot be copied, so <code>[ForecasterRnn]</code> raised `TypeError: cannot pickle 'module' object` when it was created.

+ The minimum supported version of numpy is now 1.26.1 (previously 1.26). numpy 1.26.0 cannot load objects pickled with numpy 2, so a forecaster saved with <code>[save_forecaster]</code> in an environment with numpy 2 could not be loaded with it (`ModuleNotFoundError: No module named 'numpy._core'`).

+ Removed the function `cast_exog_dtypes` from `skforecast.utils` (added in 0.8.0). It was not used by skforecast and did not work as documented: with a pandas Series it raised `AttributeError`, it modified the DataFrame passed by the user and it lost the categories. Use `exog.astype(exog_dtypes)` instead.

+ `set_out_sample_residuals` of <code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveMultiSeries]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code>, <code>[ForecasterEquivalentDate]</code> and <code>[ForecasterRnn]</code> now issues a <code>[ResidualsUsageWarning]</code> when the out-of-sample residuals have, on average, fewer than 10 residuals per bin (for example, 48 residuals with the default `n_bins=10`). With so few values per bin, the intervals obtained with `use_binned_residuals=True` are too narrow: in simulations with a nominal coverage of 95% and 24 to 60 out-of-sample residuals, the empirical coverage was 59 to 83% with 10 bins and 90 to 97% without bins, for both `'bootstrapping'` and `'conformal'`. The warning suggests providing more residuals, reducing `n_bins` in `binner_kwargs` or predicting with `use_binned_residuals=False`. The stored residuals and the predictions are unchanged. In the forecasters with several series, a single warning lists the affected levels. [User guide](../user_guides/probabilistic-forecasting-bootstrapped-residuals.ipynb#intervals-conditioned-on-predicted-values-binned-residuals)

+ <code>[show_versions]</code> also reports the versions of scipy, statsmodels, matplotlib, torch, lightgbm, xgboost, catboost, skops and cloudpickle (`None` when a package is not installed).

+ <code>[save_forecaster]</code> keeps the dots in `file_name` and adds the extension of the backend, so `'model_v1.2'` is saved as `'model_v1.2.joblib'`. Previously, everything after the last dot was replaced by the extension: `'model_v1.1'` and `'model_v1.2'` were both saved as `'model_v1.joblib'`, and the second one overwrote the first without any warning. A name that ends with a backend extension (`.joblib`, `.pkl`, `.pickle`, `.cloudpickle` or `.skops`) is saved as before, with that extension replaced by the one of the backend. Any other extension is now kept: `'model.bin'` is saved as `'model.bin.joblib'` instead of `'model.joblib'`. [User guide](../user_guides/save-load-forecaster.ipynb#pickle-backend)

+ <code>[save_forecaster]</code> saves the `.py` files of the custom weight functions defined in `'__main__'` (`save_custom_functions=True`) in the folder of the forecaster file instead of the working directory, so two forecasters saved in different folders with functions of the same name no longer overwrite each other's file. A forecaster saved in the working directory keeps its files there. If it is saved in another folder, for example `models/forecaster.joblib`, the function is imported with `from models.custom_weights import custom_weights` before loading it. [User guide](../user_guides/save-load-forecaster.ipynb#forecaster-with-custom-features)

+ <code>[Arima]</code> corrects the innovation variance `sigma2_` for the degrees of freedom, as R's `forecast::Arima` does: the sum of squared innovations is divided by the number of innovations minus the number of estimated coefficients. Before, it was divided by the number of innovations (the maximum likelihood estimate), which is biased downwards and made the prediction intervals too narrow in short series and in models with many coefficients. Prediction intervals are now slightly wider, about 1% to 4% in series of 70 to 150 observations. In a simulation with 30 observations, the coverage of the 95% intervals increased from 87.2% to 90.3% for an AR(1) with three exogenous variables and from 84.9% to 87.5% for an ARMA(2,1), and it did not exceed the nominal level in any of the scenarios. Point predictions, coefficients, log-likelihood, information criteria and the model chosen by the automatic selection are unchanged. With `lambda_bc` and `biasadj=True`, fitted values and predictions change slightly because the bias adjustment uses this variance. `sigma2_` no longer matches the value of statsmodels' SARIMAX and R's `stats::arima`; the maximum likelihood estimate is still available in `model_['sigma2_ml']`. ([#1365](https://github.com/skforecast/skforecast/issues/1365))


**Fixed**

+ The prediction intervals of <code>[Ets]</code> models with multiplicative errors and additive or no trend and seasonality (MNN, MAN, MAdN, MNA, MAA, MAdA) are now computed from the analytical forecast variance (class 2 models in Hyndman et al., 2008), as R's `forecast.ets` does, instead of from 1000 simulated paths.

+ `seas_heuristic`, used by `nsdiffs` and the automatic order selection of <code>[Arima]</code> to choose the number of seasonal differences, computed the seasonal strength with a centered moving average instead of STL, so it could choose a different `D` than R's `forecast::nsdiffs` for series near the 0.64 threshold. It now uses the same STL decomposition as R's `forecast::mstl`.

+ The AICc used by the automatic model selection of <code>[Ets]</code> did not count the variance as a parameter, so its small-sample correction was smaller than the one in the AIC and in R's `forecast::ets`. It now uses the same number of parameters as the AIC, which can change the selected model for short series.

+ `ndiffs`, used by <code>[Arima]</code> to choose the order of differencing in the automatic model selection, forced at least one lag in the KPSS test. R's `forecast::ndiffs` uses `trunc(3 * sqrt(n) / 13)` lags, which is 0 for fewer than 19 observations, so the number of differences could differ from R for short series. It now uses the same number of lags.

+ <code>[FoundationModel]</code> only routes Chronos-2 checkpoints (`amazon/chronos-2*` and `autogluon/chronos-2*`) to `ChronosAdapter`. Chronos (T5) and Chronos-Bolt checkpoints were accepted when the model was created but failed at predict time, because their pipelines do not accept the input format and the `cross_learning` argument used by the adapter. They now raise a `ValueError` when the model is created.

+ <code>[FoundationModel]</code> only routes Moirai-2 checkpoints (`Salesforce/moirai-2*`) to `MoiraiAdapter`. Moirai 1.x and Moirai-MoE checkpoints were accepted when the model was created but failed when the weights were loaded, because their configurations lack arguments required by `Moirai2Module`. They now raise a `ValueError` when the model is created.

+ The `supports_categorical_features` tag of <code>[ForecasterFoundation]</code> was always `True`. It is now read from the adapter, and is only `True` for Chronos-2, the only backend that handles non-numeric covariates natively.

+ <code>[FoundationModel]</code> `fit` now ignores `exog`, with an `IgnoredArgumentWarning`, when the model does not support exogenous variables (TimesFM 2.5 and Moirai-2), as <code>[ForecasterFoundation]</code> `fit` already did. Previously, when `FoundationModel` was used directly, the exog was stored: `exog_in_` and `exog_names_in_` reported it, `context_exog_` was not `None` as documented, and with TimesFM 2.5 every later `predict` warned that the covariates were ignored. Nothing changes when using <code>[ForecasterFoundation]</code>.

+ Fixed an issue in <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code> where, with `refit`, `use_in_sample_residuals=False` and `use_binned_residuals=True`, the out-of-sample residuals set by the user were restored after each `fit()` but the binner that created them was not, so the intervals could be built with the residuals of a different bin. The binner and its intervals are now restored together with the residuals in every fold.

+ Fixed an issue in <code>[backtesting_forecaster_multiseries]</code> where <code>[ForecasterDirectMultiVariate]</code> raised `TypeError: 'NoneType' object is not subscriptable` with `use_in_sample_residuals=False`, because only one of `out_sample_residuals_` and `out_sample_residuals_by_bin_` was restored after each `fit()` depending on `use_binned_residuals`. Both attributes are now restored.

+ Fixed two issues with time zone aware indexes in a time zone with daylight saving time (for example, `Europe/Madrid`): the dates that skforecast generates from the index of a series (for example, the index of the predictions) were wrong when they crossed a daylight saving change. Intraday frequencies and indexes without time zone were not affected. The issue depended on how the timestamps were created:
    + Timestamps created in UTC and then converted to the local time zone (for example, daily data stamped at UTC midnight), with a frequency of days or longer (`'D'`, `'W'`, `'MS'`...). Their local time of day shifts one hour at each daylight saving change, but the dates were generated keeping the local time of day constant. When the forecast horizon crossed the change in autumn, `predict` and the backtesting and search functions raised `AmbiguousTimeError: Cannot infer dst time from ...`; in spring, the predictions were returned with timestamps shifted one hour, which could later raise a `KeyError` in backtesting. It affected all the forecasters, <code>[TimeSeriesFold]</code> and <code>[OneStepAheadFold]</code> with a dictionary of series, and `steps` or `initial_train_size` given as a date. <code>[reshape_series_long_to_dict]</code>, <code>[reshape_exog_long_to_dict]</code> and <code>[reshape_series_wide_to_long]</code> raised the same error in autumn and, in spring, silently replaced the values after the change with NaN. The index is now extended in fixed UTC steps when the timestamps follow this convention.
    + Timestamps created directly in the local time zone (for example, daily data at local midnight), with a daily frequency (`'D'`, `'2D'`...), when the change fell between the last date of the series and the first step predicted. The index of the predictions was generated adding a fixed 24 hours to the last date, so all the predictions were stamped one hour off (for example, at 01:00 instead of 00:00 after the spring change), and `predict` with `exog` raised `` ValueError: To make predictions `exog` must start one step ahead of `last_window`. `` although `exog` started at the right date. It affected all the forecasters and <code>[expand_index]</code>. The local time of day is now kept.

+ Fixed an issue with time zone aware indexes where a date without time zone, given as `steps` in `predict` of <code>[ForecasterRecursive]</code> and <code>[ForecasterRecursiveClassifier]</code>, or as `initial_train_size` in <code>[TimeSeriesFold]</code>, <code>[OneStepAheadFold]</code> and the backtesting and search functions, raised `TypeError: Cannot compare tz-naive and tz-aware timestamps`. The date is now interpreted in the time zone of the index, and a date with another time zone is converted to it.

+ Fixed an issue in <code>[TimeSeriesFold]</code> and <code>[OneStepAheadFold]</code> where `initial_train_size` given as a date was converted into a wrong number of training observations, without any warning, when the index of the series had no frequency (for example, 2 instead of 25 with an hourly index without `freq`), because the index was expanded as if it were daily. The number of training observations is now the number of dates in the index up to that date.

+ Fixed an issue in <code>[ForecasterRecursive]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> with an `XGBRegressor` as estimator, where the predictions differed from those of `XGBRegressor.predict` without any warning. The optimized prediction path used all the trees even when early stopping (`early_stopping_rounds`) had selected a better iteration, and ignored a `missing` value set by the user. With `booster='gblinear'`, `predict` raised `XGBoostError: Inplace predict is not supported by the current booster`. The predictions are now computed as with `XGBRegressor.predict`, which also corrects the prediction intervals and the results of backtesting and hyperparameter search.

+ Fixed an issue in <code>[ForecasterRecursive]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> where a user subclass of a scikit-learn linear model that overrides `predict` (for example, a `Ridge` that clips its predictions at 0) was predicted as its base class, because the optimized prediction path computes the predictions directly from `coef_` and `intercept_`. Subclasses of `RandomForestRegressor` and `DecisionTreeRegressor` that keep the class name were affected in the same way. User subclasses now use their own `predict` method; scikit-learn estimators keep the optimized path.

+ Fixed an issue in <code>[ForecasterRecursiveClassifier]</code> with a `CatBoostClassifier` as estimator, where `predict`, `predict_proba` and <code>[backtesting_forecaster]</code> raised `CatBoostError: 'data' is numpy array of floating point numerical type, it means no categorical features, but 'cat_features' parameter specifies nonzero number of categorical features`. CatBoost requires its categorical features (the lags with the default `features_encoding='auto'`, and the categorical exogenous variables) as integers, but they were only cast to integer in `fit`. They are now also cast when predicting.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> with `encoding='ordinal_category'` and a `CatBoostRegressor` as estimator, where the hyperparameter search functions with <code>[OneStepAheadFold]</code> (<code>[grid_search_forecaster_multiseries]</code>, <code>[random_search_forecaster_multiseries]</code> and <code>[bayesian_search_forecaster_multiseries]</code>) computed wrong metrics without any warning when a series had no data in the test period (for example, a series that ends before it). CatBoost uses the codes of the categorical column that identifies the series, and in the test set these codes only counted the series present, so the series after the missing one were predicted as another series. The codes are now the same in the training and test sets.

+ Fixed an issue in <code>[ForecasterRecursive]</code> and <code>[ForecasterRecursiveMultiSeries]</code> with an `XGBRegressor` or an `LGBMRegressor` set to run on GPU. These forecasters predict on CPU and then restore the original device of the estimator, but the device names were translated. With XGBoost, `device='cuda:0'` (or another GPU ordinal) made `predict` and the other prediction methods raise `` ValueError: `device` must be 'gpu', 'cpu', 'cuda', or None. ``, and `device='gpu'` was restored as `'cuda'`. With LightGBM, `device='cuda'` was restored as `'gpu'`, a different backend (OpenCL), which the next `fit` then used. The original device is now restored as is, and an estimator without a device set is no longer left with `device='cpu'`. The function `set_cpu_gpu_device` of the <code>[utils]</code> module, which does this, now passes the device to the estimator as is and no longer modifies CatBoost models (it set `task_type='CPU'` on an unfitted model without restoring it).

+ Fixed an issue in <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code> with `differentiation` where the predictions were wrong, without any warning, when `steps` was not consecutive from 1 (for example, `steps=[3, 4, 5]`). Only the requested steps were predicted, but reverting the differentiation accumulates the predictions of all the previous steps, and the conformal intervals were scaled as if the requested steps were the first ones. This affected `predict`, `predict_interval`, `predict_bootstrapping`, `predict_quantiles` and `predict_dist`, and also <code>[backtesting_forecaster]</code>, <code>[backtesting_forecaster_multiseries]</code> and the hyperparameter search functions that use backtesting when `gap > 0`, because they predict only the steps after the gap. Now all the steps from 1 to `max(steps)` are predicted, the differentiation is reverted and then the requested steps are selected, so `predict(steps=[3, 4, 5])` returns the last three values of `predict(steps=5)`.

+ Fixed an issue in <code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveClassifier]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code>, <code>[ForecasterStats]</code> and <code>[ForecasterRnn]</code> where `predict` accepted an `exog` whose index did not follow the frequency of the series (gaps, duplicated dates or another frequency, for example hourly values in a daily model), as long as its first date was the first step predicted. These forecasters use `exog` by position, so the predictions used the values of other dates without any warning. `predict` now raises a `ValueError` that shows the first date that does not match and how to add the missing dates as NaN, `exog.reindex(expand_index(last_window.index, steps=...))`. An `exog` with extra rows, without `freq` (for example, read from a CSV file) or with gaps after the last step predicted is still accepted.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> where a wide `exog` (a pandas Series or DataFrame with the same values for all the series) was used by position in `predict` and the other prediction methods, while `fit`, a dict `exog` and <code>[backtesting_forecaster_multiseries]</code> align it with the series by date. When `exog` did not start at the first step predicted (for example, the full `exog` including the training period), the predictions used the values of other dates without any error. When it was shorter than `steps` or missed a column, the prediction failed after a warning that promised NaN. A wide `exog` is now aligned with the predictions by date and column, as a dict `exog`, and the missing values are filled with NaN with a `MissingValuesWarning`. For all the `exog` formats, this warning is only raised when the dates of some steps are missing (not when `exog` starts before the first step predicted) and names the first missing date, and an `exog` with duplicated dates raises a `ValueError` instead of a pandas error. Note that a wide `exog` whose index does not match the index of the predictions, but whose values are in the right order (for example, a `RangeIndex` reset to start at 0), was used by position with a warning; its values are now aligned by index, so they are NaN. Give `exog` the index of the predictions.

+ Fixed an issue in the `set_in_sample_residuals` method of <code>[ForecasterRecursiveMultiSeries]</code>, which raised `KeyError: '[...] not in index'` when the series had a `RangeIndex` and there were more than 10,000 training residuals (with a `DatetimeIndex`, a pandas `FutureWarning`). The residuals were sampled by label instead of by position; they are now the same as those stored by `fit`, and they are stored as numpy arrays, as `fit` does, instead of pandas Series.

+ Fixed an issue in <code>[grid_search_forecaster_multiseries]</code>, <code>[random_search_forecaster_multiseries]</code> and <code>[bayesian_search_forecaster_multiseries]</code> with <code>[OneStepAheadFold]</code> when the <code>[ForecasterRecursiveMultiSeries]</code> had already been fitted with other series: the series encoding kept the levels of the previous fit, so rows could be assigned to the wrong series and the metrics and sample weights could be wrong.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> with `encoding='onehot'` where the predictions were wrong when the series were not in alphabetical order: the training matrix orders the one-hot columns alphabetically, but the prediction matrix followed the order of the input. It affected `predict`, the probabilistic prediction methods, `create_predict_X` and backtesting since at least version 0.19.0, with a dict or a wide DataFrame whose series were not sorted alphabetically, which includes names in natural order such as `'s1'` to `'s10'` (they sort as `'s1'`, `'s10'`, `'s2'`). Predicting also failed when a series lost all its training rows because of missing values, because the prediction matrix had one column less than the training matrix. In `create_predict_X`, a series not seen during training now gets all its one-hot columns set to 0, as in `predict`, instead of raising a `ValueError`.

+ Fixed an issue in <code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveClassifier]</code>, <code>[ForecasterDirect]</code> and <code>[ForecasterEquivalentDate]</code> where a `last_window` DataFrame with several columns was accepted. Its values were flattened into a single array, so the lags mixed the values of the different columns and the predictions were wrong without any warning. A `ValueError` is now raised when `last_window` has more than one column.

+ <code>[RollingFeatures]</code> raised `TypeError: argument of type 'NoneType' is not iterable` when `kwargs_stats=None`, although `None` is accepted by the parameter validation and by the type hint. The default value of `kwargs_stats` is now `None`, which is replaced by the documented default `{'ewm': {'alpha': 0.3}}`.

+ The `verbose` header of <code>[select_features]</code> and <code>[select_features_multiseries]</code> is now `Feature selection (<selector name>)` instead of `Recursive feature elimination (<selector name>)`, which was wrong for selectors other than `RFE` and `RFECV`.

+ <code>[select_features]</code> and <code>[select_features_multiseries]</code> fitted the selector with a single record when `subsample=1` was passed as an integer. `subsample` is now always a proportion (type hint `float`), so `subsample=1` uses all the records.

+ Fixed an issue in <code>[select_features_multiseries]</code> with a <code>[ForecasterRecursiveMultiSeries]</code> and `encoding='onehot'`, where the one-hot column of a series without training rows could be returned in `selected_exog`.

+ Fixed an issue in <code>[Arima]</code> where the regression coefficients of the exogenous variables, the intercept and the drift were wrong whenever the model had two or more of them, for example, one exogenous variable plus the intercept (the default with `d + D = 0`) or two exogenous variables. The optimizer estimated the coefficients in the basis of the original regressors, but they were then transformed as if they had been estimated in the rotated basis used to improve numerical conditioning. In-sample fitted values were correct, but `coef_` and the predictions of `predict` and `predict_interval` were not, and could be far from the data (even negative for a positive series). The coefficients also depended on the platform, because the sign of the rotation depends on the linear algebra library. This affected <code>[ForecasterStats]</code> with an `Arima` estimator and exogenous variables, including automatic order selection (`order=None`). The coefficients now match those of `statsmodels` SARIMAX.

+ Fixed an issue in <code>[Arima]</code> and <code>[Ets]</code> where `get_params` did not return all the constructor parameters, so `sklearn.base.clone` silently reset the missing ones to their default values. <code>[ForecasterStats]</code> clones its estimators in `__init__`, `fit` and `set_params`, so these settings were ignored by the forecaster, by <code>[backtesting_stats]</code> and by the hyperparameter search functions. In `Arima`, this affected all the automatic model selection settings (`max_p`, `max_q`, `max_P`, `max_Q`, `max_order`, `max_d`, `max_D`, `start_p`, `start_q`, `start_P`, `start_Q`, `stationary`, `seasonal`, `ic`, `stepwise`, `nmodels`, `trace`, `approximation`, `truncate`, `test`, `test_kwargs`, `seasonal_test`, `seasonal_test_kwargs`, `allowdrift`, `allowmean`, `lambda_bc` and `biasadj`). In `Ets`, it affected `lambda_param`, `lambda_auto`, `bias_adjust`, `bounds` and `ic`.

+ Fixed an issue in <code>[Arima]</code> with automatic model selection (`order=None`) when the selected model included a drift term (`d + D = 1` with a constant, the usual case for series with trend). With exogenous variables, `predict` and `predict_interval` raised `ValueError: matmul: ...` because the drift column was not extended over the forecast horizon, and the drift coefficient was named `exog1`. In <code>[backtesting_stats]</code> with `freeze_params=True` (default), the frozen model lost the drift term: the first fold was predicted without it (forecasts close to the level of the detrended series instead of the level of the data) and the following refits could not include it. Now the prediction always adds the drift term when the fitted model has it, and the frozen parameters include `fit_intercept` and `include_drift`.

+ Fixed the Box-Cox transformation (`lambda_bc`, `biasadj`) in <code>[Arima]</code>. With a manual order, `lambda_bc` was silently ignored, so the model was fitted and predicted on the original scale (this also affected the refits of <code>[backtesting_stats]</code> with `freeze_params=True`). With automatic model selection, `fitted_values_` and `in_sample_residuals_` were on the transformed scale while `y_train_` was on the original one, so `get_score` and the residual statistics of `summary` were meaningless. Now the transformation is applied in both modes, fitted values are back-transformed (bias adjusted if `biasadj=True`) and residuals are computed on the original scale, as in <code>[Ets]</code>. `best_params_` also includes the `lambda_bc` used, so frozen models keep the transformation. When the differenced series was constant, the automatic selection did not back-transform the forecasts either.

+ Fixed two issues in the automatic model selection of <code>[Arima]</code> with a drift term. When the differenced series was constant (a perfectly linear series such as `np.arange(50)`, or a seasonal pattern growing linearly), the forecasts were flat. Now, as in R's `forecast::auto.arima`, the drift is fixed at the mean of the differenced series when `d + D = 1` (unless `allowdrift=False`), so the forecasts extend the trend. In addition, when the series started with missing values, the future drift was computed from the length of the series including them, which shifted the forecasts by the drift times the number of leading missing values. The drift now continues from its last training value.

+ <code>[Arima]</code> now computes the BIC (and the AICc) for models with a manual order, using the same definition as the automatic selection and R's `forecast::Arima`. Before, `bic_` was `None` and `get_info_criteria('bic')` returned NaN. It is still not available with `method='CSS'`, which does not compute a likelihood.

+ Fixed an issue in <code>[Arima]</code> where the Kalman filter did not propagate the state covariance over missing values. On a missing observation, it kept the previous filtered covariance instead of the predicted one, so the uncertainty did not grow over the gap, and when the series started with missing values the stationary and diffuse priors of the initial state were lost. This affected the likelihood, coefficients, fitted values, residuals and prediction intervals of the models estimated by maximum likelihood (`method='ML'`, and the default `'CSS-ML'`, which switches to ML when the series has missing values) on series with missing values, including <code>[ForecasterStats]</code> with an `Arima` estimator. For example, an ARIMA(1,1,1) with a manual order on a series starting with three missing values estimated coefficients of the opposite sign. The automatic order selection drops the leading missing values, so there only the gaps inside the series were affected. For stationary models, the likelihood now matches the exact Kalman filter of `statsmodels` SARIMAX.

+ <code>[Arima]</code> raised `TypeError: ufunc 'bitwise_or' not supported for the input types, ...` when it was fitted on an empty series with an intercept or exogenous variables. It now raises `ValueError: Too few non-missing observations`, as without them.

+ Fixed an issue in <code>[Arar]</code> where `fit` overwrote the `max_ar_depth` and `max_lag` parameters with the values determined from the series length when they were `None`, so refitting the same estimator on another series reused the limits of the first one. The values used are now stored in the new fitted attributes `max_ar_depth_` and `max_lag_`.

+ Fixed a confusing `NotImplementedError` about `last_window` raised by <code>[backtesting_stats]</code> when `refit` was an integer other than 1 (intermittent refit) and the forecaster contained estimators other than <code>[Sarimax]</code>. As with `refit=False`, `refit` is now set to `True` and an `IgnoredArgumentWarning` is issued.

+ Fixed an issue in <code>[backtesting_stats]</code> where a <code>[ForecasterStats]</code> with a single estimator raised `IndexingError: Too many indexers` when `gap > 0` and no prediction interval was requested (`interval=None` and `alpha=None`). In this case the predictions are a pandas Series, and the first `gap` steps were removed with DataFrame indexing. This also affected <code>[grid_search_stats]</code> and <code>[random_search_stats]</code> with a `gap`.

+ Fixed an issue in <code>[backtesting_stats]</code> with several estimators and `freeze_params=False`, where the `estimator_params` column was not aligned with the `estimator_id` column: the predictions alternate between estimators at every step, but the parameters were grouped by estimator, so many rows showed the parameters of another estimator. Predictions and metrics were not affected.

+ Fixed the estimation of the smoothing parameters of <code>[Ets]</code>. The functions that check the parameters during the optimization were compiled with `fastmath`, which assumes there are no NaN values, so the checks of the components absent from the model (no trend, no seasonality or no damping) rejected every candidate. Every model without a damped trend kept the starting values of the smoothing parameters (alpha=0.1, beta=0.01, gamma=0.01, phi=0.98); only the initial states were estimated (for seasonal models, without converging in the 2000 iterations of Nelder-Mead). The automatic model selection (`model='ZZZ'`) compared these unfitted models. Other errors of the estimation are fixed in the same change, all of them present in 0.25.0:
    + The usual bounds of beta, gamma and phi were read from the wrong positions of the bounds vector.
    + The admissibility check of seasonal models used a characteristic polynomial of the wrong degree with the reciprocal roots, and its root finding failed whenever a root was complex, which rejected the candidate. It now uses the polynomial of R's `forecast::ets`, evaluated with a Schur-Cohn stability test (about 50 times faster than computing the roots).
    + Models without damping were checked for admissibility with `phi=NaN` instead of `phi=1`.
    + Fixed smoothing parameters (`alpha`, `beta`, `gamma` and `phi`) were estimated anyway. They now keep their values, constrain the estimated ones (`beta <= alpha <= 1 - gamma` with the usual bounds) and raise a `ValueError` when they are out of range or leave no admissible value for the estimated ones (as R's `forecast::ets`).
    + The optimization started with `phi` at its upper bound. The starting values now follow R's `initparam`.
    + A trend that became non-positive in a multiplicative trend model was flagged with the value -99999, which collided with real values: an additive model of a series around -200000 got an infinite AIC. The point forecasts used the same flag; they are now NaN.
    + The initial states were bounded to [-1e6, 1e6], so series of larger magnitude could not be fitted properly. They are now unbounded, as in R.

+ <code>[Ets]</code> raises a `ValueError` when a model with multiplicative components is fitted to a series with zero or negative values (previously it was fitted, e.g. an `MAN` model on a standardized series), and the automatic model selection only considers additive models for such series, or when `lambda_auto=True`, as R's `forecast::ets`.

+ Fixed the Box-Cox transformation of <code>[Ets]</code>. `lambda_auto=True` always selected `lambda=-1` (it minimized the variance of the transformed series); it now uses Guerrero's method with the seasonal period, as R's `forecast::BoxCox.lambda(x, lower = -0.9)`. The bias-adjusted back-transformation of the point forecasts (`bias_adjust=True`) missed a factor and used the one-step variance at every horizon; it is now `y * (1 + v_h * (1 - lambda) / (2 * y^(2 * lambda)))`, with `v_h` the forecast variance of each horizon on the transformed scale (`y * (1 + v_h / 2)` when `lambda=0`), as R's `forecast.ets` and `InvBoxCox`. Long-horizon point forecasts of Box-Cox models are therefore slightly higher (AirPassengers, 24 steps ahead: 1 to 2%). The prediction intervals combined the forecast on the original scale with the standard deviation on the transformed scale (or, when simulated, were left on the transformed scale); they are now computed on the transformed scale and back-transformed. As in R's `InvBoxCox`, bounds outside the range of the transformation (`lambda * y + 1 < 0`) are NaN when `lambda < 0` and keep their sign otherwise (they were positive values with even powers, such as `lambda=0.5`).

+ The prediction intervals of <code>[Ets]</code> models without an analytical variance (multiplicative errors) were simulated without a seed, so `predict_interval` returned different values on every call. The simulation now uses a fixed seed (new argument `random_state=123` of `simulate_ets`) and is compiled with numba: about 4 ms instead of 70 ms for 1000 paths of 12 steps.

+ The automatic model selection of <code>[Ets]</code> (`model='ZZZ'`) ignored `lambda_param` (and fitted the model without the Box-Cox transformation) and `bias_adjust` (for the fitted values). Both are now applied and, as in R's `forecast::ets`, only additive models are considered when `lambda_param` is given. The internal function `ets(model='ZZZ')` raised `KeyError: 'Z'` when `m <= 24`; it now runs the automatic selection.

+ <code>[Ets]</code> now raises a descriptive `ValueError` when `model` is not valid, instead of `KeyError`. This includes partial automatic specifications such as `'ZZN'`, which are not supported: use `model='ZZZ'` and restrict the search with `seasonal`, `trend`, `damped`, `allow_multiplicative` and `allow_multiplicative_trend`.

+ <code>[save_forecaster]</code> raised a `SaveLoadSkforecastWarning`, asking to save the class manually, when the `window_features` included a <code>[RollingFeaturesClassification]</code>, which is part of skforecast. The warning is now raised only for user-defined classes.

+ Fixed an issue in <code>[save_forecaster]</code> where the `.py` files of the custom weight functions (`weight_func`) were written with the default encoding of the platform instead of UTF-8. On Windows, a character outside its code page (for example, `σ`) raised `UnicodeEncodeError` after the forecaster file was written, and other non-ASCII characters (for example, `ñ` in a string) produced a file that could not be imported.

+ Fixed an issue in <code>[save_forecaster]</code> where the `.py` files of the custom weight functions defined in `'__main__'` did not include the imports they use. The loaded forecaster predicted, but refitting it (`fit`, or a backtesting with `refit`) raised `NameError: name 'np' is not defined`, also with the function of the user guide. The files now start with the import statements of the modules, functions and classes the function uses. The other objects it uses from outside its body (global variables, functions defined in `'__main__'`) cannot be written as an import: a `SaveLoadSkforecastWarning` names them and recommends `backend='cloudpickle'`.

+ The forecasters raised `TypeError: module, class, method, function, traceback, frame, or code object was expected, got partial` when `weight_func` was a `functools.partial` or a callable object, because they read its source code. They are now accepted (`source_code_weight_func` is `None` for them). When they are defined in `'__main__'`, <code>[save_forecaster]</code> saves the function wrapped by the `functools.partial` as a `.py` file, and raises a `SaveLoadSkforecastWarning` that recommends `backend='cloudpickle'` for the callable objects and lambda functions, which cannot be saved as a `.py` file (with `backend='skops'`, a lambda function was written to a `<lambda>.py` file). A function whose source code is not available (for example, defined in the Python console, which raised `OSError: could not get source code`) is also accepted and reported in the same warning when the forecaster is saved.

+ Fixed an issue in <code>[RollingFeatures]</code> and <code>[RollingFeaturesClassification]</code> where `transform_batch` stored the pandas `Rolling` objects of the transformed series, so every forecaster fitted with these `window_features` kept a reference to its training series. <code>[save_forecaster]</code> with `backend='skops'` failed for these forecasters (`TypeError: no default __reduce__ due to non-trivial __cinit__` when saving or, with a `RangeIndex`, `TypeError: RangeIndex(...) must be called with integers` when loading), and the other backends wrote the training series to the file (with joblib and 500,000 values, 16 MB instead of 6 KB). The `Rolling` objects are no longer stored.

+ Fixed several issues in <code>[save_forecaster]</code> and <code>[load_forecaster]</code> with `backend='skops'` and a `DatetimeIndex`. The index was stored as text and is now stored as integers, with its unit, the name of its time zone and its frequency:
    + The time zone was reduced to the UTC offset of each timestamp. A forecaster trained on a series with a daylight saving time change could not be loaded (`ValueError: Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True`), and one without it was loaded with a fixed offset (for example, `UTC+01:00` instead of `Europe/Madrid`), so the predictions beyond a later change had their labels shifted.
    + Frequencies below one second (for example, `'500ms'`) could not be loaded.
    + The parameters of the frequency were lost (for example, the holidays of a `CustomBusinessDay`), so loading or `predict` failed.
    + Long series were slow to save and load: a <code>[ForecasterEquivalentDate]</code> trained on 200,000 values took 3.4 seconds to save and 3.4 seconds to load, with a 58 MB file (now 0.02 and 0.01 seconds, and 3.3 MB).
    + Saving now raises a `ValueError` when the time zone cannot be rebuilt from its name (for example, a `dateutil` time zone). Convert the index to a named time zone (for example, `'Europe/Madrid'`) or use another backend.
    + Files saved with previous versions are still loaded. Those with a frequency below one second, and most of those with a daylight saving time change, failed and are now loaded (with a fixed UTC offset, the only time zone information they stored).

+ Fixed an issue in <code>[save_forecaster]</code> with `backend='skops'` where a forecaster trained with categorical exogenous variables could not be saved (`TypeError: no default __reduce__ due to non-trivial __cinit__`). With pyarrow exogenous variables (for example, `double[pyarrow]`), the forecaster was saved and loaded, but its first `predict` crashed the Python process. The categorical, pyarrow and time zone aware dtypes of the exogenous variables are now stored as plain types.

+ Fixed an issue in <code>[save_forecaster]</code> with `backend='skops'` where a generic `pd.DateOffset` could not be saved (a `TypeError` saying that the `n` argument must be an integer). This affected the `offset` of <code>[ForecasterEquivalentDate]</code> (for example, `pd.DateOffset(days=7)`) and the forecasters trained on a series whose frequency is a `pd.DateOffset` (for example, `pd.DateOffset(months=1)`). Other offsets, such as `'D'` or `'MS'`, were not affected.

+ Fixed an issue in <code>[save_forecaster]</code> with `backend='skops'` where `last_window_` and `training_range_` were replaced with plain types while the file was written, so using the same forecaster from another thread at that time (for example, `predict`) failed. The forecaster is no longer modified.

+ Fixed an issue in <code>[plot_prediction_distribution]</code> where it raised a `KeyError` (for example, `KeyError: '103'`) when `bootstrapping_predictions` had an integer index, such as the `RangeIndex` returned by `predict_bootstrapping` when the forecaster is trained with a series without a datetime index. The rows were looked up with their labels converted to strings. They are now selected by position.

+ Fixed an issue where `fit` raised `AttributeError: Estimator functiontransformer does not provide get_feature_names_out` when `transformer_exog` (in all the forecasters) or `transformer_y` (in <code>[ForecasterRecursive]</code> and <code>[ForecasterDirect]</code>) was a scikit-learn `Pipeline` (or `ColumnTransformer`) with a step that does not implement `get_feature_names_out`, for example `make_pipeline(FunctionTransformer(np.log1p, np.expm1), StandardScaler())`. The columns of the transformed data now keep the input column names.

+ Fixed an issue in <code>[ForecasterStats]</code> where `predict` with a `last_window` of a single observation and a `transformer_y` with pandas output (`set_output(transform='pandas')`) raised `TypeError: object of type 'numpy.float64' has no len()`, because <code>[transform_series]</code> returned a scalar instead of a Series for a single row.

+ <code>[transform_series]</code> raised `AttributeError: property 'feature_names_in_' of 'Pipeline' object has no setter` when a fitted `Pipeline` was applied (`fit=False`) to a series with another name than the one used to fit it. The input is now renamed to the name seen in fit, and the transformer is no longer copied and modified. The output keeps the name of the input series, except for transformers that expand it into several columns, whose columns are now named after the name seen in fit (for example, `y_A` instead of `pred_A` with a `OneHotEncoder` fitted on `y`).

+ Fixed an issue in <code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code> and <code>[ForecasterRnn]</code> where the prediction methods raised `UnboundLocalError: cannot access local variable 'steps_direct'` when `steps` was a numpy integer, and `ValueError: min() iterable argument is empty` when it was `0` or an empty list. Numpy integers are now accepted, a type other than an int, a list or `None` raises a `TypeError`, and `0` or an empty list raise a `ValueError` that says so.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> where the prediction methods raised `ValueError: The truth value of a Index is ambiguous` when `levels` was a pandas Index or a numpy array (for example, `series.columns`). They are now converted to a list, also in <code>[ForecasterRnn]</code>, which raised a `TypeError`.

+ <code>[exog_to_direct]</code> and <code>[exog_to_direct_numpy]</code> now raise a `ValueError` when `steps` is not between 1 and the number of rows of `exog`. With more steps than rows, <code>[exog_to_direct]</code> returned missing values and <code>[exog_to_direct_numpy]</code> raised a concatenation error, and `steps=0` raised `IndexError: list index out of range`. The forecasters always call them with valid values.

+ The forecasters raised ``TypeError: `lags` argument must be an int, 1d numpy ndarray, range, tuple or list`` when `lags` was a numpy integer (`lags=np.int64(3)`, for example from `np.arange`), and accepted booleans (`lags=True` or `[True, 2]`). With lags given as unsigned integers (`np.uint8`), `predict` raised ``ValueError: `last_window` must have as many values as needed to generate the predictors``, because `-window_size` overflowed. Numpy integers are now accepted, booleans raise a `TypeError`, and the lags are stored as `int64`. The same applies to the `window_sizes` of custom window features, where an empty list raised `ValueError: max() iterable argument is empty`.

+ The forecasters removed `sample_weight` from the `fit_kwargs` dict passed by the user, and <code>[ForecasterRnn]</code> removed `series_val` and `exog_val` (when it was created and in `set_fit_kwargs`), so the same dict could not be reused. The dict is now copied. When the `fit` method of the estimator accepts `**kwargs` (for example, a scikit-learn `Pipeline`), the warning about ignored `fit_kwargs` said that they are not used by `fit`; it now says that arguments passed through `**kwargs` are not supported.

+ <code>[ForecasterRnn]</code> raised `KeyError: 'exog_val'` when the `exog_val` of `fit_kwargs` was a pandas Series without name. It is now named `'exog'`, as `exog` in `fit`.

+ The forecasters raised `TypeError: Categorical dtypes in exog must contain only integer values` with a categorical exogenous variable whose categories are nullable integers (`Int32`, for example after `convert_dtypes`), and issued a false `DataTypeWarning` with `UInt8` or pyarrow numeric columns (`double[pyarrow]`).

+ <code>[ForecasterRecursiveMultiSeries]</code> raised `TypeError: boolean value of NA is ambiguous` in `fit` when a series with a nullable or pyarrow dtype (`Float64`, `Int64`, `double[pyarrow]`) started or ended with missing values.

+ <code>[ForecasterRecursiveMultiSeries]</code> accepted a dict of series with different time zones. With series in `UTC` and `Europe/Madrid`, `predict` only returned the series in one of them; with a series without time zone, `fit` raised `TypeError: Cannot compare tz-naive and tz-aware timestamps`. A `ValueError` that lists the time zones is now raised, also in <code>[ForecasterFoundation]</code>. With series whose frequencies cannot be compared (daily and monthly, or a `DatetimeIndex` and a `RangeIndex`), the error about different frequencies raised `TypeError: '<' not supported between instances of ...` instead.

+ Fixed several issues with the `exog` of <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterFoundation]</code>:
    + A wide `exog` Series without name was converted to a column named `0`, so `fit` of <code>[ForecasterRecursiveMultiSeries]</code> failed with a scikit-learn error about feature names of mixed types. It now raises the same `ValueError` as with a dict of `exog`.
    + Duplicated column names raised an error saying that `exog` had a column named as one of the series (wide `exog`), or were accepted (dict of `exog`). They now raise a `ValueError`.
    + An `exog` with the same length as its series but different dates was not aligned by date: <code>[ForecasterRecursiveMultiSeries]</code> raised ``ValueError: Different index for `series` and `exog` after transformation``, and <code>[ForecasterFoundation]</code> used its values by position. It is now aligned by date, with the usual warning about missing values. An `exog` with duplicated dates raises a `ValueError`.
    + The order of `exog_names_in_` of <code>[FoundationModel]</code> changed between Python processes. It now follows the order of appearance of the columns.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> where `predict_interval`, `predict_quantiles`, `predict_dist` and `predict_bootstrapping` raised `ValueError: Residuals for level 'b' are None` when a level that was not predicted had no residuals, for example after calling `set_out_sample_residuals` with only some of the series. Only the residuals of the levels to predict are now checked (for a level that is not in the residuals dict, those of `'_unknown_level'`).

+ The prediction methods did not detect an `exog` Series whose name is one of the exogenous variables used in training when the forecaster was trained with more of them. <code>[ForecasterRecursive]</code>, <code>[ForecasterDirect]</code> and the other forecasters with `exog` raised `KeyError: "['exog_2'] not in index"`, and <code>[ForecasterRecursiveMultiSeries]</code> filled the missing variables with NaN (without any warning with a dict of `exog`). They now raise the `ValueError` about missing columns, or a `MissingExogWarning` in <code>[ForecasterRecursiveMultiSeries]</code>.

+ Fixed an issue in <code>[ForecasterRnn]</code> where `predict` with a `last_window` without one of the series used as input (a series not in `levels`) used another column of `last_window` in its place, so the predictions were wrong without any warning. It now raises a `ValueError`, as <code>[ForecasterDirectMultiVariate]</code>.

+ The prediction methods issued a `MissingValuesWarning` when `last_window` had missing values that are not used to predict: before the last `window_size` rows, in levels that are not predicted (<code>[ForecasterRecursiveMultiSeries]</code>) or in series without lags (<code>[ForecasterDirectMultiVariate]</code>). <code>[ForecasterStats]</code>, which uses the whole `last_window`, still checks all its values.

+ When matplotlib, statsmodels or keras were installed but failed to import (for example, statsmodels 0.13.1 with pandas 2), the <code>[plot]</code> module, <code>[ForecasterRnn]</code> and <code>[create_and_compile_model]</code> raised `ModuleNotFoundError: No module named '(/path/to/python3'`, which hid the real error. The original error is now raised, and the installation instructions are only shown when the package is not installed.

+ Fixed an issue in the backtesting, hyperparameter search and feature selection functions where, if the internal copy of the forecaster failed (for example, with an estimator that cannot be deep-copied), the forecaster passed by the user was left without its fitted estimator, residuals and last window. The copy no longer modifies the original forecaster.

+ <code>[multivariate_time_series_corr]</code> raised `TypeError: 'numpy.int64' object is not iterable` when `lags` was a numpy integer. It is now handled as an int (the lags from 0 to `lags - 1`).


## 0.25.0 <small>Sep 11, 2026</small> { id="0.25.0" }

The main changes in this release are:

+ <span class="badge text-bg-docs">Docs</span> New user guide about foundation forecasting with heterogeneous series: different lengths, exogenous variables and missing values. [User guide](../user_guides/foundation-forecasting-with-heterogeneous-series.ipynb)

+ <span class="badge text-bg-feature">Feature</span> <code>[ForecasterFoundation]</code> and <code>[FoundationModel]</code> now accept heterogeneous multi-series input: series of different lengths, a different subset of exogenous columns per series, and NaN values in the target.

+ <span class="badge text-bg-feature">Feature</span> New <code>[TimesFM3Adapter]</code> adds support for **Google TimesFM 3.0** (`'google/timesfm-3.0-*'` ids), resolved automatically from `model_id`. Unlike TimesFM 2.5, it accepts past-only and known-future exogenous variables and exposes `device` and `predict_kwargs`. [User guide](../user_guides/foundation-forecasting-models.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New <code>[LicenseWarning]</code> in the <code>[exceptions]</code> module, raised whenever a foundation model whose pre-trained weights are released under a non-commercial license (TimesFM 3.0, Moirai-2, TabPFN-TS, TS-ICL) is loaded (deduplicated to once per session by Python's default warning filter). Suppressible like any other skforecast warning (`suppress_warnings=True` or `warnings.simplefilter`).

+ <span class="badge text-bg-api-change">API Change</span> `TimesFMAdapter` renamed to <code>[TimesFM25Adapter]</code> (`'google/timesfm-2.5-*'` ids). The adapter is reached through <code>[FoundationModel]</code>, so user code is unaffected, but forecasters pickled by earlier versions with a `TimesFMAdapter` cannot be loaded.

+ <span class="badge text-bg-api-change">API Change</span> Removed support for percentiles in the `interval` argument of the `predict_interval` method of the Forecasters and of the backtesting functions. Deprecated since 0.23.0, `interval` must now be expressed as quantiles in the 0-1 range (e.g. `interval=[0.05, 0.95]`). Passing percentiles such as `interval=[5, 95]` no longer emits a `FutureWarning` and raises a `ValueError` instead.


**Added**

+ <code>[ForecasterFoundation]</code> and <code>[FoundationModel]</code> now accept heterogeneous multi-series input: series of different lengths, a different subset of exogenous columns per series, and NaN values in the target. Every series is forecast with its own exog columns only. For backends that require identical covariate columns in a batch (Chronos-2, TS-ICL, TabICL, TimesFM 3.0), the series are grouped by their exog columns and the backend is called once per group, so the prediction of a series never depends on the exog of the other series (Chronos-2 `cross_learning` applies within each group). The new read-only attributes `supports_heterogeneous_covariates` and `supports_nan_in_series` (on `FoundationModel` and `ForecasterFoundation`, together with `supports_past_only_covariates`) expose the backend constraints. [User guide](../user_guides/foundation-forecasting-models.ipynb)

+ New <code>[TimesFM3Adapter]</code> adds support for **Google TimesFM 3.0** (`'google/timesfm-3.0-*'` ids), resolved automatically from `model_id`. Unlike TimesFM 2.5, it accepts past-only and known-future exogenous variables and exposes `device` and `predict_kwargs`. [User guide](../user_guides/foundation-forecasting-models.ipynb)

+ New <code>[LicenseWarning]</code> in the <code>[exceptions]</code> module, raised whenever a foundation model whose pre-trained weights are released under a non-commercial license (TimesFM 3.0, Moirai-2, TabPFN-TS, TS-ICL) is loaded (deduplicated to once per session by Python's default warning filter). Suppressible like any other skforecast warning (`suppress_warnings=True` or `warnings.simplefilter`).

+ <code>[ForecasterFoundation]</code> exposes the read-only attribute `allow_exog` (delegates to `estimator.allow_exog`), so the four adapter capability flags (`allow_exog`, `supports_past_only_covariates`, `supports_heterogeneous_covariates`, `supports_nan_in_series`) can be inspected on the forecaster. [User guide](../user_guides/foundation-forecasting-with-heterogeneous-series.ipynb)


**Changed**

+ <code>[FoundationModel]</code> and <code>[ForecasterFoundation]</code> now validate the columns of the future `exog` against the historical exog of each series at predict time. A future column with no historical values raises a `ValueError`. A historical column with no future values is used as a past-only covariate by the adapters that support it (Chronos-2, TS-ICL, TimesFM 3.0) and ignored with an `IgnoredArgumentWarning` by the rest (TabICL, TabPFN-TS, TFC-T0, Nori). The new read-only attribute `supports_past_only_covariates` exposes which behavior applies.

+ `TimesFMAdapter` renamed to <code>[TimesFM25Adapter]</code> (`'google/timesfm-2.5-*'` ids). The adapter is reached through <code>[FoundationModel]</code>, so user code is unaffected, but forecasters pickled by earlier versions with a `TimesFMAdapter` cannot be loaded.

+ <code>[FoundationModel]</code> `set_params` now raises a `ValueError` when the new `model_id` is served by a different adapter than the one selected at construction (or by none). Previously the id was accepted and the failure surfaced only when the weights were loaded. Create a new <code>[FoundationModel]</code> to switch model family.

+ Removed support for percentiles in the `interval` argument of the `predict_interval` method of the Forecasters and of the backtesting functions. Deprecated since 0.23.0, `interval` must now be expressed as quantiles in the 0-1 range (e.g. `interval=[0.05, 0.95]`). Passing percentiles such as `interval=[5, 95]` no longer emits a `FutureWarning` and raises a `ValueError` instead.

+ Removed support for percentiles in the `level` argument of the `predict_interval` method of the statistical estimators (<code>[Arima]</code>, <code>[Arar]</code>, <code>[Ets]</code>). Deprecated since 0.23.0, `level` must now be expressed as coverage proportions in the (0, 1] range (e.g. `level=[0.8, 0.95]`). Passing percentiles such as `level=[80, 95]` no longer emits a `FutureWarning` and raises a `ValueError` instead.


**Fixed**

+ <code>[backtesting_foundation]</code> failed or produced wrongly dated predictions when a series ended before the end of the span, contained trailing NaN inside a fold, or had exogenous variables that did not cover the whole forecast horizon (`KeyError` in the metrics or in the Chronos-2 and TS-ICL adapters, `all input arrays must have the same shape` in TimesFM 3.0). The context of every series now ends at the end of the train span of the fold, so predictions always fall inside the fold's test window; a series is predicted in a fold only if it has at least one observed value in that window and its context window is not entirely NaN (a fold where no level can be predicted is skipped with a `MissingValuesWarning`, as in `backtesting_forecaster_multiseries`); and the historical and future exog are aligned to the context and to the horizon on the backtesting path as they already were in `predict`.

+ <code>[NoriAdapter]</code> failed with `Input y contains NaN` when the context contained NaN. The rows whose target or covariates are NaN are now dropped before the in-context fit.

+ <code>[backtesting_foundation]</code> silently accepted a `levels` argument with names that are not in `series` (the unknown level received a `None` metric, or every fold was skipped with a misleading `MissingValuesWarning` when none of the levels existed). It now raises a `ValueError` naming the unknown levels, as `backtesting_forecaster_multiseries` and `bayesian_search_foundation` already did.

+ Fixed an issue in <code>[QuantileBinner]</code> where quantile interpolation could create bins that no observation falls into, so the corresponding bin was missing from the residuals dictionary of the Forecasters. This caused a `KeyError`, or the use of the residuals of a different bin, in `predict_interval`, `predict_bootstrapping` and `predict_quantiles` when `use_binned_residuals=True`.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> where `predict_bootstrapping` used the requested number of bins instead of the number of bins actually learned by each series binner, raising a `KeyError` when any of them was reduced.

+ Fixed an issue in <code>[ForecasterRecursiveMultiSeries]</code> where `set_out_sample_residuals` built the binned residuals of `'_unknown_level'` by joining the bins of the known series, although each series has its own binner. The residuals of all series are now binned with the binner of `'_unknown_level'`, so `predict_interval`, `predict_bootstrapping` and `predict_quantiles` no longer raise a `KeyError` for unknown levels when `use_in_sample_residuals=False` and `use_binned_residuals=True`, and the residuals stored in each bin correspond to that bin.

+ Fixed an issue in <code>[crps_from_quantiles]</code> where the integration bounds were derived by scaling the extreme predicted quantiles by fixed factors (`0.9` and `1.1`). This made the score depend on the level of the series, return negative values for negative quantiles, under-penalize true values far outside the predicted quantiles, and return `0` when all predicted quantiles were `0`. The area outside the predicted quantiles is now computed analytically, so the score is translation invariant, non-negative, grows linearly with the distance when `y_true` falls outside the predicted range, and reduces to the absolute error when the predictive distribution is a point mass.


## 0.24.0 <small>Aug 24, 2026</small> { id="0.24.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> New function <code>[bayesian_search_foundation]</code> in the <code>[model_selection]</code> module to tune the inference-time configuration (e.g. `context_length`) of <code>[ForecasterFoundation]</code> models using optuna. [User guide](../user_guides/foundation-forecasting-models.ipynb#selection-of-context-length)

+ <span class="badge text-bg-feature">Feature</span> New function <code>[grid_search_equivalent_date]</code> in the <code>[model_selection]</code> module to search the best baseline configuration (`offset`, `n_offsets`, `agg_func`) of a <code>[ForecasterEquivalentDate]</code> using time series backtesting. [User guide](../user_guides/forecasting-baseline.ipynb#searching-for-the-best-configuration)

+ <span class="badge text-bg-feature">Feature</span> New <code>[NoriAdapter]</code> in the <code>foundation</code> module wrapping `Synthefy Nori`, registered under the `'Synthefy/Nori` `model_id` prefix. Supports future-known exogenous variables, arbitrary quantiles in the 0-1 range, and lazy backend import. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1252](https://github.com/skforecast/skforecast/issues/1252))

+ <span class="badge text-bg-feature">Feature</span> New <code>[TSICLAdapter]</code> in the <code>foundation</code> module wrapping `tsicl` (`TSICL`), registered under the `'taharnbl/TS-ICL'` `model_id` prefix. Supports past and future known exogenous variables, a 0.01 quantile grid in `[0.01, 0.99]`, and lazy import of the `tsicl` backend. Thanks to the [EDF Lab](https://github.com/EDF-Lab) team for contributing this adapter. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1265](https://github.com/skforecast/skforecast/pull/1265))

+ <span class="badge text-bg-feature">Feature</span> New functions <code>[winkler_score]</code> and <code>[weighted_interval_score]</code> in the <code>[metrics]</code> module to evaluate the quality of prediction intervals. The Winkler score assesses a single interval (balancing sharpness and calibration), while the Weighted Interval Score (WIS) aggregates several intervals together with the median forecast and approximates the CRPS. [User guide](../user_guides/probabilistic-forecasting-metrics.ipynb) ([#1254](https://github.com/skforecast/skforecast/pull/1254), [#1262](https://github.com/skforecast/skforecast/pull/1262))

+ <span class="badge text-bg-enhancement">Enhancement</span> Optimized the memory layout (`order='F'`) of the bootstrapping prediction matrix in <code>[ForecasterRecursive]</code>, giving a notable speed-up for CatBoost estimators.


**Added**

+ New function <code>[bayesian_search_foundation]</code> in the <code>[model_selection]</code> module to tune the inference-time configuration (e.g. `context_length`) of <code>[ForecasterFoundation]</code> models using optuna. [User guide](../user_guides/foundation-forecasting-models.ipynb#selection-of-context-length)

+ New function <code>[grid_search_equivalent_date]</code> in the <code>[model_selection]</code> module to search the best baseline configuration (`offset`, `n_offsets`, `agg_func`) of a <code>[ForecasterEquivalentDate]</code> using time series backtesting. [User guide](../user_guides/forecasting-baseline.ipynb#searching-for-the-best-configuration)

+ New <code>[NoriAdapter]</code> in the <code>foundation</code> module wrapping `Synthefy Nori`, registered under the `'Synthefy/Nori` `model_id` prefix. Supports future-known exogenous variables, arbitrary quantiles in the 0-1 range, and lazy backend import. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1252](https://github.com/skforecast/skforecast/issues/1252))

+ New <code>[TSICLAdapter]</code> in the <code>foundation</code> module wrapping `tsicl` (`TSICL`), registered under the `'taharnbl/TS-ICL'` `model_id` prefix. Supports past and future known exogenous variables, a 0.01 quantile grid in `[0.01, 0.99]`, and lazy import of the `tsicl` backend. Thanks to the [EDF Lab](https://github.com/EDF-Lab) team for contributing this adapter. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1265](https://github.com/skforecast/skforecast/pull/1265))

+ New functions <code>[winkler_score]</code> and <code>[weighted_interval_score]</code> in the <code>[metrics]</code> module to evaluate the quality of prediction intervals. The Winkler score assesses a single interval (balancing sharpness and calibration), while the Weighted Interval Score (WIS) aggregates several intervals together with the median forecast and approximates the CRPS. [User guide](../user_guides/probabilistic-forecasting-metrics.ipynb) ([#1254](https://github.com/skforecast/skforecast/pull/1254), [#1262](https://github.com/skforecast/skforecast/pull/1262))


**Changed**

+ The main branch of the repository has been renamed from `master` to `main`. All references to the default branch in CI, documentation, and test fixtures have been updated accordingly.

+ During backtesting and one-step-ahead validation of <code>[ForecasterRecursiveMultiSeries]</code>, a level is now kept whenever the estimator natively supports NaN inputs (LightGBM, XGBoost, CatBoost and scikit-learn's tree-based models: `DecisionTree`, `ExtraTree`, `ExtraTrees`, `RandomForest` and `HistGradientBoosting`) and no `differentiation` is applied. As a result, metrics may change for series with interspersed NaN values, as previously skipped levels now produce predictions. ([#1196](https://github.com/skforecast/skforecast/issues/1196), [#1260](https://github.com/skforecast/skforecast/pull/1260))

+ <code>[ForecasterDirectMultiVariate]</code> and <code>[ForecasterRnn]</code> build their predictors from every series, so no level can be dropped from the last window. Folds whose last window contains NaNs are now always predicted, and either return NaN predictions or raise, depending on the estimator. ([#1260](https://github.com/skforecast/skforecast/pull/1260))

+ Optimized the memory layout (`order='F'`) of the bootstrapping prediction matrix in <code>[ForecasterRecursive]</code>, giving a notable speed-up for CatBoost estimators. As a side effect, bootstrap prediction intervals produced by linear estimators may differ at floating-point precision (~1e-15) because BLAS summation order depends on the array memory layout.


**Fixed**

+ Fixed an issue where <code>[FoundationModel]</code> was not fully compatible with `sklearn.base.clone`. The `TimesFMAdapter`, `TabICLAdapter`, `TabPFNAdapter`, and `NoriAdapter` stored their configuration dictionaries (`forecast_config_kwargs`, `tabicl_config`, `tabpfn_model_config`, `nori_config`) as a fresh copy in `__init__`, which broke the parameter identity check performed by `clone` and raised a `RuntimeError` whenever a non-empty configuration dictionary was passed. Because <code>[ForecasterFoundation]</code> clones its estimator at construction, this also prevented building a forecaster from a `FoundationModel` configured with those settings. The configuration is now stored by reference, following the scikit-learn convention of keeping constructor parameters unchanged.

+ Fixed a misleading error raised by the `predict`, `predict_interval`, `predict_bootstrapping` and `create_predict_X` methods of <code>[ForecasterRecursiveMultiSeries]</code> when `levels` was an empty list and no `last_window` was passed. The internal length validation reported that `last_window` did not contain enough observations to generate the predictors, pointing at the wrong cause. A `ValueError` stating that no series were requested is now raised instead.



## 0.23.0 <small>Jul 8, 2026</small> { id="0.23.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> New `calendar_features` parameter in all ML Forecasters (<code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveMultiSeries]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code>). Users can now pass a <code>[CalendarFeatures]</code> instance to delegate the automatic creation of calendar features (e.g. month, day of week, hour) from the datetime index to the forecaster. Calendar features are generated during both training and prediction, requiring no manual feature engineering. [User guide](../user_guides/calendar-features.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New <code>[TabPFNAdapter]</code> in the <code>foundation</code> module for zero-shot forecasting with **TabPFN-TS** (Prior Labs), registered under the `'priorlabs/tabpfn'` `model_id` prefix. The adapter supports known-future exogenous variables, arbitrary quantiles in the 0-1 range, local and cloud-API inference modes, and lazy import of the `tabpfn-time-series` backend. This brings the number of foundation model adapters available out of the box to five. Thanks to the [Prior Labs](https://priorlabs.ai) team for contributing this adapter. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1206](https://github.com/skforecast/skforecast/issues/1206), [#1213](https://github.com/skforecast/skforecast/pull/1213))

+ <span class="badge text-bg-feature">Feature</span> New <code>[T0Adapter]</code> in the <code>foundation</code> module wrapping `tfc-t0` (`T0Forecaster`), registered under the `'theforecastingcompany/t0'` `model_id` prefix. Supports future-known exogenous variables, arbitrary quantiles in the 0-1 range, and lazy backend import. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1221](https://github.com/skforecast/skforecast/issues/1221), [#1219](https://github.com/skforecast/skforecast/pull/1219))

+ <span class="badge text-bg-feature">Feature</span> New functions <code>[acf]</code>, <code>[pacf]</code> and <code>[calculate_lag_autocorrelation]</code> in the <code>[stats]</code> module. Fast ACF and PACF implementations via FFT and Levinson-Durbin, removing the dependency on `statsmodels` for autocorrelation calculations. [User guide](../user_guides/autocorrelation-and-lag-selection.ipynb)

+ <span class="badge text-bg-enhancement">Enhancement</span> Refactored the calendar feature engineering toolkit (<code>[CalendarFeatures]</code>, <code>[create_calendar_features]</code>) with new `'cyclical'`, `'onehot'`, and `'spline'` encodings, fine-grained `max_values` overrides per feature, `spline_kwargs` for spline customisation, and a `keep_original_columns` option. ISO week 53 and leap-year day-of-year 366 are now handled in a fully stateless way. An <code>[IgnoredArgumentWarning]</code> is emitted when `max_values` is passed together with `encoding='onehot'`, since onehot uses a fixed known-category set.

+ <span class="badge text-bg-enhancement">Enhancement</span> New `backend` parameter in <code>[save_forecaster]</code> and <code>[load_forecaster]</code> to select the serialization engine. In addition to the default `'joblib'`, the `'pickle'` and `'cloudpickle'` backends are now supported. The `'cloudpickle'` backend embeds custom functions (e.g. `weight_func`) and user-defined classes (e.g. `window_features`) directly in the saved file, removing the need to export them as separate `.py` files. A fourth `'skops'` backend provides a secure format that does not execute arbitrary code on load, recommended when loading files from untrusted sources; the new `trusted` parameter of <code>[load_forecaster]</code> controls which types skops is allowed to reconstruct (`False` by default, the secure setting). The `'skops'` backend is not available for `ForecasterStats`, `ForecasterRnn`, or `ForecasterFoundation`, whose underlying estimators embed objects that skops cannot serialize. On load, the backend is inferred automatically from the file extension (`.joblib`, `.pkl`/`.pickle`, `.cloudpickle`, `.skops`) when `backend` is not provided. [User guide](../user_guides/save-load-forecaster.ipynb)

+ <span class="badge text-bg-api-change">API Change</span> The `interval` argument of the `predict_interval` method of the Forecasters and of the backtesting functions is now expressed as quantiles in the 0-1 range (e.g. `interval=[0.05, 0.95]`) instead of percentiles in the 0-100 range. Passing percentiles is still supported but deprecated and emits a `FutureWarning`; support will be removed in a future version.

+ <span class="badge text-bg-api-change">API Change</span> The `level` argument of the `predict_interval` method of the statistical estimators (<code>[Arima]</code>, <code>[Arar]</code>, <code>[Ets]</code>) is now expressed as quantiles in the 0-1 range (e.g. `level=[0.05, 0.95]`) instead of percentiles in the 0-100 range. Passing percentiles is still supported but deprecated and emits a `FutureWarning`; support will be removed in a future version.

+ <span class="badge text-bg-api-change">API Change</span> <code>[select_features]</code> and <code>[select_features_multiseries]</code> now support calendar features. The `select_only` argument accepts the new value `'calendar'` (and a list combining `'autoreg'`, `'exog'` and `'calendar'`), and both functions return a fourth element, `selected_calendar_features`, with the selected calendar features at the source-feature level (e.g. `month`). Calendar features are evaluated at the encoded-column level (e.g. `month_sin`, `month_cos`) and a source feature is kept whenever at least one of its encoded columns is selected.

+ <span class="badge text-bg-fix">Fix</span> Fixed parallel execution failure in single-core environments (e.g. Docker with `cpus: '1.0'`). <code>select_n_jobs_backtesting</code> and <code>select_n_jobs_fit_forecaster</code> now fall back to `n_jobs=1` instead of `0`, which raised `ValueError` in `joblib.Parallel`. ([#1197](https://github.com/skforecast/skforecast/issues/1197))

+ <span class="badge text-bg-fix">Fix</span> Fixed <code>[TimeSeriesFold]</code> `split` to clamp the start of the last window to zero when `window_size` exceeds `initial_train_size`. Previously a small negative `iloc` start was interpreted by Python as an offset from the end of the index, producing an empty last window and a downstream `TypeError`. This was hit whenever a foundation model's `context_length` exceeded the initial train size during backtesting. ([#1213](https://github.com/skforecast/skforecast/pull/1213))

!!! warning "Serialized models incompatibility"

    Forecasters that were serialized with previous versions of skforecast are **not compatible** with version 0.23.0 due to internal changes in all Forecasters (new parameters, changes in attributes, and an optimized training pipeline). Forecasters must be **retrained** after upgrading.


**Added**

+ New `calendar_features` parameter in all ML Forecasters (<code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveMultiSeries]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code>). Users can now pass a <code>[CalendarFeatures]</code> instance to delegate the automatic creation of calendar features (e.g. month, day of week, hour) from the datetime index to the forecaster. Calendar features are generated during both training and prediction, requiring no manual feature engineering. Only supported when the index of the input data is a `pandas.DatetimeIndex`. [User guide](../user_guides/calendar-features.ipynb)

+ New <code>[TabPFNAdapter]</code> in the <code>foundation</code> module wrapping `tabpfn-time-series` (`TabPFNTSPipeline`), registered under the `'priorlabs/tabpfn'` `model_id` prefix. Supports known-future exogenous variables, arbitrary quantiles in the 0-1 range, local and cloud-API inference modes, and lazy backend import. Includes a `FakeTabPFNTSPipeline` fixture and a full mock-based test suite mirroring the TabICL adapter tests. Contributed by the [Prior Labs](https://priorlabs.ai) team. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1206](https://github.com/skforecast/skforecast/issues/1206), [#1213](https://github.com/skforecast/skforecast/pull/1213))

+ New <code>[T0Adapter]</code> in the <code>foundation</code> module wrapping `tfc-t0` (`T0Forecaster`), registered under the `'theforecastingcompany/t0'` `model_id` prefix. Supports future-known exogenous variables, arbitrary quantiles in the 0-1 range, and lazy backend import. [User guide](../user_guides/foundation-forecasting-models.ipynb) ([#1221](https://github.com/skforecast/skforecast/issues/1221), [#1219](https://github.com/skforecast/skforecast/pull/1219))

+ New functions <code>[acf]</code>, <code>[pacf]</code> and <code>[calculate_lag_autocorrelation]</code> in the <code>[stats]</code> module. Fast ACF and PACF implementations via FFT and Levinson-Durbin, removing the dependency on `statsmodels` for autocorrelation calculations. [User guide](../user_guides/autocorrelation-and-lag-selection.ipynb)

+ New `backend` parameter in <code>[save_forecaster]</code> and <code>[load_forecaster]</code> to select the serialization engine. In addition to the default `'joblib'`, the `'pickle'` and `'cloudpickle'` backends are now supported. The `'cloudpickle'` backend embeds custom functions (e.g. `weight_func`) and user-defined classes (e.g. `window_features`) directly in the saved file, removing the need to export them as separate `.py` files. A fourth `'skops'` backend provides a secure format that does not execute arbitrary code on load, recommended when loading files from untrusted sources; the new `trusted` parameter of <code>[load_forecaster]</code> controls which types skops is allowed to reconstruct (`False` by default, the secure setting). The `'skops'` backend is not available for `ForecasterStats`, `ForecasterRnn`, or `ForecasterFoundation`, whose underlying estimators embed objects that skops cannot serialize. On load, the backend is inferred automatically from the file extension (`.joblib`, `.pkl`/`.pickle`, `.cloudpickle`, `.skops`) when `backend` is not provided. [User guide](../user_guides/save-load-forecaster.ipynb)

+ Added `torch 2.12` compatibility.

+ Added `matplotlib 3.11` compatibility.


**Changed**

+ The `interval` argument of the `predict_interval` method of the Forecasters and of the backtesting functions is now expressed as quantiles in the 0-1 range (e.g. `interval=[0.05, 0.95]`) instead of percentiles in the 0-100 range. Passing percentiles is still supported but deprecated and emits a `FutureWarning`; support will be removed in a future version.

+ The `level` argument of the `predict_interval` method of the statistical estimators (<code>[Arima]</code>, <code>[Arar]</code>, <code>[Ets]</code>) is now expressed as quantiles in the 0-1 range (e.g. `level=[0.05, 0.95]`) instead of percentiles in the 0-100 range. Passing percentiles is still supported but deprecated and emits a `FutureWarning`; support will be removed in a future version.

+ Refactored the calendar feature engineering toolkit (<code>[CalendarFeatures]</code>, <code>[create_calendar_features]</code>) with new `'cyclical'`, `'onehot'`, and `'spline'` encodings, fine-grained `max_values` overrides per feature, `spline_kwargs` for spline customisation, and a `keep_original_columns` option. ISO week 53 and leap-year day-of-year 366 are now handled in a fully stateless way. An <code>[IgnoredArgumentWarning]</code> is emitted when `max_values` is passed together with `encoding='onehot'`, since onehot uses a fixed known-category set.

+ <code>[select_features]</code> and <code>[select_features_multiseries]</code> now support calendar features. The `select_only` argument accepts the new value `'calendar'` (and a list combining `'autoreg'`, `'exog'` and `'calendar'`), and both functions return a fourth element, `selected_calendar_features`, with the selected calendar features at the source-feature level (e.g. `month`). Calendar features are evaluated at the encoded-column level (e.g. `month_sin`, `month_cos`) and a source feature is kept whenever at least one of its encoded columns is selected. A `ValueError` is now raised when the group(s) requested in `select_only` contain no features to evaluate.

+ <code>[calculate_distance_from_holiday]</code> moved from <code>[experimental]</code> to <code>[preprocessing]</code>. The function now accepts a `pandas.Series` or `pandas.DataFrame`, infers the time unit from the index frequency, renames its output columns to `time_to_holiday` and `time_since_holiday`, no longer mutates the input, requires `holiday_column` to be passed explicitly when `X` is a DataFrame, and emits a `UserWarning` while filling with `False` when the holiday column contains NaN values.

+ The `verbose` argument of <code>[save_forecaster]</code> now defaults to `False` (previously `True`), so saving a forecaster no longer prints its summary unless explicitly requested. `load_forecaster` is unchanged (`verbose=True`).

+ The internal preprocessing submodule was renamed from `skforecast.preprocessing.preprocessing` to `skforecast.preprocessing._preprocessing`. The public API (`from skforecast.preprocessing import …`) is unchanged; only direct imports from the submodule path are affected.

+ Removed the unused experimental `FastOrdinalEncoder`.

+ Removed `seaborn` as an optional dependency. The plotting functions in the <code>[plot]</code> module now rely only on `matplotlib`.


**Fixed**

+ Fixed parallel execution failure in single-core environments (e.g. Docker with `cpus: '1.0'`). <code>select_n_jobs_backtesting</code> and <code>select_n_jobs_fit_forecaster</code> now fall back to `n_jobs=1` instead of `0`, which raised `ValueError` in `joblib.Parallel`. ([#1197](https://github.com/skforecast/skforecast/issues/1197))

+ Fixed <code>[TimeSeriesFold]</code> `split` to clamp the start of the last window to zero when `window_size` exceeds `initial_train_size`. Previously a small negative `iloc` start was interpreted by Python as an offset from the end of the index, producing an empty last window and a downstream `TypeError`. This was hit whenever a foundation model's `context_length` exceeded the initial train size during backtesting. ([#1213](https://github.com/skforecast/skforecast/pull/1213))

+ Fix a bug in <code>[ForecasterStats]</code> where the `remove_estimators` method was not deleting the corresponding estimator parameters.


## 0.22.0 <small>Apr 23, 2026</small> { id="0.22.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> New module <code>foundation</code> for zero-shot time series forecasting using pre-trained foundation models. The module introduces <code>[FoundationModel]</code>, a scikit-learn compatible interface, and <code>[ForecasterFoundation]</code>, a high-level forecaster fully integrated with the skforecast ecosystem (backtesting, prediction intervals via native quantiles). Four adapters are included out of the box: **Chronos-2** (Amazon), **TimesFM 2.5** (Google), **Moirai-2** (Salesforce), and **TabICLv2** (Soda-Inria). Supports single-series and multi-series forecasting, exogenous variables (Chronos-2, TabICLv2), and quantile-based prediction intervals. [User guide](../user_guides/foundation-forecasting-models.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New `categorical_features` parameter in all ML Forecasters. When set to `'auto'` (default), non-numeric exogenous columns are automatically detected and encoded using an internal `OrdinalEncoder`. A list of column names can also be provided to explicitly specify which columns should be treated as categorical, including numeric columns. Native categorical support is configured automatically for compatible estimators (`LightGBM`, `CatBoost`, `XGBoost`, `HistGradientBoostingRegressor`). [User guide](../user_guides/categorical-features.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New `dropna_from_series` parameter in the <code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveClassifier]</code>, <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code>. When set to `True`, rows with NaN values generated during the construction of the training matrices are dropped before fitting. This allows training forecasters with time series that contain interspersed missing values. This parameter was already available in the <code>[ForecasterRecursiveMultiSeries]</code>. [User guide](../user_guides/handling-missing-values.ipynb)

+ <span class="badge text-bg-enhancement">Enhancement</span> Optimized the training pipeline in all Forecasters eliminating unnecessary DataFrame construction and dtype casting during `fit`. The public `create_train_X_y` method continues to return pandas objects for user inspection.

+ <span class="badge text-bg-enhancement">Enhancement</span> Significantly reduced memory consumption and improved training speed in direct Forecasters (<code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code>) when using exogenous variables. Memory usage is reduced by up to **90%** and fit times improve by **1.2x–3.8x** in large-scale scenarios, enabling training with more steps and exogenous features without running into memory limitations.

+ <span class="badge text-bg-api-change">API Change</span> The `regressor` argument has been removed, deprecated in version **0.19.0**. Use the `estimator` argument instead.

+ <span class="badge text-bg-fix">Fix</span> Fixed conformal prediction intervals with `differentiation`, categorical lags in <code>[ForecasterRecursiveClassifier]</code>, and other bug fixes. See details in the "Fixed" section below.

!!! warning "Serialized models incompatibility"

    Forecasters that were serialized with previous versions of skforecast are **not compatible** with version 0.22.0 due to internal changes in all Forecasters (new parameters, changes in attributes, and an optimized training pipeline). Forecasters must be **retrained** after upgrading.


**Added**

+ New module <code>foundation</code> for zero-shot time series forecasting using pre-trained foundation models. The module introduces <code>[FoundationModel]</code>, a scikit-learn compatible interface, and <code>[ForecasterFoundation]</code>, a high-level forecaster fully integrated with the skforecast ecosystem (backtesting, prediction intervals via native quantiles). Four adapters are included out of the box: **Chronos-2** (Amazon), **TimesFM 2.5** (Google), **Moirai-2** (Salesforce), and **TabICLv2** (Soda-Inria). Supports single-series and multi-series forecasting, exogenous variables (Chronos-2, TabICLv2), and quantile-based prediction intervals. [User guide](../user_guides/foundation-forecasting-models.ipynb)

+ New `categorical_features` parameter in all ML Forecasters. When set to `'auto'` (default), non-numeric exogenous columns are automatically detected and encoded using an internal `OrdinalEncoder`. A list of column names can also be provided to explicitly specify which columns should be treated as categorical, including numeric columns. Native categorical support is configured automatically for compatible estimators (`LightGBM`, `CatBoost`, `XGBoost`, `HistGradientBoostingRegressor`). [User guide](../user_guides/categorical-features.ipynb)

+ New `dropna_from_series` parameter in the <code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveClassifier]</code>, <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code>. When set to `True`, rows with NaN values generated during the construction of the training matrices are dropped before fitting. This allows training forecasters with time series that contain interspersed missing values. This parameter was already available in the <code>[ForecasterRecursiveMultiSeries]</code>. [User guide](../user_guides/handling-missing-values.ipynb)

+ [Binned residuals](../user_guides/probabilistic-forecasting-bootstrapped-residuals.ipynb#intervals-conditioned-on-predicted-values-binned-residuals) are now available in the <code>[ForecasterRnn]</code>. 


**Changed**

+ The `regressor` argument has been removed, deprecated in version **0.19.0**. Use the `estimator` argument instead.

+ Optimized the training pipeline in all Forecasters eliminating unnecessary DataFrame construction and dtype casting during `fit`. The public `create_train_X_y` method continues to return pandas objects for user inspection.

+ Significantly reduced memory consumption and improved training speed in direct Forecasters (<code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code>) when using exogenous variables. Memory usage is reduced by up to 90% and fit times improve by 1.2x–3.8x in large-scale scenarios, enabling training with more steps and exogenous features without running into memory limitations.


**Fixed**

+ Fixed an issue in conformal prediction intervals (`method='conformal'`) where the correction factor was incorrectly scaled when using `differentiation`. The inverse differentiation was applied to both the point predictions and the correction factor, causing the prediction intervals to grow too fast. Affected forecasters: <code>[ForecasterRecursive]</code>, <code>[ForecasterRecursiveMultiSeries]</code>, <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code>. ([#1143](https://github.com/skforecast/skforecast/pull/1143))

+ Fixed an issue in <code>[ForecasterRecursiveClassifier]</code> where the lags were not correctly passed as categorical features when using categorical exogenous variables.

+ Fixed an issue in the hyperparameter search when using a <code>[OneStepAheadFold]</code> validation. During training, the forecaster arguments `sample_weight` and `fit_kwargs` were not set correctly.

+ Fixed an issue in <code>[backtesting_forecaster_multiseries]</code> where the `tqdm` progress bar completed during data preparation instead of tracking the actual fold computation, giving the false impression that backtesting had finished.


## 0.21.0 <small>Mar 13, 2026</small> { id="0.21.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Added **AI context files** (`llms.txt`, `llms-full.txt`) following the [llmstxt.org](https://llmstxt.org) spec, IDE integration for GitHub Copilot, Claude Code, Cursor, and Aider, and 12 modular workflow skills so that AI coding assistants can generate accurate, up-to-date skforecast code. [User guide](../quick-start/ai-assisted-forecasting.md)

+ <span class="badge text-bg-enhancement">Enhancement</span> Optimized internal prediction loops in `_recursive_predict` and `_recursive_predict_bootstrapping` for <code>[ForecasterRecursive]</code> and <code>[ForecasterRecursiveMultiSeries]</code>. Changes include vectorized lag indexing for non-contiguous lags (~50% faster with 15-20 lags), pre-computation of loop-invariant values, and reduced redundant operations. These improvements result in faster `predict` methods calls, especially in scenarios with many lags and bootstrap iterations.

+ <span class="badge text-bg-enhancement">Enhancement</span> Optimized and refactored Bayesian search functions (<code>bayesian_search_forecaster</code>, <code>bayesian_search_forecaster_multiseries</code>). Key improvements include: better default TPE sampler configuration (`multivariate=True`, `group=True`, `consider_endpoints=True`) for more effective hyperparameter optimization, caching of train/test splits in `OneStepAheadFold` to avoid redundant computation when the same lag configuration is evaluated multiple times, default `n_trials` increased from 10 to 20, and `kwargs_create_study`/`kwargs_study_optimize` defaults changed from `{}` to `None`. Additionally, the `return_best` refit summary message is now controlled by the `verbose` parameter across all search functions.

+ <span class="badge text-bg-enhancement">Enhancement</span> <code>[bayesian_search_forecaster]</code> and <code>[bayesian_search_forecaster_multiseries]</code> results DataFrame now includes a `trial_number` column, allowing users to correlate result rows with specific optuna trials via `study.trials[trial_number]`.

+ <span class="badge text-bg-api-change">API Change</span> <code>[bayesian_search_forecaster]</code> and <code>[bayesian_search_forecaster_multiseries]</code> now return the full optuna `Study` object as the second element of the tuple instead of `best_trial`. The best trial is still accessible via `study.best_trial`. This enables access to all optimization trials, optuna visualizations, and study resumption.


**Added**

+ Added machine-readable AI context files (`llms.txt`, `llms-full.txt`) following the [llmstxt.org](https://llmstxt.org) spec, automatic IDE integration (`.github/copilot-instructions.md`, `AGENTS.md`), 12 workflow skills in `skills/`, and a generation script (`tools/ai/generate_ai_context_files.py`) to keep all derived files in sync. [User guide](../quick-start/ai-assisted-forecasting.md)

+ Added <code>[TimeSeriesSplitter]</code> class to the experimental module. This class provides a flexible way to split time series data into training and testing sets while respecting temporal order and allowing for various configurations of train/test sizes, gaps, and strides ([#1117](https://github.com/skforecast/skforecast/pull/1117)).


**Changed**

+ Optimized internal prediction loops in `_recursive_predict` and `_recursive_predict_bootstrapping` for <code>[ForecasterRecursive]</code> and <code>[ForecasterRecursiveMultiSeries]</code>. Changes include vectorized lag indexing for non-contiguous lags (~50% faster with 15-20 lags), pre-computation of loop-invariant values, and reduced redundant operations. These improvements result in faster `predict` methods calls, especially in scenarios with many lags and bootstrap iterations.

+ Optimized and refactored Bayesian search functions (`bayesian_search_forecaster`, `bayesian_search_forecaster_multiseries`). Key improvements include: better default TPE sampler configuration (`multivariate=True`, `group=True`, `consider_endpoints=True`) for more effective hyperparameter optimization, caching of train/test splits in `OneStepAheadFold` to avoid redundant computation when the same lag configuration is evaluated multiple times, default `n_trials` increased from 10 to 20, and `kwargs_create_study`/`kwargs_study_optimize` defaults changed from `{}` to `None`. Additionally, the `return_best` refit summary message is now controlled by the `verbose` parameter across all search functions.

+ <code>[bayesian_search_forecaster]</code> and <code>[bayesian_search_forecaster_multiseries]</code> results DataFrame now includes a `trial_number` column, allowing users to correlate result rows with specific optuna trials via `study.trials[trial_number]`.

+ <code>[bayesian_search_forecaster]</code> and <code>[bayesian_search_forecaster_multiseries]</code> now return the full optuna `Study` object as the second element of the tuple instead of `best_trial`. The best trial is still accessible via `study.best_trial`.

+ `kwargs_read_csv` has been renamed to `kwargs_read` in the `fetch_dataset` function. The new name reflects that the keyword arguments are passed to both `pd.read_csv` and `pd.read_parquet`, depending on the dataset file type.


**Fixed**

+ Fixed an issue where using a `transformer_y` or `transformer_series` that expands the target into multiple columns (e.g., `OneHotEncoder`) produced a non-descriptive internal error. Now, a clear `ValueError` is raised explaining that transformers applied to the target series must return a single column. ([#1126](https://github.com/skforecast/skforecast/pull/1126))

+ Fixed an issue where `out_sample_residuals_` and `out_sample_residuals_by_bin_` were not reset during `fit()`, causing stale residuals from a previous model to silently persist after refitting. ([#1123](https://github.com/skforecast/skforecast/pull/1123))

+ Fixed an issue in <code>expand_index</code> where the original `RangeIndex.step` was not preserved when creating future indices. Previously, `step=1` was always assumed, which could lead to incorrect prediction indices. ([#1150](https://github.com/skforecast/skforecast/pull/1150))


## 0.20.1 <small>Feb 11, 2026</small> { id="0.20.1" }

The main changes in this release are:

+ <span class="badge text-bg-fix">Fix</span> Fixed an issue in backtesting functions where passing `interval` as a single float (e.g. `interval=0.8` for 80% coverage) was not handled correctly when `interval_method` is set to `'bootstrapping'`, causing an error during prediction interval calculation.

+ <span class="badge text-bg-fix">Fix</span> Fixed an issue in <code>[reshape_exog_long_to_dict]</code> where the `fill_value` parameter was applied to all columns, causing errors with categorical columns and silent data corruption in string columns. Now, `fill_value` is only applied to numeric columns, and non-numeric columns retain NaN in the gaps. A warning is issued to inform the user.


**Added**


**Changed**


**Fixed**

+ Fixed an issue in backtesting functions where passing `interval` as a single float (e.g. `interval=0.8` for 80% coverage) was not handled correctly when `interval_method` is set to `'bootstrapping'`, causing an error during prediction interval calculation.

+ Fixed an issue in <code>[reshape_exog_long_to_dict]</code> where the `fill_value` parameter was applied to all columns, causing errors with categorical columns and silent data corruption in string columns. Now, `fill_value` is only applied to numeric columns, and non-numeric columns retain NaN in the gaps. A warning is issued to inform the user.


## 0.20.0 <small>Feb 01, 2026</small> { id="0.20.0" }

The main changes in this release are:

+ <span class="badge text-bg-enhancement">Enhancement</span> Refactored the **bootstrapped residuals calculation** in all recursive forecasters achieving **10x speedup in the interval prediction process**. This improvement significantly reduces the time required to generate prediction intervals, enhancing overall performance and user experience.

+ <span class="badge text-bg-feature">Feature</span> New skforecast <code>[Arima]</code> class in the <code>[stats]</code> module. Native and fast Python implementation of ARIMA model for time series forecasting that follows the scikit-learn interface. [User guide](../user_guides/forecasting-sarimax-arima.ipynb)

+ <span class="badge text-bg-feature">Feature</span> <code>[ForecasterStats]</code> now supports multiple estimators (<code>[Sarimax]</code>, <code>[Arima]</code>, <code>[Arar]</code>, <code>[Ets]</code>), enabling users to fit, predict, and compare several statistical models simultaneously in a unified workflow.

+ <span class="badge text-bg-feature">Feature</span> Added parameter `max_out_of_range_proportion` to <code>[PopulationDriftDetector]</code> to set the maximum allowed proportion of out-of-range observations (for numeric features) before triggering drift detection. [User guide](../user_guides/drift-detection.ipynb)

+ <span class="badge text-bg-api-change">API Change</span> <code>[ForecasterSarimax]</code> has been removed, deprecated in version **0.19.0**. Use the new <code>[ForecasterStats]</code> class in the <code>[recursive]</code> module, which offers enhanced capabilities and flexibility for statistical time series forecasting.


**Added**

+ Support for `Python 3.14`.

+ New skforecast <code>[Arima]</code> class in the <code>[stats]</code> module. Native and fast Python implementation of ARIMA model for time series forecasting that follows the scikit-learn interface. [User guide](../user_guides/forecasting-sarimax-arima.ipynb)

+ <code>[ForecasterStats]</code> now supports multiple estimators (<code>[Sarimax]</code>, <code>[Arima]</code>, <code>[Arar]</code>, <code>[Ets]</code>), enabling users to fit, predict, and compare several statistical models simultaneously in a unified workflow.

+ New argument `freeze_params` in the [backtesting_stats] function to allow freezing the parameters of the statistical models during backtesting. When set to `True`, the models will use the parameters obtained from the initial fit throughout the backtesting process, rather than re-estimating them at each step.

+ Introduced vectorized `_recursive_predict_bootstrapping` methods in <code>[ForecasterRecursive]</code> and <code>[ForecasterRecursiveMultiSeries]</code> that predict all bootstrap samples in a single batch per step instead of looping over bootstrap iterations. This achieves significant speedup in the interval prediction process.

+ Added <code>_transform_vectorized</code> method to <code>[RollingFeatures]</code> for faster computation of vectorizable statistics.
  
+ Implemented caching in <code>[ForecasterDirect]</code> to avoid repeated computation of column indices and names during backtesting.
  
+ Optimized array operations in <code>[ForecasterDirectMultiVariate]</code> and <code>[ForecasterDirect]</code> to reduce memory allocations.

+ Added parameter `max_out_of_range_proportion` to <code>[PopulationDriftDetector]</code> to set the maximum allowed proportion of out-of-range observations (for numeric features) before triggering drift detection.

+ Introduced optimized prediction paths for linear models (using numpy dot product), LightGBM (using booster API), and XGBoost (using inplace_predict)


**Changed**

+ [ForecasterSarimax] has been removed, deprecated in version 0.19.0. Use the new [ForecasterStats] class in the [recursive] module, which offers enhanced capabilities and flexibility for statistical time series forecasting.

+ Removed residual handling from `_recursive_predict` methods, separating bootstrap logic into dedicated methods.

+ `suppress_warnings_fit` parameter in <code>[backtesting_stats]</code> function has been replaced with `suppress_warnings` to control the display of skforecast warnings during the entire backtesting process.


**Fixed**

+ Fixed an issue in <code>[QuantileBinner]</code> where duplicate bin edges caused by repeated values in the data led to non-consecutive bin indices. This caused errors in `predict_bootstrapping` when using binned residuals. The fix removes duplicate edges and ensures bins are always numbered consecutively from 0 to `n_bins_-1`. A warning is now issued when the number of bins is reduced.


## 0.19.1 <small>Dec 10, 2025</small> { id="0.19.1" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Enabled thresholds based on standard deviations in the <code>[PopulationDriftDetector]</code> class. Now, users can specify thresholds using standard deviations from the mean, allowing for more flexible and statistically grounded drift detection. This is now the default behavior when thresholds are not explicitly provided. ([#1080](https://github.com/skforecast/skforecast/issues/1080))

+ <span class="badge text-bg-fix">Fix</span> Fixed an issue that prevented using Forecasters created in past versions of the library after loading them with <code>[load_forecaster]</code>. The problem occurred with the introduction of the `estimator` parameter in version `0.19.0`, which replaced the previous `regressor` parameter. This fix ensures that Forecasters saved with versions prior to `0.19.0` can be loaded and used without any issues. ([#1079](https://github.com/skforecast/skforecast/issues/1079))


**Added**

+ Enabled thresholds based on standard deviations in the <code>[PopulationDriftDetector]</code> class. Now, users can specify thresholds using standard deviations from the mean, allowing for more flexible and statistically grounded drift detection. This is now the default behavior when thresholds are not explicitly provided. ([#1080](https://github.com/skforecast/skforecast/issues/1080))


**Changed**

+ Include argument `suppress_warnings` in <code>[save_forecaster]</code> and <code>[load_forecaster]</code> functions to control the display of skforecast warnings during the save and load processes. If `suppress_warnings` is set to `True`, skforecast warnings will be suppressed. See skforecast.exceptions.warn_skforecast_categories for more information.


**Fixed**

+ Fixed an issue that prevented using Forecasters created in past versions of the library after loading them with <code>[load_forecaster]</code>. The problem occurred with the introduction of the `estimator` parameter in version `0.19.0`, which replaced the previous `regressor` parameter. This fix ensures that Forecasters saved with versions prior to `0.19.0` can be loaded and used without any issues. ([#1079](https://github.com/skforecast/skforecast/issues/1079))


## 0.19.0 <small>Nov 28, 2025</small> { id="0.19.0" }

The main changes in this release are:

+ <span class="badge text-bg-api-change">API Change</span> Parameter and attribute `regressor` has been deprecated in favor of `estimator` in all Forecasters and will be removed in future releases to align with scikit-learn terminology. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ <span class="badge text-bg-feature">Feature</span> New class <code>[ForecasterRecursiveClassifier]</code> in the <code>[recursive]</code> module. This forecaster is designed to handle time series data where the target variable is categorical, enabling the prediction of future class labels based on historical patterns. [User guide](../user_guides/autoregressive-classification-forecasting.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New class <code>[PopulationDriftDetector]</code> in the <code>[drift_detection]</code> module to detect population drift between reference and new data. Suitable to detect when forecasting models need to be retrained due to changes in the data distribution. It supports both target and exogenous variables, in single and multiseries forecasting. [User guide](../user_guides/drift-detection.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New module <code>[stats]</code>. This module contains statistical models for time series forecasting that follows the scikit-learn interface. [User guide](../user_guides/forecasting-sarimax-arima.ipynb)

+ <span class="badge text-bg-feature">Feature</span> New class <code>[Arar]</code> in the <code>[stats]</code> module. This class implements ARAR algorithm, a forecasting method that combines a "memory shortening" transformation with an autoregressive (AR) model. [User guide](../user_guides/forecasting-arar.ipynb)

+ <span class="badge text-bg-api-change">API Change</span> Class <code>[Sarimax]</code> has been moved to the new <code>[stats]</code> module. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ <span class="badge text-bg-api-change">API Change</span> Class <code>[ForecasterSarimax]</code> has been deprecated in favor of the new <code>[ForecasterStats]</code> model in the <code>[recursive]</code> module. The new forecaster is compatible with a broader range of statistical models such as: sarimax, arima, arar and ets. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ <span class="badge text-bg-fix">Fix</span> Fixed an issue that prevented using indices with frequencies containing metadata (e.g., `CustomBusinessDay`, `CustomBusinessHour`, or holiday/weekmask variants). The library now preserves full frequency metadata by using `freq` instead of `freqstr`, ensuring correct alignment and compatibility with custom date offsets. ([#1051](https://github.com/skforecast/skforecast/issues/1051))


**Added**

+ New class <code>[ForecasterRecursiveClassifier]</code> in the <code>[recursive]</code> module. This forecaster is designed to handle time series data where the target variable is categorical, enabling the prediction of future class labels based on historical patterns.

+ New class <code>[PopulationDriftDetector]</code> in the <code>[drift_detection]</code> module to detect population drift between reference and new data. Suitable to detect when forecasting models need to be retrained due to changes in the data distribution. It supports both target and exogenous variables, in single and multiseries forecasting.

+ New module <code>[stats]</code>. This module contains statistical models for time series forecasting that follows the scikit-learn interface.

+ New class <code>[Arar]</code> in the <code>[stats]</code> module. This class implements ARAR algorithm, a forecasting method that combines a "memory shortening" transformation with an autoregressive (AR) model.

+ New function <code>[reshape_series_exog_dict_to_long]</code> in the <code>[preprocessing]</code> module to reshape series and exogenous variables from a dictionary format into a long-format pandas DataFrame with a MultiIndex. The first level of the index is the series name, and the second level is the time index.

+ Added dataset `vic_electricity_classification` to the <code>[datasets]</code> module. It contains hourly electricity consumption data for households in Victoria, Australia, classified into three categories: 'low', 'medium' and 'high' according to the 20th and 80th percentiles.


**Changed**

+ Deprecated support for `Python 3.9`.

+ Parameter `regressor` has been deprecated in favor of `estimator` in all Forecasters and will be removed in future releases to align with scikit-learn terminology. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ Class <code>[Sarimax]</code> has been moved to the new <code>[stats]</code> module. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ Class <code>[ForecasterSarimax]</code> has been deprecated in favor of the new <code>[ForecasterStats]</code> model in the <code>[recursive]</code> module. The new forecaster is compatible with a broader range of statistical models such as: sarimax, arima, arar and ets. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.


**Fixed**

+ Fixed an issue that prevented using indices with frequencies containing metadata (e.g., `CustomBusinessDay`, `CustomBusinessHour`, or holiday/weekmask variants). The library now preserves full frequency metadata by using `freq` instead of `freqstr`, ensuring correct alignment and compatibility with custom date offsets. ([#1051](https://github.com/skforecast/skforecast/issues/1051))


## 0.18.0 <small>Sep 22, 2025</small> { id="0.18.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> New parameter `fold_stride` in <code>[TimeSeriesFold]</code>. This parameter controls how the start of the test set [advances between consecutive folds](../user_guides/backtesting.ipynb#backtesting-with-fold-stride) during the <code>[backtesting_forecaster]</code>, <code>[backtesting_forecaster_multiseries]</code> and <code>[backtesting_sarimax]</code> functions. By default, `fold_stride` is equal to `steps`, which means that the test sets do not overlap and there are no gaps between them. However, if `fold_stride` is set to a value less than `steps`, the test sets will overlap, resulting in multiple forecasts for the same observations. Conversely, if `fold_stride` is set to a value greater than `steps`, gaps will be left between consecutive test sets. ([#764](https://github.com/skforecast/skforecast/issues/764))

+ <span class="badge text-bg-feature">Feature</span> Added module <code>[drift_detection]</code> with class <code>[RangeDriftDetector]</code> to [detect out-of-range values](../user_guides/drift-detection.ipynb) in both target and exogenous variables during prediction. This lightweight detector checks whether new observations fall outside the ranges seen during training, making it suitable for real-time and production environments. It supports both global exogenous variables and series-specific exogenous variables in multiseries forecasting.

+ <span class="badge text-bg-feature">Feature</span> New function <code>[backtesting_gif_creator]</code> in the <code>[plot]</code> module to [create a gif](../user_guides/backtesting.ipynb#create-your-own-backtesting-gif) that visualizes the backtesting process. 

+ <span class="badge text-bg-feature">Feature</span> New function <code>[show_datasets_info]</code> to display information about all [available datasets](../user_guides/datasets.ipynb).

+ <span class="badge text-bg-feature">Feature</span> New attribute `__skforecast_tags__` and public method `get_tags()` in all forecasters which provide metadata about the forecaster, such as its capabilities and limitations. This attribute can be useful for [introspection and understanding the behavior](../quick-start/forecaster-attributes.ipynb#skforecast-tags) of different forecasters.

+ <span class="badge text-bg-api-change">API Change</span> Backtesting functions output DataFrame now includes a `fold` column to identify the fold number of each prediction.

+ <span class="badge text-bg-fix">Fix</span> Fixed a bug that caused the gap to not be applied correctly in the <code>[backtesting_forecaster_multiseries]</code> function. ([#1028](https://github.com/skforecast/skforecast/issues/1028))

+ <span class="badge text-bg-fix">Fix</span> Fixed a bug that prevented the `CatBoostRegressor` from working with the <code>[ForecasterRecursiveMultiSeries]</code>. ([#1039](https://github.com/skforecast/skforecast/issues/1039))


**Added**

+ New parameter `fold_stride` in <code>[TimeSeriesFold]</code>. This parameter controls how the start of the test set [advances between consecutive folds](../user_guides/backtesting.ipynb#backtesting-with-fold-stride) during the <code>[backtesting_forecaster]</code>, <code>[backtesting_forecaster_multiseries]</code> and <code>[backtesting_sarimax]</code> functions. By default, `fold_stride` is equal to `steps`, which means that the test sets do not overlap and there are no gaps between them. However, if `fold_stride` is set to a value less than `steps`, the test sets will overlap, resulting in multiple forecasts for the same observations. Conversely, if `fold_stride` is set to a value greater than `steps`, gaps will be left between consecutive test sets. ([#764](https://github.com/skforecast/skforecast/issues/764))

+ Added module <code>[drift_detection]</code> with class <code>[RangeDriftDetector]</code> to [detect out-of-range values](../user_guides/drift-detection.ipynb) in both target and exogenous variables during prediction. This lightweight detector checks whether new observations fall outside the ranges seen during training, making it suitable for real-time and production environments. It supports both global exogenous variables and series-specific exogenous variables in multiseries forecasting.

+ New function <code>[backtesting_gif_creator]</code> in the <code>[plot]</code> module to [create a gif](../user_guides/backtesting.ipynb#create-your-own-backtesting-gif) that visualizes the backtesting process.

+ New function <code>[show_datasets_info]</code> to display information about all [available datasets](../user_guides/datasets.ipynb).

+ New attribute `__skforecast_tags__` and public method `get_tags()` in all forecasters which provide metadata about the forecaster, such as its capabilities and limitations. This attribute can be useful for [introspection and understanding the behavior](../quick-start/forecaster-attributes.ipynb#skforecast-tags) of different forecasters.


**Changed**

+ Backtesting functions output DataFrame now includes a `fold` column to identify the fold number of each prediction.


**Fixed**

+ Fixed a bug that caused the gap to not be applied correctly in the <code>[backtesting_forecaster_multiseries]</code> function. ([#1028](https://github.com/skforecast/skforecast/issues/1028))

+ Fixed a bug that prevented the `CatBoostRegressor` from working with the <code>[ForecasterRecursiveMultiSeries]</code>. ([#1039](https://github.com/skforecast/skforecast/issues/1039))


## 0.17.0 <small>Aug 11, 2025</small> { id="0.17.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> <code>[ForecasterEquivalentDate]</code> can now predict intervals using the conformal prediction framework.

+ <span class="badge text-bg-feature">Feature</span> Created module <code>[experimental]</code>, this module contains experimental features that are not yet fully tested or may change in future releases.

+ <span class="badge text-bg-enhancement">Enhancement</span> The <code>[ForecasterRnn]</code> and the function <code>[create_and_compile_model]</code> have been refactored to allow for the inclusion of exogenous variables. The forecaster can also make interval predictions using the conformal prediction framework.

+ <span class="badge text-bg-api-change">API Change</span> Input data passed to all functions/classes must have either a pandas `RangeIndex` or `DatetimeIndex`. Previously, if the input did not meet this condition, a `RangeIndex` starting at 0 was automatically generated. This behavior has been removed to ensure consistent and explicit handling of input data.

+ <span class="badge text-bg-api-change">API Change</span> <code>[ForecasterRecursiveMultiSeries]</code> now accepts three input types for the `series` data: a wide-format DataFrame, where each column corresponds to a different time series; a long-format DataFrame with a MultiIndex, where the first level indicates the series name and the second level is the time index; or a dictionary with series names as keys and pandas `Series` as values.

+ <span class="badge text-bg-api-change">API Change</span> <code>[ForecasterRecursiveMultiSeries]</code> now accepts `exog` input as a wide-format DataFrame, where each column corresponds to a different exogenous variable; a long-format DataFrame with a MultiIndex, where the first level indicates the series name to which it belongs and the second level is the time index; or a dictionary with series names as keys and pandas `Series` or `DataFrames` as values.

+ <span class="badge text-bg-api-change">API Change</span> The functions `series_long_to_dict` and `exog_long_to_dict` have been renamed to <code>[reshape_series_long_to_dict]</code> and <code>[reshape_exog_long_to_dict]</code> in the <code>[preprocessing]</code> module.
  
+ <span class="badge text-bg-fix">Fix</span> A bug that prevented the use of `initial_train_size` as a date with the <code>[OneStepAheadFold]</code> during the hyperparameter search has been fixed.

+ <span class="badge text-bg-fix">Fix</span> A bug that caused the data types to be set incorrectly when creating the predicting matrix with the `create_predict_X` method or when `return_predictors=True` in the <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code> functions has been fixed. The dtypes of the predictors are now set to match those of the training data.

+ <span class="badge text-bg-fix">Fix</span> A bug that prevented the use of a `pd.RangeIndex` with the <code>[OneStepAheadFold]</code> during the hyperparameter search has been fixed.


**Added**

+ Added attribute `exog_dtypes_out_` in all forecasters to store the data types of the exogenous variables used in training after the transformation applied by `transformer_exog`. If `transformer_exog` is not used, it is equal to `exog_dtypes_in_`.

+ Added function <code>[reshape_series_wide_to_long]</code> in the <code>[preprocessing]</code> module. This function reshapes a wide-format DataFrame where each column corresponds to a series into a long-format DataFrame with with a MultiIndex. The first level of the index is the series name and the second level is the time index.

+ Added metric <code>[symmetric_mean_absolute_percentage_error]</code> in the <code>[metrics]</code> module. This metric calculates the symmetric mean absolute percentage error (SMAPE) between the true values and the predicted values.

+ <code>[ForecasterEquivalentDate]</code> can now predict intervals using the conformal prediction framework.

+ Created module <code>[experimental]</code>, this module contains experimental features that are not yet fully tested or may change in future releases.

+ Include function <code>[calculate_distance_from_holiday]</code> in the <code>[experimental]</code> module. It calculates the number of days to the next holiday and the number of days since the last holiday in a DataFrame with a date column.

+ The <code>[ForecasterRnn]</code> and the function <code>[create_and_compile_model]</code> now support the inclusion of exogenous variables.

+ Added method `predict_interval` to the <code>[ForecasterRnn]</code> using the conformal prediction framework.


**Changed**

+ Input data passed to all functions/classes must have either a pandas `RangeIndex` or `DatetimeIndex`. Previously, if the input did not meet this condition, a `RangeIndex` starting at 0 was automatically generated. This behavior has been removed to ensure consistent and explicit handling of input data.

+ <code>[ForecasterRecursiveMultiSeries]</code> now accepts three input types for the series data: a wide-format DataFrame, where each column corresponds to a different time series; a long-format DataFrame with a MultiIndex, where the first level indicates the series name and the second level is the time index; or a dictionary with series names as keys and pandas Series as values.

+ <code>[ForecasterRecursiveMultiSeries]</code> now accepts `exog` input as a wide-format DataFrame, where each column corresponds to a different exogenous variable; a long-format DataFrame with a MultiIndex, where the first level indicates the series name to which it belongs and the second level is the time index; or a dictionary with series names as keys and pandas `Series` or `DataFrames` as values.

+ When predicting, <code>[ForecasterRecursiveMultiSeries]</code> does not require the exog input to have the same type as the one used during training.

+ Function `series_long_to_dict` renamed to <code>[reshape_series_long_to_dict]</code> in the <code>[preprocessing]</code> module. This function reshapes a long-format DataFrame with time series data into a dictionary format where each entry corresponds to a series.
  
+ Function `exog_long_to_dict` renamed to <code>[reshape_exog_long_to_dict]</code> in the <code>[preprocessing]</code> module. This function reshapes a long-format DataFrame with exogenous variables into a dictionary format where each entry corresponds to the exogenous variables of a series.

+ The <code>[create_and_compile_model]</code> function has been refactored. All arguments related with layers and compilation are now passed as a dictionary using the following arguments: `recurrent_layers_kwargs`, `dense_layers_kwargs`, `output_dense_layer_kwargs`, and `compile_kwargs`.

+ The arguments `lags` and `steps` were removed from the <code>[ForecasterRnn]</code> initialization. These arguments are now inferred from the estimator architecture.

+ Remove `preprocess_y`, `preprocess_last_window` and `preprocess_exog` in favor of `check_extract_values_and_index` in the <code>[utils]</code> module. This function checks if the index is a pandas `DatetimeIndex` or `RangeIndex` and extracts the values and index accordingly.


**Fixed**

+ A bug that prevented the use of `initial_train_size` as a date with the <code>[OneStepAheadFold]</code> during the hyperparameter search has been fixed.

+ A bug that caused the data types to be set incorrectly when creating the predicting matrix with the `create_predict_X` method or when `return_predictors=True` in the <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code> functions has been fixed. The dtypes of the predictors are now set to match those of the training data.

+ A bug that prevented the use of a `pd.RangeIndex` with the <code>[OneStepAheadFold]</code> during the hyperparameter search has been fixed.


## 0.16.0 <small>May 01, 2025</small> { id="0.16.0" }

The main changes in this release are:

+ <span class="badge text-bg-enhancement">Enhancement</span> Refactored the internal codebase of all forecasters to enhance performance, primarily by replacing pandas DataFrames with more efficient NumPy arrays.


**Added**

+ Function `set_cpu_gpu_device()` in the <code>[utils]</code> module to set the device of the estimator to 'cpu' or 'gpu'. It is used to ensure that the recursive prediction is done in cpu even if the estimator is set to 'gpu'. This allows to avoid the bottleneck of the recursive prediction when using a gpu. Only applied to recursive forecasters when the estimator is a `XGBoost`, `LightGBM` or `CatBoost` model.

+ Added `series_name_in_` attribute in single series forecasters to store the name of the series used to fit the forecaster.

+ Added argument `return_predictors` to <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code> to return the predictors generated during the backtesting process along with the predictions.


**Changed**

+ Refactored the internal codebase of all forecasters to enhance performance, primarily by replacing pandas DataFrames with more efficient NumPy arrays.

+ In-sample residuals in direct forecasters has been simplified.

+ The method `create_predict_X` in the <code>[ForecasterRecursiveMultiSeries]</code> now returns a long-format DataFrame with the predictors. The columns are `level` and one column for each predictor. The index is the same as the prediction index.

+ The method `create_predict_X` in the <code>[ForecasterDirectMultiVariate]</code> now includes the `level` column in the returned DataFrame. The columns are `level` and one column for each predictor. The index is the same as the prediction index.


**Fixed**


## 0.15.1 <small>Mar 18, 2025</small> { id="0.15.1" }

+ <span class="badge text-bg-fix">Fix</span> Minor release to fix a bug when importing module `skforecast.sarimax`.


**Added**


**Changed**


**Fixed**

+ Fixed import error when importing the `skforecast.sarimax` module.


## 0.15.0 <small>Mar 10, 2025</small> { id="0.15.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Added [conformal framework for probabilistic forecasting](../user_guides/probabilistic-forecasting-conformal-prediction.ipynb). Generate prediction intervals using the [conformal prediction split method](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/regression/#2-the-split-method).

+ <span class="badge text-bg-feature">Feature</span> [Binned residuals](../user_guides/probabilistic-forecasting-bootstrapped-residuals.ipynb#intervals-conditioned-on-predicted-values-binned-residuals) are now available in the <code>[ForecasterRecursiveMultiSeries]</code>, <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code> forecasters. 

+ <span class="badge text-bg-feature">Feature</span> New class <code>[ConformalIntervalCalibrator]</code> to perform [conformal calibration](../user_guides/probabilistic-forecasting-conformal-calibration.ipynb). This class is used to calibrate the prediction intervals using the conformal prediction framework.

+ <span class="badge text-bg-api-change">API Change</span> Probabilistic predictions in <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> are now returned as a long format DataFrame.

+ <span class="badge text-bg-api-change">API Change</span> Fit argument `store_in_sample_residuals` has changed default value to `False`. This means in-sample residuals are not stored by default. To store them, call new method `set_in_sample_residuals` after fitting the forecaster using the same training data. 


**Added**

+ Support for `Python 3.13`.

+ Added `rich>=13.9.4` library as hard dependence.

+ New argument `method  = 'conformal'` in `predict_interval` method or `interval_method = 'conformal'` in backtesting functions to use the conformal prediction framework.

+ New class <code>[ConformalIntervalCalibrator]</code> to perform conformal calibration. This class is used to calibrate the prediction intervals using the conformal prediction framework.

+ Binned residuals are now available in the <code>[ForecasterRecursiveMultiSeries]</code>, <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code> forecasters. 

+ New method `set_in_sample_residuals` to store the in-sample residuals after fitting the forecaster using the same training data.

+ Functions `crps_from_predictions` and `crps_from_quantiles` in module `metrics` to calculate the Continuous Ranked Probability Score (CRPS).

+ Function `calculate_coverage` in module `metrics` to calculate the coverage of the predicted intervals.

+ The `differentiation` argument in <code>[ForecasterRecursiveMultiSeries]</code> can now be a dict to [differentiate each series independently](../user_guides/independent-multi-time-series-forecasting.ipynb#differentiation). This is useful if the user wants to differentiate each series with a different order or not differentiate some of them.

+ Added statistic `ewm` (exponential weighted mean) in <code>[RollingFeatures]</code>. Alpha can be specified using the new argument `kwargs_stats`, default `{'ewm': {'alpha': 0.3}}`.

+ Added method `_repr_html_` to <code>[ForecasterSarimax]</code>, <code>[TimeSeriesFold]</code> and <code>[OneStepAheadFold]</code> to display the object in HTML format.

+ Added argument `consolidate_dtypes` in <code>[exog_long_to_dict]</code> function to ensure that the data types of the exogenous variables are consistent across all series when `np.nan` values are added and integer columns are converted to float.

+ Added <code>[calculate_lag_autocorrelation]</code> function to the <code>[plot]</code> module to calculate the autocorrelation and partial autocorrelation of a time series.

+ Added datasets `m5`, `ett_m1`, `ett_m2`, `ett_m2_extended` and `expenditures_australia` and `public_transport_madrid` to the <code>[datasets]</code> module.

+ Added function `create_mean_pinball_loss` in the <code>[metrics]</code> module to create a function to calculate the mean pinball loss for a given quantile.

+ Added function `check_one_step_ahead_input` to check the input data when using a <code>[OneStepAheadFold]</code> in the <code>[model_selection]</code> functions.

+ Function `set_warnings_style` in the <code>[exceptions]</code> module to set the style of the skforecast warnings issued by the library.


**Changed**

+ <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> forecasters use conformal prediction framework as default for probabilistic forecasting, `method = 'conformal'` in `predict_interval` method.

+ <code>[backtesting_forecaster_multiseries]</code> uses conformal prediction framework as default for probabilistic forecasting, `interval_method = 'conformal'`.

+ <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code> use binned residuals as default for probabilistic forecasting, `use_binned_residuals = True`.

+ Fit argument `store_in_sample_residuals` has changed default value to `False`. This means in-sample residuals are not stored by default. To store them, call new method `set_in_sample_residuals` after fitting the forecaster using the same training data. 

+ Predictions from `predict_bootstrapping` in <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> are now returned as a long format DataFrame with the bootstrapping predictions. The columns are `level`, `pred_boot_0`, `pred_boot_1`, ..., `pred_boot_n_boot`.

+ Predictions from `predict_interval` in <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> are now returned as long format DataFrame with the predictions and the lower and upper bounds of the estimated interval. The columns are `level`, `pred`, `lower_bound`, `upper_bound`.

+ Predictions from `predict_quantiles` in <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> are now returned as long format DataFrame with the quantiles predicted by the forecaster. For example, if `quantiles = [0.05, 0.5, 0.95]`, the columns are `level`, `q_0.05`, `q_0.5`, `q_0.95`.

+ Predictions from `predict_dist` in <code>[ForecasterRecursiveMultiSeries]</code> and <code>[ForecasterDirectMultiVariate]</code> are now returned as long format DataFrame with the parameters of the fitted distribution for each step. The columns are `level`, `param_0`, `param_1`, ..., `param_n`, where `param_i` are the parameters of the distribution.

+ <code>[ForecasterAutoregCustom]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> has been deleted (deprecated since skforecast 0.14.0). Window features can be added using the `window_features` argument in the <code>[ForecasterRecursive]</code>, <code>[ForecasterDirect]</code>, <code>[ForecasterDirectMultiVariate]</code> and <code>[ForecasterRecursiveMultiSeries]</code>.

+ Argument `dropna` in <code>[exog_long_to_dict]</code> function has been renamed to `drop_all_nan_cols`.

+ Argument `initial_train_size` can be a `str` or a `pandas datetime` in <code>[TimeSeriesFold]</code> and <code>[OneStepAheadFold]</code>. If so, the cv object will use the specified date to split the data. (contribution by [@g-rubio](https://github.com/g-rubio) [#898](https://github.com/skforecast/skforecast/pull/898)).

+ `set_dark_theme` background color changed to `#001633` to improve readability.


**Fixed**

+ Now <code>[ForecasterRecursiveMultiSeries]</code> can be saved correctly when `weight_func` is a `dict` with `None` for any series. It now use the method `_weight_func_all_1` to create the weight function for these series.

+ Fix `transform_numpy` function in the <code>[utils]</code> module to work when transformers output in `scikit-learn` is `set_output(transform='pandas')`.


## 0.14.0 <small>Nov 11, 2024</small> { id="0.14.0" }

The main changes in this release are:

This release has undergone a major refactoring to improve the performance of the library. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ <span class="badge text-bg-feature">Feature</span> Window features can be added to the training matrix using the `window_features` argument in all forecasters. You can use the <code>[RollingFeatures]</code> class to create these features or create your own object. [Create window and custom features](../user_guides/window-features-and-custom-features.ipynb).

+ <span class="badge text-bg-feature">Feature</span> <code>[model_selection]</code> functions now have a new argument `cv`. This argument expect an object of type <code>[TimeSeriesFold]</code> ([backtesting](../user_guides/backtesting.ipynb)) or <code>[OneStepAheadFold]</code> which allows to define the validation strategy using the arguments `initial_train_size`, `steps`, `gap`, `refit`, `fixed_train_size`, `skip_folds` and `allow_incomplete_folds`.

+ <span class="badge text-bg-feature">Feature</span> Hyperparameter search now allows to follow a [one-step-ahead validation strategy](../user_guides/hyperparameter-tuning-and-lags-selection.ipynb#one-step-ahead-validation) using a <code>[OneStepAheadFold]</code> as `cv` argument in the <code>[model_selection]</code> functions.

+ <span class="badge text-bg-enhancement">Enhancement</span> Refactor the prediction process in <code>[ForecasterRecursiveMultiSeries]</code> to improve performance when predicting multiple series.

+ <span class="badge text-bg-enhancement">Enhancement</span> The bootstrapping process in the `predict_bootstrapping` method of all forecasters has been optimized to improve performance. This may result in slightly different results when using the same seed as in previous versions.

+ <span class="badge text-bg-enhancement">Enhancement</span> Exogenous variables can be added to the training matrix if they do not contain the first window size observations. This is useful when exogenous variables are not available in early historical data. Visit the [exogenous variables](../user_guides/exogenous-variables.ipynb#handling-missing-exogenous-data-in-initial-training-periods) section for more information.

+ <span class="badge text-bg-api-change">API Change</span> Package structure has been changed to improve code organization. The forecasters have been grouped into the `recursive`, `direct` amd `deep_learning` modules. Visit the [migration guide](../user_guides/migration-guide.ipynb) section for more information.

+ <span class="badge text-bg-api-change">API Change</span> <code>[ForecasterAutoregCustom]</code> has been deprecated. [Window features](../user_guides/window-features-and-custom-features.ipynb) can be added using the `window_features` argument in the <code>[ForecasterRecursive]</code>.

+ <span class="badge text-bg-api-change">API Change</span> Refactor the `set_out_sample_residuals` method in all forecasters, it now expects `y_true` and `y_pred` as arguments instead of `residuals`. This method is used to store the residuals of the out-of-sample predictions.

+ <span class="badge text-bg-api-change">API Change</span> The `pmdarima.ARIMA` estimator is no longer supported by the <code>[ForecasterSarimax]</code>. You can use the skforecast <code>[Sarimax]</code> model or, to continue using it, use skforecast 0.13.0 or lower.

+ <span class="badge text-bg-fix">Fix</span> Fixed a bug where the `create_predict_X` method in recursive Forecasters did not correctly generate the matrix correctly when using transformations and/or differentiations


**Added**

+ Added `numba>=0.59` as hard dependency.

+ Added `window_features` argument to all forecasters. This argument allows the user to add window features to the training matrix. See <code>[RollingFeatures]</code>.

+ Hyperparameter search now allows to follow a one-step-ahead validation strategy using a <code>[OneStepAheadFold]</code> as `cv` argument in the <code>[model_selection]</code> functions.

+ Differentiation has been extended to all forecasters. The `differentiation` argument has been added to all forecasters to model the n-order differentiated time series.

+ Create `transform_numpy` function in the <code>[utils]</code> module to carry out the transformation of the modeled time series and exogenous variables as numpy arrays.

+ `random_state` argument in the `fit` method of <code>[ForecasterRecursive]</code> to set a seed for the random generator so that the stored sample residuals are always deterministic.

+ New private method `_train_test_split_one_step_ahead` in all forecasters.

+ New private function `_calculate_metrics_one_step_ahead` to <code>[model_selection]</code> module to calculate the metrics when predicting one step ahead.

+ The `steps` argument in the predict method of the <code>[ForecasterRecursive]</code> can now be a str or a pandas datetime. If so, the method will predict up to the specified date. (contribution by [@imMoya](https://github.com/imMoya) [#811](https://github.com/skforecast/skforecast/pull/811)).

+ Exogenous variables can be added to the training matrix if they do not contain the first window size observations. This is useful when exogenous variables are not available in early historical data.

+ Added support for different activation functions in the <code>[create_and_compile_model]</code> function. (contribution by [@pablorodriper](https://github.com/pablorodriper) [#824](https://github.com/skforecast/skforecast/pull/824)).


**Changed**

+ <code>[ForecasterAutoregCustom]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> has been deprecated. Window features can be added using the `window_features` argument in the <code>[ForecasterRecursive]</code> and <code>[ForecasterRecursiveMultiSeries]</code>.

+ Refactor `recursive_predict` in <code>[ForecasterRecursiveMultiSeries]</code> to predict all series at once and include option of adding residuals. This improves performance when predicting multiple series.

+ Refactor `predict_bootstrapping` in all Forecasters. The bootstrapping process has been optimized to improve performance. This may result in slightly different results when using the same seed as in previous versions.

+ Change the default value of `encoding` to `ordinal` in <code>[ForecasterRecursiveMultiSeries]</code>. This will avoid conflicts if the estimator does not support categorical variables by default.

+ Removed argument `engine` from <code>[bayesian_search_forecaster]</code> and <code>[bayesian_search_forecaster_multiseries]</code>.

+ The `pmdarima.ARIMA` estimator is no longer supported by the <code>[ForecasterSarimax]</code>. You can use the skforecast <code>[Sarimax]</code> model or, to continue using it, use skforecast 0.13.0 or lower.

+ `initialize_lags` in <code>[utils]</code> now returns the maximum lag, `max_lag`.

+ Removed attribute `window_size_diff` from all Forecasters. The window size extended by the order of differentiation is now calculated on `window_size`.

+ `lags` can be `None` when initializing any Forecaster that includes window features.

+ <code>[model_selection]</code> module has been divided internally into different modules to improve code organization (`_validation`, `_search`, `_split`).

+ Functions from `model_selection_multiseries` and `model_selection_sarimax` modules have been moved to the <code>[model_selection]</code> module.

+ <code>[model_selection]</code> functions now have a new argument `cv`. This argument expect an object of type <code>[TimeSeriesFold]</code> or <code>[OneStepAheadFold]</code> which allows to define the validation strategy using the arguments `initial_train_size`, `steps`, `gap`, `refit`, `fixed_train_size`, `skip_folds` and `allow_incomplete_folds`.

+ Added <code>[feature_selection]</code> module. The functions <code>[select_features]</code> and <code>[select_features_multiseries]</code> have been moved to this module.

+ The functions <code>[select_features]</code> and <code>[select_features_multiseries]</code> now have 3 returns: `selected_lags`, `selected_window_features` and `selected_exog`.

+ Refactor the `set_out_sample_residuals` method in all forecasters, it now expects `y_true` and `y_pred` as arguments instead of `residuals`.

+ `exog_to_direct` and `exog_to_direct_numpy` in <code>[utils]</code> now returns a the names of the columns of the transformed exogenous variables.

+ Renamed attributes in all Forecasters:

    + `encoding_mapping` has been renamed to `encoding_mapping_`.

    + `last_window` has been renamed to `last_window_`.

    + `index_type` has been renamed to `index_type_`.

    + `index_freq` has been renamed to `index_freq_`.

    + `training_range` has been renamed to `training_range_`.

    + `series_col_names` has been renamed to `series_names_in_`.

    + `included_exog` has been renamed to `exog_in_`.

    + `exog_type` has been renamed to `exog_type_in_`.

    + `exog_dtypes` has been renamed to `exog_dtypes_in_`.

    + `exog_col_names` has been renamed to `exog_names_in_`.

    + `series_X_train` has been renamed to `X_train_series_names_in_`.

    + `X_train_col_names` has been renamed to `X_train_features_names_out_`.

    + `binner_intervals` has been renamed to `binner_intervals_`.

    + `in_sample_residuals` has been renamed to `in_sample_residuals_`.

    + `out_sample_residuals` has been renamed to `out_sample_residuals_`.

    + `fitted` has been renamed to `is_fitted`.

+ Renamed arguments in different functions and methods:

    + `in_sample_residuals` has been renamed to `use_in_sample_residuals`.

    + `binned_residuals` has been renamed to `use_binned_residuals`.

    + `series_col_names` has been renamed to `series_names_in_` in the `check_predict_input`, `check_preprocess_exog_multiseries` and `initialize_transformer_series` functions in the <code>[utils]</code> module.

    + `series_X_train` has been renamed to `X_train_series_names_in_` in the `prepare_levels_multiseries` function in the <code>[utils]</code> module.

    + `exog_col_names` has been renamed to `exog_names_in_` in the `check_predict_input` and `check_preprocess_exog_multiseries` functions in the <code>[utils]</code> module.

    + `index_type` has been renamed to `index_type_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `index_freq` has been renamed to `index_freq_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `included_exog` has been renamed to `exog_in_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `exog_type` has been renamed to `exog_type_in_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `exog_dtypes` has been renamed to `exog_dtypes_in_` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `fitted` has been renamed to `is_fitted` in the `check_predict_input` function in the <code>[utils]</code> module.

    + `use_in_sample` has been renamed to `use_in_sample_residuals` in the `prepare_residuals_multiseries` function in the <code>[utils]</code> module.

    + `in_sample_residuals` has been renamed to `use_in_sample_residuals` in the <code>[backtesting_forecaster]</code>, <code>[backtesting_forecaster_multiseries]</code> and `check_backtesting_input` (<code>[utils]</code> module) functions.

   + `binned_residuals` has been renamed to `use_binned_residuals` in the <code>[backtesting_forecaster]</code> function.

    + `in_sample_residuals` has been renamed to `in_sample_residuals_` in the `prepare_residuals_multiseries` function in the <code>[utils]</code> module.

    + `out_sample_residuals` has been renamed to `out_sample_residuals_` in the `prepare_residuals_multiseries` function in the <code>[utils]</code> module.

    + `last_window` has been renamed to `last_window_` in the `preprocess_levels_self_last_window_multiseries` function in the <code>[utils]</code> module.


**Fixed**

+ Fixed a bug where the `create_predict_X` method in recursive Forecasters did not correctly generate the matrix correctly when using transformations and/or differentiations.


## 0.13.0 <small>Aug 01, 2024</small> { id="0.13.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> are able to [predict series not seen during training](../user_guides/independent-multi-time-series-forecasting.ipynb#forecasting-unknown-series). This is useful when the user wants to predict a new series that was not included in the training data.

+ <span class="badge text-bg-feature">Feature</span> `encoding` can be set to `None` in Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. This option does [not add the encoded series ids](../user_guides/independent-multi-time-series-forecasting.ipynb#series-encoding-in-multi-series) to the estimator training matrix.

+ <span class="badge text-bg-feature">Feature</span> New `create_predict_X` method in all recursive and direct Forecasters to allow the user to inspect the matrix passed to the predict method of the estimator.

+ <span class="badge text-bg-feature">Feature</span> New module <code>[metrics]</code> with functions to calculate metrics for time series forecasting such as <code>[mean_absolute_scaled_error]</code> and <code>[root_mean_squared_scaled_error]</code>. Visit [Time Series Forecasting Metrics](../user_guides/metrics.ipynb) for more information.

+ <span class="badge text-bg-feature">Feature</span> New argument `add_aggregated_metric` in <code>[backtesting_forecaster_multiseries]</code> to include, in addition to the metrics for each level, the aggregated metric of all levels using the average (arithmetic mean), weighted average (weighted by the number of predicted values of each level) or pooling (the values of all levels are pooled and then the metric is calculated).

+ <span class="badge text-bg-feature">Feature</span> New argument `skip_folds` in <code>[model_selection]</code> and <code>[model_selection_multiseries]</code> functions. It allows the user to [skip some folds during backtesting](../user_guides/backtesting.ipynb#backtesting-with-skip-folds), which can be useful to speed up the backtesting process and thus the hyperparameter search.

+ <span class="badge text-bg-api-change">API Change</span> backtesting procedures now pass the training series to the metric functions so it can be used to calculate metrics that depend on the training series.

+ <span class="badge text-bg-api-change">API Change</span> Changed the default value of the `transformer_series` argument to `None` in the Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. In most cases, tree-based models are used as estimators in these forecasters, so no transformation is applied by default as it is not necessary.

**Added**

+ Support for `Python 3.12`.

+ `keras` has been added as an optional dependency, tag `deeplearning`, to use the <code>[ForecasterRnn]</code>.

+ `PyTorch` backend for the <code>[ForecasterRnn]</code>.

+ New `create_predict_X` method in all recursive and direct Forecasters to allow the user to inspect the matrix passed to the predict method of the estimator.

+ New `_create_predict_inputs` method in all Forecasters to unify the inputs of the predict methods.

+ New plot function <code>[plot_prediction_intervals]</code> in the <code>[plot]</code> module to plot predicted intervals.

+ New module <code>[metrics]</code> with functions to calculate metrics for time series forecasting such as `mean_absolute_scaled_error` and `root_mean_squared_scaled_error`.

+ New argument `skip_folds` in <code>[model_selection]</code> and <code>[model_selection_multiseries]</code> functions. It allows the user to skip some folds during backtesting, which can be useful to speed up the backtesting process and thus the hyperparameter search.

+ New function <code>[plot_prediction_intervals]</code> in module <code>[plot]</code>.

+ Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> are able to predict series not seen during training. This is useful when the user wants to predict a new series that was not included in the training data.

+ `encoding` can be set to `None` in Global Forecasters <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code>. This option does not add the encoded series ids to the estimator training matrix.

+ New argument `add_aggregated_metric` in <code>[backtesting_forecaster_multiseries]</code> to include, in addition to the metrics for each level, the aggregated metric of all levels using the average (arithmetic mean), weighted average (weighted by the number of predicted values of each level) or pooling (the values of all levels are pooled and then the metric is calculated).

+ New argument `aggregate_metric` in <code>[grid_search_forecaster_multiseries]</code>, <code>[random_search_forecaster_multiseries]</code> and <code>[bayesian_search_forecaster_multiseries]</code> to select the aggregation method used to combine the metric(s) of all levels during the hyperparameter search. The available methods are: mean (arithmetic mean), weighted (weighted by the number of predicted values of each level) and pool (the values of all levels are pooled and then the metric is calculated). If more than one metric and/or aggregation method is used, all are reported in the results, but the first of each is used to select the best model.

+ New class <code>DateTimeFeatureTransformer</code> and function <code>create_datetime_features</code> in the <code>[preprocessing]</code> module to create datetime and calendar features from a datetime index.

**Changed**

+ Deprecated `python 3.8` compatibility.

+ Update [project dependencies](../quick-start/how-to-install.md).

+ Change default value of `n_bins` when initializing <code>[ForecasterAutoreg]</code> from 15 to 10.

+ Refactor `_recursive_predict` in all recursive forecasters.

+ Change default value of `transformer_series` when initializing <code>[ForecasterAutoregMultiSeries]</code> and <code>[ForecasterAutoregMultiSeriesCustom]</code> from `StandardScaler()` to `None`.

+ Function `_get_metric` moved from <code>[model_selection]</code> to <code>[metrics]</code>.

+ Change information message when `verbose` is `True` in <code>[backtesting_forecaster]</code> and <code>[backtesting_forecaster_multiseries]</code>.

+ `select_n_jobs_backtesting` and `select_n_jobs_fit` in <code>[utils]</code> return `n_jobs = 1` if estimator is `LGBMRegressor`. This is because `lightgbm` is highly optimized for gradient boosting and parallelizes operations at a very fine-grained level, making additional parallelization unnecessary and potentially harmful due to resource contention.

+ `metric_values` returned by <code>[backtesting_forecaster]</code> and <code>[backtesting_sarimax]</code> is a `pandas DataFrame` with one column per metric instead of a `list`.

**Fixed**

+ Bug fix in <code>[backtesting_forecaster_multiseries]</code> using a <code>[ForecasterAutoregMultiSeries]</code> or <code>[ForecasterAutoregMultiSeriesCustom]</code> that includes differentiation.


## 0.12.1 <small>May 20, 2024</small> { id="0.12.1" }

<span class="badge text-bg-fix">Fix</span> This is a minor release to fix a bug.

**Added**


**Changed**


**Fixed**

+ Bug fix when storing `last_window` using a [`ForecasterAutoregMultiSeries`] that includes differentiation.


## 0.12.0 <small>May 05, 2024</small> { id="0.12.0" }

The main changes in this release are:

+ <span class="badge text-bg-feature">Feature</span> Multiseries forecaster (Global Models) can be trained using [series of different lengths and with different exogenous variables](https://skforecast.org/latest/user_guides/multi-series-with-different-length-and-different_exog) per series.

+ <span class="badge text-bg-feature">Feature</span> New functionality to [select features](https://skforecast.org/latest/user_guides/feature-selection) using scikit-learn selectors ([`select_features`](https://skforecast.org/latest/api/model_selection#skforecast.model_selection.model_selection.select_features) and [`select_features_multiseries`](https://skforecast.org/0.12.0/api/model_selection_multiseries#skforecast.model_selection_multiseries.model_selection_multiseries.select_features_multiseries)).

+ <span class="badge text-bg-feature">Feature</span> Added new forecaster [`ForecasterRnn`](https://skforecast.org/latest/api/forecasterrnn) to create forecasting models based on [deep learning](https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm) (RNN and LSTM).

+ <span class="badge text-bg-feature">Feature</span> New method to [predict intervals conditioned on the range of the predicted values](https://skforecast.org/0.12.0/user_guides/probabilistic-forecasting#intervals-conditioned-on-predicted-values-binned-residuals). This is can help to improve the interval coverage when the residuals are not homoscedastic ([`ForecasterAutoreg`](https://skforecast.org/0.12.0/api/forecasterautoreg)).

+ <span class="badge text-bg-enhancement">Enhancement</span> [Bayesian hyperparameter search](https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting#hyperparameter-tuning-and-lags-selection-multi-series) is now available for all multiseries forecasters using `optuna` as the search engine.

+ <span class="badge text-bg-enhancement">Enhancement</span> All Recursive Forecasters are now able to [differentiate the time series](https://skforecast.org/latest/user_guides/time-series-differentiation) before modeling it.

+ <span class="badge text-bg-api-change">API Change</span> Changed the default value of the `transformer_series` argument to use a `StandardScaler()` in the Global Forecasters ([`ForecasterAutoregMultiSeries`](https://skforecast.org/0.12.0/api/forecastermultiseries), [`ForecasterAutoregMultiSeriesCustom`](https://skforecast.org/0.12.0/api/forecastermultiseriescustom) and [`ForecasterAutoregMultiVariate`](https://skforecast.org/0.12.0/api/forecastermultivariate)).

**Added**

+ Added `bayesian_search_forecaster_multiseries` function to `model_selection_multiseries` module. This function performs a Bayesian hyperparameter search for the `ForecasterAutoregMultiSeries`, `ForecasterAutoregMultiSeriesCustom`, and `ForecasterAutoregMultiVariate` using `optuna` as the search engine.

+ `ForecasterAutoregMultiVariate` allows to include None when lags is a dict so that a series does not participate in the construction of X_train.

+ The `output_file` argument has been added to the hyperparameter search functions in the `model_selection`, `model_selection_multiseries` and `model_selection_sarimax` modules to save the results of the hyperparameter search in a tab-separated values (TSV) file.

+ New argument `binned_residuals` in method `predict_interval` allows to condition the bootstrapped residuals on range of the predicted values. 

+ Added `save_custom_functions` argument to the `save_forecaster` function in the `utils` module. If `True`, save custom functions used in the forecaster (`fun_predictors` and `weight_func`) as .py files. Custom functions must be available in the environment where the forecaster is loaded.

+ Added `select_features` and `select_features_multiseries` functions to the `model_selection` and `model_selection_multiseries` modules to perform feature selection using scikit-learn selectors.

+ Added `sort_importance` argument to `get_feature_importances` method in all Forecasters. If `True`, sort the feature importances in descending order.

+ Added `initialize_lags_grid` function to `model_selection` module. This function initializes the lags to be used in the hyperparameter search functions in `model_selection` and `model_selection_multiseries`.

+ Added `_initialize_levels_model_selection_multiseries` function to `model_selection_multiseries` module. This function initializes the levels of the series to be used in the model selection functions.

+ Added `set_dark_theme` function to the `plot` module to set a dark theme for matplotlib plots.

+ Allow tuple type for `lags` argument in all Forecasters.

+ Argument `differentiation` in all Forecasters to model the n-order differentiated time series.

+ Added `window_size_diff` attribute to all Forecasters. It stores the size of the window (`window_size`) extended by the order of differentiation. Added  to all Forecasters for API consistency.

+ Added `store_last_window` parameter to `fit` method in Forecasters. If `True`, store the last window of the training data.

+ Added `utils.set_skforecast_warnings` function to set the warnings of the skforecast package.

+ Added new forecaster `ForecasterRnn` to create forecasting models based on deep learning (RNN and LSTM).

+ Added new function `create_and_compile_model` to module `skforecast.ForecasterRnn.utils` to help to create and compile a RNN or LSTM models to be used in `ForecasterRnn`.

**Changed**

+ Deprecated argument `lags_grid` in `bayesian_search_forecaster`. Use `search_space` to define the candidate values for the lags. This allows the lags to be optimized along with the other hyperparameters of the estimator in the bayesian search.

+ `n_boot` argument in `predict_interval`changed from 500 to 250.

+ Changed the default value of the `transformer_series` argument to use a `StandardScaler()` in the Global Forecasters (`ForecasterAutoregMultiSeries`, `ForecasterAutoregMultiSeriesCustom` and `ForecasterAutoregMultiVariate`).

+ Refactor `utils.select_n_jobs_backtesting` to use the forecaster directly instead of `forecaster_name` and `estimator_name`.

+ Remove `_backtesting_forecaster_verbose` in model_selection in favor of `_create_backtesting_folds`, (deprecated since 0.8.0).

**Fixed**

+ Small bug in `utils.select_n_jobs_backtesting`, rename `ForecasterAutoregMultiseries` to `ForecasterAutoregMultiSeries`.


## 0.11.0 <small>Nov 16, 2023</small> { id="0.11.0" }

The main changes in this release are:

+ New `predict_quantiles` method in all Autoreg Forecasters to calculate the specified quantiles for each step.

+ Create `ForecasterBaseline.ForecasterEquivalentDate`, a Forecaster to create simple model that serves as a basic reference for evaluating the performance of more complex models.

**Added**

+ Added `skforecast.datasets` module. It contains functions to load data for our examples and user guides.

+ Added `predict_quantiles` method to all Autoreg Forecasters.

+ Added `SkforecastVersionWarning` to the `exception` module. This warning notify that the skforecast version installed in the environment differs from the version used to initialize the forecaster when using `load_forecaster`.

+ Create `ForecasterBaseline.ForecasterEquivalentDate`, a Forecaster to create simple model that serves as a basic reference for evaluating the performance of more complex models.

**Changed**

+ Enhance the management of internal copying in skforecast to minimize the number of copies, thereby accelerating data processing.

**Fixed**

+ Rename `self.skforecast_version` attribute to `self.skforecast_version` in all Forecasters.

+ Fixed a bug where the `create_train_X_y` method did not correctly align lags and exogenous variables when the index was not a Pandas index in all Forecasters.


## 0.10.1 <small>Sep 26, 2023</small> { id="0.10.1" }

This is a minor release to fix a bug when using `grid_search_forecaster`, `random_search_forecaster` or `bayesian_search_forecaster` with a Forecaster that includes differentiation.

**Added**


**Changed**


**Fixed**

+ Bug fix `grid_search_forecaster`, `random_search_forecaster` or `bayesian_search_forecaster` with a Forecaster that includes differentiation.


## 0.10.0 <small>Sep 07, 2023</small> { id="0.10.0" }

The main changes in this release are:

+ New `Sarimax.Sarimax` model. A wrapper of `statsmodels.SARIMAX` that follows the scikit-learn API and can be used with the `ForecasterSarimax`.

+ Added `differentiation` argument to `ForecasterAutoreg` and `ForecasterAutoregCustom` to model the n-order differentiated time series using the new skforecast preprocessor `TimeSeriesDifferentiator`.

**Added**

+ New `Sarimax.Sarimax` model. A wrapper of `statsmodels.SARIMAX` that follows the scikit-learn API.

+ Added `skforecast.preprocessing.TimeSeriesDifferentiator` to preprocess time series by differentiating or integrating them (reverse differentiation).

+ Added `differentiation` argument to `ForecasterAutoreg` and `ForecasterAutoregCustom` to model the n-order differentiated time series.

**Changed**

+ Refactor `ForecasterSarimax` to work with both skforecast Sarimax and pmdarima ARIMA models.

+ Replace `setup.py` with `pyproject.toml`.

**Fixed**


## 0.9.1 <small>Jul 14, 2023</small> { id="0.9.1" }

The main changes in this release are:

+ Fix imports in `skforecast.utils` module to correctly import `sklearn.linear_model` into the `select_n_jobs_backtesting` and `select_n_jobs_fit_forecaster` functions.

**Added**

**Changed**

**Fixed**

+ Fix imports in `skforecast.utils` module to correctly import `sklearn.linear_model` into the `select_n_jobs_backtesting` and `select_n_jobs_fit_forecaster` functions.


## 0.9.0 <small>Jul 09, 2023</small> { id="0.9.0" }

The main changes in this release are:

+ `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate` include the `n_jobs` argument in their `fit` method, allowing multi-process parallelization for improved performance.

+ All backtesting and grid search functions have been extended to include the `n_jobs` argument, allowing multi-process parallelization for improved performance.

+ Argument `refit` now can be also an `integer` in all backtesting dependent functions in modules `model_selection`, `model_selection_multiseries`, and `model_selection_sarimax`. This allows the Forecaster to be trained every this number of iterations.

+ `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiSeriesCustom` can be trained using series of different lengths. This means that the model can handle datasets with different numbers of data points in each series.

**Added**

+ Support for `scikit-learn 1.3.x`.

+ Argument `n_jobs='auto'` to `fit` method in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate` to allow multi-process parallelization.

+ Argument `n_jobs='auto'` to all backtesting dependent functions in modules `model_selection`, `model_selection_multiseries` and `model_selection_sarimax` to allow multi-process parallelization.

+ Argument `refit` now can be also an `integer` in all backtesting dependent functions in modules `model_selection`, `model_selection_multiseries`, and `model_selection_sarimax`. This allows the Forecaster to be trained every this number of iterations.

+ `ForecasterAutoregMultiSeries` and `ForecasterAutoregMultiSeriesCustom` allow to use series of different lengths for training.

+ Added `show_progress` to grid search functions.

+ Added functions `select_n_jobs_backtesting` and `select_n_jobs_fit_forecaster` to `utils` to select the number of jobs to use during multi-process parallelization.

**Changed**

+ Remove `get_feature_importance` in favor of `get_feature_importances` in all Forecasters, (deprecated since 0.8.0).

+ The `model_selection._create_backtesting_folds` function now also returns the last window indices and whether or not to train the forecaster.

+ The `model_selection` functions `_backtesting_forecaster_refit` and `_backtesting_forecaster_no_refit` have been unified in `_backtesting_forecaster`.

+ The `model_selection_multiseries` functions `_backtesting_forecaster_multiseries_refit` and `_backtesting_forecaster_multiseries_no_refit` have been unified in `_backtesting_forecaster_multiseries`.

+ The `model_selection_sarimax` functions `_backtesting_refit_sarimax` and `_backtesting_no_refit_sarimax` have been unified in `_backtesting_sarimax`.

+ `utils.preprocess_y` allows a pandas DataFrame as input.

**Fixed**

+ Ensure reproducibility of Direct Forecasters when using `predict_bootstrapping`, `predict_dist` and `predict_interval` with a `list` of steps.

+ The `create_train_X_y` method returns a dict of pandas Series as `y_train` in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`. This ensures that each series has the appropriate index according to the step to be trained.

+ The `filter_train_X_y_for_step` method in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate` now updates the index of `X_train_step` to ensure correct alignment with `y_train_step`.


## 0.8.1 <small>May 27, 2023</small> { id="0.8.1" }

**Added**

- Argument `store_in_sample_residuals=True` in `fit` method added to all forecasters to speed up functions such as backtesting.

**Changed**

- Refactor `utils.exog_to_direct` and `utils.exog_to_direct_numpy` to increase performance.

**Fixed**

- `utils.check_exog_dtypes` now compares the `dtype.name` instead of the `dtype`. (suggested by Metaming https://github.com/Metaming)


## 0.8.0 <small>May 16, 2023</small> { id="0.8.0" }

**Added**

+ Added the `fit_kwargs` argument to all forecasters to allow the inclusion of additional keyword arguments passed to the estimator's `fit` method.

+ Added the `set_fit_kwargs` method to set the `fit_kwargs` attribute.
  
+ Support for `pandas 2.0.x`.

+ Added `exceptions` module with custom warnings.

+ Added function `utils.check_exog_dtypes` to issue a warning if exogenous variables are one of type `init`, `float`, or `category`. Raise Exception if `exog` has categorical columns with non integer values.

+ Added function `utils.get_exog_dtypes` to get the data types of the exogenous variables included during the training of the forecaster model. 

+ Added function `utils.cast_exog_dtypes` to cast data types of the exogenous variables using a dictionary as a mapping.

+ Added function `utils.check_select_fit_kwargs` to check if the argument `fit_kwargs` is a dictionary and select only the keys used by the `fit` method of the estimator.

+ Added function `model_selection._create_backtesting_folds` to provide train/test indices (position) for backtesting functions.

+ Added argument `gap` to functions in `model_selection`, `model_selection_multiseries` and `model_selection_sarimax` to omit observations between training and prediction.

+ Added argument `show_progress` to functions `model_selection.backtesting_forecaster`, `model_selection_multiseries.backtesting_forecaster_multiseries` and `model_selection_sarimax.backtesting_forecaster_sarimax` to indicate weather to show a progress bar.

+ Added argument `remove_suffix`, default `False`, to the method `filter_train_X_y_for_step()` in `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`. If `remove_suffix=True` the suffix "_step_i" will be removed from the column names of the training matrices.

**Changed**

+ Rename optional dependency package `statsmodels` to `sarimax`. Now only `pmdarima` will be installed, `statsmodels` is no longer needed.

+ Rename `get_feature_importance()` to `get_feature_importances()` in all Forecasters. `get_feature_importance()` method will me removed in skforecast 0.9.0.

+ Refactor `get_feature_importances()` in all Forecasters.

+ Remove `model_selection_statsmodels` in favor of `ForecasterSarimax` and `model_selection_sarimax`, (deprecated since 0.7.0).

+ Remove attributes `create_predictors` and `source_code_create_predictors` in favor of `fun_predictors` and `source_code_fun_predictors` in `ForecasterAutoregCustom`, (deprecated since 0.7.0).

+ The `utils.check_exog` function now includes a new optional parameter, `allow_nan`, that controls whether a warning should be issued if the input `exog` contains NaN values. 

+ `utils.check_exog` is applied before and after `exog` transformations.

+ The `utils.preprocess_y` function now includes a new optional parameter, `return_values`, that controls whether to return a numpy ndarray with the values of y or not. This new option is intended to avoid copying data when it is not necessary.

+ The `utils.preprocess_exog` function now includes a new optional parameter, `return_values`, that controls whether to return a numpy ndarray with the values of y or not. This new option is intended to avoid copying data when it is not necessary.

+ Replaced `tqdm.tqdm` by `tqdm.auto.tqdm`.

+ Refactor `utils.exog_to_direct`.

**Fixed**

+ The dtypes of exogenous variables are maintained when generating the training matrices with the `create_train_X_y` method in all the Forecasters.


## 0.7.0 <small>Mar 21, 2023</small> { id="0.7.0" }

**Added**

+ Class `ForecasterAutoregMultiSeriesCustom`.

+ Class `ForecasterSarimax` and `model_selection_sarimax` (wrapper of [pmdarima](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA)).
  
+ Method `predict_interval()` to `ForecasterAutoregDirect` and `ForecasterAutoregMultiVariate`.

+ Method `predict_bootstrapping()` to all forecasters, generate multiple forecasting predictions using a bootstrapping process.

+ Method `predict_dist()` to all forecasters, fit a given probability distribution for each step using a bootstrapping process.

+ Function `plot_prediction_distribution` in module `plot`.

+ Alias `backtesting_forecaster_multivariate` for `backtesting_forecaster_multiseries` in `model_selection_multiseries` module.

+ Alias `grid_search_forecaster_multivariate` for `grid_search_forecaster_multiseries` in `model_selection_multiseries` module.

+ Alias `random_search_forecaster_multivariate` for `random_search_forecaster_multiseries` in `model_selection_multiseries` module.

+ Attribute `forecaster_id` to all Forecasters.

**Changed**

+ Deprecated `python 3.7` compatibility.

+ Added `python 3.11` compatibility.

+ `model_selection_statsmodels` is deprecated in favor of `ForecasterSarimax` and `model_selection_sarimax`. It will be removed in version 0.8.0.

+ Remove `levels_weights` argument in `grid_search_forecaster_multiseries` and `random_search_forecaster_multiseries`, deprecated since version 0.6.0. Use `series_weights` and `weights_func` when creating the forecaster instead.

+ Attributes `create_predictors` and `source_code_create_predictors` renamed to `fun_predictors` and `source_code_fun_predictors` in `ForecasterAutoregCustom`. Old names will be removed in version 0.8.0.

+ Remove engine `'skopt'` in `bayesian_search_forecaster` in favor of engine `'optuna'`. To continue using it, use skforecast 0.6.0.

+ `in_sample_residuals` and `out_sample_residuals` are stored as numpy ndarrays instead of pandas series.

+ In `ForecasterAutoregMultiSeries`, `set_out_sample_residuals()` is now expecting a `dict` for the `residuals` argument instead of a `pandas DataFrame`.

+ Remove the `scikit-optimize` dependency.

**Fixed**

+ Remove operator `**` in `set_params()` method for all forecasters.

+ Replace `getfullargspec` in favor of `inspect.signature` (contribution by @jordisilv).


## 0.6.0 <small>Nov 30, 2022</small> { id="0.6.0" }

**Added**

+ Class `ForecasterAutoregMultivariate`.

+ Function `initialize_lags` in `utils` module  to create lags values in the initialization of forecasters (applies to all forecasters).

+ Function `initialize_weights` in `utils` module to check and initialize arguments `series_weights`and `weight_func` (applies to all forecasters).

+ Argument `weights_func` in all Forecasters to allow weighted time series forecasting. Individual time based weights can be assigned to each value of the series during the model training.

+ Argument `series_weights` in `ForecasterAutoregMultiSeries` to define individual weights each series.

+ Include argument `random_state` in all Forecasters `set_out_sample_residuals` methods for random sampling with reproducible output.

+ In `ForecasterAutoregMultiSeries`, `predict` and `predict_interval` methods allow the simultaneous prediction of multiple levels.

+ `backtesting_forecaster_multiseries` allows backtesting multiple levels simultaneously.

+ `metric` argument can be a list in `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries`. If `metric` is a `list`, multiple metrics will be calculated. (suggested by Pablo Dávila Herrero https://github.com/Pablo-Davila)

+ Function `multivariate_time_series_corr` in module `utils`.

+ Function `plot_multivariate_time_series_corr` in module `plot`.
  
**Changed**

+ `ForecasterAutoregDirect` allows to predict specific steps.

+ Remove `ForecasterAutoregMultiOutput` in favor of `ForecasterAutoregDirect`, (deprecated since 0.5.0).

+ Rename function `exog_to_multi_output` to `exog_to_direct` in `utils` module.

+ In `ForecasterAutoregMultiSeries`, rename parameter `series_levels` to `series_col_names`.

+ In `ForecasterAutoregMultiSeries` change type of `out_sample_residuals` to a `dict` of numpy ndarrays.

+ In `ForecasterAutoregMultiSeries`, delete argument `level` from method `set_out_sample_residuals`.

+ In `ForecasterAutoregMultiSeries`, `level` argument of `predict` and `predict_interval` renamed to `levels`.

+ In `backtesting_forecaster_multiseries`, `level` argument of `predict` and `predict_interval` renamed to `levels`.

+ In `check_predict_input` function, argument `level` renamed to `levels` and `series_levels` renamed to `series_col_names`.

+ In `backtesting_forecaster_multiseries`, `metrics_levels` output is now a pandas DataFrame.

+ In `grid_search_forecaster_multiseries` and `random_search_forecaster_multiseries`, argument `levels_weights` is deprecated since version 0.6.0, and will be removed in version 0.7.0. Use `series_weights` and `weights_func` when creating the forecaster instead.

+ Refactor `_create_lags_` in `ForecasterAutoreg`, `ForecasterAutoregDirect` and `ForecasterAutoregMultiSeries`. (suggested by Bennett https://github.com/Bennett561)

+ Refactor `backtesting_forecaster` and `backtesting_forecaster_multiseries`.

+ In `ForecasterAutoregDirect`, `filter_train_X_y_for_step` now starts at 1 (before 0).

+ In `ForecasterAutoregDirect`, DataFrame `y_train` now start with 1, `y_step_1` (before `y_step_0`).

+ Remove `cv_forecaster` from module `model_selection`.

**Fixed**

+ In `ForecasterAutoregMultiSeries`, argument `last_window` predict method now works when it is a pandas DataFrame.

+ In `ForecasterAutoregMultiSeries`, fix bug transformers initialization.


## 0.5.1 <small>Oct 05, 2022</small> { id="0.5.1" }

**Added**

+ Check that `exog` and `y` have the same length in `_evaluate_grid_hyperparameters` and `bayesian_search_forecaster` to avoid fit exception when `return_best`.

+ Check that `exog` and `series` have the same length in `_evaluate_grid_hyperparameters_multiseries` to avoid fit exception when `return_best`.

**Changed**

+ Argument `levels_list` in `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` and `_evaluate_grid_hyperparameters_multiseries` renamed to `levels`.

**Fixed**

+ `ForecasterAutoregMultiOutput` updated to match `ForecasterAutoregDirect`.

+ Fix Exception to raise when `level_weights` does not add up to a number close to 1.0 (before was exactly 1.0) in `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` and `_evaluate_grid_hyperparameters_multiseries`.

+ `Create_train_X_y` in `ForecasterAutoregMultiSeries` now works when the forecaster is not fitted.


## 0.5.0 <small>Sep 23, 2022</small> { id="0.5.0" }

**Added**

+ New arguments `transformer_y` (`transformer_series` for multiseries) and `transformer_exog` in all forecaster classes. It is for transforming (scaling, max-min, ...) the modeled time series and exogenous variables inside the forecaster.

+ Functions in utils `transform_series` and `transform_dataframe` to carry out the transformation of the modeled time series and exogenous variables.

+ Functions `_backtesting_forecaster_verbose`, `random_search_forecaster`, `_evaluate_grid_hyperparameters`, `bayesian_search_forecaster`, `_bayesian_search_optuna` and `_bayesian_search_skopt` in model_selection.

+ Created `ForecasterAutoregMultiSeries` class for modeling multiple time series simultaneously.

+ Created module `model_selection_multiseries`. Functions: `_backtesting_forecaster_multiseries_refit`, `_backtesting_forecaster_multiseries_no_refit`, `backtesting_forecaster_multiseries`, `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries` and `_evaluate_grid_hyperparameters_multiseries`.

+ Function `_check_interval` in utils. (suggested by Thomas Karaouzene https://github.com/tkaraouzene)

+ `metric` can be a list in `backtesting_forecaster`, `grid_search_forecaster`, `random_search_forecaster`, `backtesting_forecaster_multiseries`. If `metric` is a `list`, multiple metrics will be calculated. (suggested by Pablo Dávila Herrero https://github.com/Pablo-Davila)

+ Skforecast works with python 3.10.

+ Functions `save_forecaster` and `load_forecaster` to module utils.

+ `get_feature_importance()` method checks if the forecast is fitted.

**Changed**

+ `backtesting_forecaster` change default value of argument `fixed_train_size: bool=True`.

+ Remove argument `set_out_sample_residuals` in function `backtesting_forecaster` (deprecated since 0.4.2).

+ `backtesting_forecaster` verbose now includes fold size.

+ `grid_search_forecaster` results include the name of the used metric as column name.

+ Remove `get_coef` method from `ForecasterAutoreg`, `ForecasterAutoregCustom` and `ForecasterAutoregMultiOutput` (deprecated since 0.4.3).

+ `_get_metric` now allows `mean_squared_log_error`.

+ `ForecasterAutoregMultiOutput` has been renamed to `ForecasterAutoregDirect`. `ForecasterAutoregMultiOutput` will be removed in version 0.6.0.

+ `check_predict_input` updated to check `ForecasterAutoregMultiSeries` inputs.

+ `set_out_sample_residuals` has a new argument `transform` to transform the residuals before being stored.

**Fixed**

+ `fit` now stores `last_window` values with len = forecaster.max_lag in ForecasterAutoreg and ForecasterAutoregCustom.

+ `in_sample_residuals` stored as a `pd.Series` when `len(residuals) > 1000`.


## 0.4.3 <small>Mar 18, 2022</small> { id="0.4.3" }

**Added**

+ Checks if all elements in lags are `int` when creating ForecasterAutoreg and ForecasterAutoregMultiOutput.

+ Add `fixed_train_size: bool=False` argument to `backtesting_forecaster` and `backtesting_sarimax`

**Changed**

+ Rename `get_metric` to `_get_metric`.

+ Functions in model_selection module allow custom metrics.

+ Functions in model_selection_statsmodels module allow custom metrics.

+ Change function `set_out_sample_residuals` (ForecasterAutoreg and ForecasterAutoregCustom), `residuals` argument must be a `pandas Series` (was `numpy ndarray`).

+ Returned value of backtesting functions (model_selection and model_selection_statsmodels) is now a `float` (was `numpy ndarray`).

+ `get_coef` and `get_feature_importance` methods unified in `get_feature_importance`.

**Fixed**

+ Requirements versions.

+ Method `fit` doesn't remove `out_sample_residuals` each time the forecaster is fitted.

+ Added random seed to residuals downsampling (ForecasterAutoreg and ForecasterAutoregCustom)


## 0.4.2 <small>Jan 08, 2022</small> { id="0.4.2" }

**Added**

+ Increased verbosity of function `backtesting_forecaster()`.

+ Random state argument in `backtesting_forecaster()`.

**Changed**

+ Function `backtesting_forecaster()` do not modify the original forecaster.

+ Deprecated argument `set_out_sample_residuals` in function `backtesting_forecaster()`.

+ Function `model_selection.time_series_spliter` renamed to `model_selection.time_series_splitter`

**Fixed**

+ Methods `get_coef` and `get_feature_importance` of `ForecasterAutoregMultiOutput` class return proper feature names.


## 0.4.1 <small>Dec 13, 2021</small> { id="0.4.1" }

**Added**

**Changed**

**Fixed**

+ `fit` and `predict` transform pandas Series and DataFrames to numpy arrays if estimator is XGBoost.


## 0.4.0 <small>Dec 10, 2021</small> { id="0.4.0" }

Version 0.4 has undergone a huge code refactoring. Main changes are related to input-output formats (only pandas Series and DataFrames are allowed although internally numpy arrays are used for performance) and model validation methods (unified into backtesting with and without refit).

**Added**

+ `ForecasterBase` as parent class

**Changed**

+ Argument `y` must be pandas Series. Numpy ndarrays are not allowed anymore.

+ Argument `exog` must be pandas Series or pandas DataFrame. Numpy ndarrays are not allowed anymore.

+ Output of `predict` is a pandas Series with index according to the steps predicted.

+ Scikit-learn pipelines are allowed as estimators.

+ `backtesting_forecaster` and `backtesting_forecaster_intervals` have been combined in a single function.

    + It is possible to backtest forecasters already trained.
    + `ForecasterAutoregMultiOutput` allows incomplete folds.
    + It is possible to update `out_sample_residuals` with backtesting residuals.
    
+ `cv_forecaster` has the option to update `out_sample_residuals` with backtesting residuals.

+ `backtesting_sarimax_statsmodels` and `cv_sarimax_statsmodels` have been combined in a single function.

+ `gridsearch_forecaster` use backtesting as validation strategy with the option of refit.

+ Extended information when printing `Forecaster` object.

+ All static methods for checking and preprocessing inputs moved to module utils.

+ Remove deprecated class `ForecasterCustom`.

**Fixed**


## 0.3.0 <small>Sep 01, 2021</small> { id="0.3.0" }

**Added**

+ New module model_selection_statsmodels to cross-validate, backtesting and grid search AutoReg and SARIMAX models from statsmodels library:
    + `backtesting_autoreg_statsmodels`
    + `cv_autoreg_statsmodels`
    + `backtesting_sarimax_statsmodels`
    + `cv_sarimax_statsmodels`
    + `grid_search_sarimax_statsmodels`
    
+ Added attribute window_size to `ForecasterAutoreg` and `ForecasterAutoregCustom`. It is equal to `max_lag`.

**Changed**

+ `cv_forecaster` returns cross-validation metrics and cross-validation predictions.
+ Added an extra column for each parameter in the dataframe returned by `grid_search_forecaster`.
+ statsmodels 0.12.2 added to requirements

**Fixed**


## 0.2.0 <small>Aug 26, 2021</small> { id="0.2.0" }

**Added**


+ Multiple exogenous variables can be passed as pandas DataFrame.

+ Documentation at https://skforecast.org

+ New unit test

+ Increased typing

**Changed**

+ New implementation of `ForecasterAutoregMultiOutput`. The training process in the new version creates a different X_train for each step. See [Direct multi-step forecasting](https://github.com/skforecast/skforecast#introduction) for more details. Old version can be access with `skforecast.deprecated.ForecasterAutoregMultiOutput`.

**Fixed**


## 0.1.9 <small>Jul 27, 2021</small> { id="0.1.9" }

**Added**

+ Logging total number of models to fit in `grid_search_forecaster`.

+ Class `ForecasterAutoregCustom`.

+ Method `create_train_X_y` to facilitate access to the training data matrix created from `y` and `exog`.

**Changed**


+ New implementation of `ForecasterAutoregMultiOutput`. The training process in the new version creates a different X_train for each step. See [Direct multi-step forecasting](https://github.com/skforecast/skforecast#introduction) for more details. Old version can be accessed with `skforecast.deprecated.ForecasterAutoregMultiOutput`.

+ Class `ForecasterCustom` has been renamed to `ForecasterAutoregCustom`. However, `ForecasterCustom` will still remain to keep backward compatibility.

+ Argument `metric` in `cv_forecaster`, `backtesting_forecaster`, `grid_search_forecaster` and `backtesting_forecaster_intervals` changed from 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error' to 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'.

+ Check if argument `metric` in `cv_forecaster`, `backtesting_forecaster`, `grid_search_forecaster` and `backtesting_forecaster_intervals` is one of 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'.

+ `time_series_spliter` doesn't include the remaining observations in the last complete fold but in a new one when `allow_incomplete_fold=True`. Take in consideration that incomplete folds with few observations could overestimate or underestimate the validation metric.

**Fixed**

+ Update lags of  `ForecasterAutoregMultiOutput` after `grid_search_forecaster`.


## 0.1.8.1 <small>May 17, 2021</small> { id="0.1.8.1" }

**Added**

+ `set_out_sample_residuals` method to store or update out of sample residuals used by `predict_interval`.

**Changed**

+ `backtesting_forecaster_intervals` and `backtesting_forecaster` print number of steps per fold.

+ Only stored up to 1000 residuals.

+ Improved verbose in `backtesting_forecaster_intervals`.

**Fixed**

+ Warning of incomplete folds when using `backtesting_forecast` with a  `ForecasterAutoregMultiOutput`.

+ `ForecasterAutoregMultiOutput.predict` allow exog data longer than needed (steps).

+ `backtesting_forecast` prints correctly the number of folds when remainder observations are cero.

+ Removed named argument X in `self.estimator.predict(X)` to allow using XGBoost estimator.

+ Values stored in `self.last_window` when training `ForecasterAutoregMultiOutput`. 


## 0.1.8 <small>Apr 02, 2021</small> { id="0.1.8" }

**Added**

- Class `ForecasterAutoregMultiOutput.py`: forecaster with direct multi-step predictions.
- Method `ForecasterCustom.predict_interval` and  `ForecasterAutoreg.predict_interval`: estimate prediction interval using bootstrapping.
- `skforecast.model_selection.backtesting_forecaster_intervals` perform backtesting and return prediction intervals.
 
**Changed**

 
**Fixed**


## 0.1.7 <small>Mar 19, 2021</small> { id="0.1.7" }

**Added**

- Class `ForecasterCustom`: same functionalities as `ForecasterAutoreg` but allows custom definition of predictors.
 
**Changed**

- `grid_search forecaster` adapted to work with objects `ForecasterCustom` in addition to `ForecasterAutoreg`.
 
**Fixed**
 
 
## 0.1.6 <small>Mar 14, 2021</small> { id="0.1.6" }

**Added**

- Method `get_feature_importances` to `skforecast.ForecasterAutoreg`.
- Added backtesting strategy in `grid_search_forecaster`.
- Added `backtesting_forecast` to `skforecast.model_selection`.
 
**Changed**

- Method `create_lags` return a matrix where the order of columns match the ascending order of lags. For example, column 0 contains the values of the minimum lag used as predictor.
- Renamed argument `X` to `last_window` in method `predict`.
- Renamed `ts_cv_forecaster` to `cv_forecaster`.
 
**Fixed**


## 0.1.4 <small>Feb 15, 2021</small> { id="0.1.4" }
  
**Added**

- Method `get_coef` to `skforecast.ForecasterAutoreg`.
 
**Changed**

 
**Fixed**



<!-- Links to API Reference -->
<!-- Forecasters -->
[recursive]: ../api/ForecasterRecursive.md
[ForecasterRecursive]: ../api/ForecasterRecursive.md
[ForecasterRecursiveClassifier]: ../api/ForecasterRecursiveClassifier.md
[ForecasterDirect]: ../api/ForecasterDirect.md
[ForecasterRecursiveMultiSeries]: ../api/ForecasterRecursiveMultiSeries.md
[ForecasterDirectMultiVariate]: ../api/ForecasterDirectMultiVariate.md
[ForecasterFoundation]: ../api/ForecasterFoundation.md
[ForecasterRnn]: ../api/ForecasterRnn.md
[create_and_compile_model]: ../api/ForecasterRnn.md#skforecast.deep_learning.utils.create_and_compile_model
[ForecasterStats]: ../api/ForecasterStats.md
[ForecasterEquivalentDate]: ../api/ForecasterEquivalentDate.md
[ForecasterRecursiveClassifier]: ../api/ForecasterRecursiveClassifier.md

<!-- foundation -->
[FoundationModel]: ../api/FoundationModel.md#skforecast.foundation._foundation_model.FoundationModel
[FoundationModelInfo]: ../api/FoundationModel.md#skforecast.foundation._model_info.FoundationModelInfo
[get_model_info]: ../api/FoundationModel.md#skforecast.foundation._model_info.get_model_info
[list_adapters]: ../api/FoundationModel.md#skforecast.foundation._model_info.list_adapters
[ChronosAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.ChronosAdapter
[TimesFM25Adapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TimesFM25Adapter
[TimesFM3Adapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TimesFM3Adapter
[MoiraiAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.MoiraiAdapter
[TabICLAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TabICLAdapter
[TabPFNAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TabPFNAdapter
[T0Adapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.T0Adapter
[NoriAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.NoriAdapter
[TSICLAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TSICLAdapter

<!-- stats -->
[stats]: ../api/stats.md
[Arima]: ../api/stats.md#skforecast.stats._arima.Arima
[Sarimax]: ../api/stats.md#skforecast.stats._sarimax.Sarimax
[Ets]: ../api/stats.md#skforecast.stats._ets.Ets
[Arar]: ../api/stats.md#skforecast.stats._arar.Arar
[acf]: ../api/stats.md#skforecast.stats._autocorrelation.acf
[pacf]: ../api/stats.md#skforecast.stats._autocorrelation.pacf
[calculate_lag_autocorrelation]: ../api/stats.md#skforecast.stats._autocorrelation.calculate_lag_autocorrelation

<!-- model_selection -->
[model_selection]: ../api/model_selection.md

[backtesting_forecaster]: ../api/model_selection.md#skforecast.model_selection._validation.backtesting_forecaster
[grid_search_forecaster]: ../api/model_selection.md#skforecast.model_selection._search.grid_search_forecaster
[random_search_forecaster]: ../api/model_selection.md#skforecast.model_selection._search.random_search_forecaster
[bayesian_search_forecaster]: ../api/model_selection.md#skforecast.model_selection._search.bayesian_search_forecaster

[backtesting_forecaster_multiseries]: ../api/model_selection.md#skforecast.model_selection._validation.backtesting_forecaster_multiseries
[grid_search_forecaster_multiseries]: ../api/model_selection.md#skforecast.model_selection._search.grid_search_forecaster_multiseries
[random_search_forecaster_multiseries]: ../api/model_selection.md#skforecast.model_selection._search.random_search_forecaster_multiseries
[bayesian_search_forecaster_multiseries]: ../api/model_selection.md#skforecast.model_selection._search.bayesian_search_forecaster_multiseries

[backtesting_foundation]: ../api/model_selection.md#skforecast.model_selection._validation.backtesting_foundation
[bayesian_search_foundation]: ../api/model_selection.md#skforecast.model_selection._search.bayesian_search_foundation

[backtesting_stats]: ../api/model_selection.md#skforecast.model_selection._validation.backtesting_stats
[grid_search_stats]: ../api/model_selection.md#skforecast.model_selection._search.grid_search_stats
[random_search_stats]: ../api/model_selection.md#skforecast.model_selection._search.random_search_stats

[grid_search_equivalent_date]: ../api/model_selection.md#skforecast.model_selection._search.grid_search_equivalent_date

[TimeSeriesFold]: ../api/model_selection.md#skforecast.model_selection._split.TimeSeriesFold
[OneStepAheadFold]: ../api/model_selection.md#skforecast.model_selection._split.OneStepAheadFold
[BaseFold]: ../api/model_selection.md#skforecast.model_selection._split.BaseFold

<!-- feature_selection -->
[feature_selection]: ../api/feature_selection.md
[select_features]: ../api/feature_selection.md#skforecast.feature_selection.feature_selection.select_features
[select_features_multiseries]: ../api/feature_selection.md#skforecast.feature_selection.feature_selection.select_features_multiseries

<!-- preprocessing -->
[preprocessing]: ../api/preprocessing.md
[RollingFeatures]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.RollingFeatures
[RollingFeaturesClassification]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.RollingFeaturesClassification
[reshape_series_wide_to_long]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.reshape_series_wide_to_long
[reshape_series_long_to_dict]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.reshape_series_long_to_dict
[reshape_exog_long_to_dict]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.reshape_exog_long_to_dict
[reshape_series_exog_dict_to_long]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.reshape_series_exog_dict_to_long
[TimeSeriesDifferentiator]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.TimeSeriesDifferentiator
[QuantileBinner]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.QuantileBinner
[ConformalIntervalCalibrator]: ../api/preprocessing.md#skforecast.preprocessing._preprocessing.ConformalIntervalCalibrator
[create_calendar_features]: ../api/preprocessing.md#skforecast.preprocessing._calendar.create_calendar_features
[CalendarFeatures]: ../api/preprocessing.md#skforecast.preprocessing._calendar.CalendarFeatures
[calculate_distance_from_holiday]: ../api/preprocessing.md#skforecast.preprocessing._calendar.calculate_distance_from_holiday

<!-- drift_detection -->
[drift_detection]: ../api/drift_detection.md
[RangeDriftDetector]: ../api/drift_detection.md#skforecast.drift_detection._range_drift.RangeDriftDetector
[PopulationDriftDetector]: ../api/drift_detection.md#skforecast.drift_detection._population_drift.PopulationDriftDetector

<!-- metrics -->
[metrics]: ../api/metrics.md
[mean_absolute_scaled_error]: ../api/metrics.md#skforecast.metrics.mean_absolute_scaled_error
[root_mean_squared_scaled_error]: ../api/metrics.md#skforecast.metrics.root_mean_squared_scaled_error
[symmetric_mean_absolute_percentage_error]: ../api/metrics.md#skforecast.metrics.symmetric_mean_absolute_percentage_error
[calculate_coverage]: ../api/metrics.md#skforecast.metrics.calculate_coverage
[crps_from_predictions]: ../api/metrics.md#skforecast.metrics.crps_from_predictions
[crps_from_quantiles]: ../api/metrics.md#skforecast.metrics.crps_from_quantiles
[winkler_score]: ../api/metrics.md#skforecast.metrics.winkler_score
[weighted_interval_score]: ../api/metrics.md#skforecast.metrics.weighted_interval_score
[create_mean_pinball_loss]: ../api/metrics.md#skforecast.metrics.create_mean_pinball_loss
[add_y_train_argument]: ../api/metrics.md#skforecast.metrics.add_y_train_argument

<!-- plot -->
[plot]: ../api/plot.md
[set_dark_theme]: ../api/plot.md#skforecast.plot.plot.set_dark_theme
[plot_residuals]: ../api/plot.md#skforecast.plot.plot.plot_residuals
[plot_prediction_distribution]: ../api/plot.md#skforecast.plot.plot.plot_prediction_distribution
[plot_prediction_intervals]: ../api/plot.md#skforecast.plot.plot.plot_prediction_intervals
[backtesting_gif_creator]: ../api/plot.md#skforecast.plot.plot.backtesting_gif_creator
[plot_multivariate_time_series_corr]: ../api/plot.md#skforecast.plot.plot.plot_multivariate_time_series_corr

<!-- utils -->
[utils]: ../api/utils.md
[expand_index]: ../api/utils.md#skforecast.utils.utils.expand_index
[save_forecaster]: ../api/utils.md#skforecast.utils.utils.save_forecaster
[load_forecaster]: ../api/utils.md#skforecast.utils.utils.load_forecaster
[show_versions]: ../api/utils.md#skforecast.utils.utils.show_versions
[transform_series]: ../api/utils.md#skforecast.utils.utils.transform_series
[exog_to_direct]: ../api/utils.md#skforecast.utils.utils.exog_to_direct
[exog_to_direct_numpy]: ../api/utils.md#skforecast.utils.utils.exog_to_direct_numpy
[multivariate_time_series_corr]: ../api/utils.md#skforecast.utils.utils.multivariate_time_series_corr

<!-- experimental -->
[experimental]: ../api/experimental.md
[TimeSeriesSplitter]: ../api/experimental.md#skforecast.experimental._splitter.TimeSeriesSplitter

<!-- datasets -->
[datasets]: ../api/datasets.md
[fetch_dataset]: ../api/datasets.md#skforecast.datasets.fetch_dataset
[load_demo_dataset]: ../api/datasets.md#skforecast.datasets.load_demo_dataset
[show_datasets_info]: ../api/datasets.md#skforecast.datasets.show_datasets_info

<!-- exceptions -->
[exceptions]: ../api/exceptions.md
[IgnoredArgumentWarning]: ../api/exceptions.md#skforecast.exceptions.exceptions.IgnoredArgumentWarning
[LicenseWarning]: ../api/exceptions.md#skforecast.exceptions.exceptions.LicenseWarning
[MissingValuesWarning]: ../api/exceptions.md#skforecast.exceptions.exceptions.MissingValuesWarning
[ResidualsUsageWarning]: ../api/exceptions.md#skforecast.exceptions.exceptions.ResidualsUsageWarning

<!-- OLD -->
[ForecasterAutoreg]: https://skforecast.org/0.13.0/api/forecasterautoreg
[ForecasterAutoregCustom]: https://skforecast.org/0.13.0/api/forecasterautoregcustom
[ForecasterAutoregDirect]: https://skforecast.org/0.13.0/api/forecasterautoregdirect
[ForecasterAutoregMultiSeries]: https://skforecast.org/0.13.0/api/forecastermultiseries
[ForecasterAutoregMultiSeriesCustom]: https://skforecast.org/0.13.0/api/forecastermultiseriescustom
[ForecasterAutoregMultiVariate]: https://skforecast.org/0.13.0/api/forecastermultivariate
[model_selection_multiseries]: https://skforecast.org/0.13.0/api/model_selection_multiseries
[model_selection_sarimax]: https://skforecast.org/0.13.0/api/model_selection_sarimax
[series_long_to_dict]: https://skforecast.org/0.16.0/api/preprocessing.html#skforecast.preprocessing.preprocessing.series_long_to_dict
[exog_long_to_dict]: https://skforecast.org/0.16.0/api/preprocessing.html#skforecast.preprocessing.preprocessing.exog_long_to_dict
[ForecasterSarimax]: https://skforecast.org/0.19.0/api/forecastersarimax.html
[backtesting_sarimax]: https://skforecast.org/0.19.0/api/model_selection.html#skforecast.model_selection._validation.backtesting_sarimax
[grid_search_sarimax]: https://skforecast.org/0.19.0/api/model_selection.html#skforecast.model_selection._search.grid_search_sarimax
[random_search_sarimax]: https://skforecast.org/0.19.0/api/model_selection.html#skforecast.model_selection._search.random_search_sarimax
