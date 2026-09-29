<h1 align="left">
    <img src="https://github.com/skforecast/skforecast/blob/main/images/banner-landing-page-skforecast.png?raw=true" alt="skforecast">
</h1>


| | |
| --- | --- |
| Package | ![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/) [![Conda](https://img.shields.io/conda/v/conda-forge/skforecast?logo=Anaconda)](https://anaconda.org/conda-forge/skforecast) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/skforecast?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/skforecast) |
| Meta | [![License](https://img.shields.io/github/license/skforecast/skforecast)](https://github.com/skforecast/skforecast/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787) [![NumFOCUS Affiliated](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects) [![GC.OS Affiliated](https://img.shields.io/badge/GC.OS-Affiliated%20Project-orange.svg?style=flat&colorA=0eac92&colorB=2077b4)](https://gc-os-ai.github.io/) |
| Testing | [![Build status](https://github.com/skforecast/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/skforecast/skforecast/actions/workflows/unit-tests.yml) [![codecov](https://codecov.io/gh/skforecast/skforecast/branch/main/graph/badge.svg)](https://codecov.io/gh/skforecast/skforecast) |
| Community | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.gg/3V52qpNkuj) [![!linkedin](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/skforecast/) [![Forecasting Python](https://img.shields.io/static/v1?logo=readme&logoColor=white&label=Blog&labelColor=%23333333&message=Forecasting%20Python&color=%23ffab40)](https://cienciadedatos.net/en/forecasting-python) |


## About skforecast

**Time series forecasting, from prototype to production.** Skforecast is a Python library for time series forecasting using scikit-learn compatible models, statistical methods, and foundation models. It works with any estimator compatible with the scikit-learn API, such as LightGBM, XGBoost, CatBoost or Keras.

<p align="center">
  <img src="https://github.com/skforecast/skforecast/blob/main/images/skforecast-backtesting-comparison.png?raw=true" alt="Backtesting of LightGBM, Chronos-2 and ARIMA with skforecast on daily electricity demand" width="100%">
</p>

<sub>Backtesting of LightGBM, Chronos-2 and ARIMA on daily electricity demand, with temperature and holidays as exogenous variables. Real skforecast outputs, from the animation on [skforecast.org](https://skforecast.org).</sub>


## Installation

```bash
pip install skforecast
```

Also available on conda-forge (`conda install -c conda-forge skforecast`). Optional dependencies for statistical models, plotting and deep learning are listed in the [installation guide](https://skforecast.org/latest/quick-start/how-to-install.html).


## Quick example

```python
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.datasets import load_demo_dataset

# Download demo dataset
y = load_demo_dataset()

# Create and fit the forecaster using the last 15 observations as features
forecaster = ForecasterRecursive(
                 estimator = LGBMRegressor(random_state=123, verbose=-1),
                 lags      = 15
             )
forecaster.fit(y=y)

# Predict the next 12 months
predictions = forecaster.predict(steps=12)
predictions.head()
# 2008-07-01    0.976355
# 2008-08-01    1.038296
# 2008-09-01    1.108676
# 2008-10-01    1.163778
# 2008-11-01    1.169646
# Freq: MS, Name: pred, dtype: float64
```

<details>
<summary><b>Same workflow with a foundation model (Chronos-2, zero-shot)</b></summary>

```python
# pip install chronos-forecasting
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.datasets import load_demo_dataset

# Download demo dataset
y = load_demo_dataset()

# Create the forecaster: fit only stores the context, there is no training
forecaster = ForecasterFoundation(
                 estimator = FoundationModel("autogluon/chronos-2-small")
             )
forecaster.fit(series=y)

# Predict the next 12 months
predictions = forecaster.predict(steps=12)
predictions.head()
#            level      pred
# 2008-07-01     y  1.002536
# 2008-08-01     y  1.030961
# 2008-09-01     y  1.083664
# 2008-10-01     y  1.184066
# 2008-11-01     y  1.170530
```

</details>


## Features

### One API for every kind of model

- **Machine learning**: any scikit-learn compatible regressor, such as LightGBM, XGBoost or CatBoost, with lags, rolling and calendar features. Recursive or direct strategies, for one series or thousands.
- **Foundation models**: zero-shot forecasting with pre-trained models such as Chronos-2, TimesFM, Moirai-2 or TabPFN-TS, without training them.
- **Statistical models**: ARIMA, SARIMAX, ETS and ARAR, with automatic model selection.
- **Deep learning**: recurrent neural networks (RNN, LSTM) built with Keras.

Train, predict, tune and backtest the same way, whatever the model.

### Built for production

- **Backtesting** that reproduces how the model will be used: refits, gaps, fold strides, fixed or expanding windows.
- **Probabilistic forecasting** with bootstrapping, conformal prediction and quantiles, evaluated with CRPS and coverage.
- **Hyperparameter tuning** with grid, random and Bayesian search (Optuna), including the number of lags.
- **Feature engineering**: rolling statistics, calendar features, exogenous and categorical variables, and differentiation.
- **Global models** that forecast many series with one model, even with different lengths, exogenous variables and missing values ([global forecasting guide](https://skforecast.org/latest/user_guides/global-forecasting-overview.html)).
- **Explainability and monitoring**: feature importances and SHAP values, feature selection and drift detection.


## Forecasters

All forecasters share the same API for training, prediction, backtesting and probabilistic forecasting.

| Forecaster | Estimator | Series | Strategy | Exog | Window features | Differentiation |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
|[ForecasterRecursive]| scikit-learn regressor | single | recursive | ✓ | ✓ | ✓ |
|[ForecasterDirect]| scikit-learn regressor | single | direct | ✓ | ✓ | ✓ |
|[ForecasterRecursiveMultiSeries]| scikit-learn regressor | multiple | recursive | ✓ | ✓ | ✓ |
|[ForecasterDirectMultiVariate]| scikit-learn regressor | multiple | direct | ✓ | ✓ | ✓ |
|[ForecasterFoundation]| pre-trained, zero-shot | single or multiple | multi-output | ✓ | | |
|[ForecasterStats]| Arima, Sarimax, Ets, Arar | single | recursive | ✓ | | |
|[ForecasterRnn]| Keras model (RNN/LSTM) | single or multiple | multi-output | ✓ | | |
|[ForecasterRecursiveClassifier]| scikit-learn classifier | single | recursive | ✓ | ✓ | |
|[ForecasterEquivalentDate]| Rule-based (baseline) | single | recursive | | | |

[ForecasterRecursive]: https://skforecast.org/latest/user_guides/autoregressive-forecaster.html
[ForecasterDirect]: https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html
[ForecasterRecursiveMultiSeries]: https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html
[ForecasterDirectMultiVariate]: https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html
[ForecasterFoundation]: https://skforecast.org/latest/user_guides/foundation-forecasting-models.html
[ForecasterStats]: https://skforecast.org/latest/user_guides/forecasting-sarimax-arima.html
[ForecasterRnn]: https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm.html
[ForecasterRecursiveClassifier]: https://skforecast.org/latest/user_guides/autoregressive-classification-forecasting.html
[ForecasterEquivalentDate]: https://skforecast.org/latest/user_guides/forecasting-baseline.html


## Documentation

The full documentation is available at **https://skforecast.org**.

| Documentation                           |     |
|:----------------------------------------|:----|
| :book: [Introduction to forecasting]    | Forecasting concepts and how to choose a forecaster |
| :rocket: [Quick start]                  | Install skforecast and train your first model |
| :hammer_and_wrench: [User guides]       | Detailed guides on every feature |
| :mortar_board: [Examples and tutorials] | Practical examples, in English, Spanish and Chinese |
| :question: [FAQ and tips]               | Answers and tips about forecasting |
| :books: [API Reference]                 | Reference for all functions and classes |
| :memo: [Releases]                       | What changed in each version |

[Introduction to forecasting]: https://skforecast.org/latest/introduction-forecasting/introduction-forecasting.html
[Quick start]: https://skforecast.org/latest/quick-start/quick-start-skforecast.html
[User guides]: https://skforecast.org/latest/user_guides/table-of-contents.html
[Examples and tutorials]: https://skforecast.org/latest/examples/examples_english.html
[FAQ and tips]: https://skforecast.org/latest/faq/table-of-contents.html
[API Reference]: https://skforecast.org/latest/api/forecasterrecursive.html
[Releases]: https://skforecast.org/latest/releases/releases.html


## AI and tools

- **AI-assisted coding**: skforecast publishes machine-readable context files so AI assistants (ChatGPT, Claude, Copilot and others) generate accurate code. Paste `https://skforecast.org/latest/llms-full.txt` into any LLM, or read [AI-assisted forecasting](https://skforecast.org/latest/quick-start/ai-assisted-forecasting.html).
- **[skforecast-ai](https://ai.skforecast.org/)**: an AI forecasting assistant that pairs a deterministic engine, powered by skforecast, with an LLM reasoning layer. Install it with `pip install skforecast-ai` ([GitHub](https://github.com/skforecast/skforecast-ai)).
- **[Skforecast Studio](https://studio.skforecast.org/)**: a no-code application to build forecasting models visually, which generates production-ready Python code.


## Contributing

Bug reports, feature requests, code, tests, documentation and examples are all welcome. Open an issue on [GitHub Issues](https://github.com/skforecast/skforecast/issues) or read the [Contribution Guide](https://github.com/skforecast/skforecast/blob/main/CONTRIBUTING.md) to get started.

skforecast is created and maintained by [Joaquín Amat Rodrigo](https://github.com/JoaquinAmatRodrigo) and [Javier Escobar Ortiz](https://github.com/JavierEscobarOrtiz), together with everyone who has contributed to it ([about the project](https://skforecast.org/latest/more/about-skforecast.html)).

<a href="https://github.com/skforecast/skforecast/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=skforecast/skforecast" alt="skforecast contributors">
</a>


## Citation

If you use skforecast in a scientific publication, please cite the version you used: each version has its own DOI and ready-made citations on [Zenodo](https://doi.org/10.5281/zenodo.8382787). To cite skforecast in general, use the DOI that always resolves to the latest release:

```
Amat Rodrigo, J., & Escobar Ortiz, J. skforecast [Computer software]. https://doi.org/10.5281/zenodo.8382787
```

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@software{skforecast,
  author  = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
  title   = {skforecast},
  license = {BSD-3-Clause},
  url     = {https://skforecast.org/},
  doi     = {10.5281/zenodo.8382787}
}
```

</details>

The citation metadata is also in [CITATION.cff](https://github.com/skforecast/skforecast/blob/main/CITATION.cff) (GitHub's "Cite this repository" button). skforecast is used in 70+ scientific publications: [see them on Google Scholar](https://scholar.google.com/scholar?q=%22skforecast%22).


## Sponsorship and funding

skforecast is free, open-source software supported by the [Sovereign Tech Fund](https://www.sovereign.tech/tech/skforecast) and by the organizations that sponsor it. If your company relies on skforecast, see **[Sponsorship and funding](https://skforecast.org/latest/more/funding.html)** for sponsorship tiers, support agreements, feature sponsorship and training.

Individuals can support the project through [Open Collective](https://opencollective.com/skforecast), GitHub Sponsors ([Joaquín Amat Rodrigo](https://github.com/sponsors/JoaquinAmatRodrigo), [Javier Escobar Ortiz](https://github.com/sponsors/JavierEscobarOrtiz)), [Buy Me a Coffee](https://www.buymeacoffee.com/skforecast) or [PayPal](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6).


## License

**Skforecast software**: [BSD-3-Clause License](https://github.com/skforecast/skforecast/blob/main/LICENSE)

**Skforecast documentation**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Trademark**: The trademark skforecast is registered with the European Union Intellectual Property Office (EUIPO) under the application number 019109684. Unauthorized use of this trademark, its logo, or any associated visual identity elements is strictly prohibited without the express consent of the owner.
