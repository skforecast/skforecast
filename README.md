<h1 align="left">
    <img src="https://github.com/skforecast/skforecast/blob/master/images/banner-landing-page-skforecast.png?raw=true#only-light" style= margin-top: 0px;>
</h1>


| | |
| --- | --- |
| Package | ![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/) [![Downloads](https://static.pepy.tech/badge/skforecast)](https://pepy.tech/project/skforecast) [![Downloads](https://static.pepy.tech/badge/skforecast/month)](https://pepy.tech/project/skforecast) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/skforecast/skforecast/graphs/commit-activity) [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) |
| Meta | [![License](https://img.shields.io/github/license/skforecast/skforecast)](https://github.com/skforecast/skforecast/blob/master/LICENSE) [![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787) |
| Testing | [![Build status](https://github.com/skforecast/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/skforecast/skforecast/actions/workflows/unit-tests.yml/badge.svg) [![codecov](https://codecov.io/gh/skforecast/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/skforecast/skforecast) |
|Donation | [![paypal](https://img.shields.io/static/v1?style=social&amp;label=Donate&amp;message=%E2%9D%A4&amp;logo=Paypal&amp;color&amp;link=%3curl%3e)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6) [![buymeacoffee](https://img.shields.io/badge/-Buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/skforecast) ![GitHub Sponsors](https://img.shields.io/github/sponsors/joaquinamatrodrigo?logo=github&label=Github%20sponsors&link=https%3A%2F%2Fgithub.com%2Fsponsors%2FJoaquinAmatRodrigo) |
|Community | [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/skforecast/)
|Affiliation | [![NumFOCUS Affiliated](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)


# Table of Contents

- :information_source: [About The Project](#about-the-project)
- :books: [Documentation](#documentation)
- :computer: [Installation & Dependencies](#installation--dependencies)
- :sparkles: [What is new in skforecast 0.14?](#what-is-new-in-skforecast-014)
- :crystal_ball: [Forecasters](#forecasters)
- :mortar_board: [Examples and tutorials](#examples-and-tutorials)
- :handshake: [How to contribute](#how-to-contribute)
- :memo: [Citation](#citation)
- :money_with_wings: [Donating](#donating)
- :scroll: [License](#license)


# About The Project

**Skforecast** is a Python library for time series forecasting using machine learning models. It works with any regressor compatible with the scikit-learn API, including popular options like LightGBM, XGBoost, CatBoost, Keras, and many others.

### Why use skforecast?

Skforecast simplifies time series forecasting with machine learning by providing:

- :jigsaw: **Seamless integration** with any scikit-learn compatible regressor (e.g., LightGBM, XGBoost, CatBoost, etc.).
- :repeat: **Flexible workflows** that allow for both single and multi-series forecasting.
- :hammer_and_wrench: **Comprehensive tools** for feature engineering, model selection, hyperparameter tuning, and more.
- :building_construction: **Production-ready models** with interpretability and validation methods for backtesting and realistic performance evaluation.

Whether you're building quick prototypes or deploying models in production, skforecast ensures a fast, reliable, and scalable experience.

### Get Involved

We value your input! Here are a few ways you can participate:

- **Report bugs** and suggest new features on our [GitHub Issues page](https://github.com/skforecast/skforecast/issues).
- **Contribute** to the project by [submitting code](https://github.com/skforecast/skforecast/blob/master/CONTRIBUTING.md), adding new features, or improving the documentation.
- **Share your feedback** on LinkedIn to help spread the word about skforecast!

Together, we can make time series forecasting accessible to everyone.


# Documentation

For detailed information on how to use and leverage the full potential of **skforecast** please refer to the comprehensive documentation available at:

**https://skforecast.org** :books:

| Documentation                           |     |
|:----------------------------------------|:----|
| :book: [Introduction to forecasting]    | Basics of forecasting concepts and methodologies |
| :rocket: [Quick start]                  | Get started quickly with skforecast |
| :hammer_and_wrench: [User guides]       | Detailed guides on skforecast features and functionalities |
| :mortar_board: [Examples and tutorials] | Learn through practical examples and tutorials to master skforecast |
| :question: [FAQ and tips]               | Find answers and tips about forecasting |
| :books: [API Reference]                 | Comprehensive reference for skforecast functions and classes |
| :black_nib: [Authors]                   | Meet the authors and contributors of skforecast |

[Introduction to forecasting]: https://skforecast.org/latest/introduction-forecasting/introduction-forecasting
[Quick start]: https://skforecast.org/latest/quick-start/quick-start-skforecast
[User guides]: https://skforecast.org/latest/user_guides/user-guides
[Examples and tutorials]: https://skforecast.org/latest/examples/examples
[FAQ and tips]: https://skforecast.org/latest/faq/faq
[API Reference]: https://skforecast.org/latest/api/forecasterautoreg
[Authors]: https://skforecast.org/latest/authors/authors


# Installation & Dependencies

To install the basic version of `skforecast` with core dependencies, run the following:

```bash
pip install skforecast
```

For more installation options, including dependencies and additional features, check out our [Installation Guide](https://skforecast.org/latest/quick-start/how-to-install.html).


# What is new in skforecast 0.14?

All significant changes to this project are documented in the release file.

- For updates to the **latest stable version**, see the [release notes](https://skforecast.org/latest/releases/releases.html).

- For updates on the **version in development** (unstable), see the [release notes](https://skforecast.org/dev/releases/releases.html).


# Forecasters

A **Forecaster** object in the skforecast library is a comprehensive container that provides essential functionality and methods for training a forecasting model and generating predictions for future points in time.

The **skforecast** library offers a variety of forecaster types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Time series differentiation | Exogenous features | Window features |
|:-----------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:---------------------------:|:------------------:|:---------------:|
|[ForecasterRecursive]|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterDirect]|:heavy_check_mark:|||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterRecursiveMultiSeries]||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterDirectMultiVariate]||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|
|[ForecasterRNN]||:heavy_check_mark:||:heavy_check_mark:|||||
|[ForecasterSarimax]|:heavy_check_mark:||:heavy_check_mark:||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||

[ForecasterRecursive]: https://skforecast.org/latest/user_guides/autoregresive-forecaster.html
[ForecasterDirect]: https://skforecast.org/latest/user_guides/direct-multi-step-forecasting.html
[ForecasterRecursiveMultiSeries]: https://skforecast.org/latest/user_guides/independent-multi-time-series-forecasting.html
[ForecasterDirectMultiVariate]: https://skforecast.org/latest/user_guides/dependent-multi-series-multivariate-forecasting.html
[ForecasterRNN]: https://skforecast.org/latest/user_guides/forecasting-with-deep-learning-rnn-lstm
[ForecasterSarimax]: https://skforecast.org/latest/user_guides/forecasting-sarimax-arima.html


# Examples and tutorials

Explore our extensive list of examples and tutorials (English and Spanish) to get you started with skforecast. You can find them [here](https://skforecast.org/latest/examples/examples_english).


# How to contribute

Primarily, skforecast development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/skforecast/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](https://skforecast.org/latest/examples/examples_english).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to skforecast, see our [Contribution Guide](https://github.com/skforecast/skforecast/blob/master/CONTRIBUTING.md).

Visit our [authors section](https://skforecast.org/latest/authors/authors) to meet all the contributors to skforecast.


# Citation

If you use skforecast for a scientific publication, we would appreciate citations to the published software.

**Zenodo**

```
Amat Rodrigo, Joaquin, & Escobar Ortiz, Javier. (2024). skforecast (v0.13.0). Zenodo. https://doi.org/10.5281/zenodo.8382788
```

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. (2024). skforecast (Version 0.13.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
title = {skforecast},
version = {0.13.0},
month = {8},
year = {2024},
license = {BSD-3-Clause},
url = {https://skforecast.org/},
doi = {10.5281/zenodo.8382788}
}
```

View the [citation file](https://github.com/skforecast/skforecast/blob/master/CITATION.cff).


# Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks!

<a href="https://www.buymeacoffee.com/skforecast"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=skforecast&button_colour=f79939&font_colour=000000&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>
<br>


[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


# License

[BSD-3-Clause License](https://github.com/skforecast/skforecast/blob/master/LICENSE)



<!-- Links to API Reference -->
<!-- Forecasters -->
