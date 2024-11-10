<script src="https://kit.fontawesome.com/d20edc211b.js" crossorigin="anonymous"></script>

<div style="margin-bottom: 20px;">
    <img src="img/banner-landing-page-skforecast.png#only-light" align="left" style="margin-bottom: 30px; margin-top: 0px;">
    <img src="img/banner-landing-page-dark-mode-skforecast-no-background.png#only-dark" align="left" style="margin-bottom: 30px; margin-top: 0px;">
</div>

<div style="clear: both;"></div>

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/)
[![codecov](https://codecov.io/gh/skforecast/skforecast/branch/master/graph/badge.svg)](https://codecov.io/gh/skforecast/skforecast)
[![Build status](https://github.com/skforecast/skforecast/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/skforecast/skforecast/actions/workflows/unit-tests.yml/badge.svg)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/skforecast/skforecast/graphs/commit-activity)
[![Downloads](https://static.pepy.tech/badge/skforecast)](https://pepy.tech/project/skforecast)
[![Downloads](https://static.pepy.tech/badge/skforecast/month)](https://pepy.tech/project/skforecast)
[![License](https://img.shields.io/github/license/skforecast/skforecast)](https://github.com/skforecast/skforecast/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/337705968.svg)](https://zenodo.org/doi/10.5281/zenodo.8382787)
[![paypal](https://img.shields.io/static/v1?style=social&amp;label=Donate&amp;message=%E2%9D%A4&amp;logo=Paypal&amp;color&amp;link=%3curl%3e)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)
[![buymeacoffee](https://img.shields.io/badge/-Buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/skforecast)
![GitHub Sponsors](https://img.shields.io/github/sponsors/joaquinamatrodrigo?logo=github&label=Github%20sponsors&link=https%3A%2F%2Fgithub.com%2Fsponsors%2FJoaquinAmatRodrigo)
[![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/skforecast/)
[![NumFOCUS Affiliated](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)


## About The Project

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


## Installation & Dependencies

To install the basic version of `skforecast` with core dependencies, run the following:

```bash
pip install skforecast
```

For more installation options, including dependencies and additional features, check out our [Installation Guide](../quick-start/how-to-install.html).


## Forecasters

A **Forecaster** object in the skforecast library is a comprehensive container that provides essential functionality and methods for training a forecasting model and generating predictions for future points in time.

The **skforecast** library offers a variety of forecaster types, each tailored to specific requirements such as single or multiple time series, direct or recursive strategies, or custom predictors. Regardless of the specific forecaster type, all instances share the same API.

| Forecaster                      | Single series | Multiple series | Recursive strategy | Direct strategy | Probabilistic prediction | Time series differentiation | Exogenous features | Window features |
|:--------------------------------|:-------------:|:---------------:|:------------------:|:---------------:|:------------------------:|:---------------------------:|:------------------:|:---------------:|
|[ForecasterRecursive]            |✔️||✔️||✔️|✔️|✔️|✔️|
|[ForecasterDirect]               |✔️|||✔️|✔️||✔️|✔️|
|[ForecasterRecursiveMultiSeries] ||✔️|✔️||✔️|✔️|✔️|✔️|
|[ForecasterDirectMultiVariate]   ||✔️||✔️|✔️||✔️|✔️|
|[ForecasterRNN]                  ||✔️||✔️|||||
|[ForecasterSarimax]              |✔️||✔️||✔️|✔️|✔️||

[ForecasterRecursive]: ../user_guides/autoregresive-forecaster.html
[ForecasterDirect]: ../user_guides/direct-multi-step-forecasting.html
[ForecasterRecursiveMultiSeries]: ../user_guides/independent-multi-time-series-forecasting.html
[ForecasterDirectMultiVariate]: ../user_guides/dependent-multi-series-multivariate-forecasting.html
[ForecasterRNN]: ../user_guides/forecasting-with-deep-learning-rnn-lstm.html
[ForecasterSarimax]: ../user_guides/forecasting-sarimax-arima.html


## Features

Skforecast provides a set of key features designed to make time series forecasting with machine learning easy and efficient. For a detailed overview, see the [User Guides](../user_guides/table-of-contents.html).


## Examples and tutorials

Explore our extensive list of examples and tutorials (English and Spanish) to get you started with skforecast. You can find them [here](../examples/examples_english.html).


## How to contribute

Primarily, skforecast development consists of adding and creating new *Forecasters*, new validation strategies, or improving the performance of the current code. However, there are many other ways to contribute:

- Submit a bug report or feature request on [GitHub Issues](https://github.com/skforecast/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](../examples/examples_english).
- Write [unit or integration tests](https://docs.pytest.org/en/latest/) for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

For more information on how to contribute to skforecast, see our [Contribution Guide](https://github.com/skforecast/skforecast/blob/master/CONTRIBUTING.md).

Visit our [authors section](../authors/authors.html) to meet all the contributors to skforecast.


## Citation

If you use skforecast for a scientific publication, we would appreciate citations to the published software.

**Zenodo**

```
Amat Rodrigo, Joaquin, & Escobar Ortiz, Javier. (2024). skforecast (v0.14.0). Zenodo. https://doi.org/10.5281/zenodo.8382788
```

**APA**:
```
Amat Rodrigo, J., & Escobar Ortiz, J. (2024). skforecast (Version 0.14.0) [Computer software]. https://doi.org/10.5281/zenodo.8382788
```

**BibTeX**:
```
@software{skforecast,
author = {Amat Rodrigo, Joaquin and Escobar Ortiz, Javier},
title = {skforecast},
version = {0.14.0},
month = {11},
year = {2024},
license = {BSD-3-Clause},
url = {https://skforecast.org/},
doi = {10.5281/zenodo.8382788}
}
```

### Publications citing skforecast

<ul>

<li><p style="color:#808080; font-size:0.95em;">
Sanan, O., Sperling, J., Greene, D., & Greer, R. (2024, April). Forecasting Weather and Energy Demand for Optimization of Renewable Energy and Energy Storage Systems for Water Desalination. In 2024 IEEE Conference on Technologies for Sustainability (SusTech) (pp. 175-182). IEEE. <a href="https://doi.org/10.1109/SusTech60925.2024.10553570">https://doi.org/10.1109/SusTech60925.2024.10553570</a>
</p></li>

<li><p style="color:#808080; font-size:0.95em;">
Bojer, A. K., Biru, B. H., Al-Quraishi, A. M. F., Debelee, T. G., Negera, W. G., Woldesillasie, F. F., & Esubalew, S. Z. (2024). Machine learning and remote sensing based time series analysis for drought risk prediction in Borena Zone, Southwest Ethiopia. Journal of Arid Environments, 222, 105160. <a href="https://doi.org/10.1016/j.jaridenv.2024.105160">https://doi.org/10.1016/j.jaridenv.2024.105160</a>
</p></li>

<li><p style="color:#808080; font-size:0.95em;">
V. Negri, A. Mingotti, R. Tinarelli and L. Peretto, "Comparison Between the Machine Learning and the Statistical Approach to the Forecasting of Voltage, Current, and Frequency," 2023 IEEE 13th International Workshop on Applied Measurements for Power Systems (AMPS), Bern, Switzerland, 2023, pp. 01-06, doi: 10.1109/AMPS59207.2023.10297192. <a href="https://doi.org/10.1109/AMPS59207.2023.10297192">https://doi.org/10.1109/AMPS59207.2023.10297192</a>
</p></li>

<li><p style="color:#808080; font-size:0.95em;">
Marcillo Vera, F., Rosado, R., Zambrano, P., Velastegui, J., Morales, G., Lagla, L., & Herrera, A. (2024). Forecasting con Python, caso de estudio: visitas a las redes sociales en Ecuador con machine learning. CONECTIVIDAD, 5(2), 15-29.<a href=https://doi.org/10.37431/conectividad.v5i2.126 a>
</p></li>

<li><p style="color:#808080; font-size:0.95em;">OUKHOUYA, H., KADIRI, H., EL HIMDI, K., & GUERBAZ, R. (2023). Forecasting International Stock Market Trends: XGBoost, LSTM, LSTM-XGBoost, and Backtesting XGBoost Models. Statistics, Optimization & Information Computing, 12(1), 200-209. <a href="https://doi.org/10.19139/soic-2310-5070-1822">https://doi.org/10.19139/soic-2310-5070-1822</a></p>
</li>

<li><p style="color:#808080; font-size:0.95em;">DUDZIK, S., & Kowalczyk, B. (2023). Prognozowanie produkcji energii fotowoltaicznej z wykorzystaniem platformy NEXO i VRM Portal. Przeglad Elektrotechniczny, 2023(11). doi:10.15199/48.2023.11.41 </p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Polo J, Martín-Chivelet N, Alonso-Abella M, Sanz-Saiz C, Cuenca J, de la Cruz M. Exploring the PV Power Forecasting at Building Façades Using Gradient Boosting Methods. Energies. 2023; 16(3):1495. <a href="https://doi.org/10.3390/en16031495">https://doi.org/10.3390/en16031495</a></p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Popławski T, Dudzik S, Szeląg P. Forecasting of Energy Balance in Prosumer Micro-Installations Using Machine Learning Models. Energies. 2023; 16(18):6726. <a href="https://doi.org/10.3390/en16186726">https://doi.org/10.3390/en16186726</a></p>
</li>
<li><p style="color:#808080; font-size:0.95em;">Harrou F, Sun Y, Taghezouit B, Dairi A. Artificial Intelligence Techniques for Solar Irradiance and PV Modeling and Forecasting. Energies. 2023; 16(18):6731. <a href="https://doi.org/10.3390/en16186731">https://doi.org/10.3390/en16186731</a></p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Amara-Ouali, Y., Goude, Y., Doumèche, N., Veyret, P., Thomas, A., Hebenstreit, D., ... &amp; Phe-Neau, T. (2023). Forecasting Electric Vehicle Charging Station Occupancy: Smarter Mobility Data Challenge. arXiv preprint arXiv:2306.06142.</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Emami, P., Sahu, A., &amp; Graf, P. (2023). BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting. arXiv preprint arXiv:2307.00142.</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Dang, HA., Dao, VD. (2023). Building Power Demand Forecasting Using Machine Learning: Application for an Office Building in Danang. In: Nguyen, D.C., Vu, N.P., Long, B.T., Puta, H., Sattler, KU. (eds) Advances in Engineering Research and Application. ICERA 2022. Lecture Notes in Networks and Systems, vol 602. Springer, Cham. <a href="https://doi.org/10.1007/978-3-031-22200-9_32">https://doi.org/10.1007/978-3-031-22200-9_32</a></p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Morate del Moral, Iván (2023). Predición de llamadas realizadas a un Call Center. Proyecto Fin de Carrera / Trabajo Fin de Grado, E.T.S.I. de Sistemas Informáticos (UPM), Madrid.</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Lopez Vega, A., &amp; Villanueva Vargas, R. A. (2022). Sistema para la automatización de procesos hospitalarios de control para pacientes para COVID-19 usando machine learning para el Centro de Salud San Fernando.</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">García Álvarez, J. D. (2022). Modelo predictivo de rentabilidad de criptomonedas para un futuro cercano.</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Chilet Vera, Á. (2023). Elaboración de un algoritmo predictivo para la reposición de hipoclorito en los depósitos mediante técnicas de Machine Learning (Doctoral dissertation, Universitat Politècnica de València).</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Bustinza Barrial, A. A., Bautista Abanto, A. M., Alva Alfaro, D. A., Villena Sotomayor, G. M., &amp; Trujillo Sabrera, J. M. (2022). Predicción de los valores de la demanda máxima de energía eléctrica empleando técnicas de machine learning para la empresa Nexa Resources–Cajamarquilla.</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Morgado, K. Desarrollo de una técnica de gestión de activos para transformadores de distribución basada en sistema de monitoreo (Doctoral dissertation, Universidad Nacional de Colombia).</p>
</li>

<li><p style="color:#808080; font-size:0.95em;">Zafeiriou A., Chantzis G., Jonkaitis T., Fokaides P., Papadopoulos A., 2023, Smart Energy Strategy - A Comparative Study of Energy Consumption Forecasting Machine Learning Models, Chemical Engineering Transactions, 103, 691-696.</p>
</li>

</ul>


## Donating

If you found skforecast useful, you can support us with a donation. Your contribution will help to continue developing and improving this project. Many thanks! :hugging_face: :heart_eyes:

<a href="https://www.buymeacoffee.com/skforecast"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=skforecast&button_colour=f79939&font_colour=000000&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>
<br>

[![paypal](https://www.paypalobjects.com/en_US/ES/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=D2JZSWRLTZDL6)


## License

[BSD-3-Clause License](https://github.com/skforecast/skforecast/blob/master/LICENSE)
