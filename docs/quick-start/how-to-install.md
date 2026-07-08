# Installation Guide

This guide will help you install `skforecast`, a powerful library for time series forecasting in Python. The default installation of `skforecast` includes only the essential dependencies required for basic functionality. Additional optional dependencies can be installed for extended features.

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue) [![PyPI](https://img.shields.io/pypi/v/skforecast)](https://pypi.org/project/skforecast/) [![Conda](https://img.shields.io/conda/v/conda-forge/skforecast?logo=Anaconda)](https://anaconda.org/conda-forge/skforecast)


## **Basic installation**

**Skforecast** requires Python 3.10 or higher. It is available on PyPI and can be installed using `pip`. You can also install it via conda from the conda-forge channel.

To install the basic version of `skforecast` with its core dependencies, run:

```bash
pip install skforecast
```

Specific version:

```bash
pip install skforecast==0.23.0
```

Latest (unstable):

```bash
pip install git+https://github.com/skforecast/skforecast@master
```

The following dependencies are installed with the default installation:

+ numpy>=1.22
+ pandas>=2.1
+ tqdm>=4.66
+ scikit-learn>=1.4
+ scipy>=1.12
+ optuna>=4.0
+ joblib>=1.3
+ numba>=0.59
+ rich>=13.9


## **Optional dependencies**

To install the full version with all optional dependencies:

```bash
pip install skforecast[full]
```

For specific use cases, you can install these dependencies as needed:

### Stats

```bash
pip install skforecast[stats]
```

+ statsmodels>=0.13, <0.15


### Plotting

```bash
pip install skforecast[plotting]
```

+ matplotlib>=3.7, <3.12
+ statsmodels>=0.13, <0.15


### Deep Learning

```bash
pip install skforecast[deeplearning]
```

+ keras>=3.0, <4.0
+ matplotlib>=3.7, <3.12
