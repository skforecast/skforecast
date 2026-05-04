"""
Stats module for time series analysis and forecasting.

This module provides various statistical models and utilities including:
- ARIMA and Auto-ARIMA models
- Exponential Smoothing (ETS)
- SARIMAX
- ARAR
- Seasonal analysis utilities
- Data transformations (Box-Cox)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    # Model classes
    "Arar",
    "Arima",
    "Sarimax",
    "Ets",
    # Autocorrelation functions
    "acf",
    "pacf",
    "calculate_lag_autocorrelation",
    # Submodules
    "arima",
    "seasonal",
    "transformations",
]

_MODEL_IMPORTS = {
    "Arar": ("._arar", "Arar"),
    "Arima": ("._arima", "Arima"),
    "Sarimax": ("._sarimax", "Sarimax"),
    "Ets": ("._ets", "Ets"),
}

_AUTOCORRELATION_IMPORTS = {
    "acf",
    "pacf",
    "calculate_lag_autocorrelation",
}

_SUBMODULES = {
    "arima",
    "seasonal",
    "transformations",
}

_STATSMODELS_IMPORTS = {
    "Arima",
    "Sarimax",
    "Ets",
    "arima",
    "seasonal",
}


def __getattr__(name: str):
    try:
        if name in _MODEL_IMPORTS:
            module_name, attr_name = _MODEL_IMPORTS[name]
            value = getattr(import_module(module_name, __name__), attr_name)
        elif name in _AUTOCORRELATION_IMPORTS:
            value = getattr(import_module("._autocorrelation", __name__), name)
        elif name in _SUBMODULES:
            value = import_module(f".{name}", __name__)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    except ModuleNotFoundError as error:
        if error.name == "statsmodels" and name in _STATSMODELS_IMPORTS:
            from ..utils import check_optional_dependency
            check_optional_dependency(package_name="statsmodels")
        raise

    globals()[name] = value
    return value


if TYPE_CHECKING:
    from . import arima
    from . import seasonal
    from . import transformations
    from ._arar import Arar
    from ._arima import Arima
    from ._autocorrelation import acf, calculate_lag_autocorrelation, pacf
    from ._ets import Ets
    from ._sarimax import Sarimax
