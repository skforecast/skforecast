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

from ._arar import Arar
from ._arima import Arima
from ._sarimax import Sarimax
from ._ets import Ets
from .autocorrelation import acf, pacf, calculate_lag_autocorrelation

# Import submodules for namespace access
from . import arima
from . import seasonal
from . import transformations
from . import autocorrelation

__all__ = [
    # Model classes
    'Arar',
    'Arima',
    'Sarimax',
    'Ets',
    # Autocorrelation functions
    'acf',
    'pacf',
    'calculate_lag_autocorrelation',

    # Submodules
    'arima',
    'seasonal',
    'transformations',
    'autocorrelation',
]
