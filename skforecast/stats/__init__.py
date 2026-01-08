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

# Import submodules for namespace access
from . import arima
from . import seasonal
from . import transformations

__all__ = [
    # Model classes
    'Arar',
    'Arima',
    'Sarimax',
    'Ets',

    # Submodules
    'arima',
    'seasonal',
    'transformations',
]
