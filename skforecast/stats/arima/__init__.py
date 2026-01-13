"""
ARIMA module for time series forecasting.

This module provides ARIMA model fitting, prediction, and automatic
model selection utilities.
"""

from ._arima_base import (
    arima,
    predict_arima,
    fitted_values,
    residuals_arima,
    ar_check,
    ma_invert,
    diff,
    na_omit,
    initialize_arima_state,
    compute_arima_likelihood,
    kalman_forecast,
)

from ._auto_arima import (
    auto_arima,
    search_arima,
    fit_custom_arima,
    analyze_series,
    arima_rjh,
    forecast_arima,
)

# Re-export from seasonal module for backward compatibility
from ..seasonal import (
    ndiffs,
    nsdiffs,
    is_constant,
    seas_heuristic,
)

# Re-export from transformations module for backward compatibility
from ..transformations import (
    box_cox,
    inv_box_cox,
    box_cox_lambda,
    guerrero,
    bcloglik,
    box_cox_biasadj,
)

__all__ = [
    # Core ARIMA functions
    'arima',
    'predict_arima',
    'fitted_values',
    'residuals_arima',
    'arima_rjh',

    # Auto ARIMA functions
    'auto_arima',
    'search_arima',
    'fit_custom_arima',
    'forecast_arima',

    # Differencing utilities (from seasonal module)
    'ndiffs',
    'nsdiffs',
    'diff',

    # Seasonal strength (from seasonal module)
    'seas_heuristic',

    # Box-Cox functions (from transformations module)
    'box_cox',
    'inv_box_cox',
    'box_cox_lambda',
    'guerrero',
    'bcloglik',
    'box_cox_biasadj',

    # Other utilities
    'analyze_series',
    'is_constant',
    'ar_check',
    'ma_invert',
    'na_omit',

    # Advanced/internal
    'initialize_arima_state',
    'compute_arima_likelihood',
    'kalman_forecast',
]
