"""
Seasonal analysis module for time series.

This module provides utilities for analyzing seasonal patterns in time series,
including seasonal strength measures and differencing utilities for achieving
stationarity.
"""

from ._seasonal_strength import (
    seas_heuristic,
    _seas_heuristic_jit,
    _moving_average_jit,
    _seasonal_component_jit,
)

from ._differencing import (
    ndiffs,
    nsdiffs,
    is_constant,
    diff,
    _is_constant_jit,
)

__all__ = [
    # Seasonal strength
    'seas_heuristic',

    # Differencing utilities
    'ndiffs',
    'nsdiffs',
    'is_constant',
    'diff',
]
