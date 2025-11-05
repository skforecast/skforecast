################################################################################
#                                 Sarimax                                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import warnings
import inspect
from typing_extensions import deprecated
import numpy as np
from ..exceptions import runtime_deprecated
from ..stats import Sarimax as SarimaxNew


# TODO: Remove in version 0.20.0
@runtime_deprecated(replacement="skforecast.stats.Sarimax", version="0.19.0", removal="0.20.0")
@deprecated("`skforecast.sarimax.Sarimax` is deprecated since version 0.19.0; use `skforecast.stats.Sarimax` instead. It will be removed in version 0.20.0.")
class Sarimax():
    """
    !!! warning "Deprecated"
        This class is deprecated since skforecast 0.19. Please use `skforecast.stats.Sarimax` instead.

    """

    def __new__(
        self,
        order: tuple = (1, 0, 0),
        seasonal_order: tuple = (0, 0, 0, 0),
        trend: str = None,
        measurement_error: bool = False,
        time_varying_regression: bool = False,
        mle_regression: bool = True,
        simple_differencing: bool = False,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        hamilton_representation: bool = False,
        concentrate_scale: bool = False,
        trend_offset: int = 1,
        use_exact_diffuse: bool = False,
        dates = None,
        freq = None,
        missing = 'none',
        validate_specification: bool = True,
        method: str = 'lbfgs',
        maxiter: int = 50,
        start_params: np.ndarray = None,
        disp: bool = False,
        sm_init_kwargs: dict[str, object] = {},
        sm_fit_kwargs: dict[str, object] = {},
        sm_predict_kwargs: dict[str, object] = {}
    ):

        return SarimaxNew(
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            measurement_error=measurement_error,
            time_varying_regression=time_varying_regression,
            mle_regression=mle_regression,
            simple_differencing=simple_differencing,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            hamilton_representation=hamilton_representation,
            concentrate_scale=concentrate_scale,
            trend_offset=trend_offset,
            use_exact_diffuse=use_exact_diffuse,
            dates=dates,
            freq=freq,
            missing=missing,
            validate_specification=validate_specification,
            method=method,
            maxiter=maxiter,
            start_params=start_params,
            disp=disp,
            sm_init_kwargs=sm_init_kwargs,
            sm_fit_kwargs=sm_fit_kwargs,
            sm_predict_kwargs=sm_predict_kwargs
        )
