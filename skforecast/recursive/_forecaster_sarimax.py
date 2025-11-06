################################################################################
#                            ForecasterSarimax                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing_extensions import deprecated
from ..exceptions import runtime_deprecated
from ._forecaster_stats import ForecasterStats


# TODO: Remove in version 0.20.0
@runtime_deprecated(replacement="ForecasterStats", version="0.19.0", removal="0.20.0")
@deprecated("`ForecasterSarimax` is deprecated since version 0.19.0; use `ForecasterStats` instead. It will be removed in version 0.20.0.")
class ForecasterSarimax():
    """
    !!! warning "Deprecated"
        This class is deprecated since skforecast 0.19. Please use
        `skforecast.recursive.ForecasterStats` instead.
    
    """
    
    def __new__(
        self,
        regressor: object,
        transformer_y: object | None = None,
        transformer_exog: object | None = None,
        fit_kwargs: dict[str, object] | None = None,
        forecaster_id: str | int | None = None
    ) -> None:
        
        return ForecasterStats(
            regressor=regressor,
            transformer_y=transformer_y,
            transformer_exog=transformer_exog,
            fit_kwargs=fit_kwargs,
            forecaster_id=forecaster_id
        )
