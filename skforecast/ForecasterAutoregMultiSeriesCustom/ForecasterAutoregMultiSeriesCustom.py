################################################################################
#                   ForecasterAutoregMultiSeriesCustom                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Any
import warnings


# TODO: Update warning with user guide link
class ForecasterAutoregMultiSeriesCustom():
    """
    This class is deprecated since skforecast 0.14. To create window features,
    use the class `ForecasterRecursiveMultiSeries` and the argument `window_features`. For
    more information, see the user guide:
    https://skforecast.org/latest/user_guides/window-features-and-custom-features
    """

    def __init__(
        self,
        regressor: Any = None,
        fun_predictors: Any = None, 
        window_size: Any = None,
        name_predictors: Any = None,
        encoding: Any = None,
        transformer_series: Any = None,
        transformer_exog: Any = None,
        weight_func: Any = None,
        series_weights: Any = None,
        differentiation: Any = None,
        dropna_from_series: Any = None,
        fit_kwargs: Any = None,
        forecaster_id: Any = None
    ) -> None:
        
        warnings.warn(
            "This class is deprecated since skforecast 0.14. To create window "
            "features, use the class `ForecasterRecursiveMultiSeries` and the argument "	
            "`window_features`. For more information, see the user guide: "
            "https://skforecast.org/latest/user_guides/window-features-and-custom-features"
        )
