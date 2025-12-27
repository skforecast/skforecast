# Unit test set_fit_kwargs ForecasterStats
# ==============================================================================
import re
import pytest
from skforecast.stats import Sarimax
from skforecast.recursive import ForecasterStats
from skforecast.exceptions import IgnoredArgumentWarning


def test_set_fit_kwargs_skforecast():
    """
    Test set_fit_kwargs method using skforecast.
    fit_kwargs is ignored and set to None for ForecasterStats.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(1, 0, 1))
                 )
    new_fit_kwargs = {'warning': 1}
    
    warn_msg = re.escape(
        "When using the skforecast Sarimax model, the fit kwargs should "
        "be passed using the model parameter `sm_fit_kwargs`."
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        forecaster.set_fit_kwargs(new_fit_kwargs)
    
    # fit_kwargs is ignored in ForecasterStats, remains None
    assert forecaster.fit_kwargs is None
