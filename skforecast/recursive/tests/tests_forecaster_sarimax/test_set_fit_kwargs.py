# Unit test set_fit_kwargs ForecasterSarimax
# ==============================================================================
import re
import pytest
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.exceptions import IgnoredArgumentWarning


def test_set_fit_kwargs_skforecast():
    """
    Test set_fit_kwargs method using skforecast.
    """
    forecaster = ForecasterSarimax(
                     regressor  = Sarimax(order=(1, 0, 1))
                 )
    new_fit_kwargs = {'warning': 1}
    
    warn_msg = re.escape(
        ("When using the skforecast Sarimax model, the fit kwargs should "
         "be passed using the model parameter `sm_fit_kwargs`.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        forecaster.set_fit_kwargs(new_fit_kwargs)
    
    assert forecaster.fit_kwargs == {}
