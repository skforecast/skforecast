# Unit test __init__ method - Arima
# ==============================================================================
from ..._arima import Arima


def test_arima_init_default_params():
    """
    Test Arima initialization with default parameters.
    """
    model = Arima()
    
    assert model.order == (0, 0, 0)
    assert model.seasonal_order == (0, 0, 0)
    assert model.m == 1
    assert model.include_mean is True
    assert model.transform_pars is True
    assert model.method == "CSS-ML"
    assert model.n_cond is None
    assert model.SSinit == "Gardner1980"
    assert model.optim_method == "BFGS"
    assert model.optim_kwargs is None
    assert model.kappa == 1e6
    assert model.is_memory_reduced is False


def test_arima_init_with_explicit_params():
    """
    Test Arima initialization with explicit parameters.
    """
    model = Arima(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1),
        m=12,
        include_mean=False,
        transform_pars=False,
        method="ML",
        n_cond=10,
        SSinit="Rossignol2011",
        optim_method="L-BFGS-B",
        optim_kwargs={'maxiter': 100},
        kappa=1e5
    )
    
    assert model.order == (1, 1, 1)
    assert model.seasonal_order == (1, 1, 1)
    assert model.m == 12
    assert model.include_mean is False
    assert model.transform_pars is False
    assert model.method == "ML"
    assert model.n_cond == 10
    assert model.SSinit == "Rossignol2011"
    assert model.optim_method == "L-BFGS-B"
    assert model.optim_kwargs == {'maxiter': 100}
    assert model.kappa == 1e5


def test_arima_init_order_validation():
    """
    Test that initialization validates order parameter length.
    """
    import pytest
    
    # Invalid order length
    msg = "`order` must be a tuple of length 3, got length 2"
    with pytest.raises(ValueError, match=msg):
        Arima(order=(1, 1))
    
    msg = "`order` must be a tuple of length 3, got length 4"
    with pytest.raises(ValueError, match=msg):
        Arima(order=(1, 1, 1, 1))


def test_arima_init_seasonal_order_validation():
    """
    Test that initialization validates seasonal_order parameter length.
    """
    import pytest
    
    # Invalid seasonal_order length
    msg = "`seasonal_order` must be a tuple of length 3, got length 2"
    with pytest.raises(ValueError, match=msg):
        Arima(seasonal_order=(1, 1))
    
    msg = "`seasonal_order` must be a tuple of length 3, got length 1"
    with pytest.raises(ValueError, match=msg):
        Arima(seasonal_order=(1,))


def test_arima_init_all_attributes_before_fitting():
    """
    Test that all attributes have expected initial values before fitting.
    """
    model = Arima(order=(1, 0, 1))
    
    # Parameter attributes are set
    assert model.order == (1, 0, 1)
    assert model.seasonal_order == (0, 0, 0)
    assert model.m == 1
    assert model.is_memory_reduced is False


def test_arima_repr_non_seasonal():
    """
    Test __repr__ for non-seasonal ARIMA model.
    """
    model = Arima(order=(2, 1, 1))
    assert repr(model) == "Arima(2,1,1)"


def test_arima_repr_seasonal():
    """
    Test __repr__ for seasonal ARIMA model.
    """
    model = Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12)
    assert repr(model) == "Arima(1,1,1)(1,1,1)[12]"
    
    model2 = Arima(order=(2, 0, 2), seasonal_order=(2, 1, 0), m=4)
    assert repr(model2) == "Arima(2,0,2)(2,1,0)[4]"


def test_arima_repr_no_seasonal_component():
    """
    Test __repr__ when seasonal_order is (0, 0, 0).
    """
    model = Arima(order=(3, 2, 1), seasonal_order=(0, 0, 0), m=12)
    assert repr(model) == "Arima(3,2,1)"
