# Unit test __init__ method - Arima
# ==============================================================================
import re
import pytest
from ..._arima import Arima


def test_arima_init_order_validation():
    """
    Test that initialization validates order parameter length.
    """
    msg = re.escape("`order` must be a tuple of length 3, got length 2")
    with pytest.raises(ValueError, match=msg):
        Arima(order=(1, 1))
    
    msg = re.escape("`order` must be a tuple of length 3, got length 4")
    with pytest.raises(ValueError, match=msg):
        Arima(order=(1, 1, 1, 1))


def test_arima_init_seasonal_order_and_m_validation():
    """
    Test that initialization validates seasonal_order parameter length.
    """
    msg = re.escape("`seasonal_order` must be a tuple of length 3, got length 2")
    with pytest.raises(ValueError, match=msg):
        Arima(order=(0, 0, 0), seasonal_order=(1, 1))
    
    msg = re.escape("`seasonal_order` must be a tuple of length 3, got length 1")
    with pytest.raises(ValueError, match=msg):
        Arima(order=(0, 0, 0), seasonal_order=(1,))
    
    msg = re.escape("`m` must be a positive integer (seasonal period).")
    with pytest.raises(ValueError, match=msg):
        Arima(order=(0, 0, 0), seasonal_order=(1, 1, 1), m=0)
    
    msg = re.escape("`m` must be a positive integer (seasonal period).")
    with pytest.raises(ValueError, match=msg):
        Arima(order=(0, 0, 0), seasonal_order=(1, 1, 1), m='not_int')

@pytest.mark.parametrize(
    "order",
    [(3, 0, 2), (0, 1, 0), (0, 0, 0)],
    ids=lambda v: f"order: {v}"
)
def test_arima_init_ValueError_when_seasonal_order_is_None_and_order_is_not(order):
    """
    Test that `seasonal_order=None` with an `order` raises an error: the
    automatic selection chooses both orders and cannot keep a given one.
    """
    msg = re.escape(
        f"`seasonal_order=None` requires `order=None`, got `order` = "
        f"{order}. The automatic selection chooses both orders and "
        f"cannot keep a given `order`. Set `order=None` to select both, or "
        f"pass `seasonal_order` as a tuple to fit a given model."
    )
    with pytest.raises(ValueError, match=msg):
        Arima(order=order, seasonal_order=None, m=12)


@pytest.mark.parametrize(
    "seasonal_order",
    [(1, 1, 0), (0, 1, 0), (0, 0, 2)],
    ids=lambda v: f"seasonal_order: {v}"
)
def test_arima_init_ValueError_when_order_is_None_and_seasonal_order_is_not_zero(
    seasonal_order
):
    """
    Test that `order=None` with a `seasonal_order` other than (0, 0, 0)
    raises an error: the automatic selection cannot keep a given seasonal
    order.
    """
    msg = re.escape(
        f"`order=None` cannot be combined with `seasonal_order` = "
        f"{seasonal_order}. The automatic selection chooses both "
        f"orders and cannot keep a given `seasonal_order`. Set "
        f"`seasonal_order=None` to select both, or pass `order` as a tuple "
        f"to fit a given model."
    )
    with pytest.raises(ValueError, match=msg):
        Arima(order=None, seasonal_order=seasonal_order, m=12)


def test_arima_init_ValueError_when_order_is_None_m_greater_than_1_and_no_seasonal_order():
    """
    Test that `order=None` with the default `seasonal_order=(0, 0, 0)` and
    `m` greater than 1 raises an error: it would select a non-seasonal model
    for data declared as seasonal.
    """
    msg = re.escape(
        "`order=None` with `seasonal_order` = (0, 0, 0) selects a "
        "non-seasonal model, but `m` = 12 indicates seasonal data. Set "
        "`seasonal_order=None` to select the seasonal order too, or "
        "`m=1` for a non-seasonal model."
    )
    with pytest.raises(ValueError, match=msg):
        Arima(order=None, m=12)


@pytest.mark.parametrize(
    "order, seasonal_order, m, is_auto",
    [(None, (0, 0, 0), 1, True),
     (None, None, 1, True),
     (None, None, 12, True),
     ((2, 1, 1), (1, 1, 1), 12, False),
     ((2, 1, 1), (0, 0, 0), 12, False)],
    ids=lambda v: f"{v}"
)
def test_arima_init_accepted_combinations_of_order_seasonal_order_and_m(
    order, seasonal_order, m, is_auto
):
    """
    Test the combinations that are accepted: the two forms of the automatic
    selection (non-seasonal and with the seasonal order) and a given model.
    """
    model = Arima(order=order, seasonal_order=seasonal_order, m=m)

    assert model.order == order
    assert model.seasonal_order == seasonal_order
    assert model.is_auto is is_auto


def test_arima_init_default_params():
    """
    Test Arima initialization with default parameters.
    """
    model = Arima(order=(0, 0, 0), seasonal_order=(0, 0, 0), m=1)
    
    assert model.order == (0, 0, 0)
    assert model.seasonal_order == (0, 0, 0)
    assert model.m == 1
    assert model.fit_intercept is True
    assert model.enforce_stationarity is True
    assert model.method == "CSS-ML"
    assert model.n_cond is None
    assert model.optim_method == "BFGS"
    assert model.optim_kwargs == {'maxiter': 1000}
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
        fit_intercept=False,
        enforce_stationarity=False,
        method="ML",
        n_cond=10,
        optim_method="L-BFGS-B",
        optim_kwargs={'maxiter': 100},
        kappa=1e5
    )
    
    assert model.order == (1, 1, 1)
    assert model.seasonal_order == (1, 1, 1)
    assert model.m == 12
    assert model.fit_intercept is False
    assert model.enforce_stationarity is False
    assert model.method == "ML"
    assert model.n_cond == 10
    assert model.optim_method == "L-BFGS-B"
    assert model.optim_kwargs == {'maxiter': 100}
    assert model.kappa == 1e5


def test_arima_init_all_attributes_before_fitting():
    """
    Test that all attributes have expected initial values before fitting.
    """
    model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
    
    # Parameter attributes are set
    assert model.order == (1, 0, 1)
    assert model.seasonal_order == (0, 0, 0)
    assert model.m == 1
    assert model.is_memory_reduced is False


def test_arima_repr_non_seasonal():
    """
    Test __repr__ for non-seasonal ARIMA model.
    """
    model = Arima(order=(2, 1, 1), seasonal_order=(0, 0, 0), m=1)
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
