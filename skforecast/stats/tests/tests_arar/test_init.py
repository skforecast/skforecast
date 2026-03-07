# Unit test __init__ method - Arar
# ==============================================================================
from ..._arar import Arar


def test_arar_init_default_params():
    """
    Test Arar initialization with default parameters.
    """
    model = Arar()
    
    assert model.max_ar_depth is None
    assert model.max_lag is None
    assert model.safe is True


def test_arar_init_with_explicit_params():
    """
    Test Arar initialization with explicit parameters.
    """
    model = Arar(max_ar_depth=8, max_lag=15, safe=False)
    
    assert model.max_ar_depth == 8
    assert model.max_lag == 15
    assert model.safe is False


def test_arar_init_all_attributes():
    """
    Test that all internal attributes are properly initialized to None or defaults.
    """
    model = Arar()
    
    # Parameter attributes
    assert model.max_ar_depth is None
    assert model.max_lag is None
    assert model.safe is True
    
    # Model state attributes (should be None before fitting)
    assert model.model_ is None
    assert model.n_features_in_ is None
    assert model.y_train_ is None
    assert model.coef_ is None
    assert model.lags_ is None
    assert model.sigma2_ is None
    assert model.psi_ is None
    assert model.sbar_ is None
    
    # Exogenous model attributes (should be None before fitting)
    assert model.exog_model_ is None
    assert model.coef_exog_ is None
    assert model.n_exog_features_in_ is None
    
    # Memory management attribute
    assert model.is_memory_reduced is False
