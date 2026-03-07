# Unit test __init__ method - Ets
# ==============================================================================
import re
import pytest
from ..._ets import Ets


@pytest.mark.parametrize(
    'm', 
    [0, -1, 1.5, "12"],
    ids=lambda x: f"m={x}"
)
def test_ets_init_ValueError_when_m_invalid(m):
    """Test Ets raises ValueError when m is not a positive integer >= 1."""
    err_msg = re.escape(
        f"`m` must be a positive integer greater than or equal to 1."
        f" Got {m}."
    )
    with pytest.raises(ValueError, match=err_msg):
        Ets(m=m)


def test_ets_init_default_params():
    """Test Ets initialization with default parameters"""
    est = Ets()
    
    assert est.m == 1
    assert est.model == "ZZZ"
    assert est.damped is None
    assert est.alpha is None
    assert est.beta is None
    assert est.gamma is None
    assert est.phi is None


def test_ets_init_with_explicit_params():
    """Test Ets initialization with explicit parameters"""
    est = Ets(m=12, model="AAA", damped=True, alpha=0.3, beta=0.1)
    
    assert est.m == 12
    assert est.model == "AAA"
    assert est.damped is True
    assert est.alpha == 0.3
    assert est.beta == 0.1


def test_ets_init_with_bounds():
    """Test Ets initialization with bounds parameter"""
    est = Ets(m=1, model="ANN", bounds="usual")
    
    assert est.bounds == "usual"
