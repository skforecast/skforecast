# Unit test __init__ method - Ets
# ==============================================================================
import numpy as np
import pytest
from ..._ets import Ets


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
