# Unit test Arar
# ==============================================================================
import math
import numpy as np
import pytest
from ..._arar import Arar


def ar1_series(n=80, phi=0.7, sigma=1.0, seed=123):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def test_estimator_fit_and_attributes():
    y = ar1_series(100)
    est = Arar()
    est.fit(y)

    assert hasattr(est, "model_")
    assert est.y_.shape == y.shape
    assert est.coef_.shape == (4,)
    assert isinstance(est.lags_, tuple) and len(est.lags_) == 4
    assert np.isscalar(est.sigma2_) and est.sigma2_ >= 1e-12
    assert est.psi_.ndim == 1 and est.psi_.size >= 1
    assert isinstance(est.sbar_, float)
    assert est.n_features_in_ == 1

    assert est.fitted_values_.shape == y.shape
    assert est.residuals_in_.shape == y.shape


def test_estimator_predict_and_intervals():
    y = ar1_series(120)
    est = Arar()
    est.fit(y)

    mean = est.predict(steps=8)
    assert mean.shape == (8,)

    df = est.predict_interval(steps=5, level=(50, 80, 95), as_frame=True)
    assert list(df.columns) == ["mean", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95"]
    assert df.shape == (5, 1 + 2 * 3)

    raw = est.predict_interval(steps=3, level=(90,), as_frame=False)
    assert raw["mean"].shape == (3,)
    assert raw["upper"].shape == (3, 1)
    assert raw["lower"].shape == (3, 1)
    assert raw["level"] == [90]


def test_estimator_invalid_steps_and_unfitted():
    est = Arar()
    with pytest.raises(Exception):
        est.predict(steps=1)

    y = ar1_series(50)
    est.fit(y)
    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=-2)
    with pytest.raises(ValueError, match="positive integer"):
        est.predict(steps=1.5)


def test_estimator_residuals_and_fitted_helpers():
    y = ar1_series(70)
    est = Arar().fit(y)
    r = est.residuals_()
    f = est.fitted_()
    assert r.shape == y.shape
    assert f.shape == y.shape
    mask = ~np.isnan(f)
    assert np.allclose(r[mask], y[mask] - f[mask])


def test_estimator_summary_and_score(capsys):
    y = ar1_series(100)
    est = Arar().fit(y)
    est.summary()
    out = capsys.readouterr().out
    assert "ARAR Model Summary" in out

    score = est.score()
    assert np.isfinite(score) or np.isnan(score)


def test_estimator_safe_false_too_short_raises():
    y = np.array([1.0]) 
    est = Arar(safe=False)
    with pytest.raises(ValueError, match="Series too short"):
        est.fit(y)


def test_arar_with_explicit_params_propagated_to_estimator():
    y = ar1_series(60)
    est = Arar(max_ar_depth=8, max_lag=15, safe=True).fit(y)
    assert isinstance(est.max_ar_depth, int)
    assert isinstance(est.max_lag, int)
    assert est.max_ar_depth >= 4
    assert est.max_lag >= est.max_ar_depth

