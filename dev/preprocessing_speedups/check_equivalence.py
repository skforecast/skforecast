"""
Equivalence of the prototypes (P1 encoder, P2 scaler shortcut) against the
current implementation. Both must be bit-identical, so the checks use
`np.testing.assert_array_equal`, not tolerances.

python check_equivalence.py            # all cases, size T (seconds)
python check_equivalence.py --size S   # same cases with more data

Run it with pandas 3 and with pandas 2.3 (separate environment).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import SIZES, env_report, make_data, make_forecaster, split_train_future  # noqa: E402
from prototypes import use_fast_encoder, use_fast_scaler  # noqa: E402

from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries  # noqa: E402

warnings.filterwarnings("ignore")


def assert_same(a, b, what):
    a = a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)
    b = b.to_numpy() if hasattr(b, "to_numpy") else np.asarray(b)
    np.testing.assert_array_equal(a, b, err_msg=what)


def run_multiseries(case, s_tr, e_tr, e_fut, est="ridge", enc="ordinal", fast_scaler=True):
    f0, f1 = make_forecaster(est, enc), make_forecaster(est, enc)
    X0, y0 = f0.create_train_X_y(s_tr, exog=e_tr)
    with use_fast_encoder(f1):
        X1, y1 = f1.create_train_X_y(s_tr, exog=e_tr)
    assert list(X0.columns) == list(X1.columns), case
    assert (X0.dtypes == X1.dtypes).all(), case
    assert_same(X0, X1, f"{case}: X_train")
    assert_same(y0, y1, f"{case}: y_train")

    f0, f1 = make_forecaster(est, enc), make_forecaster(est, enc)
    f0.fit(s_tr, exog=e_tr, store_in_sample_residuals=True)
    with use_fast_encoder(f1):
        f1.fit(s_tr, exog=e_tr, store_in_sample_residuals=True)
    if f0.exog_in_ and f0.categorical_features_names_in_:
        for c0, c1 in zip(f0.categorical_encoder.categories_, f1.categorical_encoder.categories_):
            # NOTE: Compared as text: in object arrays NaN != NaN, and `categories_`
            # ends with NaN when the training data has missing values.
            assert [str(v) for v in c0] == [str(v) for v in c1], f"{case}: categories_"
            assert c0.dtype == c1.dtype, f"{case}: dtype of categories_ {c0.dtype} != {c1.dtype}"
    ctx = use_fast_scaler() if fast_scaler else warnings.catch_warnings()
    with ctx:
        for steps in (24, 168):
            assert_same(f0.predict(steps=steps, exog=e_fut), f1.predict(steps=steps, exog=e_fut),
                        f"{case}: predict {steps}")
        assert_same(
            f0.predict_interval(steps=24, exog=e_fut, method="conformal").iloc[:, 1:],
            f1.predict_interval(steps=24, exog=e_fut, method="conformal").iloc[:, 1:],
            f"{case}: conformal",
        )
        assert_same(
            f0.predict_interval(steps=24, exog=e_fut, method="bootstrapping", n_boot=50).iloc[:, 1:],
            f1.predict_interval(steps=24, exog=e_fut, method="bootstrapping", n_boot=50).iloc[:, 1:],
            f"{case}: bootstrapping",
        )
    print(f"OK {case} ({est}, {enc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="T")
    args = ap.parse_args()
    print(env_report())

    s, e = make_data(**SIZES[args.size])
    s_tr, e_tr, e_fut = split_train_future(s, e, 168)
    keys = list(s_tr)

    # 1. Base cases
    for est in ("ridge", "lgbm"):
        for enc in ("ordinal", "onehot"):
            run_multiseries("base", s_tr, e_tr, e_fut, est, enc)

    # 2. Missing categorical values in training and in the future exog
    # (LightGBM in the cases with NaN predictors: Ridge does not accept them)
    e2 = {k: v.copy() for k, v in e_tr.items()}
    ef2 = {k: v.copy() for k, v in e_fut.items()}
    e2[keys[0]].iloc[100:150, e2[keys[0]].columns.get_loc("cat_promo")] = np.nan
    ef2[keys[1]].iloc[:10, ef2[keys[1]].columns.get_loc("cat_common")] = np.nan
    run_multiseries("NaN categories", s_tr, e2, ef2, est="lgbm")

    # 3. Unseen category in predict (declared in the dtype, absent in training)
    ef3 = {k: v.copy() for k, v in e_fut.items()}
    for k in ef3:
        col = ef3[k]["cat_promo"].cat.add_categories(["cat_promo_new"])
        col.iloc[:5] = "cat_promo_new"
        ef3[k]["cat_promo"] = col
    e3 = {k: v.copy() for k, v in e_tr.items()}
    for k in e3:
        e3[k]["cat_promo"] = e3[k]["cat_promo"].cat.add_categories(["cat_promo_new"])
    run_multiseries("unseen category", s_tr, e3, ef3, est="lgbm")

    # 4. Same categories in another order in the future exog
    ef4 = {k: v.copy() for k, v in e_fut.items()}
    for k in ef4:
        cats = list(ef4[k]["cat_store"].cat.categories)[::-1]
        ef4[k]["cat_store"] = ef4[k]["cat_store"].cat.reorder_categories(cats)
    run_multiseries("reordered categories", s_tr, e_tr, ef4)

    # 5. Ordered categorical and integer categories
    e5 = {k: v.copy() for k, v in e_tr.items()}
    ef5 = {k: v.copy() for k, v in e_fut.items()}
    for d in (e5, ef5):
        for k in d:
            d[k]["cat_common"] = d[k]["cat_common"].cat.as_ordered()
            d[k]["cat_store"] = d[k]["cat_store"].cat.rename_categories(
                lambda c: int(c.split("_")[-1])
            )
    run_multiseries("ordered + integer categories", s_tr, e5, ef5)

    # 6. Exog missing in some series and exog starting later than the series
    e6 = dict(e_tr)
    e6[keys[0]] = None
    e6[keys[1]] = e_tr[keys[1]].iloc[200:]
    run_multiseries("missing exog", s_tr, e6, e_fut, est="lgbm")

    # 7. Text columns (fallback to scikit-learn, must not change anything)
    e7 = {k: v.assign(cat_store=v["cat_store"].astype(str)) for k, v in e_tr.items()}
    ef7 = {k: v.assign(cat_store=v["cat_store"].astype(str)) for k, v in e_fut.items()}
    run_multiseries("text column (fallback)", s_tr, e7, ef7)

    # 8. Backtesting with refit, both prototypes
    n_span = max(len(v) for v in s.values())
    cv = TimeSeriesFold(steps=24, initial_train_size=n_span - 72, refit=True, verbose=False)
    m0, p0 = backtesting_forecaster_multiseries(
        make_forecaster(), series=s, exog=e, cv=cv, metric="mean_absolute_error",
        n_jobs=1, show_progress=False, suppress_warnings=True)
    f1 = make_forecaster()
    with use_fast_encoder(f1), use_fast_scaler():
        m1, p1 = backtesting_forecaster_multiseries(
            f1, series=s, exog=e, cv=cv, metric="mean_absolute_error",
            n_jobs=1, show_progress=False, suppress_warnings=True)
    assert_same(p0["pred"], p1["pred"], "backtesting refit=True")
    print("OK backtesting refit=True")

    # 9. ForecasterRecursive (single series) with the same exog and scaler
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from skforecast.recursive import ForecasterRecursive

    k = keys[0]
    fr = [ForecasterRecursive(Ridge(), lags=48, transformer_y=StandardScaler()) for _ in range(2)]
    fr[0].fit(s_tr[k], exog=e_tr[k])
    with use_fast_encoder(fr[1]):
        fr[1].fit(s_tr[k], exog=e_tr[k])
    with use_fast_scaler():
        assert_same(fr[0].predict(steps=168, exog=e_fut[k]), fr[1].predict(steps=168, exog=e_fut[k]),
                    "ForecasterRecursive predict")
    print("OK ForecasterRecursive")
    print("ALL CHECKS PASSED (bit-identical)")


if __name__ == "__main__":
    main()
