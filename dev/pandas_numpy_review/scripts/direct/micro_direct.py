import sys, warnings, timeit
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.base import clone
from skforecast.direct import ForecasterDirect
from skforecast.utils import check_predict_input, expand_index, input_to_frame, transform_dataframe
from skforecast.utils.utils import check_exog, check_exog_dtypes

warnings.filterwarnings("ignore")
N = 2000
idx = pd.date_range("2020-01-01", periods=N + 24, freq="h")
rng = np.random.default_rng(0)
y = pd.Series(rng.normal(size=N + 24), index=idx, name="y")
exnum = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["e1", "e2", "e3"])
excat = pd.DataFrame({"cat": pd.Categorical(np.array(list("abcdefg"))[idx.dayofweek])}, index=idx)
excat_obj = excat.astype({"cat": str})


def t(label, fn, n=2000):
    r = min(timeit.repeat(fn, number=n, repeat=5)) / n
    print(f"{label:70s} {r*1e6:8.1f} us")
    return r


for name, ex in [("num3", exnum), ("cat1(category)", excat), ("cat1(str)", excat_obj)]:
    print(f"\n== {name}")
    f = ForecasterDirect(Ridge(), steps=24, lags=24, n_jobs=1)
    f.fit(y.iloc[:N], exog=ex.iloc[:N])
    ef = ex.iloc[N:]
    lw = f.last_window_
    t("predict total", lambda: f.predict(exog=ef))
    t("predict check_inputs=False", lambda: f.predict(exog=ef, check_inputs=False))
    t("check_predict_input", lambda: check_predict_input(
        forecaster_name="ForecasterDirect", steps=list(range(1, 25)), is_fitted=True, exog_in_=True,
        index_type_=f.index_type_, index_freq_=f.index_freq_, window_size=f.window_size,
        last_window=lw, exog=ef, exog_names_in_=f.exog_names_in_, max_step=24))
    t("input_to_frame(exog)", lambda: input_to_frame(ef, "exog"))
    t("exog.copy(deep=False)", lambda: ef.copy(deep=False))
    if name.startswith("cat"):
        enc = f.categorical_encoder
        t("OrdinalEncoder.transform(exog[[cat]]) (pandas output)", lambda: enc.transform(ef[["cat"]]))
        def assign():
            e2 = ef.copy(deep=False)
            e2[["cat"]] = enc.transform(e2[["cat"]])
            return e2
        t("copy + encoder + assign (as in _create_predict_inputs)", assign)
        cats = enc.categories_[0]
        cat_index = pd.Index(cats)
        def np_codes():
            v = ef["cat"].to_numpy()
            codes = cat_index.get_indexer(v).astype(float)
            codes[codes < 0] = np.nan
            return codes
        t("alternative: Index.get_indexer on values -> float codes", np_codes)
        a = assign().to_numpy(); b = np_codes()
        print("   same values:", np.array_equal(a[:, 0], b, equal_nan=True))
        e2 = assign()
        t("exog.to_numpy() after encode", lambda: e2.to_numpy())
        t("dtypes.tolist() compare", lambda: e2.dtypes.tolist() == list(f.exog_dtypes_out_.values()))
    else:
        t("exog.to_numpy()", lambda: ef.to_numpy())
        t("dtypes.tolist() compare", lambda: ef.dtypes.tolist() == list(f.exog_dtypes_out_.values()))
    t("expand_index(lw.index, 24)", lambda: expand_index(lw.index, 24))
    p = np.zeros(24); pidx = expand_index(lw.index, 24)
    t("pd.Series(pred, index, name) output", lambda: pd.Series(p, index=pidx, name="pred"))
    t("last_window.to_numpy(copy=True).ravel()", lambda: lw.to_numpy(copy=True).ravel())

print("\n== fit pieces")
t("clone(Ridge())", lambda: clone(Ridge()), n=2000)
f = ForecasterDirect(Ridge(), steps=24, lags=24, n_jobs=1)
t("fit num3", lambda: f.fit(y.iloc[:N], exog=exnum.iloc[:N]), n=10)
for nm, exx in [("none", None), ("num3", exnum), ("cat1", excat)]:
    f = ForecasterDirect(Ridge(), steps=24, lags=24, n_jobs=1)
    t(f"_create_train_X_y {nm} (not fitted)", lambda: f._create_train_X_y(y.iloc[:N], exog=None if exx is None else exx.iloc[:N]), n=50)
    f.fit(y.iloc[:N], exog=None if exx is None else exx.iloc[:N])
    t(f"fit {nm}", lambda: f.fit(y.iloc[:N], exog=None if exx is None else exx.iloc[:N]), n=5)

print("\n== predict_interval pieces (num3)")
f = ForecasterDirect(Ridge(), steps=24, lags=24, n_jobs=1)
f.fit(y.iloc[:N], exog=exnum.iloc[:N], store_in_sample_residuals=True)
ef = exnum.iloc[N:]
bp = f.predict_bootstrapping(exog=ef, n_boot=250)
t("predict_interval bootstrapping", lambda: f.predict_interval(exog=ef, n_boot=250), n=300)
t("predict_bootstrapping", lambda: f.predict_bootstrapping(exog=ef, n_boot=250), n=300)
t("predict(check_inputs=False)", lambda: f.predict(exog=ef, check_inputs=False), n=300)
t("boot.quantile(q=[.025,.975],axis=1).T", lambda: bp.quantile(q=[0.025, 0.975], axis=1).transpose(), n=300)
arr = bp.to_numpy()
t("np.quantile(arr, [.025,.975], axis=1).T", lambda: np.quantile(arr, [0.025, 0.975], axis=1).T, n=300)
q1 = bp.quantile(q=[0.025, 0.975], axis=1).transpose().to_numpy()
q2 = np.quantile(arr, [0.025, 0.975], axis=1).T
print("   identical:", np.array_equal(q1, q2), np.abs(q1 - q2).max())
pr = f.predict(exog=ef, check_inputs=False)
qi = bp.quantile(q=[0.025, 0.975], axis=1).transpose(); qi.columns = ["lower_bound", "upper_bound"]
t("pd.concat((pred, interval), axis=1)", lambda: pd.concat((pr, qi), axis=1), n=300)
t("pd.DataFrame from numpy (alt)", lambda: pd.DataFrame(np.column_stack([pr.to_numpy(), q2]), index=pr.index, columns=["pred", "lower_bound", "upper_bound"]), n=300)
