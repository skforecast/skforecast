import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from skforecast.recursive import ForecasterRecursive
from skforecast.utils import check_predict_input, expand_index
from skforecast.utils.utils import _build_predict_function

for exog_kind in [None, "num", "cat"]:
    y, exog = make_data(exog_kind=exog_kind)
    train_n = N - STEPS
    for name, est in estimators().items():
        f = ForecasterRecursive(estimator=est, lags=LAGS)
        f.fit(y=y.iloc[:train_n], exog=None if exog is None else exog.iloc[:train_n])
        ex_test = None if exog is None else exog.iloc[train_n:]
        lw = y.iloc[train_n - LAGS:train_n]
        t_pred = timeit(lambda: f.predict(steps=STEPS, last_window=lw, exog=ex_test), n=50)
        t_pred_nc = timeit(lambda: f.predict(steps=STEPS, last_window=lw, exog=ex_test, check_inputs=False), n=50)
        t_cpi = timeit(lambda: check_predict_input(
            forecaster_name="ForecasterRecursive", steps=STEPS, is_fitted=True,
            exog_in_=f.exog_in_, index_type_=f.index_type_, index_freq_=f.index_freq_,
            window_size=f.window_size, last_window=lw, exog=ex_test,
            exog_names_in_=f.exog_names_in_), n=50)
        t_inputs = timeit(lambda: f._create_predict_inputs(steps=STEPS, last_window=lw, exog=ex_test, check_inputs=False), n=50)
        lwv, exv, calv, pidx, st, _ = f._create_predict_inputs(steps=STEPS, last_window=lw, exog=ex_test, check_inputs=False)
        t_rec = timeit(lambda: f._recursive_predict(steps=STEPS, last_window_values=lwv, exog_values=exv, calendar_values=calv, prediction_index=pidx), n=50)
        pfn = _build_predict_function(f.estimator, f)
        X1 = np.zeros((1, len(f.X_train_features_names_out_)))
        t_est1 = timeit(lambda: pfn(X1), n=200)
        t_series = timeit(lambda: pd.Series(np.zeros(STEPS), index=pidx, name="pred"), n=200)
        t_expand = timeit(lambda: expand_index(lw.index, STEPS), n=200)
        print(f"exog={exog_kind} {name}: predict={t_pred*1e3:.3f}ms  predict(check_inputs=False)={t_pred_nc*1e3:.3f}ms  "
              f"check_predict_input={t_cpi*1e3:.3f}ms  _create_predict_inputs(nocheck)={t_inputs*1e3:.3f}ms  "
              f"_recursive_predict={t_rec*1e3:.3f}ms  (est 1-row x{STEPS}={t_est1*STEPS*1e3:.3f}ms)  "
              f"pd.Series out={t_series*1e3:.3f}ms expand_index={t_expand*1e3:.3f}ms")
        if exog is not None:
            # exog sub-steps
            from skforecast.utils import input_to_frame, transform_dataframe
            def exog_part():
                e = input_to_frame(ex_test, 'exog')
                if f.categorical_features_names_in_:
                    e = e.copy(deep=False)
                    e[f.categorical_features_names_in_] = f.categorical_encoder.transform(e[f.categorical_features_names_in_])
                v = e.to_numpy()
                e.dtypes.tolist() == list(f.exog_dtypes_out_.values())
                return v
            t_ex = timeit(exog_part, n=50)
            if f.categorical_features_names_in_:
                t_enc = timeit(lambda: f.categorical_encoder.transform(ex_test[f.categorical_features_names_in_]), n=50)
                codes = ex_test["dow"].cat.codes.to_numpy()
                t_np = timeit(lambda: np.where(codes >= 0, codes, np.nan).astype(float), n=200)
                print(f"   exog block={t_ex*1e3:.3f}ms  OrdinalEncoder.transform(pandas out)={t_enc*1e3:.3f}ms  numpy codes alt={t_np*1e3:.4f}ms")
            else:
                print(f"   exog block={t_ex*1e3:.3f}ms")

        tot, out, st = profile_breakdown(lambda: f.predict(steps=STEPS, last_window=lw, exog=ex_test), n_rep=200)
        print("   profile predict x200:", fmt_breakdown(tot, out))
