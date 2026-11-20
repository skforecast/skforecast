import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from skforecast.recursive import ForecasterRecursive

exog_kind = sys.argv[1] if len(sys.argv) > 1 else "num"
exog_kind = None if exog_kind == "none" else exog_kind
y, exog = make_data(exog_kind=exog_kind)
train_n = N - STEPS
f = ForecasterRecursive(estimator=Ridge(), lags=LAGS)
f.fit(y=y.iloc[:train_n], exog=None if exog is None else exog.iloc[:train_n])
ex_test = None if exog is None else exog.iloc[train_n:]
lw = y.iloc[train_n - LAGS:train_n]
for i in range(3):
    t = timeit(lambda: f.predict(steps=STEPS, last_window=lw, exog=ex_test), n=300)
    t2 = timeit(lambda: f.predict(steps=STEPS, last_window=lw, exog=ex_test, check_inputs=False), n=300)
    print(f"predict {t*1e3:.3f}ms nocheck {t2*1e3:.3f}ms")
tot, out, st = profile_breakdown(lambda: f.predict(steps=STEPS, last_window=lw, exog=ex_test), n_rep=500)
print(fmt_breakdown(tot, out))
st.sort_stats(sys.argv[2] if len(sys.argv)>2 else "tottime").print_stats(int(sys.argv[3]) if len(sys.argv)>3 else 25)
