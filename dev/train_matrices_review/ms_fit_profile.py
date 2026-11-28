import cProfile, pstats, warnings, io
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
n_series, n_days = 5000, 300
idx = pd.date_range('2020-01-01', periods=n_days, freq='D')
wide = pd.DataFrame(rng.poisson(3, (n_days, n_series)).astype(float), index=idx,
                    columns=[f's{i}' for i in range(n_series)])
series = {c: wide[c].rename(c) for c in wide.columns}
f = ForecasterRecursiveMultiSeries(estimator=LGBMRegressor(n_estimators=200, verbose=-1, random_state=0),
                                   lags=list(range(1, 15)) + [21, 28])
print('probabilistic mode:', f._probabilistic_mode)
pr = cProfile.Profile(); pr.enable(); f.fit(series=series); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(35); print(s.getvalue()[:9000])
