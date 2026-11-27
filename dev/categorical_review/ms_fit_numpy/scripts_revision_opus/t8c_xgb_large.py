"""T8c: XGBoost with a user-configured categorical column (feature_types 'c')
and categorical_features=None: fit time and peak memory vs the magnitude of
the integer categories. Usage: python t8c_xgb_large.py <max_label>"""
import sys, os, time, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
from skforecast.recursive import ForecasterRecursiveMultiSeries
from xgboost import XGBRegressor

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
def peak_mb():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20

m = int(sys.argv[1])
series, exog_train, exog_pred = make_case("df", ["f", "ci"], effect_col="ci", exog_kind="df")
labels = np.array([m // 3, 2 * m // 3, m])
ex = exog_train.assign(ci=pd.Categorical(labels[exog_train["ci"].cat.codes.to_numpy()]))
f = ForecasterRecursiveMultiSeries(
    XGBRegressor(n_estimators=20, random_state=1, enable_categorical=True, feature_types=["q"] * 5 + ["c"]),
    lags=3, encoding="ordinal", categorical_features=None)
t = time.perf_counter()
st, r, w = run(lambda: f.fit(series=series, exog=ex))
print(f"max label {m}: fit={st} {r[:100] if st=='err' else ''} {time.perf_counter()-t:.2f}s peak={peak_mb():.0f}MB", flush=True)
