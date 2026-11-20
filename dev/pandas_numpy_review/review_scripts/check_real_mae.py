"""
Fidelity of `real_data.py` to the articles: the metrics of the current code
next to the values printed in each article (LightGBM with its default threads,
as in the articles; skforecast 0.27.0, scikit-learn 1.9.1, LightGBM 4.7.0 and,
in py44 and py39, pandas 2.3.3 there).

Usage: REAL_DATA_DIR=... python check_real_mae.py
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import warnings

from real_data import CASES
from skforecast.model_selection import backtesting_forecaster, backtesting_forecaster_multiseries

warnings.simplefilter("ignore")
ARTICLE = {
    "store": {"item_1": 5.512613, "item_2": 9.218515, "item_3": 7.310630},
    "demo": {"id_1000": 150.894697, "id_1001": 863.158413, "id_1003": 265.134384,
             "id_1004": 921.148294},
    "bdg2": {"Bear_assembly_Angel": 1225.568428, "Bear_assembly_Beatrice": 238.254156,
             "Bear_assembly_Diana": 32.173603, "average": 307},
    "bike": {"": 49.442977},
}
for name, ref in ARTICLE.items():
    c = CASES[name]()
    if c["kind"] == "ms":
        m, _ = backtesting_forecaster_multiseries(
            forecaster=c["forecaster"](), series=c["series"], exog=c["exog"], cv=c["cv"],
            show_progress=False, **c["kwargs"])
        got = dict(zip(m["levels"], m["mean_absolute_error"]))
    else:
        m, _ = backtesting_forecaster(
            forecaster=c["forecaster"](), y=c["y"], exog=c["exog"], cv=c["cv"],
            show_progress=False, **c["kwargs"])
        got = {"": m["mean_absolute_error"].iloc[0]}
    for level, value in ref.items():
        print(f"{name:<6} {level:<24} article {value:12.4f}   here {got[level]:12.4f}   "
              f"diff {got[level] - value:+.4g} ({(got[level] / value - 1) * 100:+.2f} %)")
print(_env.versions())
